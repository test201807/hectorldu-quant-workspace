"""CLI entry point: python -m strategylab.cli run|wfo|mc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .backtest_engine import run_engine, trades_to_dataframe
from .config import StrategyLabConfig
from .data_loader import make_synthetic_bars
from .metrics import compute_kpis
from .monte_carlo import run_all_mc
from .report import generate_full_report, snapshot_json
from .signals_trend_v2 import add_trend_features, compute_regime_gate, confirm_signals
from .wfo import run_wfo


def _load_config(config_path: str | None) -> StrategyLabConfig:
    if config_path:
        return StrategyLabConfig.from_yaml(config_path)
    return StrategyLabConfig()


def _ensure_output_dir(out: str | None) -> Path:
    p = Path(out) if out else Path("outputs/strategylab_cli")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _write_run_manifest(cfg: StrategyLabConfig, out_dir: Path, command: str) -> Path:
    """Write a run manifest with full config snapshot and metadata."""
    import hashlib
    from datetime import datetime, timezone

    manifest = {
        "command": command,
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": hashlib.sha256(cfg.snapshot_json().encode()).hexdigest()[:12],
        "config": cfg.to_dict(),
    }
    return snapshot_json(manifest, out_dir, "run_manifest")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single backtest (trend strategy on synthetic data if no file)."""
    cfg = _load_config(args.config)
    out_dir = _ensure_output_dir(args.output)
    _write_run_manifest(cfg, out_dir, "run")

    if args.data:
        df = pl.read_parquet(args.data)
        if "time_utc" in df.columns:
            df = df.with_columns(
                pl.col("time_utc").cast(pl.Datetime("us", "UTC"), strict=False)
            )
        symbol = args.symbol or "UNKNOWN"
    else:
        print("[cli] No data file provided, using synthetic bars.")
        df = make_synthetic_bars(symbol="SYNTH", n_bars=20_000, seed=42)
        symbol = "SYNTH"

    # Add features
    df = add_trend_features(df)

    # Compute ATR if missing
    if "atr_price" not in df.columns:
        df = df.with_columns(
            (pl.col("close") * 0.005).alias("atr_price")
        )

    # Add ER/mom/vol proxies for gate
    if "er_288" not in df.columns:
        df = df.with_columns([
            pl.lit(0.5).alias("er_288"),
            pl.lit(0.0).alias("mom_bps_288"),
            pl.lit(50.0).alias("vol_bps_288"),
        ])

    # Regime gate
    gate_long, gate_short, gate_params = compute_regime_gate(df)

    # Confirm signals
    sig_long = confirm_signals(gate_long, min_bars=cfg.engine.entry_confirm_bars)
    sig_short = confirm_signals(gate_short, min_bars=cfg.engine.entry_confirm_bars)

    # WFO fold bounds (use full data as single IS+OOS)
    t_list = df.get_column("time_utc").to_list()
    t_min, t_max = min(t_list), max(t_list)
    mid = t_list[len(t_list) // 2]

    trades = run_engine(
        df=df,
        fold_id="F00",
        is_start=t_min,
        is_end=mid,
        oos_start=mid,
        oos_end=t_max,
        signal_long=sig_long,
        signal_short=sig_short,
        engine_cfg=cfg.engine,
        costs_cfg=cfg.costs,
        risk_cfg=cfg.risk,
        symbol=symbol,
    )

    trades_df = trades_to_dataframe(trades)
    print(f"[cli] {len(trades)} trades generated.")

    kpis = compute_kpis(trades_df) if not trades_df.is_empty() else {}
    print(f"[cli] KPIs: {json.dumps(kpis, indent=2, default=str)}")

    paths = generate_full_report(trades_df, out_dir, symbol=symbol, strategy="TREND_v2_cli")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("[cli] Run complete.")


def cmd_wfo(args: argparse.Namespace) -> None:
    """Run walk-forward optimization."""
    cfg = _load_config(args.config)
    out_dir = _ensure_output_dir(args.output)
    _write_run_manifest(cfg, out_dir, "wfo")

    if args.data:
        df = pl.read_parquet(args.data)
        if "time_utc" in df.columns:
            df = df.with_columns(
                pl.col("time_utc").cast(pl.Datetime("us", "UTC"), strict=False)
            )
        symbol = args.symbol or "UNKNOWN"
    else:
        print("[wfo] No data file, using synthetic bars.")
        df = make_synthetic_bars(symbol="SYNTH", n_bars=50_000, seed=42)
        symbol = "SYNTH"

    # Features + gate (simplified)
    df = add_trend_features(df)
    if "atr_price" not in df.columns:
        df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))
    if "er_288" not in df.columns:
        df = df.with_columns([
            pl.lit(0.5).alias("er_288"),
            pl.lit(0.0).alias("mom_bps_288"),
            pl.lit(50.0).alias("vol_bps_288"),
        ])

    gate_long, gate_short, _ = compute_regime_gate(df)
    sig_long = confirm_signals(gate_long, min_bars=cfg.engine.entry_confirm_bars)
    sig_short = confirm_signals(gate_short, min_bars=cfg.engine.entry_confirm_bars)

    # Use param_grid from config (not hardcoded)
    param_grid = cfg.wfo.param_grid

    result = run_wfo(
        df=df,
        signal_long=sig_long,
        signal_short=sig_short,
        param_grid=param_grid,
        costs_cfg=cfg.costs,
        risk_cfg=cfg.risk,
        symbol=symbol,
        is_months=cfg.wfo.is_months,
        oos_months=cfg.wfo.oos_months,
        step_months=cfg.wfo.step_months,
        embargo_days=cfg.wfo.embargo_days,
        max_combos=cfg.wfo.max_combos_per_symbol,
    )

    print(f"[wfo] {len(result.folds)} folds, {len(result.best_per_fold)} with results.")
    print(f"[wfo] OOS trades: {result.oos_trades.height}")
    if result.oos_kpis:
        print(f"[wfo] OOS KPIs: {json.dumps(result.oos_kpis, indent=2, default=str)}")

    if not result.oos_trades.is_empty():
        generate_full_report(result.oos_trades, out_dir, symbol=symbol, strategy="WFO_OOS", prefix="wfo")

    # Snapshot
    snap = {
        "n_folds": len(result.folds),
        "n_folds_with_results": len(result.best_per_fold),
        "oos_kpis": result.oos_kpis,
        "best_params_per_fold": {
            fid: gr.params for fid, gr in result.best_per_fold.items()
        },
    }
    snapshot_json(snap, out_dir, "wfo_snapshot")
    print("[wfo] Complete.")


def cmd_mc(args: argparse.Namespace) -> None:
    """Run Monte Carlo simulation on trade returns."""
    cfg = _load_config(getattr(args, "config", None))
    out_dir = _ensure_output_dir(args.output)
    _write_run_manifest(cfg, out_dir, "mc")

    if args.trades:
        trades_df = pl.read_parquet(args.trades)
        returns = trades_df.get_column("net_pnl").to_list()
    else:
        # Demo with random returns
        import random
        rng = random.Random(42)
        returns = [rng.gauss(0.0005, 0.005) for _ in range(200)]
        print("[mc] No trades file, using random demo returns.")

    n_sims = args.n_sims or 1000
    results = run_all_mc(returns, n_sims=n_sims, seed=args.seed or 42)

    for r in results:
        print(f"\n[mc] {r.method}: {r.n_sims} sims")
        print(f"     Final equity percentiles: {json.dumps(r.percentiles, indent=2)}")
        dd_pcts = {
            k: sorted(r.max_drawdowns)[min(int(len(r.max_drawdowns) * float(k[1:]) / 100),
                                            len(r.max_drawdowns) - 1)]
            for k in ["p5", "p50", "p95"]
        } if r.max_drawdowns else {}
        print(f"     MDD percentiles: {json.dumps(dd_pcts, indent=2, default=str)}")

    snap = {
        "n_sims": n_sims,
        "n_trades": len(returns),
        "results": [
            {"method": r.method, "percentiles": r.percentiles}
            for r in results
        ],
    }
    snapshot_json(snap, out_dir, "mc_snapshot")
    print("\n[mc] Complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="strategylab",
        description="StrategyLab CLI — institutional backtest framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run single backtest")
    p_run.add_argument("--config", help="Path to YAML config")
    p_run.add_argument("--data", help="Path to bars parquet")
    p_run.add_argument("--symbol", default=None, help="Symbol name")
    p_run.add_argument("--output", default=None, help="Output directory")

    # wfo
    p_wfo = sub.add_parser("wfo", help="Walk-forward optimization")
    p_wfo.add_argument("--config", help="Path to YAML config")
    p_wfo.add_argument("--data", help="Path to bars parquet")
    p_wfo.add_argument("--symbol", default=None, help="Symbol name")
    p_wfo.add_argument("--output", default=None, help="Output directory")

    # mc
    p_mc = sub.add_parser("mc", help="Monte Carlo simulation")
    p_mc.add_argument("--trades", help="Path to trades parquet with net_pnl")
    p_mc.add_argument("--n-sims", type=int, default=1000, help="Number of simulations")
    p_mc.add_argument("--seed", type=int, default=42, help="Random seed")
    p_mc.add_argument("--output", default=None, help="Output directory")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "wfo":
        cmd_wfo(args)
    elif args.command == "mc":
        cmd_mc(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
