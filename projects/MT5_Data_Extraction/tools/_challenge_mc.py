"""
Challenge MC Simulator — Probabilidad de PASS en funding challenge.

Responde la pregunta: "¿Con qué probabilidad paso el challenge FTMO
dado el historial de trades OOS?"

Método:
  1. Carga OOS trades del último run NB3 (overlay_trades_v2.parquet)
  2. Computa sizing (pos_notional) una sola vez sobre todos los OOS trades
  3. Genera N=1000 secuencias bootstrap de PnL (block bootstrap, block=10 trades)
     manteniendo las mismas fechas → solo cambia el orden de los trades
  4. Aplica reglas FTMO a cada secuencia:
       - Daily max loss:   $1,250 (5% de $25k)
       - Total max loss:   $2,500 (10% de $25k)
       - Profit target:    $1,250 (5% de $25k)
       - Min trading days: 2
  5. Reporta distribución de resultados y P(PASS)

Uso:
  cd C:\\Quant\\projects\\MT5_Data_Extraction
  venv1\\Scripts\\python.exe tools\\_challenge_mc.py
  venv1\\Scripts\\python.exe tools\\_challenge_mc.py --risk 100 --sims 2000
  venv1\\Scripts\\python.exe tools\\_challenge_mc.py --mode stress --block 5 --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).parent.parent

# ── FTMO Challenge rules ($25k account) ─────────────────────────────
CHALLENGE_CAPITAL        = 25_000
CHALLENGE_DAILY_MAX_LOSS = 1_250   # 5% daily
CHALLENGE_TOTAL_MAX_LOSS = 2_500   # 10% total
CHALLENGE_PROFIT_TARGET  = 1_250   # 5% target
CHALLENGE_MIN_DAYS       = 2
DEFAULT_RISK             = 75
DEFAULT_SIMS             = 1_000
DEFAULT_BLOCK            = 10      # block size (trades) para bootstrap


# ── Data ────────────────────────────────────────────────────────────

def find_latest_trend_run() -> Path | None:
    trend_dir = PROJECT / "outputs" / "trend_v2"
    runs = sorted([d for d in trend_dir.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def load_oos_trades(run_dir: Path, mode: str = "base") -> tuple[pl.DataFrame, str]:
    path = run_dir / "overlay_trades_v2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No overlay trades at {path}")
    df = pl.read_parquet(path)
    oos = df.filter(pl.col("segment") == "OOS").sort("entry_time_utc")
    pnl_col = "net_pnl_base" if mode == "base" else "net_pnl_stress"
    if pnl_col not in oos.columns:
        pnl_col = "net_pnl"
    return oos, pnl_col


def compute_sizing(oos: pl.DataFrame, pnl_col: str, risk_per_trade_usd: float) -> tuple[float, float]:
    """Devuelve (pos_notional, sl_return_median)."""
    sl_trades = oos.filter(pl.col("exit_reason") == "SL")
    if sl_trades.height > 0:
        sl_return = float(sl_trades[pnl_col].abs().median())
    else:
        losers = oos.filter(pl.col(pnl_col) < 0)
        sl_return = float(losers[pnl_col].abs().median()) if losers.height > 0 else 0.003
    sl_return = max(sl_return, 1e-8)
    return risk_per_trade_usd / sl_return, sl_return


# ── Core simulation (una secuencia de PnL en USD) ───────────────────

def simulate_one(
    pnl_usd: np.ndarray,
    dates: list,
    initial_capital: float = CHALLENGE_CAPITAL,
    daily_max_loss: float = CHALLENGE_DAILY_MAX_LOSS,
    total_max_loss: float = CHALLENGE_TOTAL_MAX_LOSS,
    profit_target: float = CHALLENGE_PROFIT_TARGET,
    min_days: int = CHALLENGE_MIN_DAYS,
) -> dict:
    """Simula el challenge con un array de PnL en USD y sus fechas."""
    equity = initial_capital
    daily_pnl: dict = {}
    trading_days: set = set()
    target_reached = False
    violated_daily = False
    violated_total = False
    max_daily_loss = 0.0
    max_total_dd = 0.0

    for pnl, date in zip(pnl_usd, dates):
        if date not in daily_pnl:
            daily_pnl[date] = 0.0

        # Bloqueo diario ANTES de ejecutar
        if daily_pnl[date] <= -daily_max_loss:
            continue

        # Bloqueo total ANTES de ejecutar
        if equity - initial_capital <= -total_max_loss:
            violated_total = True
            break

        equity += pnl
        daily_pnl[date] += pnl
        trading_days.add(date)

        # Tracks
        max_daily_loss = min(max_daily_loss, daily_pnl[date])
        max_total_dd = min(max_total_dd, equity - initial_capital)

        # Violaciones
        if daily_pnl[date] <= -daily_max_loss:
            violated_daily = True
        if equity - initial_capital <= -total_max_loss:
            violated_total = True
            break

        # Target
        if (equity - initial_capital >= profit_target
                and len(trading_days) >= min_days):
            target_reached = True
            break

    passed = (
        not violated_daily
        and not violated_total
        and target_reached
        and len(trading_days) >= min_days
    )
    return {
        "passed": passed,
        "final_pnl": equity - initial_capital,
        "target_reached": target_reached,
        "violated_daily": violated_daily,
        "violated_total": violated_total,
        "trading_days": len(trading_days),
        "max_daily_loss": max_daily_loss,
        "max_total_dd": max_total_dd,
    }


# ── Block bootstrap ──────────────────────────────────────────────────

def block_bootstrap(
    pnl_arr: np.ndarray,
    dates_arr: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample PnL con block bootstrap manteniendo estructura de bloques.

    Mantiene las MISMAS FECHAS en orden — solo cambia el PnL de cada
    trade. Así preservamos la estructura diaria del calendario.
    """
    n = len(pnl_arr)
    n_blocks = (n + block_size - 1) // block_size
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    resampled_pnl = np.concatenate([pnl_arr[s:s + block_size] for s in starts])[:n]
    return resampled_pnl, dates_arr  # fechas sin cambio


# ── Monte Carlo ──────────────────────────────────────────────────────

def run_mc(
    oos: pl.DataFrame,
    pnl_col: str,
    risk_per_trade_usd: float = DEFAULT_RISK,
    n_sims: int = DEFAULT_SIMS,
    block_size: int = DEFAULT_BLOCK,
    seed: int = 42,
) -> dict:
    """Ejecuta Monte Carlo bootstrap y retorna distribución de resultados."""
    pos_notional, sl_return = compute_sizing(oos, pnl_col, risk_per_trade_usd)

    # Arrays numpy para velocidad
    pnl_frac = oos[pnl_col].to_numpy()
    pnl_usd  = pnl_frac * pos_notional
    dates    = oos["entry_time_utc"].cast(pl.Date).to_numpy()

    rng = np.random.default_rng(seed)

    results = []
    for _ in range(n_sims):
        boot_pnl, boot_dates = block_bootstrap(pnl_usd, dates, block_size, rng)
        r = simulate_one(boot_pnl, boot_dates.tolist())
        results.append(r)

    # Métricas agregadas
    n = len(results)
    passed       = [r["passed"] for r in results]
    final_pnls   = np.array([r["final_pnl"] for r in results])
    max_dds      = np.array([r["max_total_dd"] for r in results])
    max_daily    = np.array([r["max_daily_loss"] for r in results])
    tgt_reached  = [r["target_reached"] for r in results]
    viol_daily   = [r["violated_daily"] for r in results]
    viol_total   = [r["violated_total"] for r in results]

    p_pass = sum(passed) / n
    p_tgt  = sum(tgt_reached) / n
    p_dviol = sum(viol_daily) / n
    p_tviol = sum(viol_total) / n

    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    pnl_q  = {f"p{int(q*100):02d}": round(float(np.quantile(final_pnls, q)), 2) for q in quantiles}
    dd_q   = {f"p{int(q*100):02d}": round(float(np.quantile(max_dds,    q)), 2) for q in quantiles}

    return {
        "inputs": {
            "n_sims": n_sims,
            "block_size": block_size,
            "risk_per_trade_usd": risk_per_trade_usd,
            "sl_return_median": round(sl_return, 6),
            "pos_notional": round(pos_notional, 2),
            "pnl_col": pnl_col,
            "n_oos_trades": len(pnl_usd),
        },
        "challenge_rules": {
            "capital": CHALLENGE_CAPITAL,
            "daily_max_loss": CHALLENGE_DAILY_MAX_LOSS,
            "total_max_loss": CHALLENGE_TOTAL_MAX_LOSS,
            "profit_target": CHALLENGE_PROFIT_TARGET,
            "min_trading_days": CHALLENGE_MIN_DAYS,
        },
        "mc_summary": {
            "p_pass":          round(p_pass, 4),
            "p_target_hit":    round(p_tgt, 4),
            "p_daily_violated": round(p_dviol, 4),
            "p_total_violated": round(p_tviol, 4),
            "pnl_usd_quantiles": pnl_q,
            "max_dd_usd_quantiles": dd_q,
            "avg_final_pnl_usd": round(float(final_pnls.mean()), 2),
            "median_max_dd_usd": round(float(np.median(max_dds)), 2),
            "median_max_daily_loss_usd": round(float(np.median(max_daily)), 2),
        },
        "raw_counts": {
            "pass": sum(passed),
            "fail": n - sum(passed),
            "target_hit": sum(tgt_reached),
            "daily_viol": sum(viol_daily),
            "total_viol": sum(viol_total),
        },
    }


# ── Deterministic reference (sin shuffle) ───────────────────────────

def run_deterministic(
    oos: pl.DataFrame,
    pnl_col: str,
    risk_per_trade_usd: float = DEFAULT_RISK,
) -> dict:
    """Corre el challenge una sola vez en orden cronológico (referencia)."""
    pos_notional, sl_return = compute_sizing(oos, pnl_col, risk_per_trade_usd)
    pnl_usd = (oos[pnl_col].to_numpy() * pos_notional)
    dates   = oos["entry_time_utc"].cast(pl.Date).to_numpy().tolist()
    result  = simulate_one(pnl_usd, dates)
    result["pos_notional"] = round(pos_notional, 2)
    result["sl_return_median"] = round(sl_return, 6)
    return result


# ── Display ──────────────────────────────────────────────────────────

def print_report(mc: dict, det: dict) -> None:
    inp = mc["inputs"]
    rules = mc["challenge_rules"]
    s = mc["mc_summary"]
    rc = mc["raw_counts"]
    pq = s["pnl_usd_quantiles"]
    dq = s["max_dd_usd_quantiles"]

    print(f"\n{'='*65}")
    print(f"  CHALLENGE MC SIMULATOR — {inp['n_sims']:,} iteraciones")
    print(f"{'='*65}")
    print(f"  Capital:        ${rules['capital']:,}")
    print(f"  Daily limit:    ${rules['daily_max_loss']:,}  ({rules['daily_max_loss']/rules['capital']:.0%})")
    print(f"  Total limit:    ${rules['total_max_loss']:,}  ({rules['total_max_loss']/rules['capital']:.0%})")
    print(f"  Profit target:  ${rules['profit_target']:,}  ({rules['profit_target']/rules['capital']:.0%})")
    print(f"  Trades OOS:     {inp['n_oos_trades']}")
    print(f"  Risk/trade:     ${inp['risk_per_trade_usd']:,}")
    print(f"  Pos notional:   ${inp['pos_notional']:,.0f}")
    print(f"  Block size:     {inp['block_size']} trades")
    print(f"  Mode:           {inp['pnl_col']}")
    print(f"{'─'*65}")
    print(f"  REFERENCIA (orden cronologico real):")
    status = "PASS" if det["passed"] else "FAIL"
    print(f"    [{status}]  PnL=${det['final_pnl']:>+,.2f}  "
          f"target={'Y' if det['target_reached'] else 'N'}  "
          f"daily_viol={'Y' if det['violated_daily'] else 'N'}  "
          f"total_viol={'Y' if det['violated_total'] else 'N'}")
    print(f"{'─'*65}")
    print(f"  MONTE CARLO (N={inp['n_sims']:,}):")
    print(f"")
    print(f"    P(PASS)              = {s['p_pass']:>7.1%}   "
          f"({rc['pass']:>5} PASS / {rc['fail']:>5} FAIL)")
    print(f"    P(Target hit)        = {s['p_target_hit']:>7.1%}   "
          f"(% secuencias que alcanzan el target)")
    print(f"    P(Daily violated)    = {s['p_daily_violated']:>7.1%}   "
          f"(% secuencias con daily cap breach)")
    print(f"    P(Total violated)    = {s['p_total_violated']:>7.1%}   "
          f"(% secuencias con total DD breach)")
    print(f"")
    print(f"  PnL final USD (distribucion):")
    print(f"    p05=${pq['p05']:>+8,.0f}  p25=${pq['p25']:>+8,.0f}  "
          f"p50=${pq['p50']:>+8,.0f}  p75=${pq['p75']:>+8,.0f}  p95=${pq['p95']:>+8,.0f}")
    print(f"")
    print(f"  Max DD total USD (distribucion):")
    print(f"    p05=${dq['p05']:>+8,.0f}  p25=${dq['p25']:>+8,.0f}  "
          f"p50=${dq['p50']:>+8,.0f}  p75=${dq['p75']:>+8,.0f}  p95=${dq['p95']:>+8,.0f}")
    print(f"{'─'*65}")

    # Interpretacion
    p = s["p_pass"]
    if p >= 0.60:
        verdict = "VERDE — Alta probabilidad de PASS (>60%)"
    elif p >= 0.40:
        verdict = "AMARILLO — Probabilidad moderada (40-60%). Monitorear."
    elif p >= 0.20:
        verdict = "NARANJA — Probabilidad baja (20-40%). Mejorar edge primero."
    else:
        verdict = "ROJO — Muy baja probabilidad (<20%). NO entrar al challenge."

    print(f"  VEREDICTO: {verdict}")
    print(f"{'='*65}")

    # Riesgo principal
    if s["p_daily_violated"] > 0.30:
        print(f"\n  ADVERTENCIA: El daily cap es el mayor riesgo "
              f"({s['p_daily_violated']:.0%} secuencias lo violan).")
        print(f"  Considera reducir risk/trade o filtrar dias de alta volatilidad.")
    if s["p_total_violated"] > 0.20:
        print(f"\n  ADVERTENCIA: Riesgo de violacion total significativo "
              f"({s['p_total_violated']:.0%}).")
        print(f"  El drawdown max mediano es ${abs(s['median_max_dd_usd']):,.0f} "
              f"(limite ${rules['total_max_loss']:,}).")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Challenge MC Simulator")
    parser.add_argument("--risk",  type=float, default=DEFAULT_RISK,
                        help=f"Risk per trade USD (default {DEFAULT_RISK})")
    parser.add_argument("--sims",  type=int,   default=DEFAULT_SIMS,
                        help=f"Numero de simulaciones MC (default {DEFAULT_SIMS})")
    parser.add_argument("--block", type=int,   default=DEFAULT_BLOCK,
                        help=f"Block size para bootstrap (default {DEFAULT_BLOCK})")
    parser.add_argument("--mode",  choices=["base", "stress"], default="base")
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--run",   default="latest")
    parser.add_argument("--save",  action="store_true",
                        help="Guardar JSON en outputs/challenge_eval/")
    args = parser.parse_args()

    run_dir = (find_latest_trend_run() if args.run == "latest"
               else PROJECT / "outputs" / "trend_v2" / args.run)
    if not run_dir or not run_dir.exists():
        print(f"ERROR: Run no encontrado: {run_dir}")
        sys.exit(1)

    print(f"Run: {run_dir.name}")
    oos, pnl_col = load_oos_trades(run_dir, args.mode)
    print(f"OOS trades: {oos.height}  |  mode: {pnl_col}")

    # Referencia determinista
    det = run_deterministic(oos, pnl_col, args.risk)

    # Monte Carlo
    print(f"Corriendo {args.sims:,} simulaciones MC (block={args.block})...")
    mc = run_mc(
        oos, pnl_col,
        risk_per_trade_usd=args.risk,
        n_sims=args.sims,
        block_size=args.block,
        seed=args.seed,
    )

    print_report(mc, det)

    if args.save:
        out_dir = PROJECT / "outputs" / "challenge_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_dir.name}_mc_report.json"
        payload = {"deterministic": det, "monte_carlo": mc}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
