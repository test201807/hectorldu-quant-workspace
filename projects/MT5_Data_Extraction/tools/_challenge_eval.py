"""
Challenge Evaluator v2 — Prop-firm funding challenge simulator.

Sizing method (stable, testable):
  sl_return   = median( abs(net_pnl) for SL-exit trades )
  pos_notional = risk_per_trade_usd / sl_return
  pnl_usd     = net_pnl_fractional * pos_notional

Sanity: 1 median-SL-hit costs exactly risk_per_trade_usd.

Usage:
  python tools/_challenge_eval.py --risk 75 --mode base
  python tools/_challenge_eval.py --sweep
  python tools/_challenge_eval.py --risk 75 --fold-analysis
"""
import sys, json, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl
import numpy as np

PROJECT = Path(__file__).parent.parent

# ── Challenge defaults — leídos del ruleset único ───────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from challenge_config import (  # noqa: E402
    CAPITAL as CHALLENGE_CAPITAL,
    DAILY_MAX as CHALLENGE_DAILY_MAX_LOSS,
    MAX_LOSS as CHALLENGE_TOTAL_MAX_LOSS,
    TARGET as CHALLENGE_PROFIT_TARGET,
    MIN_DAYS as CHALLENGE_MIN_DAYS,
    RISK_USD as DEFAULT_RISK,
)


def find_latest_trend_run():
    trend_dir = PROJECT / "outputs" / "trend_v2"
    runs = sorted([d for d in trend_dir.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None


def load_oos_trades(run_dir, mode="base"):
    """Load OOS trades from overlay output."""
    path = run_dir / "overlay_trades_v2.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No overlay trades at {path}")
    df = pl.read_parquet(path)
    oos = df.filter(pl.col("segment") == "OOS").sort("entry_time_utc")
    pnl_col = "net_pnl_base" if mode == "base" else "net_pnl_stress"
    return oos, pnl_col


def compute_sizing(oos, pnl_col, risk_per_trade_usd):
    """
    Compute position notional from SL-exit trades.
    Returns (pos_notional, sl_return_median).
    """
    sl_trades = oos.filter(pl.col("exit_reason") == "SL")
    if sl_trades.height > 0:
        sl_return = float(sl_trades[pnl_col].abs().median())
    else:
        # Fallback: all losers
        losers = oos.filter(pl.col(pnl_col) < 0)
        sl_return = float(losers[pnl_col].abs().median()) if losers.height > 0 else 0.003
    sl_return = max(sl_return, 1e-8)
    pos_notional = risk_per_trade_usd / sl_return
    return pos_notional, sl_return


def sizing_sanity_check(oos, pnl_col, pos_notional, risk_per_trade_usd):
    """Verify 1 SL ≈ -risk_per_trade_usd. Returns (actual_usd, pass)."""
    sl_trades = oos.filter(pl.col("exit_reason") == "SL")
    if sl_trades.height == 0:
        return 0.0, True
    median_sl_usd = float(sl_trades[pnl_col].abs().median()) * pos_notional
    tolerance = risk_per_trade_usd * 0.05  # 5% tolerance
    ok = abs(median_sl_usd - risk_per_trade_usd) <= tolerance
    return median_sl_usd, ok


# ── Core simulation ─────────────────────────────────────────────────

def simulate_challenge(
    oos, pnl_col,
    initial_capital=CHALLENGE_CAPITAL,
    daily_max_loss_usd=CHALLENGE_DAILY_MAX_LOSS,
    total_max_loss_usd=CHALLENGE_TOTAL_MAX_LOSS,
    profit_target_usd=CHALLENGE_PROFIT_TARGET,
    min_trading_days=CHALLENGE_MIN_DAYS,
    risk_per_trade_usd=DEFAULT_RISK,
):
    pos_notional, sl_return = compute_sizing(oos, pnl_col, risk_per_trade_usd)

    equity = initial_capital
    peak_equity = initial_capital
    trading_days = set()
    daily_log = {}
    trades_taken = 0
    trades_skipped_daily = 0
    target_reached = False
    target_day = None
    violated_daily = False
    violated_total = False
    max_daily_loss_seen = 0.0
    max_total_dd_seen = 0.0
    total_wins = 0
    total_win_usd = 0.0
    total_loss_usd = 0.0

    sim_df = oos.with_columns([
        pl.col("entry_time_utc").cast(pl.Date).alias("_trade_date"),
        (pl.col(pnl_col) * pos_notional).alias("_pnl_usd"),
    ])

    for row in sim_df.iter_rows(named=True):
        trade_date = row["_trade_date"]
        pnl_usd = row["_pnl_usd"]

        if trade_date not in daily_log:
            daily_log[trade_date] = {"n_trades": 0, "pnl_usd": 0.0,
                                     "equity_start": equity, "skipped": 0}
        day_info = daily_log[trade_date]

        # Daily stop BEFORE trade
        if day_info["pnl_usd"] <= -daily_max_loss_usd:
            day_info["skipped"] += 1
            trades_skipped_daily += 1
            continue

        # Total stop BEFORE trade
        total_dd = equity - initial_capital
        if total_dd <= -total_max_loss_usd:
            violated_total = True
            break

        # Take trade
        equity += pnl_usd
        day_info["n_trades"] += 1
        day_info["pnl_usd"] += pnl_usd
        trades_taken += 1
        trading_days.add(trade_date)

        if pnl_usd > 0:
            total_wins += 1
            total_win_usd += pnl_usd
        else:
            total_loss_usd += abs(pnl_usd)

        if equity > peak_equity:
            peak_equity = equity

        # Daily violation check AFTER trade
        if day_info["pnl_usd"] <= -daily_max_loss_usd:
            violated_daily = True

        # Total DD tracking
        total_dd_now = equity - initial_capital
        max_total_dd_seen = min(max_total_dd_seen, total_dd_now)
        if total_dd_now <= -total_max_loss_usd:
            violated_total = True
            break

        # Target check
        if (equity - initial_capital) >= profit_target_usd and len(trading_days) >= min_trading_days:
            target_reached = True
            target_day = trade_date
            break

    # Worst daily loss
    for info in daily_log.values():
        max_daily_loss_seen = min(max_daily_loss_seen, info["pnl_usd"])

    # Discipline
    rule_min_days = len(trading_days) >= min_trading_days
    rule_daily = max_daily_loss_seen > -daily_max_loss_usd
    rule_total = max_total_dd_seen > -total_max_loss_usd
    rule_target = target_reached
    discipline = sum([rule_min_days, rule_daily, rule_total, rule_target]) * 25

    final_pnl = equity - initial_capital
    wr = total_wins / trades_taken if trades_taken > 0 else 0
    avg_win = total_win_usd / total_wins if total_wins > 0 else 0
    total_losses = trades_taken - total_wins
    avg_loss = total_loss_usd / total_losses if total_losses > 0 else 0

    # Daily summary
    daily_summary = []
    for d in sorted(daily_log.keys()):
        info = daily_log[d]
        if info["n_trades"] > 0 or info["skipped"] > 0:
            daily_summary.append({
                "date": str(d), "n_trades": info["n_trades"],
                "pnl_usd": round(info["pnl_usd"], 2),
                "skipped": info["skipped"],
            })

    return {
        "sizing": {
            "risk_per_trade_usd": risk_per_trade_usd,
            "sl_return_median": round(sl_return, 6),
            "position_notional": round(pos_notional, 2),
            "pnl_col": pnl_col,
        },
        "challenge": {
            "initial_capital": initial_capital,
            "daily_max_loss_usd": daily_max_loss_usd,
            "total_max_loss_usd": total_max_loss_usd,
            "profit_target_usd": profit_target_usd,
            "min_trading_days": min_trading_days,
        },
        "results": {
            "final_equity": round(equity, 2),
            "final_pnl_usd": round(final_pnl, 2),
            "trades_taken": trades_taken,
            "trades_skipped_daily_stop": trades_skipped_daily,
            "trading_days": len(trading_days),
            "win_rate": round(wr, 4),
            "avg_win_usd": round(avg_win, 2),
            "avg_loss_usd": round(avg_loss, 2),
            "payoff_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
            "max_daily_loss_usd": round(max_daily_loss_seen, 2),
            "max_total_dd_usd": round(max_total_dd_seen, 2),
            "target_reached": target_reached,
            "target_day": str(target_day) if target_day else None,
            "violated_daily_limit": violated_daily,
            "violated_total_limit": violated_total,
            "discipline_pct": discipline,
        },
        "rules": {
            "min_2_days": rule_min_days,
            "daily_loss_ok": rule_daily,
            "total_loss_ok": rule_total,
            "target_hit": rule_target,
        },
        "daily_summary": daily_summary,
    }


# ── Per-fold analysis ────────────────────────────────────────────────

def simulate_per_fold(oos, pnl_col, risk_per_trade_usd, **kwargs):
    fold_col = "fold_id"
    if fold_col not in oos.columns:
        return []
    results = []
    for fold in sorted(oos[fold_col].unique().to_list()):
        fold_oos = oos.filter(pl.col(fold_col) == fold).sort("entry_time_utc")
        if fold_oos.height < 3:
            continue
        r = simulate_challenge(fold_oos, pnl_col,
                               risk_per_trade_usd=risk_per_trade_usd, **kwargs)
        r["fold"] = fold
        results.append(r)
    return results


# ── Display ──────────────────────────────────────────────────────────

def print_dashboard(result, label=""):
    s = result["sizing"]
    c = result["challenge"]
    r = result["results"]
    rules = result["rules"]

    print(f"\n{'='*65}")
    print(f"  CHALLENGE DASHBOARD {label}")
    print(f"{'='*65}")
    print(f"  Capital:     ${c['initial_capital']:,}")
    print(f"  Risk/trade:  ${s['risk_per_trade_usd']:,}  "
          f"(SL ret={s['sl_return_median']:.4%}, notional=${s['position_notional']:,.0f})")
    print(f"  Mode:        {s['pnl_col']}")
    print(f"{'─'*65}")
    print(f"  Balance:            ${r['final_equity']:>10,.2f}  (PnL ${r['final_pnl_usd']:>+,.2f})")
    print(f"  Trades:             {r['trades_taken']:>10}  "
          f"(skipped daily-stop: {r['trades_skipped_daily_stop']})")
    print(f"  Win rate:           {r['win_rate']:>10.1%}")
    print(f"  Avg win / loss:     ${r['avg_win_usd']:>8,.2f} / ${r['avg_loss_usd']:>,.2f}  "
          f"(payoff {r['payoff_ratio']:.1f}x)")
    print(f"  Trading days:       {r['trading_days']:>10}")
    print(f"  Max daily loss:     ${r['max_daily_loss_usd']:>10,.2f}  "
          f"(limit ${c['daily_max_loss_usd']:,})")
    print(f"  Max total DD:       ${r['max_total_dd_usd']:>10,.2f}  "
          f"(limit ${c['total_max_loss_usd']:,})")
    print(f"{'─'*65}")
    print(f"  OBJETIVOS:")
    for label_r, key, val in [
        ("Min 2 dias trading", "min_2_days", rules["min_2_days"]),
        ("Daily max loss OK",  "daily_loss_ok", rules["daily_loss_ok"]),
        ("Total max loss OK",  "total_loss_ok", rules["total_loss_ok"]),
        ("Profit target",      "target_hit", rules["target_hit"]),
    ]:
        status = "PASS" if val else "FAIL"
        print(f"    [{status}]  {label_r}")
    print(f"{'─'*65}")
    print(f"  DISCIPLINA: {r['discipline_pct']}%")
    if r["target_day"]:
        print(f"  TARGET alcanzado dia: {r['target_day']}")
    print(f"{'='*65}")

    ds = result.get("daily_summary", [])
    if ds:
        print(f"\n  {'Fecha':<12} {'Trades':>7} {'Skip':>5} {'PnL USD':>12}")
        print(f"  {'─'*38}")
        for d in ds[:25]:
            print(f"  {d['date']:<12} {d['n_trades']:>7} {d['skipped']:>5} "
                  f"${d['pnl_usd']:>+10,.2f}")
        if len(ds) > 25:
            print(f"  ... ({len(ds)-25} dias mas)")


def print_sweep_table(sweep_results, label=""):
    print(f"\n{'='*105}")
    print(f"  RISK SWEEP {label}")
    print(f"{'='*105}")
    hdr = (f"  {'Risk':>6} {'Notional':>10} {'PnL$':>9} {'Trd':>5} {'Skip':>5} "
           f"{'Days':>5} {'MaxDayL':>9} {'MaxTotDD':>10} "
           f"{'Tgt':>4} {'DyOK':>5} {'TtOK':>5} {'Disc':>5} {'PASS':>5}")
    print(hdr)
    print(f"  {'─'*101}")
    for r in sweep_results:
        print(f"  ${r['risk']:>5} ${r['notional']:>9,.0f} ${r['pnl']:>+8,.0f} "
              f"{r['trades']:>5} {r['skipped']:>5} {r['days']:>5} "
              f"${r['max_day_loss']:>+8,.0f} ${r['max_tot_dd']:>+9,.0f} "
              f"{'Y' if r['target'] else 'n':>4} "
              f"{'Y' if r['daily_ok'] else 'F':>5} "
              f"{'Y' if r['total_ok'] else 'F':>5} "
              f"{r['disc']:>4}% "
              f"{'PASS' if r['pass'] else 'FAIL':>5}")


def run_sweep(oos, pnl_col, risks=None, **kwargs):
    if risks is None:
        risks = [25, 50, 75, 100, 125, 150, 200, 250, 300]
    results = []
    for risk in risks:
        r = simulate_challenge(oos, pnl_col, risk_per_trade_usd=risk, **kwargs)
        res = r["results"]
        rules = r["rules"]
        results.append({
            "risk": risk,
            "notional": r["sizing"]["position_notional"],
            "pnl": res["final_pnl_usd"],
            "trades": res["trades_taken"],
            "skipped": res["trades_skipped_daily_stop"],
            "days": res["trading_days"],
            "max_day_loss": res["max_daily_loss_usd"],
            "max_tot_dd": res["max_total_dd_usd"],
            "target": res["target_reached"],
            "daily_ok": rules["daily_loss_ok"],
            "total_ok": rules["total_loss_ok"],
            "disc": res["discipline_pct"],
            "pass": all([rules["min_2_days"], rules["daily_loss_ok"],
                         rules["total_loss_ok"], rules["target_hit"]]),
        })
    return results


def run_sweep_with_folds(oos, pnl_col, risks=None, **kwargs):
    """Sweep + per-fold pass-rate and hard-violation count."""
    if risks is None:
        risks = [25, 50, 75, 100, 125, 150, 200, 250, 300]
    results = []
    for risk in risks:
        agg = simulate_challenge(oos, pnl_col, risk_per_trade_usd=risk, **kwargs)
        folds = simulate_per_fold(oos, pnl_col, risk, **kwargs)
        n_folds = len(folds)
        fold_passes = sum(1 for f in folds if f["results"]["discipline_pct"] == 100)
        fold_daily_viol = sum(1 for f in folds if f["results"]["violated_daily_limit"])
        fold_total_viol = sum(1 for f in folds if f["results"]["violated_total_limit"])
        res = agg["results"]
        rules = agg["rules"]
        results.append({
            "risk": risk,
            "pnl": res["final_pnl_usd"],
            "trades": res["trades_taken"],
            "disc": res["discipline_pct"],
            "pass_agg": all([rules["min_2_days"], rules["daily_loss_ok"],
                             rules["total_loss_ok"], rules["target_hit"]]),
            "fold_pass": fold_passes,
            "fold_total": n_folds,
            "fold_daily_viol": fold_daily_viol,
            "fold_total_viol": fold_total_viol,
            "max_tot_dd": res["max_total_dd_usd"],
        })
    return results


def print_fold_sweep(results, label=""):
    print(f"\n{'='*90}")
    print(f"  RISK SWEEP + FOLD ROBUSTNESS {label}")
    print(f"{'='*90}")
    print(f"  {'Risk':>6} {'PnL$':>9} {'Trd':>5} {'Disc':>5} {'AggPASS':>8} "
          f"{'FoldPASS':>10} {'DyViol':>7} {'TtViol':>7} {'MaxTotDD':>10}")
    print(f"  {'─'*86}")
    for r in results:
        print(f"  ${r['risk']:>5} ${r['pnl']:>+8,.0f} {r['trades']:>5} "
              f"{r['disc']:>4}% {'PASS' if r['pass_agg'] else 'FAIL':>8} "
              f"{r['fold_pass']:>4}/{r['fold_total']:<4} "
              f"{r['fold_daily_viol']:>7} {r['fold_total_viol']:>7} "
              f"${r['max_tot_dd']:>+9,.0f}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Challenge Evaluator v2")
    parser.add_argument("--risk", type=float, default=DEFAULT_RISK)
    parser.add_argument("--mode", choices=["base", "stress"], default="base")
    parser.add_argument("--run", default="latest")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--fold-analysis", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Full report: sweep + folds + sanity for both modes")
    args = parser.parse_args()

    run_dir = (find_latest_trend_run() if args.run == "latest"
               else PROJECT / "outputs" / "trend_v2" / args.run)
    if not run_dir or not run_dir.exists():
        print(f"ERROR: Run not found: {run_dir}")
        sys.exit(1)
    print(f"Run: {run_dir.name}")

    oos, pnl_col = load_oos_trades(run_dir, args.mode)
    print(f"OOS trades: {oos.height} ({pnl_col})")

    # Sizing sanity
    pos_notional, sl_ret = compute_sizing(oos, pnl_col, args.risk)
    actual_sl_usd, sanity_ok = sizing_sanity_check(oos, pnl_col, pos_notional, args.risk)
    print(f"Sizing: risk=${args.risk}, SL_ret={sl_ret:.4%}, "
          f"notional=${pos_notional:,.0f}, 1-SL=${actual_sl_usd:,.2f} "
          f"[{'OK' if sanity_ok else 'WARN'}]")

    if args.full:
        for mode in ["base", "stress"]:
            oos_m, col_m = load_oos_trades(run_dir, mode)
            print(f"\n{'='*65}")
            print(f"  MODE: {mode.upper()}")
            print(f"{'='*65}")

            # Dashboard at optimal risk
            r = simulate_challenge(oos_m, col_m, risk_per_trade_usd=args.risk)
            print_dashboard(r, f"(risk=${args.risk}, {mode})")

            # Sweep
            sweep = run_sweep(oos_m, col_m)
            print_sweep_table(sweep, f"({mode})")

            # Fold sweep
            fsweep = run_sweep_with_folds(oos_m, col_m, risks=[50, 75, 100, 125])
            print_fold_sweep(fsweep, f"({mode})")

        # Save
        out_dir = PROJECT / "outputs" / "challenge_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_dir.name}_full_report.json"
        # Build save dict
        save_data = {}
        for mode in ["base", "stress"]:
            oos_m, col_m = load_oos_trades(run_dir, mode)
            r = simulate_challenge(oos_m, col_m, risk_per_trade_usd=args.risk)
            sweep = run_sweep(oos_m, col_m)
            save_data[mode] = {"dashboard": r, "sweep": sweep}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    elif args.sweep:
        for mode in ["base", "stress"]:
            oos_m, col_m = load_oos_trades(run_dir, mode)
            sweep = run_sweep(oos_m, col_m)
            print_sweep_table(sweep, f"({mode})")
            fsweep = run_sweep_with_folds(oos_m, col_m, risks=[50, 75, 100, 125])
            print_fold_sweep(fsweep, f"({mode})")

        out_dir = PROJECT / "outputs" / "challenge_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        for mode in ["base", "stress"]:
            oos_m, col_m = load_oos_trades(run_dir, mode)
            sweep = run_sweep(oos_m, col_m)
        out_path = out_dir / f"{run_dir.name}_sweep.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"base": run_sweep(*load_oos_trades(run_dir, "base")),
                        "stress": run_sweep(*load_oos_trades(run_dir, "stress"))},
                       f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    elif args.fold_analysis:
        folds = simulate_per_fold(oos, pnl_col, args.risk)
        passes = sum(1 for f in folds if f["results"]["discipline_pct"] == 100)
        daily_viols = sum(1 for f in folds if f["results"]["violated_daily_limit"])
        total_viols = sum(1 for f in folds if f["results"]["violated_total_limit"])
        print(f"\nPer-fold (risk=${args.risk}, {pnl_col}):")
        print(f"  PASS: {passes}/{len(folds)} | daily violations: {daily_viols} | total violations: {total_viols}")
        for f in folds:
            r = f["results"]
            print(f"  fold={f['fold']:2d}: pnl=${r['final_pnl_usd']:>+7,.0f} "
                  f"trd={r['trades_taken']:>3} "
                  f"maxDD=${r['max_total_dd_usd']:>+7,.0f} "
                  f"maxDL=${r['max_daily_loss_usd']:>+7,.0f} "
                  f"tgt={'Y' if r['target_reached'] else 'n'} "
                  f"dyV={'Y' if r['violated_daily_limit'] else 'n'} "
                  f"ttV={'Y' if r['violated_total_limit'] else 'n'} "
                  f"disc={r['discipline_pct']}%")

    else:
        result = simulate_challenge(oos, pnl_col, risk_per_trade_usd=args.risk)
        print_dashboard(result, f"(risk=${args.risk}, {args.mode})")

        out_dir = PROJECT / "outputs" / "challenge_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_dir.name}_challenge_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
