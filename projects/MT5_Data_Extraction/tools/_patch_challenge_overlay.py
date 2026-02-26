"""
Patch: Integrate ChallengeOverlayStateMachine into NB3 Cell 16.

Replaces old daily caps (fractional, effectively disabled) with
USD-based challenge rules that match the prop-firm funding exam:
  - daily stop: -$1,250
  - total stop: -$2,500
  - profit target: +$1,250
  - min 2 trading days

Sizing: pos_notional = risk_per_trade_usd / median_sl_loss
  where median_sl_loss is computed from SL-exit trades.

Key difference from old overlay:
  Old: DAILY_MAX_LOSS=-5% -> 1-2 SL hits cap the day -> KILLED edge
  New: DAILY_MAX_LOSS=$1,250 / risk=$75 -> need ~16 SL/day -> virtually never triggers

Usage: python tools/_patch_challenge_overlay.py [--dry-run]
"""
import json, sys, pathlib

DRY_RUN = "--dry-run" in sys.argv
PROJECT = pathlib.Path(__file__).parent.parent
NB3 = PROJECT / "03_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb"

NEW_CELL16_SOURCE = r'''# ======================================================================================
# Celda 16 v3.0.0 — Challenge-Ready Overlay (ChallengeOverlayStateMachine)
# Edge filter: BTCUSD LONG only (fwd_ret +0.356% @24h)
# Challenge rules: daily -$1,250 / total -$2,500 / target +$1,250 / min 2 days
# Sizing: risk_per_trade / median_SL_loss -> 1 SL ~ $risk
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 16 v3.0.0 :: Challenge-Ready Overlay")

if "RUN" not in globals():
    raise RuntimeError("[Celda 16] ERROR: RUN no existe.")

if RUN.get("_overlay_applied"):
    raise RuntimeError("[Celda 16] Overlay ya aplicado en este run. Re-ejecutar desde Cell 00.")
RUN["_overlay_applied"] = True

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_OVERLAY_TRADES = ARTIFACTS.get("overlay_trades", RUN_DIR / "overlay_trades_v2.parquet")
OUT_OVERLAY_SUMMARY = ARTIFACTS.get("overlay_summary", RUN_DIR / "overlay_summary_v2.parquet")
OUT_SNAP = ARTIFACTS.get("overlay_snapshot", RUN_DIR / "overlay_snapshot_v2.json")
OUT_CHALLENGE = RUN_DIR / "challenge_dashboard_v2.json"

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ── Edge filter params ──
SYMBOL_WHITELIST = ["BTCUSD"]
SIDE_FILTER = "LONG"
ENTRY_WEEKDAYS_ONLY = True

# ── Challenge params (prop-firm exam) ──
CHALLENGE_CAPITAL        = 25_000
CHALLENGE_DAILY_MAX_LOSS = 1_250   # USD
CHALLENGE_TOTAL_MAX_LOSS = 2_500   # USD
CHALLENGE_PROFIT_TARGET  = 1_250   # USD
CHALLENGE_MIN_DAYS       = 2
RISK_PER_TRADE_USD       = 75     # optimal from sweep (worst fold DD = -$1,955, no violations)

if not TRADES_PATH.exists():
    print("[Celda 16] WARNING: trades_engine no existe, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    df = pl.read_parquet(TRADES_PATH)
    n_engine = df.height

    # ── Step 1: Edge filter ──
    df = df.filter(
        pl.col("symbol").is_in(SYMBOL_WHITELIST) &
        (pl.col("side") == SIDE_FILTER)
    )
    n_after_edge = df.height
    print(f"[Celda 16] Edge filter: {n_engine} -> {n_after_edge} (BTCUSD LONG only)")

    # ── Step 2: Weekday filter ──
    df = df.with_columns([
        pl.col("entry_time_utc").cast(pl.Date).alias("_date"),
        pl.col("entry_time_utc").dt.weekday().alias("_dow"),
    ])
    if ENTRY_WEEKDAYS_ONLY:
        df = df.filter(pl.col("_dow") <= 5)
    n_after_weekday = df.height

    # ── Step 3: Sizing (from SL-exit trades) ──
    sl_trades = df.filter(pl.col("exit_reason") == "SL")
    if sl_trades.height > 0:
        sl_return_median = float(sl_trades["net_pnl_base"].abs().median())
    else:
        sl_return_median = 0.003  # fallback
    sl_return_median = max(sl_return_median, 1e-8)
    pos_notional = RISK_PER_TRADE_USD / sl_return_median

    # Sanity check: 1 SL should cost ~$RISK_PER_TRADE_USD
    if sl_trades.height > 0:
        actual_sl_usd = sl_return_median * pos_notional
        print(f"[Celda 16] Sizing: risk=${RISK_PER_TRADE_USD}, SL_ret={sl_return_median:.4%}, "
              f"notional=${pos_notional:,.0f}, 1-SL=${actual_sl_usd:,.2f}")

    # ── Step 4: ChallengeOverlayStateMachine (OOS simulation) ──
    # Save ALL trades (IS+OOS) for overlay output, but simulate challenge on OOS only
    df_sorted = df.sort("entry_time_utc")

    # For overlay_trades: keep all (no filtering by challenge rules on the parquet)
    df_sorted.write_parquet(str(OUT_OVERLAY_TRADES), compression="zstd")
    n_overlay = df_sorted.height

    # Challenge simulation on OOS
    oos = df_sorted.filter(pl.col("segment") == "OOS")

    challenge_result = None
    if oos.height > 0:
        equity = CHALLENGE_CAPITAL
        trading_days = set()
        daily_log = {}
        trades_taken = 0
        trades_skipped = 0
        target_reached = False
        target_day = None
        violated_daily = False
        violated_total = False
        max_daily_loss_seen = 0.0
        max_total_dd_seen = 0.0
        total_wins = 0
        total_win_usd = 0.0
        total_loss_usd = 0.0

        for row in oos.iter_rows(named=True):
            trade_date = row["_date"]
            pnl_usd = row["net_pnl_base"] * pos_notional

            if trade_date not in daily_log:
                daily_log[trade_date] = {"n_trades": 0, "pnl_usd": 0.0, "skipped": 0}
            day = daily_log[trade_date]

            # Daily stop BEFORE trade
            if day["pnl_usd"] <= -CHALLENGE_DAILY_MAX_LOSS:
                day["skipped"] += 1
                trades_skipped += 1
                continue

            # Total stop BEFORE trade
            if (equity - CHALLENGE_CAPITAL) <= -CHALLENGE_TOTAL_MAX_LOSS:
                violated_total = True
                break

            # Take trade
            equity += pnl_usd
            day["n_trades"] += 1
            day["pnl_usd"] += pnl_usd
            trades_taken += 1
            trading_days.add(trade_date)

            if pnl_usd > 0:
                total_wins += 1
                total_win_usd += pnl_usd
            else:
                total_loss_usd += abs(pnl_usd)

            # Daily violation check
            if day["pnl_usd"] <= -CHALLENGE_DAILY_MAX_LOSS:
                violated_daily = True

            # Total DD tracking
            total_dd = equity - CHALLENGE_CAPITAL
            max_total_dd_seen = min(max_total_dd_seen, total_dd)
            if total_dd <= -CHALLENGE_TOTAL_MAX_LOSS:
                violated_total = True
                break

            # Target check
            if total_dd >= CHALLENGE_PROFIT_TARGET and len(trading_days) >= CHALLENGE_MIN_DAYS:
                target_reached = True
                target_day = str(trade_date)
                break

        # Worst daily loss
        for info in daily_log.values():
            max_daily_loss_seen = min(max_daily_loss_seen, info["pnl_usd"])

        # Discipline
        r_days = len(trading_days) >= CHALLENGE_MIN_DAYS
        r_daily = max_daily_loss_seen > -CHALLENGE_DAILY_MAX_LOSS
        r_total = max_total_dd_seen > -CHALLENGE_TOTAL_MAX_LOSS
        r_target = target_reached
        discipline = sum([r_days, r_daily, r_total, r_target]) * 25

        wr = total_wins / trades_taken if trades_taken > 0 else 0
        avg_win = total_win_usd / total_wins if total_wins > 0 else 0
        n_losses = trades_taken - total_wins
        avg_loss = total_loss_usd / n_losses if n_losses > 0 else 0

        # Daily summary
        daily_summary = []
        for d in sorted(daily_log.keys()):
            info = daily_log[d]
            if info["n_trades"] > 0 or info["skipped"] > 0:
                daily_summary.append({
                    "date": str(d), "n_trades": info["n_trades"],
                    "pnl_usd": round(info["pnl_usd"], 2), "skipped": info["skipped"],
                })

        challenge_result = {
            "created_utc": _now_utc_iso(),
            "version": "v3.0.0",
            "sizing": {
                "risk_per_trade_usd": RISK_PER_TRADE_USD,
                "sl_return_median": round(sl_return_median, 6),
                "position_notional": round(pos_notional, 2),
            },
            "challenge": {
                "initial_capital": CHALLENGE_CAPITAL,
                "daily_max_loss_usd": CHALLENGE_DAILY_MAX_LOSS,
                "total_max_loss_usd": CHALLENGE_TOTAL_MAX_LOSS,
                "profit_target_usd": CHALLENGE_PROFIT_TARGET,
                "min_trading_days": CHALLENGE_MIN_DAYS,
            },
            "results": {
                "final_equity": round(equity, 2),
                "final_pnl_usd": round(equity - CHALLENGE_CAPITAL, 2),
                "trades_taken": trades_taken,
                "trades_skipped_daily_stop": trades_skipped,
                "trading_days": len(trading_days),
                "win_rate": round(wr, 4),
                "avg_win_usd": round(avg_win, 2),
                "avg_loss_usd": round(avg_loss, 2),
                "payoff_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
                "max_daily_loss_usd": round(max_daily_loss_seen, 2),
                "max_total_dd_usd": round(max_total_dd_seen, 2),
                "target_reached": target_reached,
                "target_day": target_day,
                "violated_daily_limit": violated_daily,
                "violated_total_limit": violated_total,
                "discipline_pct": discipline,
            },
            "rules": {
                "min_2_days": r_days,
                "daily_loss_ok": r_daily,
                "total_loss_ok": r_total,
                "target_hit": r_target,
            },
            "daily_summary": daily_summary,
        }

        # Print dashboard
        res = challenge_result["results"]
        print(f"[Celda 16] Challenge OOS (base): PnL=${res['final_pnl_usd']:+,.0f}, "
              f"disc={res['discipline_pct']}%, target={'SI' if res['target_reached'] else 'NO'}")
        print(f"[Celda 16] MaxDayLoss=${res['max_daily_loss_usd']:,.0f} "
              f"MaxTotDD=${res['max_total_dd_usd']:,.0f} "
              f"trades={res['trades_taken']} days={res['trading_days']}")

    # Save challenge dashboard
    if challenge_result:
        Path(OUT_CHALLENGE).write_text(
            json.dumps(challenge_result, indent=2, default=str), encoding="utf-8")
        print(f"[Celda 16] OUT: {OUT_CHALLENGE}")

    # Summary
    if n_overlay > 0:
        summary = (
            df_sorted.group_by(["symbol", "segment"])
            .agg([
                pl.len().alias("n_trades"),
                pl.col("net_pnl_base").sum().alias("total_ret"),
                pl.col("net_pnl_base").mean().alias("mean_ret"),
                (pl.col("net_pnl_base") > 0).mean().alias("win_rate"),
            ])
            .sort(["symbol", "segment"])
        )
    else:
        summary = pl.DataFrame()
    summary.write_parquet(str(OUT_OVERLAY_SUMMARY), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v3.0.0",
        "edge_filter": {"symbols": SYMBOL_WHITELIST, "side": SIDE_FILTER},
        "challenge": {
            "capital": CHALLENGE_CAPITAL,
            "daily_max_loss": CHALLENGE_DAILY_MAX_LOSS,
            "total_max_loss": CHALLENGE_TOTAL_MAX_LOSS,
            "profit_target": CHALLENGE_PROFIT_TARGET,
            "risk_per_trade": RISK_PER_TRADE_USD,
        },
        "sizing": {"sl_return_median": round(sl_return_median, 6),
                   "pos_notional": round(pos_notional, 2)},
        "n_engine": n_engine, "n_after_edge": n_after_edge,
        "n_after_weekday": n_after_weekday, "n_overlay": n_overlay,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 16] trades: {n_engine} -> {n_overlay} (edge+weekday filter)")
    print(f"[Celda 16] OUT: {OUT_OVERLAY_TRADES}")

print(">>> Celda 16 v3.0.0 :: OK")
'''


def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def find_cell(nb, cell_id):
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            return c
    raise KeyError(f"Cell {cell_id} not found")


def set_source(cell, code):
    lines = code.split("\n")
    new_src = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            new_src.append(line + "\n")
        else:
            new_src.append(line)
    cell["source"] = new_src


nb3 = load_nb(NB3)
cell = find_cell(nb3, "ab67bb0e")

old_src = "".join(cell["source"])
set_source(cell, NEW_CELL16_SOURCE)

if DRY_RUN:
    print("[DRY RUN] NB3 not saved.")
    print(f"Old Cell 16: {len(old_src)} chars")
    print(f"New Cell 16: {len(NEW_CELL16_SOURCE)} chars")
else:
    save_nb(NB3, nb3)
    print(f"Saved NB3: {NB3}")
    print(f"Cell 16 replaced: v2.0.1 -> v3.0.0 (ChallengeOverlayStateMachine)")
    print(f"  Edge filter: BTCUSD LONG (unchanged)")
    print(f"  Challenge: $25k capital, daily -$1,250, total -$2,500, target +$1,250")
    print(f"  Sizing: risk=$75, pos_notional from median SL loss")
    print(f"  Daily caps: USD-based (need ~16 SL/day to trigger, vs old ~1-2 SL)")
