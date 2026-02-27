"""
Patch NB4: insertar Celda 17b Challenge Simulation después de Cell 17 (id=709ba063).

Uso:
    python tools/_patch_nb4_challenge.py [--dry-run]
"""
import json
import sys
from pathlib import Path

NB4_PATH = Path(__file__).parent.parent / "03_STRATEGY_LAB" / "notebooks" / "04_RANGE_M5_Strategy_v1.ipynb"
AFTER_CELL_ID = "709ba063"   # Cell 17 Selection — insertar DESPUÉS de este
NEW_CELL_ID   = "d41ef001"

NEW_CELL_SOURCE = """\
# ======================================================================================
# Celda 17b v1.0.0 — Challenge Simulation RANGE (ChallengeOverlayStateMachine)
# Challenge rules: daily -$500 / total -$1,000 / target +$500 / min 2 days (from ruleset)
# Sides: XAUUSD LONG + SHORT combinados cronologicamente
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 17b v1.0.0 :: Challenge Simulation RANGE")

RUN_DIR   = RUN["RUN_DIR"]
ARTIFACTS = RUN["ARTIFACTS"]

OVERLAY_PATH  = ARTIFACTS["overlay_trades"]
SEL_SNAP_PATH = ARTIFACTS["selection_snapshot"]
OUT_CHALLENGE = RUN_DIR / "challenge_dashboard_range_v1.json"

def _now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# -- Challenge params (ruleset) --
_ruleset_path = Path.cwd().parent / "configs" / "challenge_ruleset.json"
_ruleset = json.loads(_ruleset_path.read_text(encoding="utf-8"))
CHALLENGE_CAPITAL        = _ruleset["starting_balance"]
CHALLENGE_DAILY_MAX_LOSS = _ruleset["daily_max_loss"]
CHALLENGE_TOTAL_MAX_LOSS = _ruleset["max_loss"]
CHALLENGE_PROFIT_TARGET  = _ruleset["profit_target"]
CHALLENGE_MIN_DAYS       = _ruleset["min_trading_days"]
RISK_PER_TRADE_USD       = _ruleset["risk_per_trade_usd"]
ACCOUNT_TYPE             = _ruleset["account_type"]
_CB_THRESHOLD            = CHALLENGE_DAILY_MAX_LOSS * _ruleset.get("circuit_breaker_pct", 1.0)

assert CHALLENGE_CAPITAL > 0
assert 0 < CHALLENGE_DAILY_MAX_LOSS <= CHALLENGE_TOTAL_MAX_LOSS < CHALLENGE_CAPITAL
assert 0 < CHALLENGE_PROFIT_TARGET <= CHALLENGE_CAPITAL
assert RISK_PER_TRADE_USD < CHALLENGE_DAILY_MAX_LOSS, "risk/trade >= daily limit"
assert ACCOUNT_TYPE in ("standard", "swing"), f"unknown account_type: {ACCOUNT_TYPE!r}"
print(f"[Celda 17b] Ruleset: ${CHALLENGE_CAPITAL:,} | daily-${CHALLENGE_DAILY_MAX_LOSS} "
      f"| total-${CHALLENGE_TOTAL_MAX_LOSS} | target+${CHALLENGE_PROFIT_TARGET} "
      f"| risk=${RISK_PER_TRADE_USD}/trade | type={ACCOUNT_TYPE}")

if not Path(OVERLAY_PATH).exists() or not Path(SEL_SNAP_PATH).exists():
    print("[Celda 17b] SKIP: overlay_trades o selection_snapshot no existe")
else:
    df = pl.read_parquet(OVERLAY_PATH)
    sel_snap = json.loads(Path(SEL_SNAP_PATH).read_text(encoding="utf-8"))
    go_keys = {(s["symbol"], s["side"]) for s in sel_snap.get("selections", [])
               if s.get("decision") == "GO"}

    df = df.with_columns(pl.col("entry_time_utc").cast(pl.Date).alias("_date"))
    oos = df.filter(pl.col("segment") == "OOS")
    if go_keys:
        mask = pl.lit(False)
        for sym, side in go_keys:
            mask = mask | ((pl.col("symbol") == sym) & (pl.col("side") == side))
        oos = oos.filter(mask)
    oos = oos.sort("entry_time_utc")
    print(f"[Celda 17b] OOS GO trades: {oos.height} (sides: {sorted(go_keys)})")

    challenge_result = None
    if oos.height > 0:
        sl_trades = oos.filter(pl.col("exit_reason") == "SL")
        sl_ret_median = float(sl_trades["net_pnl_base"].abs().median()) if sl_trades.height > 0 else 0.003
        sl_ret_median = max(sl_ret_median, 1e-8)
        pos_notional = RISK_PER_TRADE_USD / sl_ret_median
        if sl_trades.height > 0:
            print(f"[Celda 17b] Sizing: risk=${RISK_PER_TRADE_USD}, SL_med={sl_ret_median:.4%}, "
                  f"notional=${pos_notional:,.0f}, 1-SL=${sl_ret_median*pos_notional:,.2f}")

        equity = CHALLENGE_CAPITAL
        trading_days = set()
        daily_log = {}
        trades_taken = 0; trades_skipped = 0
        target_reached = False; target_day = None
        violated_daily = False; violated_total = False
        max_daily_loss_seen = 0.0; max_total_dd_seen = 0.0
        total_wins = 0; total_win_usd = 0.0; total_loss_usd = 0.0

        for row in oos.iter_rows(named=True):
            trade_date = row["_date"]
            pnl_usd = row["net_pnl_base"] * pos_notional
            if trade_date not in daily_log:
                daily_log[trade_date] = {"n_trades": 0, "pnl_usd": 0.0, "skipped": 0}
            day = daily_log[trade_date]

            if day["pnl_usd"] <= -_CB_THRESHOLD:
                day["skipped"] += 1; trades_skipped += 1; continue
            if (equity - CHALLENGE_CAPITAL) <= -CHALLENGE_TOTAL_MAX_LOSS:
                violated_total = True; break

            equity += pnl_usd
            day["n_trades"] += 1; day["pnl_usd"] += pnl_usd
            trades_taken += 1; trading_days.add(trade_date)

            if pnl_usd > 0:
                total_wins += 1; total_win_usd += pnl_usd
            else:
                total_loss_usd += abs(pnl_usd)

            if day["pnl_usd"] <= -_CB_THRESHOLD:
                violated_daily = True

            total_dd = equity - CHALLENGE_CAPITAL
            max_total_dd_seen = min(max_total_dd_seen, total_dd)
            if total_dd <= -CHALLENGE_TOTAL_MAX_LOSS:
                violated_total = True; break
            if total_dd >= CHALLENGE_PROFIT_TARGET and len(trading_days) >= CHALLENGE_MIN_DAYS:
                target_reached = True; target_day = str(trade_date); break

        for info in daily_log.values():
            max_daily_loss_seen = min(max_daily_loss_seen, info["pnl_usd"])

        r_days  = len(trading_days) >= CHALLENGE_MIN_DAYS
        r_daily = max_daily_loss_seen > -CHALLENGE_DAILY_MAX_LOSS
        r_total = max_total_dd_seen > -CHALLENGE_TOTAL_MAX_LOSS
        r_target = target_reached
        discipline = sum([r_days, r_daily, r_total, r_target]) * 25

        wr = total_wins / trades_taken if trades_taken > 0 else 0
        avg_win  = total_win_usd  / total_wins if total_wins > 0 else 0
        n_losses = trades_taken - total_wins
        avg_loss = total_loss_usd / n_losses if n_losses > 0 else 0

        daily_summary = [
            {"date": str(d), "n_trades": daily_log[d]["n_trades"],
             "pnl_usd": round(daily_log[d]["pnl_usd"], 2),
             "skipped": daily_log[d]["skipped"]}
            for d in sorted(daily_log)
            if daily_log[d]["n_trades"] > 0 or daily_log[d]["skipped"] > 0
        ]

        challenge_result = {
            "created_utc": _now_utc_iso(), "version": "v1.0.0",
            "sizing": {
                "risk_per_trade_usd": RISK_PER_TRADE_USD,
                "sl_return_median": round(sl_ret_median, 6),
                "position_notional": round(pos_notional, 2),
            },
            "challenge": {
                "account_type": ACCOUNT_TYPE,
                "initial_capital": CHALLENGE_CAPITAL,
                "daily_max_loss_usd": CHALLENGE_DAILY_MAX_LOSS,
                "total_max_loss_usd": CHALLENGE_TOTAL_MAX_LOSS,
                "profit_target_usd": CHALLENGE_PROFIT_TARGET,
                "min_trading_days": CHALLENGE_MIN_DAYS,
                "circuit_breaker_pct": _ruleset.get("circuit_breaker_pct", 1.0),
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
                "min_2_days": r_days, "daily_loss_ok": r_daily,
                "total_loss_ok": r_total, "target_hit": r_target,
            },
            "daily_summary": daily_summary,
        }

        res = challenge_result["results"]
        print(f"[Celda 17b] Challenge OOS: PnL=${res['final_pnl_usd']:+,.0f}, "
              f"disc={res['discipline_pct']}%, target={'SI' if res['target_reached'] else 'NO'}")
        print(f"[Celda 17b] MaxDayLoss=${res['max_daily_loss_usd']:,.0f} "
              f"MaxTotDD=${res['max_total_dd_usd']:,.0f} "
              f"trades={res['trades_taken']} days={res['trading_days']}")

        OUT_CHALLENGE.write_text(
            json.dumps(challenge_result, indent=2, default=str), encoding="utf-8")
        print(f"[Celda 17b] OUT: {OUT_CHALLENGE}")

print(">>> Celda 17b v1.0.0 :: OK")
"""


def patch(dry_run: bool = False) -> None:
    nb = json.loads(NB4_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Verificar que la celda nueva no exista ya
    existing_ids = [c.get("id") for c in cells]
    if NEW_CELL_ID in existing_ids:
        print(f"[patch] SKIP: celda {NEW_CELL_ID} ya existe en NB4")
        return

    # Encontrar índice de la celda ancla
    anchor_idx = next(
        (i for i, c in enumerate(cells) if c.get("id") == AFTER_CELL_ID), None
    )
    if anchor_idx is None:
        raise ValueError(f"Celda ancla {AFTER_CELL_ID!r} no encontrada en NB4")

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": NEW_CELL_ID,
        "metadata": {},
        "outputs": [],
        "source": NEW_CELL_SOURCE,
    }

    insert_at = anchor_idx + 1
    cells.insert(insert_at, new_cell)
    print(f"[patch] Celda {NEW_CELL_ID!r} insertada en índice {insert_at} "
          f"(después de {AFTER_CELL_ID!r} @ {anchor_idx})")
    print(f"[patch] Total celdas: {len(nb['cells'])}")

    if dry_run:
        print("[patch] DRY-RUN: no se escribió nada")
        return

    NB4_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[patch] NB4 guardado: {NB4_PATH}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    patch(dry_run=dry)
