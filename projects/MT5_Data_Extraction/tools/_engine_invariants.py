"""
Engine invariant checks (B.2 from sprint plan).
3 checks:
  1) Higher costs => performance does NOT improve
  2) No future data leakage (entry_time > signal_time)
  3) Weekend gate + exec-bar gate active (0 weekend entries)
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8')

import polars as pl
from pathlib import Path

PROJECT = Path(__file__).parent.parent

def find_latest_run(strategy_dir):
    runs = sorted([d for d in strategy_dir.glob("run_*") if d.is_dir()])
    return runs[-1] if runs else None

print("=" * 70)
print("ENGINE INVARIANT CHECKS")
print("=" * 70)

passed = 0
failed = 0

# --- TREND ---
trend_run = find_latest_run(PROJECT / "outputs" / "trend_v2")
if trend_run:
    print(f"\nTREND run: {trend_run.name}")
    trades = pl.read_parquet(trend_run / "trades_engine_v2.parquet")

    # Check 1: Higher costs => worse performance
    base_pnl = trades["net_pnl_base"].sum()
    stress_pnl = trades["net_pnl_stress"].sum()
    print(f"\n  CHECK 1: Higher costs => worse performance")
    print(f"    net_pnl_base:   {base_pnl:.4f}")
    print(f"    net_pnl_stress: {stress_pnl:.4f}")
    if stress_pnl <= base_pnl:
        print(f"    PASS: stress ({stress_pnl:.4f}) <= base ({base_pnl:.4f})")
        passed += 1
    else:
        print(f"    FAIL: stress > base (IMPOSSIBLE if costs are higher)")
        failed += 1

    # Check 2: No future data leakage
    print(f"\n  CHECK 2: No future data leakage (entry_time > signal_time)")
    leaks = trades.filter(pl.col("entry_time_utc") <= pl.col("signal_time_utc"))
    print(f"    Trades with entry <= signal: {leaks.height}")
    if leaks.height == 0:
        print(f"    PASS: all {trades.height} trades have entry > signal")
        passed += 1
    else:
        print(f"    FAIL: {leaks.height} trades have entry <= signal (LOOKAHEAD)")
        failed += 1

    # Check 3: Weekend gate
    print(f"\n  CHECK 3: Weekend gate (0 weekend entries)")
    qa_path = trend_run / "engine_qa_report_v2.json"
    if qa_path.exists():
        with open(qa_path) as f:
            qa = json.load(f)
        weekend = qa.get("n_weekend", qa.get("weekend_entries", "N/A"))
        lookahead = qa.get("n_lookahead", qa.get("lookahead_entries", "N/A"))
        status = qa.get("status", "UNKNOWN")
        print(f"    QA status: {status}")
        print(f"    Weekend entries: {weekend}")
        print(f"    Lookahead entries: {lookahead}")
        if status == "PASS":
            print(f"    PASS")
            passed += 1
        else:
            print(f"    FAIL: QA status is {status}")
            failed += 1
    else:
        print(f"    SKIP: QA report not found")
else:
    print("\nTREND: No runs found")

# --- RANGE ---
range_run = find_latest_run(PROJECT / "outputs" / "range_v1")
if range_run:
    print(f"\nRANGE run: {range_run.name}")
    trades = pl.read_parquet(range_run / "trades_engine_range_v1.parquet")

    # Check 1
    base_pnl = trades["net_pnl_base"].sum()
    stress_pnl = trades["net_pnl_stress"].sum()
    print(f"\n  CHECK 1: Higher costs => worse performance")
    print(f"    net_pnl_base:   {base_pnl:.4f}")
    print(f"    net_pnl_stress: {stress_pnl:.4f}")
    if stress_pnl <= base_pnl:
        print(f"    PASS: stress ({stress_pnl:.4f}) <= base ({base_pnl:.4f})")
        passed += 1
    else:
        print(f"    FAIL: stress > base")
        failed += 1

    # Check 2
    print(f"\n  CHECK 2: No future data leakage")
    leaks = trades.filter(pl.col("entry_time_utc") <= pl.col("signal_time_utc"))
    print(f"    Trades with entry <= signal: {leaks.height}")
    if leaks.height == 0:
        print(f"    PASS: all {trades.height} trades have entry > signal")
        passed += 1
    else:
        print(f"    FAIL: {leaks.height} trades have entry <= signal")
        failed += 1

    # Check 3
    print(f"\n  CHECK 3: Weekend gate")
    qa_path = range_run / "engine_qa_report_range_v1.json"
    if qa_path.exists():
        with open(qa_path) as f:
            qa = json.load(f)
        status = qa.get("status", "UNKNOWN")
        print(f"    QA status: {status}")
        if status == "PASS":
            print(f"    PASS")
            passed += 1
        else:
            print(f"    FAIL: QA status is {status}")
            failed += 1
    else:
        print(f"    SKIP: QA report not found")
else:
    print("\nRANGE: No runs found")

print(f"\n{'=' * 70}")
print(f"RESULT: {passed} PASSED, {failed} FAILED")
if failed > 0:
    print("*** ENGINE INTEGRITY COMPROMISED ***")
    sys.exit(1)
else:
    print("*** ALL INVARIANTS HOLD ***")
print(f"{'=' * 70}")
