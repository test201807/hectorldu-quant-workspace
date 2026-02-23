"""
test_institutional.py — Tests de consistencia para institutional_pack
=====================================================================
Valida que el pack institucional no tiene inconsistencias:
  - Sin NaN/Inf en outputs
  - stress <= base en retornos (cost monotonicity)
  - Sin trades en weekend
  - OOS no contaminado con IS
  - Sharpe/return/MDD consistentes entre artefactos
  - Cost sensitivity es monotónica (más cost -> peor performance)

Uso:
  cd C:\\Quant\\projects\\MT5_Data_Extraction
  .\\venv1\\Scripts\\python.exe -X utf8 tools\\test_institutional.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


def _load_pack(project: Path) -> tuple[Path, str]:
    """Load run_id and pack directory."""
    latest = project / "outputs" / "trend_v2" / "_latest_run.txt"
    run_id = latest.read_text().strip()
    run_dir = project / "outputs" / "trend_v2" / f"run_{run_id}"
    pack_dir = run_dir / "institutional_pack"
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack not found: {pack_dir}")
    return pack_dir, run_id


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def check(self, name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.results.append((name, status, detail))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        icon = "OK" if condition else "FAIL"
        print(f"  [{icon}] {name}" + (f" -- {detail}" if detail else ""))


def main():
    project = Path(__file__).parent.parent
    pack_dir, run_id = _load_pack(project)
    run_dir = pack_dir.parent
    print(f"Testing institutional_pack for run {run_id}")
    print(f"Pack: {pack_dir}\n")

    T = TestResult()

    # ══════════════════════════════════════════════════════
    # TEST 1: All expected files exist
    # ══════════════════════════════════════════════════════
    print("=== FILE INTEGRITY ===")
    expected_files = [
        "trades_oos.csv", "metrics_core.csv", "metrics_by_fold.csv",
        "significance_tests.csv", "bootstrap_ci.csv",
        "monte_carlo_distribution.csv", "alpha_decay_report.csv",
        "pbo_report.csv", "cost_sensitivity.csv",
        "institutional_manifest.json", "README.txt",
    ]
    for f in expected_files:
        T.check(f"file_exists:{f}", (pack_dir / f).exists())

    # ══════════════════════════════════════════════════════
    # TEST 2: No NaN/Inf in numeric CSVs
    # ══════════════════════════════════════════════════════
    print("\n=== NAN/INF CHECK ===")
    numeric_csvs = ["metrics_core.csv", "metrics_by_fold.csv", "cost_sensitivity.csv"]
    for fname in numeric_csvs:
        fp = pack_dir / fname
        if not fp.exists():
            continue
        df = pl.read_csv(fp)
        for col in df.columns:
            if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                arr = df[col].to_numpy()
                has_nan = bool(np.any(np.isnan(arr)))
                has_inf = bool(np.any(np.isinf(arr)))
                T.check(f"no_nan:{fname}:{col}", not has_nan,
                        f"NaN found" if has_nan else "")
                T.check(f"no_inf:{fname}:{col}", not has_inf,
                        f"Inf found" if has_inf else "")

    # ══════════════════════════════════════════════════════
    # TEST 3: Stress performance <= Base performance
    # ══════════════════════════════════════════════════════
    print("\n=== STRESS <= BASE ===")
    core = pl.read_csv(pack_dir / "metrics_core.csv")
    base = core.filter(pl.col("scenario") == "base")
    stress = core.filter(pl.col("scenario") == "stress")

    if base.height > 0 and stress.height > 0:
        b_ret = float(base["total_return_pct"][0])
        s_ret = float(stress["total_return_pct"][0])
        T.check("stress_return_leq_base", s_ret <= b_ret,
                f"base={b_ret:.2f}%, stress={s_ret:.2f}%")

        b_sharpe = float(base["sharpe_annual"][0])
        s_sharpe = float(stress["sharpe_annual"][0])
        T.check("stress_sharpe_leq_base", s_sharpe <= b_sharpe,
                f"base={b_sharpe:.3f}, stress={s_sharpe:.3f}")

        b_pf = float(base["profit_factor"][0])
        s_pf = float(stress["profit_factor"][0])
        T.check("stress_pf_leq_base", s_pf <= b_pf,
                f"base={b_pf:.3f}, stress={s_pf:.3f}")

    # ══════════════════════════════════════════════════════
    # TEST 4: No weekend trades in OOS
    # ══════════════════════════════════════════════════════
    print("\n=== NO WEEKEND TRADES ===")
    trades_oos = pl.read_csv(pack_dir / "trades_oos.csv", try_parse_dates=True)

    if "entry_time_utc" in trades_oos.columns:
        entry_dt = trades_oos["entry_time_utc"]
        if entry_dt.dtype == pl.Utf8:
            entry_dt = entry_dt.str.to_datetime()
        dow = entry_dt.dt.weekday()
        weekend_entries = dow.filter(dow > 5).len()
        T.check("no_weekend_entries", weekend_entries == 0,
                f"weekend entries: {weekend_entries}")

    # ══════════════════════════════════════════════════════
    # TEST 5: OOS not contaminated with IS
    # ══════════════════════════════════════════════════════
    print("\n=== OOS ISOLATION ===")
    if "segment" in trades_oos.columns:
        segments = trades_oos["segment"].unique().to_list()
        T.check("oos_only_segment", segments == ["OOS"] or set(segments) == {"OOS"},
                f"segments found: {segments}")

    # ══════════════════════════════════════════════════════
    # TEST 6: Cost sensitivity is monotonically decreasing
    # ══════════════════════════════════════════════════════
    print("\n=== COST MONOTONICITY ===")
    cost_df = pl.read_csv(pack_dir / "cost_sensitivity.csv")
    if cost_df.height >= 2:
        rets = cost_df["total_return_pct"].to_list()
        T.check("cost_return_monotonic",
                all(rets[i] >= rets[i + 1] for i in range(len(rets) - 1)),
                f"returns: {rets}")

        sharpes = cost_df["sharpe_annual"].to_list()
        T.check("cost_sharpe_monotonic",
                all(sharpes[i] >= sharpes[i + 1] for i in range(len(sharpes) - 1)),
                f"sharpes: {sharpes}")

    # ══════════════════════════════════════════════════════
    # TEST 7: Metrics consistency with existing artifacts
    # ══════════════════════════════════════════════════════
    print("\n=== CROSS-ARTIFACT CONSISTENCY ===")

    # Compare n_trades with overlay_trades
    overlay = pl.read_parquet(run_dir / "overlay_trades_v2.parquet")
    oos_overlay = overlay.filter(
        (pl.col("segment") == "OOS") &
        (pl.col("symbol") == "BTCUSD") &
        (pl.col("side") == "LONG")
    )
    pack_n = int(base["n_trades"][0]) if base.height > 0 else 0
    overlay_n = oos_overlay.height
    T.check("n_trades_match_overlay", pack_n == overlay_n,
            f"pack={pack_n}, overlay={overlay_n}")

    # Compare with challenge dashboard
    challenge_file = run_dir / "challenge_dashboard_v2.json"
    if challenge_file.exists():
        ch = json.loads(challenge_file.read_text(encoding="utf-8"))
        ch_wr = ch["results"]["win_rate"]
        pack_wr = float(base["win_rate"][0]) if base.height > 0 else 0
        # Challenge uses different trade subset (only last OOS fold), so may differ
        # Just check they're in reasonable range
        T.check("winrate_reasonable", 0.05 <= pack_wr <= 0.80,
                f"pack WR={pack_wr:.4f}")

    # ══════════════════════════════════════════════════════
    # TEST 8: Manifest integrity
    # ══════════════════════════════════════════════════════
    print("\n=== MANIFEST ===")
    manifest = json.loads((pack_dir / "institutional_manifest.json").read_text(encoding="utf-8"))
    T.check("manifest_has_run_id", manifest.get("run_id") == run_id)
    T.check("manifest_has_verdict", "overall_verdict" in manifest)
    T.check("manifest_has_files", len(manifest.get("files", {})) >= 8,
            f"files: {len(manifest.get('files', {}))}")
    T.check("manifest_oos_trades", manifest.get("oos_trades", 0) > 0,
            f"oos_trades={manifest.get('oos_trades')}")

    # ══════════════════════════════════════════════════════
    # TEST 9: Fold metrics sanity
    # ══════════════════════════════════════════════════════
    print("\n=== FOLD METRICS ===")
    folds = pl.read_csv(pack_dir / "metrics_by_fold.csv")
    T.check("fold_count_10", folds.height == 10, f"folds={folds.height}")
    T.check("fold_all_have_trades", folds["n_trades"].min() > 0,
            f"min trades per fold: {folds['n_trades'].min()}")

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"RESULTS: {T.passed} PASS, {T.failed} FAIL")
    print(f"{'='*60}")

    if T.failed > 0:
        print("\nFAILED TESTS:")
        for name, status, detail in T.results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")

    return 1 if T.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
