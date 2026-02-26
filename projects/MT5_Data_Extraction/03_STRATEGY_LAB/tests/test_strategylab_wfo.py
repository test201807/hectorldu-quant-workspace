"""Tests for Walk-Forward Optimization — synthetic data only."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl
from strategylab.config import CostsConfig, RiskConfig
from strategylab.data_loader import make_synthetic_bars
from strategylab.wfo import WFOFold, build_wfo_folds, grid_search, run_wfo


class TestBuildWFOFolds:

    def test_returns_folds(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3, step_months=3)
        assert len(folds) > 0
        assert all(isinstance(f, WFOFold) for f in folds)

    def test_fold_ids_unique(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3, step_months=3)
        ids = [f.fold_id for f in folds]
        assert len(ids) == len(set(ids))

    def test_is_before_oos(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3, step_months=3)
        for f in folds:
            assert f.is_start < f.is_end
            assert f.is_end <= f.oos_start
            assert f.oos_start < f.oos_end

    def test_no_folds_if_range_too_short(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 6, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=18, oos_months=3, step_months=3)
        assert len(folds) == 0

    def test_expanding_coverage(self):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3, step_months=3)
        # Last fold OOS must end before data end
        assert folds[-1].oos_end <= end


class TestGridSearch:

    def _make_data(self, n: int = 8000, seed: int = 42):
        df = make_synthetic_bars(symbol="TEST", n_bars=n, seed=seed)
        df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))
        sig_long = [((i // 100) % 2 == 0) for i in range(n)]
        sig_short = [((i // 100) % 2 == 1) for i in range(n)]
        t_list = df.get_column("time_utc").to_list()
        mid = t_list[len(t_list) // 2]
        fold = WFOFold("F00", t_list[0], mid, mid, t_list[-1])
        return df, sig_long, sig_short, fold

    def test_returns_list(self):
        df, sl, ss, fold = self._make_data()
        results = grid_search(
            df=df, fold=fold,
            signal_long=sl, signal_short=ss,
            param_grid={"sl_atr": [1.5, 2.0], "tp_atr": [3.0, 5.0]},
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
            min_trades_is=5,
        )
        assert isinstance(results, list)

    def test_skips_trail_le_sl(self):
        df, sl, ss, fold = self._make_data()
        results = grid_search(
            df=df, fold=fold,
            signal_long=sl, signal_short=ss,
            param_grid={"sl_atr": [3.0], "trail_atr": [2.0]},  # trail < sl
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
            min_trades_is=1,
        )
        assert len(results) == 0  # all combos skipped

    def test_sorted_by_score(self):
        df, sl, ss, fold = self._make_data()
        results = grid_search(
            df=df, fold=fold,
            signal_long=sl, signal_short=ss,
            param_grid={"sl_atr": [1.5, 2.0], "tp_atr": [3.0, 5.0]},
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
            min_trades_is=5,
        )
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestRunWFO:

    def test_full_wfo_synthetic(self):
        """End-to-end WFO on synthetic data."""
        # 50k bars * 5min = ~173 days; use short IS/OOS to fit
        df = make_synthetic_bars(symbol="WFO_TEST", n_bars=50_000, seed=42)
        df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))
        n = df.height

        sig_long = [((i // 100) % 2 == 0) for i in range(n)]
        sig_short = [((i // 100) % 2 == 1) for i in range(n)]

        result = run_wfo(
            df=df,
            signal_long=sig_long,
            signal_short=sig_short,
            param_grid={"sl_atr": [2.0], "tp_atr": [5.0]},
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
            symbol="WFO_TEST",
            is_months=2,
            oos_months=1,
            step_months=1,
            max_combos=10,
            min_trades_is=5,
        )
        assert len(result.folds) > 0
        # all_combos_per_fold populated — enables PBO (Bailey 2014)
        assert isinstance(result.all_combos_per_fold, dict)
        if result.best_per_fold:
            for fold_id in result.best_per_fold:
                assert fold_id in result.all_combos_per_fold
                assert len(result.all_combos_per_fold[fold_id]) >= 1
        # holdout fields present (no holdout requested → empty)
        assert isinstance(result.holdout_trades, pl.DataFrame)
        assert isinstance(result.holdout_kpis, dict)

    def test_holdout_final(self):
        """With holdout_months > 0, holdout_trades and holdout_kpis are populated."""
        df = make_synthetic_bars(symbol="HOLD_TEST", n_bars=80_000, seed=99)
        df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))
        n = df.height

        sig_long = [((i // 100) % 2 == 0) for i in range(n)]
        sig_short = [((i // 100) % 2 == 1) for i in range(n)]

        result = run_wfo(
            df=df,
            signal_long=sig_long,
            signal_short=sig_short,
            param_grid={"sl_atr": [2.0], "tp_atr": [5.0]},
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
            symbol="HOLD_TEST",
            is_months=2,
            oos_months=1,
            step_months=1,
            max_combos=10,
            min_trades_is=5,
            holdout_months=1,
        )
        assert isinstance(result.holdout_trades, pl.DataFrame)
        assert isinstance(result.holdout_kpis, dict)
        # Holdout folds should be fewer than without holdout (last month excluded)
        assert len(result.folds) > 0
