"""Tests for institutional guards: costs, NaN handling, dedup, schema, VaR."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl
from strategylab.backtest_engine import run_engine, trades_to_dataframe
from strategylab.config import CostsConfig, EngineConfig, RiskConfig
from strategylab.costs import compute_fill_cost, roundtrip_cost_dec, stressed_config
from strategylab.data_loader import make_synthetic_bars
from strategylab.monte_carlo import iid_bootstrap, var_cvar
from strategylab.schema import BARS_SCHEMA, TRADES_SCHEMA, validate_columns


def _make_data_and_signals(n: int = 5000, seed: int = 42):
    df = make_synthetic_bars(symbol="TEST", n_bars=n, seed=seed)
    df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))
    sig_long = [((i // 50) % 2 == 0) for i in range(n)]
    sig_short = [((i // 50) % 2 == 1) for i in range(n)]
    return df, sig_long, sig_short


# ---------------------------------------------------------------------------
# Cost model correctness
# ---------------------------------------------------------------------------

class TestCostModel:

    def test_roundtrip_cost_formula(self):
        """Verify roundtrip cost = 2 * (spread/2 + commission + slippage) / 10000."""
        cfg = CostsConfig(spread_bps=4.0, commission_bps=2.0, slippage_bps=1.0)
        # one_way = 4/2 + 2 + 1 = 5 bps
        # roundtrip = 2 * 5 / 10000 = 0.001
        assert abs(roundtrip_cost_dec(cfg) - 0.001) < 1e-12

    def test_net_pnl_includes_cost(self):
        """Every trade's net_pnl must equal gross_pnl minus roundtrip cost."""
        cfg_costs = CostsConfig(spread_bps=10.0, commission_bps=5.0, slippage_bps=3.0)
        cost_rt = roundtrip_cost_dec(cfg_costs)
        assert cost_rt > 0

        df, sl, ss = _make_data_and_signals()
        t_list = df.get_column("time_utc").to_list()
        mid = t_list[len(t_list) // 2]
        trades = run_engine(
            df=df, fold_id="F00",
            is_start=t_list[0], is_end=mid,
            oos_start=mid, oos_end=t_list[-1],
            signal_long=sl, signal_short=ss,
            engine_cfg=EngineConfig(),
            costs_cfg=cfg_costs,
            risk_cfg=RiskConfig(),
        )
        for t in trades:
            expected_net = t.gross_pnl - cost_rt
            assert abs(t.net_pnl - expected_net) < 1e-12, (
                f"net_pnl={t.net_pnl} != gross-cost={expected_net}"
            )

    def test_stressed_config_multiplies(self):
        base = CostsConfig(spread_bps=4.0, commission_bps=2.0, slippage_bps=1.0)
        stress = stressed_config(base, cost_factor=2.0, slippage_factor=3.0)
        assert stress.spread_bps == 8.0
        assert stress.commission_bps == 4.0
        assert stress.slippage_bps == 3.0

    def test_fill_cost_components(self):
        cfg = CostsConfig(spread_bps=4.0, commission_bps=2.0, slippage_bps=1.0)
        fill = compute_fill_cost(100.0, 1.0, cfg)
        assert fill.spread_cost > 0
        assert fill.commission_cost > 0
        assert fill.slippage_cost > 0
        assert abs(fill.total - (fill.spread_cost + fill.commission_cost + fill.slippage_cost)) < 1e-12

    def test_borrow_in_config(self):
        """Borrow field exists in CostsConfig (for future use)."""
        cfg = CostsConfig(borrow_bps_annual=50.0)
        assert cfg.borrow_bps_annual == 50.0


# ---------------------------------------------------------------------------
# NaN / None handling in engine
# ---------------------------------------------------------------------------

class TestNaNHandling:

    def test_engine_handles_nan_prices(self):
        """Engine should not crash with NaN values in OHLC."""
        df = make_synthetic_bars(symbol="NAN_TEST", n_bars=500, seed=99)
        df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))

        # Inject some NaN values
        h = df.get_column("high").to_list()
        h[10] = float("nan")
        h[20] = None
        df = df.with_columns(pl.Series("high", h))

        sig_long = [True] * 500
        sig_short = [False] * 500
        t_list = df.get_column("time_utc").to_list()
        mid = t_list[250]

        # Should not raise
        trades = run_engine(
            df=df, fold_id="F00",
            is_start=t_list[0], is_end=mid,
            oos_start=mid, oos_end=t_list[-1],
            signal_long=sig_long, signal_short=sig_short,
            engine_cfg=EngineConfig(),
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
        )
        assert isinstance(trades, list)

    def test_engine_handles_nan_atr(self):
        """Engine should handle NaN in ATR gracefully."""
        df = make_synthetic_bars(symbol="ATR_NAN", n_bars=500, seed=77)
        atr = [float("nan")] * 50 + [0.5] * 450
        df = df.with_columns(pl.Series("atr_price", atr))

        sig_long = [True] * 500
        sig_short = [False] * 500
        t_list = df.get_column("time_utc").to_list()
        mid = t_list[250]

        trades = run_engine(
            df=df, fold_id="F00",
            is_start=t_list[0], is_end=mid,
            oos_start=mid, oos_end=t_list[-1],
            signal_long=sig_long, signal_short=sig_short,
            engine_cfg=EngineConfig(),
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
        )
        assert isinstance(trades, list)


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestDedup:

    def test_load_bars_deduplicates(self):
        """load_bars should remove duplicate (symbol, time_utc) keeping last."""
        import tempfile
        df = make_synthetic_bars(symbol="DUP", n_bars=100, seed=42)
        # Duplicate first 10 rows with different close
        dup = df.head(10).with_columns(pl.col("close") + 999.0)
        combined = pl.concat([df, dup])
        assert combined.height == 110

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            combined.write_parquet(f.name)
            from strategylab.data_loader import load_bars
            loaded = load_bars(f.name)

        assert loaded.height == 100  # duplicates removed
        # Check that "last" was kept (the +999 values)
        first_10_close = loaded.head(10).get_column("close").to_list()
        for c in first_10_close:
            assert c > 900  # the +999 values should be kept


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_valid_bars_schema(self):
        df = make_synthetic_bars(n_bars=10)
        validate_columns(df, BARS_SCHEMA, "bars")  # should not raise

    def test_missing_column_raises(self):
        df = pl.DataFrame({"symbol": ["A"], "time_utc": [None]})
        try:
            validate_columns(df, BARS_SCHEMA, "bars")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "missing required columns" in str(e)

    def test_trades_schema(self):
        df, sl, ss = _make_data_and_signals()
        t_list = df.get_column("time_utc").to_list()
        mid = t_list[len(t_list) // 2]
        trades = run_engine(
            df=df, fold_id="F00",
            is_start=t_list[0], is_end=mid,
            oos_start=mid, oos_end=t_list[-1],
            signal_long=sl, signal_short=ss,
            engine_cfg=EngineConfig(),
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
        )
        if trades:
            tdf = trades_to_dataframe(trades)
            validate_columns(tdf, TRADES_SCHEMA, "trades")  # should not raise


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

class TestVaRCVaR:

    def test_var_cvar_basic(self):
        vals = list(range(100))  # 0..99
        var5, cvar5 = var_cvar(vals, alpha=0.05)
        assert var5 == 5  # 5th percentile of 0..99
        assert cvar5 <= var5  # CVaR is mean of tail, should be <= VaR

    def test_var_cvar_empty(self):
        var, cvar = var_cvar([], alpha=0.05)
        assert var == 0.0
        assert cvar == 0.0

    def test_var_from_mc(self):
        """VaR/CVaR from MC results should be consistent with percentiles."""
        import random
        rng = random.Random(42)
        rets = [rng.gauss(0.001, 0.01) for _ in range(200)]
        result = iid_bootstrap(rets, n_sims=500, seed=42)
        var5, cvar5 = var_cvar(result.final_equities, alpha=0.05)
        assert var5 <= result.percentiles["p25"]  # VaR5 should be below p25
        assert cvar5 <= var5


# ---------------------------------------------------------------------------
# Embargo in WFO
# ---------------------------------------------------------------------------

class TestEmbargoWFO:

    def test_embargo_creates_gap(self):
        from datetime import datetime, timezone

        from strategylab.wfo import build_wfo_folds

        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3,
                                step_months=3, embargo_days=5)
        for f in folds:
            gap = (f.oos_start - f.is_end).days
            assert gap >= 5, f"Embargo gap is {gap} days, expected >= 5"

    def test_no_embargo_no_gap(self):
        from datetime import datetime, timezone

        from strategylab.wfo import build_wfo_folds

        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        folds = build_wfo_folds(start, end, is_months=12, oos_months=3,
                                step_months=3, embargo_days=0)
        for f in folds:
            gap = (f.oos_start - f.is_end).days
            assert gap == 0


# ---------------------------------------------------------------------------
# Column rename (timestamp_utc → time_utc)
# ---------------------------------------------------------------------------

class TestColumnRename:

    def test_timestamp_utc_renamed(self):
        """load_bars should rename timestamp_utc → time_utc transparently."""
        import tempfile
        from datetime import datetime, timezone

        from strategylab.data_loader import load_bars

        df = pl.DataFrame({
            "symbol": ["TEST"] * 5,
            "timestamp_utc": [
                datetime(2024, 1, i, tzinfo=timezone.utc) for i in range(1, 6)
            ],
            "open": [1.0] * 5,
            "high": [1.1] * 5,
            "low": [0.9] * 5,
            "close": [1.05] * 5,
            "tick_volume": [100] * 5,
            "spread_points": [2] * 5,
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.write_parquet(f.name)
            loaded = load_bars(f.name)

        assert "time_utc" in loaded.columns
        assert "timestamp_utc" not in loaded.columns
        assert "volume" in loaded.columns
        assert "tick_volume" not in loaded.columns
        assert "spread" in loaded.columns
        assert "spread_points" not in loaded.columns
        assert loaded.schema["time_utc"] == pl.Datetime("us", "UTC")

    def test_epoch_seconds_coerced(self):
        """Integer epoch timestamps should be auto-detected and coerced."""
        import tempfile

        from strategylab.data_loader import load_bars

        df = pl.DataFrame({
            "symbol": ["TEST"] * 3,
            "timestamp_utc": [1704067200, 1704153600, 1704240000],  # epoch seconds
            "open": [1.0] * 3,
            "high": [1.1] * 3,
            "low": [0.9] * 3,
            "close": [1.05] * 3,
            "tick_volume": [100] * 3,
            "spread_points": [2] * 3,
        })
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.write_parquet(f.name)
            loaded = load_bars(f.name)

        assert loaded.schema["time_utc"] == pl.Datetime("us", "UTC")
        assert loaded["time_utc"].dt.year().to_list() == [2024, 2024, 2024]
