"""Tests for backtest engine — uses only synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategylab.backtest_engine import Trade, run_engine, trades_to_dataframe
from strategylab.config import CostsConfig, EngineConfig, RiskConfig
from strategylab.data_loader import make_synthetic_bars


def _make_data_and_signals(n: int = 5000, seed: int = 42):
    """Create synthetic bars + simple signals for testing."""
    df = make_synthetic_bars(symbol="TEST", n_bars=n, seed=seed)
    # Add ATR column
    import polars as pl
    df = df.with_columns((pl.col("close") * 0.005).alias("atr_price"))

    # Simple alternating signals (every 50 bars switch)
    sig_long = [((i // 50) % 2 == 0) for i in range(n)]
    sig_short = [((i // 50) % 2 == 1) for i in range(n)]
    return df, sig_long, sig_short


class TestEngineBasic:
    """Basic engine behaviour tests."""

    def test_returns_list_of_trades(self):
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
            symbol="TEST",
        )
        assert isinstance(trades, list)
        assert all(isinstance(t, Trade) for t in trades)

    def test_minimum_bars_guard(self):
        """Engine returns empty for very small DataFrames."""
        df, sl, ss = _make_data_and_signals(n=5)
        t_list = df.get_column("time_utc").to_list()
        trades = run_engine(
            df=df, fold_id="F00",
            is_start=t_list[0], is_end=t_list[-1],
            oos_start=t_list[0], oos_end=t_list[-1],
            signal_long=sl, signal_short=ss,
            engine_cfg=EngineConfig(),
            costs_cfg=CostsConfig(),
            risk_cfg=RiskConfig(),
        )
        assert trades == []

    def test_trades_have_correct_fields(self):
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
            symbol="TEST",
        )
        if trades:
            t = trades[0]
            assert hasattr(t, "symbol")
            assert hasattr(t, "side")
            assert hasattr(t, "net_pnl")
            assert hasattr(t, "exit_reason")
            assert t.side in ("LONG", "SHORT")

    def test_no_look_ahead_bias(self):
        """Entry time must be after signal time."""
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
        for t in trades:
            assert t.entry_time_utc >= t.signal_time_utc

    def test_trail_greater_than_sl(self):
        """Verify default config has trail_atr > sl_atr (bug fix)."""
        cfg = EngineConfig()
        assert cfg.trail_atr is not None
        assert cfg.trail_atr > cfg.sl_atr, "Trail must be > SL to avoid unreachable SL"


class TestTradesToDataframe:
    """Tests for trade serialisation."""

    def test_empty_trades(self):
        result = trades_to_dataframe([])
        assert result.is_empty()

    def test_roundtrip(self):
        df, sl, ss = _make_data_and_signals(n=3000)
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
            assert tdf.height == len(trades)
            assert "net_pnl" in tdf.columns
            assert "exit_reason" in tdf.columns


class TestExitReasons:
    """Verify exit reasons are valid."""

    VALID_REASONS = {"SL", "TP", "TRAIL", "TIME", "REGIME_OFF", "WEEKEND", "DD_KILL", "DAILY_CAP"}

    def test_all_exit_reasons_valid(self):
        df, sl, ss = _make_data_and_signals(n=5000, seed=123)
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
        for t in trades:
            assert t.exit_reason in self.VALID_REASONS, f"Invalid exit: {t.exit_reason}"
