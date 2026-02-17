"""Tests for risk management module — synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategylab.config import RiskConfig
from strategylab.risk import DailyRiskTracker, compute_position_size


class TestPositionSizing:
    """Position sizing tests."""

    def test_basic_sizing(self):
        cfg = RiskConfig(risk_per_trade=0.01, min_pos_size=0.25, max_pos_size=3.0)
        size = compute_position_size(cfg, entry_price=100.0, sl_distance=2.0)
        # risk_amount = 1.0 * 0.01 = 0.01; raw_size = 0.01 / 2.0 = 0.005
        # clamped to min_pos_size = 0.25
        assert size == 0.25

    def test_zero_sl_returns_zero(self):
        cfg = RiskConfig()
        assert compute_position_size(cfg, entry_price=100.0, sl_distance=0.0) == 0.0

    def test_negative_sl_returns_zero(self):
        cfg = RiskConfig()
        assert compute_position_size(cfg, entry_price=100.0, sl_distance=-1.0) == 0.0

    def test_zero_price_returns_zero(self):
        cfg = RiskConfig()
        assert compute_position_size(cfg, entry_price=0.0, sl_distance=1.0) == 0.0

    def test_max_clamp(self):
        cfg = RiskConfig(risk_per_trade=0.50, max_pos_size=3.0)
        size = compute_position_size(cfg, entry_price=100.0, sl_distance=0.001)
        assert size == 3.0

    def test_size_in_bounds(self):
        cfg = RiskConfig()
        size = compute_position_size(cfg, entry_price=100.0, sl_distance=1.0)
        assert cfg.min_pos_size <= size <= cfg.max_pos_size


class TestDailyRiskTracker:
    """DailyRiskTracker tests."""

    def test_initial_state(self):
        tracker = DailyRiskTracker(RiskConfig())
        assert tracker.equity == 1.0
        assert tracker.can_trade()
        assert not tracker.is_killed()

    def test_daily_trade_limit(self):
        cfg = RiskConfig(max_trades_per_day=2)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        assert tracker.can_trade()
        tracker.record_trade(0.001)
        assert tracker.can_trade()
        tracker.record_trade(0.001)
        assert not tracker.can_trade()  # hit limit

    def test_daily_loss_cap(self):
        cfg = RiskConfig(daily_loss_cap=-0.02, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(-0.03)  # exceed daily loss cap
        assert not tracker.can_trade()

    def test_daily_profit_cap(self):
        cfg = RiskConfig(daily_profit_cap=0.03, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(0.04)  # exceed daily profit cap
        assert not tracker.can_trade()

    def test_date_reset(self):
        cfg = RiskConfig(max_trades_per_day=1)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(0.001)
        assert not tracker.can_trade()
        tracker.new_bar("2024-01-02")  # new day
        assert tracker.can_trade()

    def test_drawdown_kill(self):
        cfg = RiskConfig(max_drawdown_cap=-0.10)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(-0.12)  # exceed max DD
        assert tracker.is_killed()

    def test_drawdown_calculation(self):
        cfg = RiskConfig()
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(0.10)  # equity = 1.10, peak = 1.10
        tracker.record_trade(-0.05)  # equity = 1.05, peak = 1.10
        dd = tracker.drawdown()
        expected = (1.05 - 1.10) / 1.10
        assert abs(dd - expected) < 1e-10

    def test_reset(self):
        cfg = RiskConfig()
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(-0.05)
        tracker.reset(equity=2.0)
        assert tracker.equity == 2.0
        assert tracker.can_trade()
