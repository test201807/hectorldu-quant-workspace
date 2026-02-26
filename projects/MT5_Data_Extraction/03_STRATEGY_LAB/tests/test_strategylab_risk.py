"""Tests for risk management module — synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategylab.config import RiskConfig
from strategylab.risk import DailyRiskTracker, compute_mt5_lots, compute_position_size


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


class TestFTMOPositionSizing:
    """FTMO-aware position sizing (account_size_usd > 0)."""

    def test_ftmo_basic_lots(self):
        """account=$10k, risk=1%, SL=$5 → lots = 100/5 = 20, clamped to max=100."""
        cfg = RiskConfig(account_size_usd=10_000, risk_per_trade=0.01,
                         min_pos_size=1.0, max_pos_size=100.0)
        lots = compute_position_size(cfg, entry_price=140.0, sl_distance=5.0)
        assert abs(lots - 20.0) < 1e-9

    def test_ftmo_clamped_to_max(self):
        """Very tight SL → lots clipped to max_pos_size."""
        cfg = RiskConfig(account_size_usd=10_000, risk_per_trade=0.05,
                         min_pos_size=1.0, max_pos_size=50.0)
        lots = compute_position_size(cfg, entry_price=100.0, sl_distance=0.10)
        assert lots == 50.0  # clamped: raw = 10000*0.05/0.1 = 5000 → clipped

    def test_ftmo_contract_size_forex(self):
        """Forex lot: 1 lot = 100_000 units. Risk=$100, SL=10pips=$0.0010/unit.
        lots = 100 / (0.0010 * 100_000) = 100 / 100 = 1.0 lot."""
        cfg = RiskConfig(account_size_usd=10_000, risk_per_trade=0.01,
                         contract_size=100_000, min_pos_size=0.01, max_pos_size=10.0)
        lots = compute_position_size(cfg, entry_price=1.10, sl_distance=0.0010)
        assert abs(lots - 1.0) < 1e-9

    def test_ftmo_zero_account_uses_normalized(self):
        """account_size_usd=0 uses original normalized formula."""
        cfg = RiskConfig(account_size_usd=0, risk_per_trade=0.01,
                         min_pos_size=0.25, max_pos_size=3.0)
        size = compute_position_size(cfg, entry_price=100.0, sl_distance=2.0)
        assert size == 0.25  # clamped to min (0.01/2 = 0.005 < 0.25)

    def test_ftmo_config_fields_exist(self):
        """RiskConfig has FTMO fields."""
        cfg = RiskConfig()
        assert hasattr(cfg, "account_size_usd")
        assert hasattr(cfg, "contract_size")
        assert cfg.account_size_usd == 0.0
        assert cfg.contract_size == 1.0


class TestComputeMT5Lots:
    """compute_mt5_lots() utility tests."""

    def test_basic_stock_cfd(self):
        """NVDA: $10k account, 1% risk, SL=$6 → lots=16.67."""
        lots = compute_mt5_lots(
            account_usd=10_000, risk_per_trade=0.01,
            sl_distance_price=6.0, contract_size=1.0,
        )
        assert abs(lots - 100.0 / 6.0) < 1e-9

    def test_forex_100k_contract(self):
        """EURUSD: $10k, 1% risk, SL=10pips=0.0010, contract=100_000 → 1 lot."""
        lots = compute_mt5_lots(
            account_usd=10_000, risk_per_trade=0.01,
            sl_distance_price=0.0010, contract_size=100_000,
        )
        assert abs(lots - 1.0) < 1e-9

    def test_zero_sl_returns_zero(self):
        assert compute_mt5_lots(10_000, 0.01, 0.0) == 0.0

    def test_zero_account_returns_zero(self):
        assert compute_mt5_lots(0.0, 0.01, 5.0) == 0.0

    def test_scales_with_account(self):
        """Double account → double lots."""
        l1 = compute_mt5_lots(10_000, 0.01, 5.0)
        l2 = compute_mt5_lots(20_000, 0.01, 5.0)
        assert abs(l2 - 2 * l1) < 1e-9


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


class TestDailyRiskTrackerFloating:
    """Tests for FTMO-style floating PnL tracking."""

    def test_effective_equity_no_floating(self):
        tracker = DailyRiskTracker(RiskConfig())
        assert tracker.effective_equity == 1.0

    def test_effective_equity_with_floating(self):
        tracker = DailyRiskTracker(RiskConfig())
        tracker.update_floating_pnl(-0.03)
        assert abs(tracker.effective_equity - 0.97) < 1e-10

    def test_floating_blocks_can_trade_via_daily_cap(self):
        cfg = RiskConfig(daily_loss_cap=-0.02, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        # No closed trade, but open position is down -3%
        tracker.update_floating_pnl(-0.03)
        assert not tracker.can_trade()  # -0.03 < -0.02 cap

    def test_floating_does_not_block_when_within_cap(self):
        cfg = RiskConfig(daily_loss_cap=-0.02, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.update_floating_pnl(-0.01)  # within cap
        assert tracker.can_trade()

    def test_drawdown_includes_floating(self):
        cfg = RiskConfig()
        tracker = DailyRiskTracker(cfg)
        # gain via closed trade → peak = 1.10
        tracker.record_trade(0.10)
        # open position now down -0.08 (floating)
        tracker.update_floating_pnl(-0.08)
        dd = tracker.drawdown()
        expected = (1.10 - 0.08 - 1.10) / 1.10  # = -0.08/1.10
        assert abs(dd - expected) < 1e-10

    def test_is_killed_via_floating(self):
        cfg = RiskConfig(max_drawdown_cap=-0.10)
        tracker = DailyRiskTracker(cfg)
        # open position is down -12% — no closed trades
        tracker.update_floating_pnl(-0.12)
        assert tracker.is_killed()

    def test_is_daily_capped(self):
        cfg = RiskConfig(daily_loss_cap=-0.02, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.record_trade(-0.01)       # -1% closed
        tracker.update_floating_pnl(-0.015)  # -1.5% floating → total -2.5%
        assert tracker.is_daily_capped()

    def test_update_floating_updates_peak(self):
        tracker = DailyRiskTracker(RiskConfig())
        # floating gain pushes effective equity above 1.0
        tracker.update_floating_pnl(0.05)
        assert tracker._peak_equity == 1.05

    def test_new_bar_preserves_floating(self):
        """Floating PnL is NOT reset on new day (position stays open)."""
        cfg = RiskConfig(daily_loss_cap=-0.02, max_trades_per_day=100)
        tracker = DailyRiskTracker(cfg)
        tracker.new_bar("2024-01-01")
        tracker.update_floating_pnl(-0.03)
        tracker.new_bar("2024-01-02")  # new day
        # daily_pnl resets to 0, but floating persists
        assert tracker._floating_pnl == -0.03

    def test_reset_clears_floating(self):
        tracker = DailyRiskTracker(RiskConfig())
        tracker.update_floating_pnl(-0.05)
        tracker.reset(equity=1.0)
        assert tracker._floating_pnl == 0.0
        assert tracker.effective_equity == 1.0

    def test_update_floating_zero_after_close(self):
        """After closing a position, floating resets to 0."""
        tracker = DailyRiskTracker(RiskConfig())
        tracker.update_floating_pnl(-0.03)
        assert tracker.effective_equity < 1.0
        tracker.update_floating_pnl(0.0)
        assert tracker.effective_equity == 1.0
