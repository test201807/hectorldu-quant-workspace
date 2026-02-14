"""Risk management: position sizing, exposure limits, kill-switches."""

from __future__ import annotations

from .config import RiskConfig


def compute_position_size(
    risk_config: RiskConfig,
    entry_price: float,
    sl_distance: float,
    equity: float = 1.0,
) -> float:
    """Risk-based position sizing. Returns size in units."""
    if sl_distance <= 0 or entry_price <= 0:
        return 0.0
    risk_amount = equity * risk_config.risk_per_trade
    raw_size = risk_amount / sl_distance
    clamped = max(risk_config.min_pos_size, min(risk_config.max_pos_size, raw_size))
    return clamped


class DailyRiskTracker:
    """Track daily PnL and enforce caps."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._current_date: object = None
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._peak_equity: float = 1.0
        self._equity: float = 1.0

    def new_bar(self, date: object) -> None:
        """Call at each new bar to reset daily counters if date changed."""
        if date != self._current_date:
            self._current_date = date
            self._daily_pnl = 0.0
            self._daily_trades = 0

    def can_trade(self) -> bool:
        """Check if daily limits allow a new trade."""
        if self._daily_trades >= self.config.max_trades_per_day:
            return False
        if self._daily_pnl <= self.config.daily_loss_cap:
            return False
        if self._daily_pnl >= self.config.daily_profit_cap:
            return False
        return True

    def record_trade(self, pnl: float) -> None:
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._equity += pnl
        self._peak_equity = max(self._peak_equity, self._equity)

    def drawdown(self) -> float:
        """Current drawdown (negative number)."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._equity - self._peak_equity) / self._peak_equity

    def is_killed(self) -> bool:
        """Check if max drawdown cap breached."""
        return self.drawdown() <= self.config.max_drawdown_cap

    @property
    def equity(self) -> float:
        return self._equity

    def reset(self, equity: float = 1.0) -> None:
        self._equity = equity
        self._peak_equity = equity
        self._current_date = None
        self._daily_pnl = 0.0
        self._daily_trades = 0
