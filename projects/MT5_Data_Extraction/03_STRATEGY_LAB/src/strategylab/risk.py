"""Risk management: position sizing, exposure limits, kill-switches."""

from __future__ import annotations

from .config import RiskConfig


def compute_position_size(
    risk_config: RiskConfig,
    entry_price: float,
    sl_distance: float,
    equity: float = 1.0,
) -> float:
    """Risk-based position sizing.

    Modo normalizado (account_size_usd = 0, default):
        Devuelve tamaño normalizado (fracción de equity = 1.0) para el motor
        interno de backtest. Fórmula: equity * risk% / sl_distance.

    Modo FTMO-aware (account_size_usd > 0):
        Devuelve lotes MT5 directamente: (account * risk%) / (sl_distance * contract_size)
        Para acciones US CFD (contract_size=1): lots = risk_USD / sl_distance_precio
        Ejemplo: account=$10k, risk=1%, SL=$5/acción → 20 lotes (acciones)
    """
    if sl_distance <= 0 or entry_price <= 0:
        return 0.0
    if risk_config.account_size_usd > 0:
        # FTMO-aware: lotes MT5 = riesgo_USD / (distancia_SL_precio × contract_size)
        risk_usd = risk_config.account_size_usd * risk_config.risk_per_trade
        raw_size = risk_usd / (sl_distance * risk_config.contract_size)
    else:
        risk_amount = equity * risk_config.risk_per_trade
        raw_size = risk_amount / sl_distance
    return max(risk_config.min_pos_size, min(risk_config.max_pos_size, raw_size))


def compute_mt5_lots(
    account_usd: float,
    risk_per_trade: float,
    sl_distance_price: float,
    contract_size: float = 1.0,
) -> float:
    """Calcula lotes MT5 para stock CFDs con riesgo fijo en USD.

    Fórmula directa para colocar órdenes en MT5 desde una cuenta FTMO:
        lots = (account_usd × risk_per_trade) / (sl_distance_price × contract_size)

    Args:
        account_usd:        Balance de la cuenta FTMO (e.g. 10_000)
        risk_per_trade:     Fracción del account a arriesgar (e.g. 0.01 = 1%)
        sl_distance_price:  Distancia del SL en precio (ATR × sl_mult, en $)
        contract_size:      Unidades por lote en MT5 (1.0 para acciones US CFD,
                            100_000 para pares forex)

    Returns:
        Lotes para la orden MT5 (float, sin clipear — el trader decide el max).

    Ejemplo:
        NVDA, account=$10k, risk=1%, ATR=$3, SL=2×ATR=$6, contract_size=1
        → lots = 10_000 × 0.01 / (6 × 1) = 16.67 lotes (acciones)
    """
    if sl_distance_price <= 0 or contract_size <= 0 or account_usd <= 0:
        return 0.0
    return (account_usd * risk_per_trade) / (sl_distance_price * contract_size)


class DailyRiskTracker:
    """Track daily PnL and enforce caps.

    Tracks both closed PnL (via record_trade) and open floating PnL
    (via update_floating_pnl) to match FTMO/prop-firm daily loss
    accounting, which measures equity including unrealized positions.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._current_date: object = None
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._peak_equity: float = 1.0
        self._equity: float = 1.0
        self._floating_pnl: float = 0.0  # unrealized PnL of open position

    def new_bar(self, date: object) -> None:
        """Call at each new bar to reset daily counters if date changed.

        Note: _floating_pnl is NOT reset — open positions carry to next day.
        """
        if date != self._current_date:
            self._current_date = date
            self._daily_pnl = 0.0
            self._daily_trades = 0

    def update_floating_pnl(self, floating_pnl: float) -> None:
        """Update unrealized PnL of the current open position.

        Call on every bar while in a position with the mark-to-market
        fractional return (net of costs). Call with 0.0 when position closes.

        This makes can_trade() and drawdown() reflect FTMO-style equity
        accounting where open losses count against daily and total limits.
        """
        self._floating_pnl = floating_pnl
        eff = self._equity + self._floating_pnl
        self._peak_equity = max(self._peak_equity, eff)

    def can_trade(self) -> bool:
        """Check if daily limits allow a new trade.

        Uses effective daily PnL (closed + floating) for the loss cap so
        that an open losing position blocks new entries — matching FTMO rules.
        """
        if self._daily_trades >= self.config.max_trades_per_day:
            return False
        if self._daily_pnl + self._floating_pnl <= self.config.daily_loss_cap:
            return False
        if self._daily_pnl >= self.config.daily_profit_cap:
            return False
        return True

    def is_daily_capped(self) -> bool:
        """True when daily loss cap is breached including floating PnL."""
        return self._daily_pnl + self._floating_pnl <= self.config.daily_loss_cap

    def record_trade(self, pnl: float) -> None:
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._equity += pnl
        self._peak_equity = max(self._peak_equity, self._equity)

    @property
    def effective_equity(self) -> float:
        """Closed equity plus current floating PnL."""
        return self._equity + self._floating_pnl

    def drawdown(self) -> float:
        """Current drawdown including floating PnL (negative number)."""
        if self._peak_equity <= 0:
            return 0.0
        return (self.effective_equity - self._peak_equity) / self._peak_equity

    def is_killed(self) -> bool:
        """Check if max drawdown cap breached (includes floating PnL)."""
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
        self._floating_pnl = 0.0
