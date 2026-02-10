"""Fill model: next-bar execution with slippage."""

from __future__ import annotations

from .config import CostsConfig


def fill_price(bar_open: float, side: int, costs: CostsConfig) -> float:
    """Compute fill price at bar open with slippage.

    side: +1 for buy, -1 for sell.
    """
    slip_dec = costs.slippage_bps / 10_000
    spread_half = (costs.spread_bps / 2) / 10_000
    adverse = slip_dec + spread_half
    return bar_open * (1 + side * adverse)
