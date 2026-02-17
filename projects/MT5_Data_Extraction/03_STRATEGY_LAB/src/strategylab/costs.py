"""Cost model: spread + commission + slippage + borrow."""

from __future__ import annotations

from dataclasses import dataclass

from .config import CostsConfig


@dataclass
class FillCosts:
    """Costs for a single fill (one way)."""
    spread_cost: float
    commission_cost: float
    slippage_cost: float
    total: float


def compute_fill_cost(price: float, size: float, config: CostsConfig, is_entry: bool = True) -> FillCosts:
    """Compute cost for a single fill at given price and size."""
    notional = abs(price * size)
    spread = (config.spread_bps / 2) / 10_000 * notional
    commission = config.commission_bps / 10_000 * notional
    slippage = config.slippage_bps / 10_000 * notional
    total = spread + commission + slippage
    return FillCosts(spread_cost=spread, commission_cost=commission,
                     slippage_cost=slippage, total=total)


def roundtrip_cost_dec(config: CostsConfig) -> float:
    """Total roundtrip cost as a decimal fraction of notional."""
    return config.total_roundtrip_dec


def stressed_config(config: CostsConfig, cost_factor: float = 2.0,
                    slippage_factor: float = 3.0) -> CostsConfig:
    """Return a stressed copy of costs config."""
    return CostsConfig(
        spread_bps=config.spread_bps * cost_factor,
        commission_bps=config.commission_bps * cost_factor,
        slippage_bps=config.slippage_bps * slippage_factor,
        borrow_bps_annual=config.borrow_bps_annual,
    )
