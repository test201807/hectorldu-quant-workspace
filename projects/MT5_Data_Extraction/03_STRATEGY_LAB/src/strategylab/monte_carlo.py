"""Monte Carlo simulation: IID bootstrap, block bootstrap, stress."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MCResult:
    """Result of a Monte Carlo simulation."""
    method: str
    n_sims: int
    percentiles: dict[str, float]   # e.g. {"p5": ..., "p25": ..., "p50": ..., "p75": ..., "p95": ...}
    final_equities: list[float]
    max_drawdowns: list[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_dd(equity: list[float]) -> float:
    """Max drawdown as negative decimal."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def _equity_from_returns(returns: list[float], start: float = 1.0) -> list[float]:
    eq = [start]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _percentiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}
    s = sorted(vals)
    n = len(s)

    def _p(q: float) -> float:
        idx = int(n * q)
        return s[min(idx, n - 1)]

    return {
        "p5": _p(0.05),
        "p25": _p(0.25),
        "p50": _p(0.50),
        "p75": _p(0.75),
        "p95": _p(0.95),
    }


# ---------------------------------------------------------------------------
# Method 1: IID Bootstrap
# ---------------------------------------------------------------------------

def iid_bootstrap(
    trade_returns: list[float],
    n_sims: int = 1000,
    n_trades: int | None = None,
    seed: int = 42,
) -> MCResult:
    """IID bootstrap: resample trades with replacement.

    Each simulation draws n_trades (default = len(trade_returns))
    trades randomly with replacement and builds an equity curve.
    """
    rng = random.Random(seed)
    if not trade_returns:
        return MCResult("iid_bootstrap", 0, _percentiles([]), [], [])

    k = n_trades or len(trade_returns)
    finals: list[float] = []
    mdds: list[float] = []

    for _ in range(n_sims):
        sample = rng.choices(trade_returns, k=k)
        eq = _equity_from_returns(sample)
        finals.append(eq[-1])
        mdds.append(_max_dd(eq))

    return MCResult(
        method="iid_bootstrap",
        n_sims=n_sims,
        percentiles=_percentiles(finals),
        final_equities=finals,
        max_drawdowns=mdds,
    )


# ---------------------------------------------------------------------------
# Method 2: Block Bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap(
    trade_returns: list[float],
    block_size: int = 10,
    n_sims: int = 1000,
    n_trades: int | None = None,
    seed: int = 42,
) -> MCResult:
    """Block bootstrap: resample in contiguous blocks to preserve autocorrelation.

    Blocks of `block_size` consecutive trades are drawn with replacement.
    """
    rng = random.Random(seed)
    if not trade_returns:
        return MCResult("block_bootstrap", 0, _percentiles([]), [], [])

    src_len = len(trade_returns)
    k = n_trades or src_len
    finals: list[float] = []
    mdds: list[float] = []

    for _ in range(n_sims):
        sample: list[float] = []
        while len(sample) < k:
            start = rng.randint(0, src_len - 1)
            for j in range(block_size):
                if len(sample) >= k:
                    break
                sample.append(trade_returns[(start + j) % src_len])
        eq = _equity_from_returns(sample)
        finals.append(eq[-1])
        mdds.append(_max_dd(eq))

    return MCResult(
        method="block_bootstrap",
        n_sims=n_sims,
        percentiles=_percentiles(finals),
        final_equities=finals,
        max_drawdowns=mdds,
    )


# ---------------------------------------------------------------------------
# Method 3: Stress Test
# ---------------------------------------------------------------------------

def stress_test(
    trade_returns: list[float],
    cost_multiplier: float = 2.0,
    slippage_add_bps: float = 5.0,
    adverse_pct: float = 0.10,
    n_sims: int = 1000,
    seed: int = 42,
) -> MCResult:
    """Stress test: apply adverse modifications to returns.

    - All returns are penalised by cost_multiplier of average cost.
    - A random `adverse_pct` fraction of trades gets extra slippage.
    - Uses IID bootstrap on the stressed returns.
    """
    rng = random.Random(seed)
    if not trade_returns:
        return MCResult("stress", 0, _percentiles([]), [], [])

    # Base penalty: cost_multiplier as fraction of mean absolute return
    mean_abs = sum(abs(r) for r in trade_returns) / len(trade_returns) if trade_returns else 0
    cost_penalty = mean_abs * (cost_multiplier - 1.0) * 0.5  # half as extra cost
    slip_penalty = slippage_add_bps / 10_000

    stressed: list[float] = []
    for r in trade_returns:
        s = r - cost_penalty
        if rng.random() < adverse_pct:
            s -= slip_penalty
        stressed.append(s)

    # Run IID bootstrap on stressed returns
    return MCResult(
        method="stress",
        n_sims=n_sims,
        percentiles=_percentiles([]),  # will be filled below
        final_equities=[],
        max_drawdowns=[],
    ) if not stressed else _stress_sim(stressed, n_sims, seed + 1)


def _stress_sim(stressed: list[float], n_sims: int, seed: int) -> MCResult:
    rng = random.Random(seed)
    k = len(stressed)
    finals: list[float] = []
    mdds: list[float] = []

    for _ in range(n_sims):
        sample = rng.choices(stressed, k=k)
        eq = _equity_from_returns(sample)
        finals.append(eq[-1])
        mdds.append(_max_dd(eq))

    return MCResult(
        method="stress",
        n_sims=n_sims,
        percentiles=_percentiles(finals),
        final_equities=finals,
        max_drawdowns=mdds,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def var_cvar(sorted_values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

    Args:
        sorted_values: Sorted list of final equities or returns.
        alpha: Confidence level (e.g. 0.05 for 5th percentile).

    Returns:
        (VaR, CVaR) where VaR is the alpha-percentile value and
        CVaR is the mean of all values below VaR.
    """
    if not sorted_values:
        return 0.0, 0.0
    s = sorted(sorted_values)
    n = len(s)
    var_idx = max(0, min(int(n * alpha), n - 1))
    var_val = s[var_idx]
    tail = s[:var_idx + 1]
    cvar_val = sum(tail) / len(tail) if tail else var_val
    return var_val, cvar_val


def run_all_mc(
    trade_returns: list[float],
    n_sims: int = 1000,
    block_size: int = 10,
    cost_multiplier: float = 2.0,
    seed: int = 42,
) -> list[MCResult]:
    """Run all three MC methods and return results."""
    return [
        iid_bootstrap(trade_returns, n_sims=n_sims, seed=seed),
        block_bootstrap(trade_returns, block_size=block_size, n_sims=n_sims, seed=seed + 100),
        stress_test(trade_returns, cost_multiplier=cost_multiplier, n_sims=n_sims, seed=seed + 200),
    ]
