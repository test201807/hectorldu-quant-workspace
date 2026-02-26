"""Performance metrics: CAGR, Sharpe, Sortino, Calmar, MDD, hit rate, expectancy."""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(equity: list[float] | pl.Series) -> float:
    """Maximum drawdown as a negative decimal (e.g. -0.15 = -15%)."""
    vals = list(equity)
    if len(vals) < 2:
        return 0.0
    peak = vals[0]
    mdd = 0.0
    for v in vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def drawdown_series(equity: list[float]) -> list[float]:
    """Return drawdown at each point (negative decimals)."""
    if not equity:
        return []
    peak = equity[0]
    out: list[float] = []
    for v in equity:
        if v > peak:
            peak = v
        out.append((v - peak) / peak if peak > 0 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Annualised metrics
# ---------------------------------------------------------------------------

BARS_PER_YEAR_5M = 252 * 288  # 5-min bars, Mon-Fri


def cagr(equity_start: float, equity_end: float, n_bars: int,
         bars_per_year: int = BARS_PER_YEAR_5M) -> float:
    """Compound annual growth rate."""
    if equity_start <= 0 or n_bars <= 0:
        return 0.0
    years = n_bars / bars_per_year
    if years <= 0:
        return 0.0
    ratio = equity_end / equity_start
    if ratio <= 0:
        return -1.0
    return ratio ** (1.0 / years) - 1.0


def sharpe_ratio(returns: list[float], bars_per_year: int = BARS_PER_YEAR_5M) -> float:
    """Annualised Sharpe ratio (Rf=0)."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean_r / std) * math.sqrt(bars_per_year)


def sharpe_daily_from_trades(trades_df: pl.DataFrame) -> float:
    """Annualised Sharpe computed on daily aggregated returns (sqrt(252)).

    Avoids the ~269x inflation that occurs when using sqrt(252*288) on sparse
    per-trade PnLs (trades are not 5-min observations). Aggregates net_pnl by
    exit date, then annualises with sqrt(252) trading days per year.

    Falls back to trade-frequency annualisation when exit_time_utc is absent.
    """
    if trades_df.is_empty():
        return 0.0
    if "exit_time_utc" not in trades_df.columns:
        # Fallback: annualise by trade frequency (trades/year)
        pnls = trades_df.get_column("net_pnl").to_list()
        hold = (trades_df.get_column("hold_bars").to_list()
                if "hold_bars" in trades_df.columns else [1] * len(pnls))
        n_bars = max(sum(hold), 1)
        years = n_bars / BARS_PER_YEAR_5M
        tpy = len(pnls) / years if years > 0 else float(len(pnls))
        return sharpe_ratio(pnls, bars_per_year=max(round(tpy), 2))
    # Aggregate PnL by exit date, annualise with sqrt(252) trading days/year
    daily = (
        trades_df
        .with_columns(pl.col("exit_time_utc").cast(pl.Date).alias("_exit_date"))
        .group_by("_exit_date")
        .agg(pl.col("net_pnl").sum())
        .sort("_exit_date")
    )
    daily_pnls = daily.get_column("net_pnl").to_list()
    return sharpe_ratio(daily_pnls, bars_per_year=252)


def sortino_ratio(returns: list[float], bars_per_year: int = BARS_PER_YEAR_5M) -> float:
    """Annualised Sortino ratio (Rf=0, downside deviation)."""
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 0.0 if mean_r == 0 else float("inf")
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 0.0
    if down_std == 0:
        return 0.0
    return (mean_r / down_std) * math.sqrt(bars_per_year)


def calmar_ratio(total_return: float, mdd: float) -> float:
    """Calmar = CAGR / |MDD|."""
    if mdd == 0:
        return 0.0
    return _safe_div(total_return, abs(mdd))


# ---------------------------------------------------------------------------
# Trade-level
# ---------------------------------------------------------------------------

def hit_rate(pnls: list[float]) -> float:
    """Win rate (fraction of trades with pnl > 0)."""
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def expectancy(pnls: list[float]) -> float:
    """Average PnL per trade (net expectancy)."""
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


def profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    return _safe_div(gross_profit, gross_loss)


def turnover(n_trades: int, n_bars: int) -> float:
    """Trade frequency: trades per bar."""
    return _safe_div(n_trades, n_bars)


def exposure(hold_bars_total: int, n_bars: int) -> float:
    """Fraction of time in market."""
    return _safe_div(hold_bars_total, n_bars)


# ---------------------------------------------------------------------------
# Aggregate from trades DataFrame
# ---------------------------------------------------------------------------

def compute_kpis(trades_df: pl.DataFrame,
                 bars_per_year: int = BARS_PER_YEAR_5M) -> dict:
    """Compute full KPI dict from trades DataFrame.

    Expects columns: net_pnl, gross_pnl, hold_bars, exit_reason.
    """
    if trades_df.is_empty():
        return {
            "n_trades": 0, "total_return": 0.0, "cagr": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
            "mdd": 0.0, "hit_rate": 0.0, "expectancy": 0.0,
            "profit_factor": 0.0, "turnover": 0.0, "exposure": 0.0,
        }

    pnls = trades_df.get_column("net_pnl").to_list()
    hold = trades_df.get_column("hold_bars").to_list()

    # Build equity curve from cumulative returns
    eq = [1.0]
    for p in pnls:
        eq.append(eq[-1] * (1.0 + p))

    n_bars_total = sum(hold) if hold else 1
    tot_ret = eq[-1] / eq[0] - 1.0

    return {
        "n_trades": len(pnls),
        "total_return": tot_ret,
        "cagr": cagr(eq[0], eq[-1], n_bars_total, bars_per_year),
        "sharpe": sharpe_daily_from_trades(trades_df),
        "sortino": sortino_ratio(pnls, bars_per_year),
        "calmar": calmar_ratio(tot_ret, max_drawdown(eq)),
        "mdd": max_drawdown(eq),
        "hit_rate": hit_rate(pnls),
        "expectancy": expectancy(pnls),
        "profit_factor": profit_factor(pnls),
        "turnover": turnover(len(pnls), n_bars_total),
        "exposure": exposure(sum(hold), n_bars_total),
    }
