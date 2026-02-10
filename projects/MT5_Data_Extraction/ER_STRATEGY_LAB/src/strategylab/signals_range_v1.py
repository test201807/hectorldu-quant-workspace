"""RANGE v1 signal generation — mean-reversion with Bollinger bands."""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_range_features(
    df: pl.DataFrame,
    bb_period: int = 96,
    bb_std: float = 2.0,
) -> pl.DataFrame:
    """Add range-specific features to bars DataFrame.

    Adds: bb_mid, bb_upper, bb_lower, pct_b, dist_mean_atr, range_width_atr.
    Expects columns: close, high, low, atr_price.
    """
    atr_col = "atr_price" if "atr_price" in df.columns else None

    out = df.with_columns([
        pl.col("close").rolling_mean(bb_period).alias("bb_mid"),
        pl.col("close").rolling_std(bb_period).alias("bb_std_raw"),
    ]).with_columns([
        (pl.col("bb_mid") + bb_std * pl.col("bb_std_raw")).alias("bb_upper"),
        (pl.col("bb_mid") - bb_std * pl.col("bb_std_raw")).alias("bb_lower"),
    ]).with_columns([
        # %B = (close - lower) / (upper - lower)
        ((pl.col("close") - pl.col("bb_lower"))
         / (pl.col("bb_upper") - pl.col("bb_lower")).clip(lower_bound=1e-10)).alias("pct_b"),
        # Distance to mean in ATR units
        ((pl.col("close") - pl.col("bb_mid"))
         / pl.col(atr_col).clip(lower_bound=1e-10)).alias("dist_mean_atr")
        if atr_col else pl.lit(0.0).alias("dist_mean_atr"),
        # Range width in ATR units
        ((pl.col("high").rolling_max(bb_period) - pl.col("low").rolling_min(bb_period))
         / pl.col(atr_col).clip(lower_bound=1e-10)).alias("range_width_atr")
        if atr_col else pl.lit(0.0).alias("range_width_atr"),
    ]).drop("bb_std_raw")

    return out


# ---------------------------------------------------------------------------
# Regime gate (ranging = low ER)
# ---------------------------------------------------------------------------

def compute_range_gate(
    df: pl.DataFrame,
    q_er_upper: float = 0.40,
    q_vol: float = 0.90,
    er_col: str = "er_288",
    vol_col: str = "vol_bps_288",
) -> tuple[list[bool], dict]:
    """Compute ranging regime gate.

    For mean-reversion, we want LOW efficiency ratio (= no trend).
    Gate: ER <= thr_er_upper AND vol <= thr_vol.

    Returns:
        (gate, params_dict)
    """
    n = df.height
    if n == 0:
        return [], {}

    er = df.get_column(er_col).to_list()
    vol = df.get_column(vol_col).to_list()

    er_vals = sorted(v for v in er if v is not None)
    vol_vals = sorted(v for v in vol if v is not None)

    if not er_vals or not vol_vals:
        return [False] * n, {}

    def _percentile(vals: list[float], q: float) -> float:
        idx = min(int(len(vals) * q), len(vals) - 1)
        return vals[idx]

    thr_er = _percentile(er_vals, q_er_upper)
    thr_vol = _percentile(vol_vals, q_vol)

    gate: list[bool] = []
    for i in range(n):
        e, v = er[i], vol[i]
        if e is None or v is None:
            gate.append(False)
            continue
        gate.append(e <= thr_er and v <= thr_vol)

    params = {
        "thr_er_upper": thr_er,
        "thr_vol": thr_vol,
        "coverage": sum(gate) / n if n > 0 else 0.0,
    }
    return gate, params


# ---------------------------------------------------------------------------
# Mean-reversion entry signals
# ---------------------------------------------------------------------------

def mean_reversion_signals(
    dist_mean_atr: list[float],
    band_k: float = 1.5,
) -> tuple[list[bool], list[bool]]:
    """Generate mean-reversion entry signals.

    LONG: price below lower band (dist_mean_atr <= -band_k).
    SHORT: price above upper band (dist_mean_atr >= +band_k).

    Returns:
        (signal_long, signal_short)
    """
    n = len(dist_mean_atr)
    sig_long = [False] * n
    sig_short = [False] * n

    for i in range(n):
        d = dist_mean_atr[i]
        if d is None or not math.isfinite(d):
            continue
        if d <= -band_k:
            sig_long[i] = True
        elif d >= band_k:
            sig_short[i] = True

    return sig_long, sig_short


def confirm_signals(raw: list[bool], min_bars: int = 6) -> list[bool]:
    """Rolling confirmation for range signals (shorter window than trend)."""
    n = len(raw)
    confirmed = [False] * n
    run = 0
    for i in range(n):
        run = run + 1 if raw[i] else 0
        confirmed[i] = run >= min_bars
    return confirmed
