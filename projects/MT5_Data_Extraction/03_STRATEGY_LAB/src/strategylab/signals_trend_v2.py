"""TREND v2 signal generation — pure functions, no side effects."""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_trend_features(df: pl.DataFrame, ema_fast: int = 12, ema_slow: int = 48) -> pl.DataFrame:
    """Add trend-specific features to bars DataFrame.

    Expects columns: close, atr_price (or atr_bps_96 + close).
    Adds: ema_fast, ema_slow, ema_cross, trend_slope.
    """
    out = df.with_columns([
        pl.col("close").ewm_mean(span=ema_fast).alias("ema_fast"),
        pl.col("close").ewm_mean(span=ema_slow).alias("ema_slow"),
    ]).with_columns([
        (pl.col("ema_fast") - pl.col("ema_slow")).alias("ema_cross"),
        pl.col("close").diff(12).alias("trend_slope"),
    ])
    return out


# ---------------------------------------------------------------------------
# Regime gate
# ---------------------------------------------------------------------------

def compute_regime_gate(
    df: pl.DataFrame,
    q_er: float = 0.60,
    q_mom_long: float = 0.55,
    q_mom_short: float = 0.45,
    q_vol: float = 0.90,
    er_col: str = "er_288",
    mom_col: str = "mom_bps_288",
    vol_col: str = "vol_bps_288",
) -> tuple[list[bool], list[bool], dict]:
    """Compute regime gate signals for LONG and SHORT independently.

    Bug fix: SHORT momentum threshold calibrated independently
    (percentile on negative side of mom distribution).

    Returns:
        (gate_long, gate_short, params_dict)
    """
    n = df.height
    if n == 0:
        return [], [], {}

    er = df.get_column(er_col).to_list()
    mom = df.get_column(mom_col).to_list()
    vol = df.get_column(vol_col).to_list()

    # Compute thresholds from IS data (caller should pre-filter)
    er_vals = [v for v in er if v is not None]
    mom_vals = [v for v in mom if v is not None]
    vol_vals = [v for v in vol if v is not None]

    if not er_vals or not mom_vals or not vol_vals:
        return [False] * n, [False] * n, {}

    er_vals.sort()
    mom_vals.sort()
    vol_vals.sort()

    def _percentile(vals: list[float], q: float) -> float:
        idx = int(len(vals) * q)
        idx = min(idx, len(vals) - 1)
        return vals[idx]

    thr_er = _percentile(er_vals, q_er)
    thr_mom_long = _percentile(mom_vals, q_mom_long)
    # SHORT: independent calibration — use lower percentile of mom
    thr_mom_short = _percentile(mom_vals, 1.0 - q_mom_long)
    thr_vol = _percentile(vol_vals, q_vol)

    gate_long: list[bool] = []
    gate_short: list[bool] = []

    for i in range(n):
        e, m, v = er[i], mom[i], vol[i]
        if e is None or m is None or v is None:
            gate_long.append(False)
            gate_short.append(False)
            continue
        # LONG: high ER + positive momentum + vol below cap
        gl = (e >= thr_er) and (m >= thr_mom_long) and (v <= thr_vol)
        # SHORT: high ER + negative momentum + vol below cap
        gs = (e >= thr_er) and (m <= thr_mom_short) and (v <= thr_vol)
        gate_long.append(gl)
        gate_short.append(gs)

    params = {
        "thr_er": thr_er,
        "thr_mom_long": thr_mom_long,
        "thr_mom_short": thr_mom_short,
        "thr_vol": thr_vol,
        "coverage_long": sum(gate_long) / n if n > 0 else 0.0,
        "coverage_short": sum(gate_short) / n if n > 0 else 0.0,
    }
    return gate_long, gate_short, params


# ---------------------------------------------------------------------------
# Signal confirmation
# ---------------------------------------------------------------------------

def confirm_signals(raw: list[bool], min_bars: int = 12) -> list[bool]:
    """Rolling confirmation: True only after min_bars consecutive True."""
    n = len(raw)
    confirmed = [False] * n
    run = 0
    for i in range(n):
        run = run + 1 if raw[i] else 0
        confirmed[i] = run >= min_bars
    return confirmed
