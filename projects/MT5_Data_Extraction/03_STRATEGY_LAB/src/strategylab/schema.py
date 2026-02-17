"""Canonical schemas for bars, features, signals, trades, positions."""

from __future__ import annotations

from typing import Any

import polars as pl


def validate_columns(df: pl.DataFrame, schema: dict[str, Any], label: str = "DataFrame") -> list[str]:
    """Validate that a DataFrame has all required columns from a schema.

    Returns list of missing columns. Raises ValueError if any are missing.
    """
    required = schema.get("required", [])
    present = set(df.columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}. Has: {sorted(present)}")
    return missing


BARS_SCHEMA = {
    "required": ["symbol", "time_utc", "open", "high", "low", "close"],
    "optional": ["volume", "spread"],
}

FEATURES_TREND = {
    "required": [
        "symbol", "time_utc", "open", "high", "low", "close",
        "er_288", "mom_bps_288", "vol_bps_288", "atr_bps_96",
    ],
}

FEATURES_RANGE = {
    "required": [
        "symbol", "time_utc", "open", "high", "low", "close",
        "er_288", "vol_bps_288", "atr_bps_96",
        "pct_b", "dist_mean_atr", "range_width_atr",
    ],
}

TRADES_SCHEMA = {
    "required": [
        "symbol", "fold_id", "segment", "side",
        "signal_time_utc", "entry_time_utc", "exit_time_utc",
        "entry_price", "exit_price",
        "gross_pnl", "net_pnl",
        "hold_bars", "exit_reason",
    ],
}

EQUITY_SCHEMA = {
    "required": ["time_utc", "cum_ret", "peak", "drawdown"],
}
