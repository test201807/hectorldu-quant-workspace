"""Load bars/features from disk using polars lazy scans."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_bars(path: str | Path, symbols: list[str] | None = None) -> pl.DataFrame:
    """Load OHLCV bars from parquet. Optionally filter symbols."""
    df = pl.read_parquet(str(path))
    if "time_utc" in df.columns:
        df = df.with_columns(pl.col("time_utc").cast(pl.Datetime("us", "UTC"), strict=False))
    if symbols:
        symbols_upper = [s.upper() for s in symbols]
        df = df.filter(pl.col("symbol").is_in(symbols_upper))
    return df.sort(["symbol", "time_utc"])


def load_features(path: str | Path, symbols: list[str] | None = None) -> pl.DataFrame:
    """Load features parquet."""
    return load_bars(path, symbols)


def make_synthetic_bars(
    symbol: str = "SYNTH",
    n_bars: int = 10_000,
    start_price: float = 100.0,
    volatility: float = 0.001,
    trend: float = 0.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic OHLCV for testing. Deterministic with seed."""
    from datetime import datetime, timedelta, timezone  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, volatility, n_bars)
    closes = start_price * np.exp(np.cumsum(rets))
    highs = closes * (1 + rng.uniform(0, volatility * 2, n_bars))
    lows = closes * (1 - rng.uniform(0, volatility * 2, n_bars))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    base = datetime(2022, 1, 3, 0, 0, tzinfo=timezone.utc)
    times = []
    t = base
    for _ in range(n_bars):
        # Skip weekends
        while t.weekday() >= 5:
            t += timedelta(days=1)
            t = t.replace(hour=0, minute=0)
        times.append(t)
        t += timedelta(seconds=300)

    return pl.DataFrame({
        "symbol": [symbol] * n_bars,
        "time_utc": times,
        "open": opens.tolist(),
        "high": highs.tolist(),
        "low": lows.tolist(),
        "close": closes.tolist(),
        "volume": rng.integers(100, 10000, n_bars).tolist(),
        "spread": [0.0002] * n_bars,
    }).with_columns(pl.col("time_utc").cast(pl.Datetime("us", "UTC")))
