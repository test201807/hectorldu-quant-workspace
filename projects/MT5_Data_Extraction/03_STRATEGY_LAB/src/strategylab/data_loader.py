"""Load bars/features from disk — flat parquet and hive-partitioned."""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Column normalisation (NB1 schema → canonical schema)
# ---------------------------------------------------------------------------
_RENAME_MAP: dict[str, str] = {
    "timestamp_utc": "time_utc",
    "tick_volume": "volume",
    "spread_points": "spread",
}


def _normalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename NB1-produced columns to canonical names and coerce types."""
    # --- Renames (only if target does not already exist) ---
    renames = {
        old: new
        for old, new in _RENAME_MAP.items()
        if old in df.columns and new not in df.columns
    }
    if renames:
        df = df.rename(renames)

    # --- Coerce time_utc to Datetime(us, UTC) ---
    if "time_utc" in df.columns:
        dtype = df.schema["time_utc"]
        if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
            sample = df["time_utc"].drop_nulls().head(1).to_list()
            v = sample[0] if sample else 0
            if v > 1e17:
                unit = "ns"
            elif v > 1e14:
                unit = "us"
            elif v > 1e11:
                unit = "ms"
            else:
                unit = "s"
            df = df.with_columns(
                pl.from_epoch(pl.col("time_utc"), time_unit=unit)
                .cast(pl.Datetime("us", "UTC"))
                .alias("time_utc")
            )
        elif not (dtype == pl.Datetime("us", "UTC")):
            df = df.with_columns(
                pl.col("time_utc").cast(pl.Datetime("us", "UTC"), strict=False)
            )

    # --- Numeric casts ---
    if "volume" in df.columns and df.schema["volume"] != pl.Float64:
        df = df.with_columns(pl.col("volume").cast(pl.Float64))
    if "spread" in df.columns and df.schema["spread"] != pl.Float64:
        df = df.with_columns(pl.col("spread").cast(pl.Float64))

    return df


def _sort_and_dedup(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (symbol, time_utc) and dedup keeping last."""
    if "symbol" in df.columns and "time_utc" in df.columns:
        df = df.sort(["symbol", "time_utc"])
        df = df.unique(subset=["symbol", "time_utc"], keep="last", maintain_order=True)
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_bars(path: str | Path, symbols: list[str] | None = None) -> pl.DataFrame:
    """Load OHLCV bars from a flat parquet file.

    Handles NB1 column names (timestamp_utc, tick_volume, spread_points)
    transparently.  Applies: column rename, UTC cast, symbol filter, sort,
    dedup (keep last).
    """
    df = pl.read_parquet(str(path))
    df = _normalise_columns(df)
    if symbols and "symbol" in df.columns:
        df = df.filter(pl.col("symbol").is_in([s.upper() for s in symbols]))
    return _sort_and_dedup(df)


def load_bars_hive(
    base_dir: str | Path,
    symbols: list[str] | None = None,
) -> pl.DataFrame:
    """Load OHLCV from a hive-partitioned directory tree.

    Expected layout::

        base_dir/symbol=XXX/year=YYYY/month=MM/part=YYYYMMDD.parquet

    If *symbols* is given only the matching ``symbol=`` subdirectories are read,
    avoiding a full scan of all partitions.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"M5 data directory not found: {base}")

    if symbols:
        dfs: list[pl.LazyFrame] = []
        for sym in symbols:
            sym_dir = base / f"symbol={sym.upper()}"
            if not sym_dir.is_dir():
                continue
            pattern = str(sym_dir / "**" / "*.parquet")
            dfs.append(pl.scan_parquet(pattern, hive_partitioning=True))
        if not dfs:
            raise FileNotFoundError(
                f"No parquet data found for symbols {symbols} in {base}"
            )
        df = pl.concat(dfs).collect()
    else:
        pattern = str(base / "**" / "*.parquet")
        df = pl.scan_parquet(pattern, hive_partitioning=True).collect()

    df = _normalise_columns(df)
    if symbols and "symbol" in df.columns:
        df = df.filter(pl.col("symbol").is_in([s.upper() for s in symbols]))
    return _sort_and_dedup(df)


def load_features(path: str | Path, symbols: list[str] | None = None) -> pl.DataFrame:
    """Load features parquet (alias for load_bars)."""
    return load_bars(path, symbols)


# ---------------------------------------------------------------------------
# Synthetic data (for tests and CLI demo)
# ---------------------------------------------------------------------------

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
