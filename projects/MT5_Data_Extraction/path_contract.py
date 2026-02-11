"""Unified Path Contract for MT5_Data_Extraction.

Single source of truth used by NB3 (TREND), NB4 (RANGE), and strategylab.paths.
Compatible with the CLOSED contracts of NB1 (MT5_DE_DATA_ROOT) and NB2 (marker-based auto-detect).

All functions accept an optional ``project_root`` parameter so callers that have
already resolved it can skip re-detection.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str) -> str | None:
    """Return non-empty env var or None."""
    v = os.getenv(name)
    return v if v not in (None, "") else None


# ---------------------------------------------------------------------------
# PROJECT_ROOT detection
# ---------------------------------------------------------------------------

def detect_project_root() -> Path:
    """Detect PROJECT_ROOT for the MT5_Data_Extraction project.

    Priority:
      1. Env vars: MT5_PROJECT_ROOT or MT5_DE_PROJECT_ROOT
      2. Walk up from CWD matching directory name ``mt5_data_extraction`` (case-insensitive)
      3. Walk up from CWD looking for NB2 marker dirs (data/, outputs/)
      4. Fallback: CWD
    """
    forced = _env("MT5_PROJECT_ROOT") or _env("MT5_DE_PROJECT_ROOT")
    if forced:
        return Path(forced).resolve()

    start = Path.cwd().resolve()

    # Strategy 1: directory name match (NB3/NB4 convention)
    for p in [start, *list(start.parents)]:
        if p.name.lower() == "mt5_data_extraction":
            return p

    # Strategy 2: marker directories (NB2 convention)
    markers = {"data", "outputs"}
    for p in [start, *list(start.parents)]:
        try:
            children = {c.name for c in p.iterdir() if c.is_dir()}
        except PermissionError:
            continue
        if markers <= children:
            return p

    return start


# ---------------------------------------------------------------------------
# Canonical paths (derived from PROJECT_ROOT)
# ---------------------------------------------------------------------------

def data_root(project_root: Path | None = None) -> Path:
    """DATA_ROOT = PROJECT_ROOT/data  (matches NB1 contract)."""
    return (project_root or detect_project_root()) / "data"


def m5_clean_dir(project_root: Path | None = None) -> Path:
    """Gold M5 hive directory: data/historical_data/m5_clean/symbol=XXX/..."""
    return data_root(project_root) / "historical_data" / "m5_clean"


def m5_raw_dir(project_root: Path | None = None) -> Path:
    """Raw M5 hive directory: data/bulk_data/m5_raw/symbol=XXX/..."""
    return data_root(project_root) / "bulk_data" / "m5_raw"


def metadata_dir(project_root: Path | None = None) -> Path:
    return data_root(project_root) / "metadata"


def processed_data_dir(project_root: Path | None = None) -> Path:
    return data_root(project_root) / "processed_data"


def config_dir(project_root: Path | None = None) -> Path:
    """Persistent config: config/er_filter_5m.json, etc."""
    return (project_root or detect_project_root()) / "config"


def outputs_root(project_root: Path | None = None) -> Path:
    """All notebook/strategy outputs live under PROJECT_ROOT/outputs/."""
    return (project_root or detect_project_root()) / "outputs"


def nb2_outputs_dir(project_root: Path | None = None) -> Path:
    """NB2 (ER Filter) runs: outputs/er_filter_5m/<RUN_ID>/."""
    return outputs_root(project_root) / "er_filter_5m"


def trend_outputs_dir(project_root: Path | None = None) -> Path:
    """NB3 (TREND v2) runs: outputs/trend_v2/<RUN_ID>/."""
    return outputs_root(project_root) / "trend_v2"


def range_outputs_dir(project_root: Path | None = None) -> Path:
    """NB4 (RANGE v1) runs: outputs/range_v1/<RUN_ID>/."""
    return outputs_root(project_root) / "range_v1"


def strategy_lab_root(project_root: Path | None = None) -> Path:
    return (project_root or detect_project_root()) / "ER_STRATEGY_LAB"


# ---------------------------------------------------------------------------
# NB2 discovery helpers
# ---------------------------------------------------------------------------

def nb2_latest_run_dir(project_root: Path | None = None) -> Path | None:
    """Find the most recent NB2 run directory (sorted by YYYYMMDD_HHMMSS name)."""
    nb2_dir = nb2_outputs_dir(project_root)
    if not nb2_dir.is_dir():
        return None
    runs = sorted(
        [d for d in nb2_dir.iterdir() if d.is_dir() and not d.name.startswith("_")],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0] if runs else None


def nb2_basket(strategy: str, project_root: Path | None = None) -> Path | None:
    """Return path to the latest NB2 basket parquet for *strategy* ('trend' or 'range').

    Looks in: ``<latest_run>/baskets/basket_{strategy}_core.parquet``
    """
    run_dir = nb2_latest_run_dir(project_root)
    if run_dir is None:
        return None
    basket_name = f"basket_{strategy}_core.parquet"
    p = run_dir / "baskets" / basket_name
    if p.exists():
        return p
    # Fallback: flat in run dir
    p2 = run_dir / basket_name
    return p2 if p2.exists() else None


# ---------------------------------------------------------------------------
# M5 data discovery
# ---------------------------------------------------------------------------

def m5_data_dir(project_root: Path | None = None) -> Path | None:
    """Return the first existing M5 data directory.

    Priority: m5_clean (gold) > m5_raw (raw).
    """
    pr = project_root or detect_project_root()
    for candidate in (m5_clean_dir(pr), m5_raw_dir(pr)):
        if candidate.is_dir():
            try:
                if any(candidate.iterdir()):
                    return candidate
            except PermissionError:
                continue
    return None
