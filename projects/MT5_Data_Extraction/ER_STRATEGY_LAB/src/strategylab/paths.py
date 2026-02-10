"""Path resolution: PROJECT_ROOT, data roots, output roots."""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def detect_project_root() -> Path:
    """Walk up from CWD looking for the MT5_Data_Extraction directory."""
    forced = _env("MT5_PROJECT_ROOT") or _env("MT5_DE_PROJECT_ROOT")
    if forced:
        return Path(forced).resolve()
    start = Path.cwd().resolve()
    for p in [start, *list(start.parents)]:
        if p.name.lower() == "mt5_data_extraction":
            return p
    return start


def strategy_lab_root() -> Path:
    """Return the ER_STRATEGY_LAB directory."""
    forced = _env("STRATEGY_LAB_ROOT")
    if forced:
        return Path(forced).resolve()
    return detect_project_root() / "ER_STRATEGY_LAB"


def outputs_root(strategy: str = "default", version: str = "v1") -> Path:
    """Return the outputs directory for a strategy, creating if needed."""
    root = Path(
        _env("STRATEGYLAB_OUTPUTS_ROOT", str(strategy_lab_root() / "outputs" / "strategylab"))
    ).resolve()
    out = root / strategy / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_root() -> Path:
    """Return the processed_data root (never in git)."""
    return detect_project_root() / "processed_data"
