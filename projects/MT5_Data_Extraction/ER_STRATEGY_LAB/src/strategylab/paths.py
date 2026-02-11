"""Path resolution — delegates to PROJECT_ROOT/path_contract.py."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_path_contract() -> None:
    """Add path_contract.py's parent to sys.path if not already importable."""
    try:
        import path_contract  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "path_contract.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return


_ensure_path_contract()
import path_contract  # type: ignore[import-untyped]  # noqa: E402

# Re-export for backward compatibility
detect_project_root = path_contract.detect_project_root
strategy_lab_root = path_contract.strategy_lab_root


def outputs_root(strategy: str = "default", version: str = "v1") -> Path:
    """Return the outputs directory for a strategy, creating if needed."""
    out = path_contract.outputs_root() / strategy / version
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_root() -> Path:
    """Return DATA_ROOT = PROJECT_ROOT/data."""
    return path_contract.data_root()
