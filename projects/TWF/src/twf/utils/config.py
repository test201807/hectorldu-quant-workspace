from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE = Path(r'C:\Quant\TWF')

@dataclass(frozen=True)
class Flags:
    STRICT_CHRONO: bool = True
    NO_OVERLAP_EVAL: bool = True
    BLOCK_BOOTSTRAP_B: int = 2000

FLAGS = Flags()

def outputs_root(symbol: str, tf: str) -> Path:
    return BASE / "outputs" / symbol / tf

def ensure_output_tree(symbol: str, tf: str) -> dict[str, Path]:
    root = outputs_root(symbol, tf)
    paths = {
        "root": root,
        "stats": root / "stats",
        "figures": root / "figures",
        "deliverables": root / "deliverables",
        "logs": root / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
