"""
Phase 5b: Fix selection gate MIN_WINRATE for high-payoff strategies.

Problem: MIN_WINRATE=0.30 blocks BTCUSD LONG (WR=20.9%)
  but with TP=14/SL=2, BE_WR = 2/(2+14) = 12.5%
  WR=20.9% is 67% above breakeven — clearly profitable.

Fix: MIN_WINRATE 0.30 -> 0.15 (still above BE_WR of 12.5%)

Usage: python tools/_patch_phase5b_selection.py [--dry-run]
"""
import json, sys, pathlib

DRY_RUN = "--dry-run" in sys.argv
PROJECT = pathlib.Path(__file__).parent.parent
NB3 = PROJECT / "03_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb"


def load_nb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nb(path, nb):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def find_cell(nb, cell_id):
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            return c
    raise KeyError(f"Cell {cell_id} not found")


def get_source(cell):
    return "".join(cell["source"])


def set_source(cell, code):
    lines = code.split("\n")
    new_src = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            new_src.append(line + "\n")
        else:
            new_src.append(line)
    cell["source"] = new_src


def assert_replace(src, old, new, label=""):
    if old not in src:
        raise AssertionError(f"Pattern not found ({label}): {old!r}")
    return src.replace(old, new, 1)


nb3 = load_nb(NB3)

cell = find_cell(nb3, "cf14989a")
src = get_source(cell)

src = assert_replace(src,
    "MIN_WINRATE = 0.30",
    "MIN_WINRATE = 0.15   # BE_WR=12.5% for SL=2/TP=14, 0.15 gives safety margin",
    "NB3.17 MIN_WINRATE")

# Update header
src = src.replace(
    "# Gates: min_oos_trades=30, max_mdd=-0.20, min_totret=0.0, min_wr=0.40, max_exposure=0.65",
    "# Gates: min_oos_trades=30, max_mdd=-0.20, min_totret=-0.05, min_wr=0.15, max_exposure=0.65"
)

set_source(cell, src)

if DRY_RUN:
    print("[DRY RUN] NB3 not saved.")
else:
    save_nb(NB3, nb3)
    print(f"Saved NB3: {NB3}")
    print("MIN_WINRATE: 0.30 -> 0.15")
