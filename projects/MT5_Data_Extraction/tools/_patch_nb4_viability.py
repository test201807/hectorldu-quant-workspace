"""
Adjust NB4 default TP for mathematical viability.

Observed WR = 35.5%
Required: TP/SL >= (1-p)/p = 0.645/0.355 = 1.817
With SL=1.5: TP_min = 1.5 * 1.817 = 2.725

Change: TP default 2.5 -> 3.0 (gives BE_WR = 33.3%, margin of 2.2pp)
Also: add TP=2.75 to grid for finer search near the viability boundary.

Usage: python tools/_patch_nb4_viability.py [--dry-run]
"""
import json, sys, pathlib

DRY_RUN = "--dry-run" in sys.argv
NB4 = pathlib.Path(r"03_STRATEGY_LAB/notebooks/04_RANGE_M5_Strategy_v1.ipynb")


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


def print_diff(cell_id, old_src, new_src, label=""):
    import difflib
    old_lines = old_src.split("\n")
    new_lines = new_src.split("\n")
    print(f"\n{'='*60}")
    print(f"Cell {cell_id} {label}")
    print(f"{'='*60}")
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
    for d in diff[2:]:
        print(f"  {d}")
    if not diff:
        print("  (no changes)")


nb = load_nb(NB4)

# ── Cell 10: TP default 2.5 -> 3.0 ──
cell = find_cell(nb, "8f3f03ad")
src = get_source(cell)
orig = src

src = assert_replace(src, "TP_ATR     = 2.5", "TP_ATR     = 3.0", "Cell10 TP default")

set_source(cell, src)
print_diff("8f3f03ad", orig, src, "(Cell 10 - TP 2.5->3.0)")

# ── Cell 14: add TP=2.75 to grid ──
cell = find_cell(nb, "b41156f4")
src = get_source(cell)
orig = src

src = assert_replace(src,
    "TP_GRID    = [2.0, 2.5, 3.0, 3.5]",
    "TP_GRID    = [2.0, 2.5, 2.75, 3.0, 3.5]",
    "Cell14 TP grid +2.75")

# Update header
src = src.replace(
    "# Grid: SL=[1.0-2.0] TP=[2.0-3.5]",
    "# Grid: SL=[1.0-2.0] TP=[2.0-3.5+2.75]"
)

set_source(cell, src)
print_diff("b41156f4", orig, src, "(Cell 14 - Grid +TP=2.75)")

# ── Summary ──
print("\n=== VIABILITY MATH ===")
sl, tp_new = 1.5, 3.0
be_new = sl / (sl + tp_new)
print(f"New default: SL={sl} TP={tp_new} => BE_WR={be_new:.1%}")
print(f"WR observado: 35.5%")
print(f"Margen: {0.355 - be_new:+.1%} (viable)")

import itertools
sl_g = [1.0, 1.5, 2.0]
tp_g = [2.0, 2.5, 2.75, 3.0, 3.5]
bk_g = [1.5, 2.0, 2.5, 3.0]
ts_g = [48, 96, 144]
combos = list(itertools.product(sl_g, tp_g, bk_g, ts_g))
viable = [c for c in combos if c[0]/(c[0]+c[1]) <= 0.355]
print(f"Grid: {len(combos)} combos, {len(viable)} viable (BE<=35.5%): {len(viable)/len(combos):.0%}")

if DRY_RUN:
    print("\n[DRY RUN] No files written.")
else:
    save_nb(NB4, nb)
    print(f"\nSaved {NB4}")
