"""
Fix NB4 SL/TP breakeven error.

Original plan had formula inverted: said BE WR = TP/(SL+TP) = 37.5%
Correct formula: BE WR = SL/(SL+TP) = 62.5% -- inviable with WR=40%

This script fixes:
  Cell 10: SL 2.5->1.5, TP 1.5->2.5 (BE WR=37.5%, viable with WR~40%)
  Cell 14: Grid inverted back to TP > SL ratios
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
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == cell_id:
            return i, c
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

# ── Cell 10 (8f3f03ad): Fix defaults ──
idx, cell = find_cell(nb, "8f3f03ad")
src = get_source(cell)
orig = src

# Current (wrong): SL=2.5 TP=1.5 => BE WR=62.5%
# Fixed:           SL=1.5 TP=2.5 => BE WR=37.5%
src = assert_replace(src, "SL_ATR     = 2.5", "SL_ATR     = 1.5", "NB4.10 SL fix")
src = assert_replace(src, "TP_ATR     = 1.5", "TP_ATR     = 2.5", "NB4.10 TP fix")

set_source(cell, src)
print_diff("8f3f03ad", orig, src, "(NB4 Cell 10 - SL/TP fix)")

# ── Cell 14 (b41156f4): Fix grid ──
idx, cell = find_cell(nb, "b41156f4")
src = get_source(cell)
orig = src

# Current (wrong): SL=[2.0,2.5,3.0] TP=[1.0,1.5,2.0] => all BE WR > 50%
# Fixed:           SL=[1.0,1.5,2.0] TP=[2.0,2.5,3.0,3.5] => most BE WR < 42%
src = assert_replace(src,
    "SL_GRID    = [2.0, 2.5, 3.0]",
    "SL_GRID    = [1.0, 1.5, 2.0]",
    "NB4.14 SL_GRID fix")
src = assert_replace(src,
    "TP_GRID    = [1.0, 1.5, 2.0]",
    "TP_GRID    = [2.0, 2.5, 3.0, 3.5]",
    "NB4.14 TP_GRID fix")

# Fix header comment
src = src.replace(
    "# Grid: SL=[2.0-3.0] TP=[1.0-2.0]",
    "# Grid: SL=[1.0-2.0] TP=[2.0-3.5]"
)

set_source(cell, src)
print_diff("b41156f4", orig, src, "(NB4 Cell 14 - Grid fix)")

if DRY_RUN:
    print("\n[DRY RUN] No files written.")
else:
    save_nb(NB4, nb)
    print(f"\nSaved {NB4}")

print("\nDone.")
