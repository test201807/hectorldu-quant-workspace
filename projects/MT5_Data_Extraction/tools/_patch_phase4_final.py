"""
Phase 4: Final Edge-Based Implementation

Justified by edge diagnosis + tuning results:

TREND (NB3) — 3 changes:
  1) Cell 10: TP_ATR 7.0 -> 14.0
     Evidence: IS tuning top combo for BTCUSD is SL=2.0, TP=14.0, TRAIL=0
     sum_ret=+0.5494 (highest of 2000 combos)
     BE_WR = 2.0/(2.0+14.0) = 12.5% — very safe margin vs WR~28%

  2) Cell 16: Add SYMBOL_WHITELIST=["BTCUSD"] + SIDE_FILTER="LONG"
     Evidence: Edge diagnosis shows:
       BTCUSD LONG: +0.356% fwd_ret at 24h (OOS)
       BTCUSD SHORT: +0.074% (marginal)
       XAUAUD LONG: -0.003% (no edge)
       XAUAUD SHORT: -0.114% (negative)
     Also: BTCUSD LONG OOS base PnL = +0.1990 (6/10 folds stress-positive)

  3) Cell 14: Grid focused on SL=[1.5,2.0,2.5], TP=[10.0,14.0], TRAIL=[0]
     Evidence: Top 10 IS combos all use TRAIL=0 and TP>=10

RANGE (NB4) — No changes (edge diagnosis = NO-ALPHA).
  Leave as damage-limiter with current restrictive params.

Usage: python tools/_patch_phase4_final.py [--dry-run]
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


nb3 = load_nb(NB3)

# ── Change 1: Cell 10 — TP_ATR 7.0 -> 14.0 ──
cell = find_cell(nb3, "be93fd2e")
src = get_source(cell)
orig = src

src = assert_replace(src, "TP_ATR     = 7.0", "TP_ATR     = 14.0", "NB3.10 TP_ATR")

set_source(cell, src)
print_diff("be93fd2e", orig, src, "(Cell 10 - TP 7.0 -> 14.0)")

# ── Change 2: Cell 14 — Grid focused ──
cell = find_cell(nb3, "515f5bb0")
src = get_source(cell)
orig = src

# SL grid: keep [1.5, 2.0, 2.5, 3.0]  (already fine)
# TP grid: focus on high values
src = assert_replace(src,
    "TP_ATR_GRID    = [5.0, 7.0, 10.0, 14.0]",
    "TP_ATR_GRID    = [7.0, 10.0, 14.0]",
    "NB3.14 TP_GRID")

# TRAIL grid: only 0 (no trail) — all top IS combos use TRAIL=0
src = assert_replace(src,
    "TRAIL_ATR_GRID = [0, 7.0, 10.0]",
    "TRAIL_ATR_GRID = [0]",
    "NB3.14 TRAIL_GRID")

# TS grid: keep [288, 576]
# MIN_HOLD grid: reduce to [3, 6] (12 not needed, tuning shows same score)
src = assert_replace(src,
    "MIN_HOLD_GRID  = [3, 6, 12]",
    "MIN_HOLD_GRID  = [3, 6]",
    "NB3.14 MH_GRID")

# Update header
src = src.replace(
    "# Grid: SL=[1.5-3.0] TP=[5.0-14.0] Trail=[0,7.0,10.0] time_stop=[288-576]",
    "# Grid: SL=[1.5-3.0] TP=[7.0-14.0] Trail=[0] time_stop=[288-576]"
)

set_source(cell, src)
print_diff("515f5bb0", orig, src, "(Cell 14 - Focused grid, TRAIL=0 only)")

# ── Change 3: Cell 16 — Add BTCUSD LONG filter ──
cell = find_cell(nb3, "ab67bb0e")
src = get_source(cell)
orig = src

# Add filter after reading trades, before applying overlay logic
# Find the anchor point: "n_before = df.height"
ANCHOR = "n_before = df.height"
if ANCHOR not in src:
    raise AssertionError(f"Anchor not found: {ANCHOR!r}")

FILTER_CODE = """n_before = df.height

    # Edge-based filter: BTCUSD LONG only (justified by edge diagnosis)
    # BTCUSD LONG fwd_ret@24h = +0.356%, all others <= 0
    SYMBOL_WHITELIST = ["BTCUSD"]
    SIDE_FILTER = "LONG"
    df = df.filter(
        pl.col("symbol").is_in(SYMBOL_WHITELIST) &
        (pl.col("side") == SIDE_FILTER)
    )
    n_after_edge_filter = df.height
    print(f"[Celda 16] Edge filter: {n_before} -> {n_after_edge_filter} (BTCUSD LONG only)")
    n_before = n_after_edge_filter"""

src = src.replace(ANCHOR, FILTER_CODE, 1)

set_source(cell, src)
print_diff("ab67bb0e", orig, src, "(Cell 16 - BTCUSD LONG edge filter)")


if DRY_RUN:
    print("\n[DRY RUN] NB3 not saved.")
else:
    save_nb(NB3, nb3)
    print(f"\nSaved NB3: {NB3}")

# Summary
print(f"\n{'='*60}")
print("PHASE 4 CHANGES")
print(f"{'='*60}")
print("""
NB3 TREND:
  Cell 10: TP_ATR 7.0 -> 14.0 (IS top combo SL=2.0/TP=14.0/TRAIL=0)
  Cell 14: Grid focused: TP=[7,10,14], TRAIL=[0], MH=[3,6]
  Cell 16: BTCUSD LONG filter (edge diagnosis: only segment with edge)

NB4 RANGE:
  No changes (NO-ALPHA confirmed by edge diagnosis)
""")
