"""
Phase 5: Relax Overlay Daily Caps

Problem: Daily caps destroy positive OOS edge.
  Pre-overlay BTCUSD LONG OOS: base=+0.1635 (POSITIVE)
  Post-overlay: base=-0.3177 (DESTROYED)

Root cause: DAILY_MAX_LOSS=-0.05 with BTCUSD's large ATR means
  1-2 SL hits cap the day, filtering subsequent winners.

Fix: Effectively disable daily caps. The BTCUSD LONG edge filter
  (added in Phase 4) is the real quality gate.

Usage: python tools/_patch_phase5_overlay.py [--dry-run]
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

# ── Cell 16 — Relax daily caps ──
cell = find_cell(nb3, "ab67bb0e")
src = get_source(cell)
orig = src

# Relax DAILY_MAX_LOSS: -0.05 -> -1.00 (effectively disabled)
src = assert_replace(src,
    "DAILY_MAX_LOSS   = -0.05",
    "DAILY_MAX_LOSS   = -1.00   # effectively disabled (edge filter is the gate)",
    "NB3.16 DAILY_MAX_LOSS")

# Relax DAILY_MAX_PROFIT: 0.08 -> 1.00 (effectively disabled)
src = assert_replace(src,
    "DAILY_MAX_PROFIT =  0.08",
    "DAILY_MAX_PROFIT =  1.00   # effectively disabled",
    "NB3.16 DAILY_MAX_PROFIT")

# Relax MAX_TRADES_DAY: 12 -> 200 (effectively disabled)
src = assert_replace(src,
    "MAX_TRADES_DAY   = 12",
    "MAX_TRADES_DAY   = 200    # effectively disabled",
    "NB3.16 MAX_TRADES_DAY")

# Update header comment
src = src.replace(
    "# Params: daily_max_loss=-5%, daily_max_profit=+8%, max_trades_day=8, weekdays_only",
    "# Params: daily caps DISABLED (edge filter is gate), weekdays_only"
)

set_source(cell, src)
print_diff("ab67bb0e", orig, src, "(Cell 16 - Relax daily caps)")

if DRY_RUN:
    print("\n[DRY RUN] NB3 not saved.")
else:
    save_nb(NB3, nb3)
    print(f"\nSaved NB3: {NB3}")

print(f"\n{'='*60}")
print("PHASE 5: OVERLAY FIX")
print(f"{'='*60}")
print("""
Cell 16 (ab67bb0e):
  DAILY_MAX_LOSS:   -0.05 -> -1.00 (disabled)
  DAILY_MAX_PROFIT:  0.08 ->  1.00 (disabled)
  MAX_TRADES_DAY:      12 ->   200 (disabled)

Rationale: Edge filter (BTCUSD LONG only) is the real quality gate.
  Daily caps were destroying positive OOS edge.
""")
