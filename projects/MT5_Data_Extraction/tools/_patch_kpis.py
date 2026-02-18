"""
Patch NB3 and NB4 cells to improve KPIs.
Usage: python tools/_patch_kpis.py [--dry-run]

Changes:
  NB3 Cell 10: TRAIL_ATR 3→5, ENTRY_CONFIRM 12→6, TRAIL=0 support
  NB3 Cell 14: Expanded grid, TRAIL=0 in combos, MAX_COMBOS 100→200
  NB3 Cell 16: Overlay relaxed (-5% loss, +8% profit, 8 trades/day)
  NB3 Cell 17: Selection relaxed (MIN_OOS 80→30, MIN_WR 48→40%)
  NB4 Cell  5: BB_WIN/MEAN_WIN/RANGE_WIN 96→288
  NB4 Cell  6: Q_ER_HIGH 0.40→0.25, Q_VOL 0.90→0.80
  NB4 Cell 10: SL 1.5→2.5, TP 2.0→1.5, TIME_STOP 144→96, ENTRY_CONFIRM 6→12,
               BAND_K 1.5→2.0, MIN_HOLD 3→6, COOLDOWN 12→24
  NB4 Cell 14: Inverted SL/TP grid, BAND_K expanded, TS+48
  NB4 Cell 16: Overlay relaxed (same as NB3)
  NB4 Cell 17: Selection relaxed (same as NB3)
"""
import json, sys, pathlib, re

DRY_RUN = "--dry-run" in sys.argv

NB3 = pathlib.Path(r"03_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb")
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
    """Replace old→new in src. Raises if old not found."""
    if old not in src:
        raise AssertionError(f"Pattern not found ({label}): {old!r}")
    return src.replace(old, new, 1)


def print_diff(cell_id, old_src, new_src, label=""):
    """Print only the changed lines."""
    old_lines = old_src.split("\n")
    new_lines = new_src.split("\n")
    print(f"\n{'='*60}")
    print(f"Cell {cell_id} {label}")
    print(f"{'='*60}")
    # Show changed lines
    import difflib
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=1))
    for d in diff[2:]:  # skip --- and +++
        print(f"  {d}")
    if not diff:
        print("  (no changes)")


# ──────────────────────────────────────────────────────────────
# NB3 PATCHES
# ──────────────────────────────────────────────────────────────
def patch_nb3(nb):
    changes = 0

    # ── Cell 10 (be93fd2e): Engine defaults ──
    idx, cell = find_cell(nb, "be93fd2e")
    src = get_source(cell)
    orig = src

    # 1. TRAIL_ATR 3.0 → 5.0
    src = assert_replace(src, "TRAIL_ATR  = 3.0", "TRAIL_ATR  = 5.0", "NB3.10 TRAIL_ATR")

    # 2. ENTRY_CONFIRM 12 → 6
    src = assert_replace(src, "ENTRY_CONFIRM = 12", "ENTRY_CONFIRM = 6", "NB3.10 ENTRY_CONFIRM")

    # 3. TRAIL=0 support: after _TRAIL assignment, add None guard
    old_trail = "    _TRAIL = trail_atr if trail_atr is not None else TRAIL_ATR"
    new_trail = (
        "    _TRAIL = trail_atr if trail_atr is not None else TRAIL_ATR\n"
        "    _TRAIL = None if _TRAIL == 0 else _TRAIL   # TRAIL=0 -> sin trailing stop"
    )
    src = assert_replace(src, old_trail, new_trail, "NB3.10 _TRAIL guard")

    # 4. Guard trail_dist computation for _TRAIL=None
    #    Line pattern: "trail_dist = _TRAIL * atr_val"
    #    Replace with conditional
    old_td = "                trail_dist = _TRAIL * atr_val"
    new_td = "                trail_dist = _TRAIL * atr_val if _TRAIL is not None else None"
    # This appears twice (LONG and SHORT entries)
    count = src.count(old_td)
    assert count == 2, f"Expected 2 occurrences of trail_dist assignment, found {count}"
    src = src.replace(old_td, new_td)

    set_source(cell, src)
    print_diff("be93fd2e", orig, src, "(NB3 Cell 10 Engine)")
    changes += 1

    # ── Cell 14 (515f5bb0): Tuning grid ──
    idx, cell = find_cell(nb, "515f5bb0")
    src = get_source(cell)
    orig = src

    # Grid variable names: SL_ATR_GRID, TP_ATR_GRID, TRAIL_ATR_GRID, TIME_STOP_GRID, MIN_HOLD_GRID
    src = assert_replace(src,
        "SL_ATR_GRID    = [1.5, 2.0, 2.5]",
        "SL_ATR_GRID    = [1.5, 2.0, 2.5, 3.0]",
        "NB3.14 SL_GRID")
    src = assert_replace(src,
        "TP_ATR_GRID    = [3.0, 5.0, 7.0]",
        "TP_ATR_GRID    = [3.0, 5.0, 7.0, 10.0]",
        "NB3.14 TP_GRID")
    src = assert_replace(src,
        "TRAIL_ATR_GRID = [3.0, 4.0, 5.0]",
        "TRAIL_ATR_GRID = [0, 4.0, 5.0, 7.0]",
        "NB3.14 TRAIL_GRID")
    src = assert_replace(src,
        "TIME_STOP_GRID = [144, 288]",
        "TIME_STOP_GRID = [144, 288, 576]",
        "NB3.14 TS_GRID")
    src = assert_replace(src,
        "MIN_HOLD_GRID  = [6, 12]",
        "MIN_HOLD_GRID  = [3, 6, 12]",
        "NB3.14 MH_GRID")

    # MAX_COMBOS 100 → 200
    src = assert_replace(src, "MAX_COMBOS = 100", "MAX_COMBOS = 200", "NB3.14 MAX_COMBOS")

    # Constraint: "if tr > sl]" → "if tr == 0 or tr > sl]"
    src = assert_replace(src,
        "if tr > sl]",
        "if tr == 0 or tr > sl]",
        "NB3.14 trail constraint")

    # Update comment
    src = assert_replace(src,
        "Enforce Trail > SL",
        "Enforce Trail > SL or Trail=0 (no trail)",
        "NB3.14 comment")

    # Update header comment
    src = src.replace(
        "# Grid: SL=[1.5,2.0,2.5] TP=[3.0,5.0,7.0] Trail=[3.0,4.0,5.0] time_stop=[144,288]",
        "# Grid: SL=[1.5-3.0] TP=[3.0-10.0] Trail=[0,4.0-7.0] time_stop=[144-576]"
    )
    src = src.replace(
        "#   min_hold=[6,12].",
        "#   min_hold=[3,6,12]."
    )

    set_source(cell, src)
    print_diff("515f5bb0", orig, src, "(NB3 Cell 14 Tuning)")
    changes += 1

    # ── Cell 16 (ab67bb0e): Overlay ──
    idx, cell = find_cell(nb, "ab67bb0e")
    src = get_source(cell)
    orig = src

    src = assert_replace(src, "DAILY_MAX_LOSS   = -0.02", "DAILY_MAX_LOSS   = -0.05", "NB3.16 MAX_LOSS")
    src = assert_replace(src, "DAILY_MAX_PROFIT =  0.03", "DAILY_MAX_PROFIT =  0.08", "NB3.16 MAX_PROFIT")
    src = assert_replace(src, "MAX_TRADES_DAY   = 3", "MAX_TRADES_DAY   = 8", "NB3.16 MAX_TRADES")

    # Update comment
    src = src.replace(
        "# Params: daily_max_loss=-2%, daily_max_profit=+3%, max_trades_day=3",
        "# Params: daily_max_loss=-5%, daily_max_profit=+8%, max_trades_day=8"
    )

    set_source(cell, src)
    print_diff("ab67bb0e", orig, src, "(NB3 Cell 16 Overlay)")
    changes += 1

    # ── Cell 17 (cf14989a): Selection ──
    idx, cell = find_cell(nb, "cf14989a")
    src = get_source(cell)
    orig = src

    src = assert_replace(src, "MIN_OOS_TRADES = 80", "MIN_OOS_TRADES = 30", "NB3.17 MIN_OOS")
    src = assert_replace(src, "MIN_WINRATE = 0.48", "MIN_WINRATE = 0.40", "NB3.17 MIN_WINRATE")

    # Update comment
    src = src.replace(
        "# Gates: min_oos_trades=80, max_mdd=-0.20, min_totret=0.0, min_wr=0.48",
        "# Gates: min_oos_trades=30, max_mdd=-0.20, min_totret=0.0, min_wr=0.40"
    )

    set_source(cell, src)
    print_diff("cf14989a", orig, src, "(NB3 Cell 17 Selection)")
    changes += 1

    return changes


# ──────────────────────────────────────────────────────────────
# NB4 PATCHES
# ──────────────────────────────────────────────────────────────
def patch_nb4(nb):
    changes = 0

    # ── Cell 5 (16060e55): Features ──
    idx, cell = find_cell(nb, "16060e55")
    src = get_source(cell)
    orig = src

    # These use os.getenv pattern: BB_WIN = int(os.getenv("RANGE_M5_BB_WIN", "96"))
    src = assert_replace(src, '"RANGE_M5_BB_WIN", "96"', '"RANGE_M5_BB_WIN", "288"', "NB4.5 BB_WIN")
    src = assert_replace(src, '"RANGE_M5_MEAN_WIN", "96"', '"RANGE_M5_MEAN_WIN", "288"', "NB4.5 MEAN_WIN")
    src = assert_replace(src, '"RANGE_M5_RANGE_WIN", "96"', '"RANGE_M5_RANGE_WIN", "288"', "NB4.5 RANGE_WIN")

    set_source(cell, src)
    print_diff("16060e55", orig, src, "(NB4 Cell 5 Features)")
    changes += 1

    # ── Cell 6 (defd40fc): Regime gate ──
    idx, cell = find_cell(nb, "defd40fc")
    src = get_source(cell)
    orig = src

    src = assert_replace(src, "Q_ER_HIGH = 0.40", "Q_ER_HIGH = 0.25", "NB4.6 Q_ER_HIGH")
    src = assert_replace(src, "Q_VOL = 0.90", "Q_VOL = 0.80", "NB4.6 Q_VOL")

    # Update comments
    src = src.replace(
        "# ER at or below 40th percentile = ranging",
        "# ER at or below 25th percentile = truly ranging"
    )
    src = src.replace(
        "# vol below 90th percentile = not volatile",
        "# vol below 80th percentile = low volatility"
    )

    set_source(cell, src)
    print_diff("defd40fc", orig, src, "(NB4 Cell 6 Regime)")
    changes += 1

    # ── Cell 10 (8f3f03ad): Engine defaults ──
    idx, cell = find_cell(nb, "8f3f03ad")
    src = get_source(cell)
    orig = src

    # Exact spacing from source:
    # SL_ATR     = 1.5
    # TP_ATR     = 2.0
    # TIME_STOP  = 144    # 12h
    # ENTRY_CONFIRM = 6
    # MIN_HOLD   = 3
    # COOLDOWN   = 12
    # BAND_K     = 1.5
    src = assert_replace(src, "SL_ATR     = 1.5", "SL_ATR     = 2.5", "NB4.10 SL_ATR")
    src = assert_replace(src, "TP_ATR     = 2.0", "TP_ATR     = 1.5", "NB4.10 TP_ATR")
    src = assert_replace(src, "TIME_STOP  = 144    # 12h", "TIME_STOP  = 96     # 8h (aligned with alpha=48-96)", "NB4.10 TIME_STOP")
    src = assert_replace(src, "ENTRY_CONFIRM = 6", "ENTRY_CONFIRM = 12", "NB4.10 ENTRY_CONFIRM")
    src = assert_replace(src, "BAND_K     = 1.5", "BAND_K     = 2.0", "NB4.10 BAND_K")
    src = assert_replace(src, "MIN_HOLD   = 3", "MIN_HOLD   = 6", "NB4.10 MIN_HOLD")
    src = assert_replace(src, "COOLDOWN   = 12", "COOLDOWN   = 24", "NB4.10 COOLDOWN")

    set_source(cell, src)
    print_diff("8f3f03ad", orig, src, "(NB4 Cell 10 Engine)")
    changes += 1

    # ── Cell 14 (b41156f4): Tuning grid ──
    idx, cell = find_cell(nb, "b41156f4")
    src = get_source(cell)
    orig = src

    # Variable names: SL_GRID, TP_GRID, BAND_K_GRID, TS_GRID
    src = assert_replace(src,
        "SL_GRID    = [1.0, 1.5, 2.0]",
        "SL_GRID    = [2.0, 2.5, 3.0]",
        "NB4.14 SL_GRID")
    src = assert_replace(src,
        "TP_GRID    = [1.5, 2.0, 3.0]",
        "TP_GRID    = [1.0, 1.5, 2.0]",
        "NB4.14 TP_GRID")
    src = assert_replace(src,
        "BAND_K_GRID = [1.0, 1.5, 2.0]",
        "BAND_K_GRID = [1.5, 2.0, 2.5, 3.0]",
        "NB4.14 BK_GRID")
    src = assert_replace(src,
        "TS_GRID    = [96, 144]",
        "TS_GRID    = [48, 96, 144]",
        "NB4.14 TS_GRID")

    # Update header comment
    src = src.replace(
        "# Grid: SL=[1.0,1.5,2.0] TP=[1.5,2.0,3.0] BAND_K=[1.0,1.5,2.0] time_stop=[96,144]",
        "# Grid: SL=[2.0-3.0] TP=[1.0-2.0] BAND_K=[1.5-3.0] time_stop=[48-144]"
    )

    set_source(cell, src)
    print_diff("b41156f4", orig, src, "(NB4 Cell 14 Tuning)")
    changes += 1

    # ── Cell 16 (75cbc797): Overlay ──
    idx, cell = find_cell(nb, "75cbc797")
    src = get_source(cell)
    orig = src

    # NB4 format: single line with semicolons
    src = assert_replace(src,
        "DAILY_MAX_LOSS = -0.02; DAILY_MAX_PROFIT = 0.03; MAX_TRADES_DAY = 3",
        "DAILY_MAX_LOSS = -0.05; DAILY_MAX_PROFIT = 0.08; MAX_TRADES_DAY = 8",
        "NB4.16 overlay params")

    set_source(cell, src)
    print_diff("75cbc797", orig, src, "(NB4 Cell 16 Overlay)")
    changes += 1

    # ── Cell 17 (709ba063): Selection ──
    idx, cell = find_cell(nb, "709ba063")
    src = get_source(cell)
    orig = src

    # NB4 format: single line with semicolons
    src = assert_replace(src,
        "MIN_OOS_TRADES = 80; MAX_MDD = -0.20; MIN_TOTRET = 0.0; MIN_WR = 0.48",
        "MIN_OOS_TRADES = 30; MAX_MDD = -0.20; MIN_TOTRET = 0.0; MIN_WR = 0.40",
        "NB4.17 selection params")

    set_source(cell, src)
    print_diff("709ba063", orig, src, "(NB4 Cell 17 Selection)")
    changes += 1

    return changes


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading NB3...")
    nb3 = load_nb(NB3)
    n3 = patch_nb3(nb3)
    print(f"\nNB3: {n3} cells patched")

    print("\n\nLoading NB4...")
    nb4 = load_nb(NB4)
    n4 = patch_nb4(nb4)
    print(f"\nNB4: {n4} cells patched")

    if DRY_RUN:
        print("\n[DRY RUN] No files written.")
    else:
        save_nb(NB3, nb3)
        save_nb(NB4, nb4)
        print(f"\nSaved {NB3}")
        print(f"Saved {NB4}")

    print(f"\nTotal: {n3 + n4} cell patches applied")
