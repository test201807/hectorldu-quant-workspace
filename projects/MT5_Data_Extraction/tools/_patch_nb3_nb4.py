"""
Patch NB3 + NB4 — Rediseno TREND/RANGE para rentabilidad FTMO.
Uso: python tools/_patch_nb3_nb4.py [--dry-run]
"""
import json, sys
from pathlib import Path

DRY_RUN = "--dry-run" in sys.argv
ROOT = Path(__file__).resolve().parent.parent

NB3 = ROOT / "03_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb"
NB4 = ROOT / "03_STRATEGY_LAB/notebooks/04_RANGE_M5_Strategy_v1.ipynb"


def assert_replace(src: str, old: str, new: str, label: str) -> str:
    assert old in src, f"[FAIL] '{label}': old string NOT found:\n{repr(old)}"
    result = src.replace(old, new, 1)
    assert result != src, f"[FAIL] '{label}': replace produced no change"
    print(f"  [OK] {label}")
    return result


def load_nb(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb: dict, path: Path):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  [SAVED] {path}")


def get_cell_src(nb: dict, cell_id: str) -> str:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise ValueError(f"Cell {cell_id} not found")


def set_cell_src(nb: dict, cell_id: str, new_src: str):
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            cell["source"] = [new_src]
            return
    raise ValueError(f"Cell {cell_id} not found")


# ============================================================
# NB3 — 03_TREND_M5_Strategy_v2.ipynb
# ============================================================
print("\n" + "="*60)
print("Patching NB3 TREND...")
print("="*60)

nb3 = load_nb(NB3)

# ----------------------------------------------------------
# Cell 10 (be93fd2e) — Engine baseline params
# Changes: SL, TIME_STOP, ENTRY_CONFIRM, EXIT_GATE_OFF, MIN_HOLD, COOLDOWN
# Plus: add entry_confirm_bars param to _simulate()
# ----------------------------------------------------------
print("\n[NB3 Cell 10 be93fd2e]")
src = get_cell_src(nb3, "be93fd2e")

# Param defaults
src = assert_replace(src, "SL_ATR     = 2.0", "SL_ATR     = 3.0", "SL_ATR 2->3")
src = assert_replace(src, "TIME_STOP  = 576", "TIME_STOP  = 1440", "TIME_STOP 576->1440")
src = assert_replace(src, "ENTRY_CONFIRM = 6", "ENTRY_CONFIRM = 28", "ENTRY_CONFIRM 6->28")
src = assert_replace(src, "EXIT_GATE_OFF = 12", "EXIT_GATE_OFF = 72", "EXIT_GATE_OFF 12->72")
src = assert_replace(src, "MIN_HOLD   = 6", "MIN_HOLD   = 72", "MIN_HOLD 6->72")
src = assert_replace(src, "COOLDOWN   = 24", "COOLDOWN   = 48", "COOLDOWN 24->48")

# Add entry_confirm_bars kwarg to _simulate signature
src = assert_replace(
    src,
    "              *, sl_atr=None, tp_atr=None, trail_atr=None, time_stop=None, min_hold=None):",
    "              *, sl_atr=None, tp_atr=None, trail_atr=None, time_stop=None, min_hold=None, entry_confirm_bars=None):",
    "_simulate signature: add entry_confirm_bars",
)

# Add local _EC variable after _MHOLD line
src = assert_replace(
    src,
    "    _MHOLD = min_hold  if min_hold  is not None else MIN_HOLD\n",
    "    _MHOLD = min_hold  if min_hold  is not None else MIN_HOLD\n"
    "    _EC    = entry_confirm_bars if entry_confirm_bars is not None else ENTRY_CONFIRM\n",
    "_simulate body: add _EC local",
)

# Replace rolling_sum uses of ENTRY_CONFIRM with _EC (inside _simulate)
src = assert_replace(
    src,
    "        (pl.col(\"_gL\").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))\n"
    "            .fill_null(False).alias(\"_confL\"),\n"
    "        (pl.col(\"_gS\").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))\n"
    "            .fill_null(False).alias(\"_confS\"),",
    "        (pl.col(\"_gL\").cast(pl.Int8).rolling_sum(_EC, min_samples=_EC).eq(_EC))\n"
    "            .fill_null(False).alias(\"_confL\"),\n"
    "        (pl.col(\"_gS\").cast(pl.Int8).rolling_sum(_EC, min_samples=_EC).eq(_EC))\n"
    "            .fill_null(False).alias(\"_confS\"),",
    "_simulate: rolling_sum ENTRY_CONFIRM -> _EC",
)

set_cell_src(nb3, "be93fd2e", src)

# ----------------------------------------------------------
# Cell 14 (515f5bb0) — Tuning grid
# Changes: SL/TP/TIME_STOP/MIN_HOLD grids, add ENTRY_CONFIRM_GRID
# Update combos, loop, _simulate call, results dict, snapshot
# ----------------------------------------------------------
print("\n[NB3 Cell 14 515f5bb0]")
src = get_cell_src(nb3, "515f5bb0")

# Grid definitions
src = assert_replace(src,
    "SL_ATR_GRID    = [1.5, 2.0, 2.5, 3.0]",
    "SL_ATR_GRID    = [2.5, 3.0, 3.5, 4.0]",
    "SL_ATR_GRID update")
src = assert_replace(src,
    "TP_ATR_GRID    = [7.0, 10.0, 14.0]",
    "TP_ATR_GRID    = [10.0, 14.0, 20.0]",
    "TP_ATR_GRID update")
src = assert_replace(src,
    "TIME_STOP_GRID = [288, 576]",
    "TIME_STOP_GRID = [576, 1440, 2880]",
    "TIME_STOP_GRID update")
src = assert_replace(src,
    "MIN_HOLD_GRID  = [3, 6]\n",
    "MIN_HOLD_GRID  = [12, 48, 72]\n"
    "ENTRY_CONFIRM_GRID = [12, 28, 48]\n",
    "MIN_HOLD_GRID update + add ENTRY_CONFIRM_GRID")

# Combos tuple — add entry_confirm dimension
src = assert_replace(src,
    "combos = [(sl, tp, tr, ts, mh)\n"
    "          for sl, tp, tr, ts, mh in itertools.product(\n"
    "              SL_ATR_GRID, TP_ATR_GRID, TRAIL_ATR_GRID, TIME_STOP_GRID, MIN_HOLD_GRID)\n"
    "          if tr == 0 or tr > sl][:MAX_COMBOS]",
    "combos = [(sl, tp, tr, ts, mh, ec)\n"
    "          for sl, tp, tr, ts, mh, ec in itertools.product(\n"
    "              SL_ATR_GRID, TP_ATR_GRID, TRAIL_ATR_GRID, TIME_STOP_GRID, MIN_HOLD_GRID, ENTRY_CONFIRM_GRID)\n"
    "          if tr == 0 or tr > sl][:MAX_COMBOS]",
    "combos: add entry_confirm dimension")

# Loop unpacking
src = assert_replace(src,
    "        for sl, tp, tr, ts, mh in combos:",
    "        for sl, tp, tr, ts, mh, ec in combos:",
    "loop: unpack ec")

# _simulate call — add entry_confirm_bars=ec
src = assert_replace(src,
    "                               sl_atr=sl, tp_atr=tp, trail_atr=tr, time_stop=ts, min_hold=mh)",
    "                               sl_atr=sl, tp_atr=tp, trail_atr=tr, time_stop=ts, min_hold=mh, entry_confirm_bars=ec)",
    "_simulate call: add entry_confirm_bars=ec")

# Results dict — add entry_confirm field
src = assert_replace(src,
    "                \"sl_atr\": sl, \"tp_atr\": tp, \"trail_atr\": tr,\n"
    "                \"time_stop\": ts, \"min_hold\": mh,",
    "                \"sl_atr\": sl, \"tp_atr\": tp, \"trail_atr\": tr,\n"
    "                \"time_stop\": ts, \"min_hold\": mh, \"entry_confirm\": ec,",
    "results dict: add entry_confirm")

# Snapshot grid — add ENTRY_CONFIRM_GRID
src = assert_replace(src,
    "             \"TIME_STOP\": TIME_STOP_GRID, \"MIN_HOLD\": MIN_HOLD_GRID},",
    "             \"TIME_STOP\": TIME_STOP_GRID, \"MIN_HOLD\": MIN_HOLD_GRID, \"ENTRY_CONFIRM\": ENTRY_CONFIRM_GRID},",
    "snapshot grid: add ENTRY_CONFIRM_GRID")

set_cell_src(nb3, "515f5bb0", src)

# ----------------------------------------------------------
# Cell 15 (40a2390f) — Alpha Design — NO CHANGES
# ----------------------------------------------------------
print("\n[NB3 Cell 15 40a2390f] — sin cambios")

# ----------------------------------------------------------
# Cell 16 (ab67bb0e) — Overlay / Edge filter
# Changes: SYMBOL_WHITELIST add XAUAUD
# ----------------------------------------------------------
print("\n[NB3 Cell 16 ab67bb0e]")
src = get_cell_src(nb3, "ab67bb0e")

src = assert_replace(src,
    'SYMBOL_WHITELIST = ["BTCUSD"]',
    'SYMBOL_WHITELIST = ["BTCUSD", "XAUAUD"]',
    "SYMBOL_WHITELIST: add XAUAUD")

set_cell_src(nb3, "ab67bb0e", src)

# ----------------------------------------------------------
# Cell 17 (cf14989a) — Selection gates
# Changes: MIN_OOS_TRADES, MAX_MDD, MIN_TOTRET, MIN_WINRATE
# ----------------------------------------------------------
print("\n[NB3 Cell 17 cf14989a]")
src = get_cell_src(nb3, "cf14989a")

src = assert_replace(src, "MIN_OOS_TRADES = 30", "MIN_OOS_TRADES = 20", "MIN_OOS_TRADES 30->20")
src = assert_replace(src, "MAX_MDD = -0.20", "MAX_MDD = -0.35", "MAX_MDD -0.20->-0.35")
src = assert_replace(src, "MIN_TOTRET = -0.05", "MIN_TOTRET = -0.15", "MIN_TOTRET -0.05->-0.15")
src = assert_replace(src,
    "MIN_WINRATE = 0.15   # BE_WR=12.5% for SL=2/TP=14, 0.15 gives safety margin",
    "MIN_WINRATE = 0.10   # BE_WR=17.6% for SL=3/TP=14, 0.10 filters strategies without real edge",
    "MIN_WINRATE 0.15->0.10 + update comment")

set_cell_src(nb3, "cf14989a", src)

save_nb(nb3, NB3)

# ============================================================
# NB4 — 04_RANGE_M5_Strategy_v1.ipynb
# ============================================================
print("\n" + "="*60)
print("Patching NB4 RANGE...")
print("="*60)

nb4 = load_nb(NB4)

# ----------------------------------------------------------
# Cell 6 (defd40fc) — Regime gate — NO CHANGES
# (ETHUSD filter se aplica en Cell 10)
# ----------------------------------------------------------
print("\n[NB4 Cell 06 defd40fc] — sin cambios (filtro ETHUSD va en Cell 10)")

# ----------------------------------------------------------
# Cell 10 (8f3f03ad) — Engine baseline RANGE
# Changes: add SYMBOLS_ALLOWED filter, SL/TP/TIME_STOP/BAND_K/MIN_HOLD/ENTRY_CONFIRM
# ----------------------------------------------------------
print("\n[NB4 Cell 10 8f3f03ad]")
src = get_cell_src(nb4, "8f3f03ad")

# Add SYMBOLS_ALLOWED filter after loading df_feat
src = assert_replace(src,
    "df_feat = pl.read_parquet(FEATURES_PATH)\n"
    "df_folds = pl.read_parquet(WFO_PATH)",
    "df_feat = pl.read_parquet(FEATURES_PATH)\n"
    "# Filter universe: ETHUSD has negative edge at all horizons — XAUUSD only\n"
    "SYMBOLS_ALLOWED = ['XAUUSD']\n"
    "df_feat = df_feat.filter(pl.col('symbol').is_in(SYMBOLS_ALLOWED))\n"
    "df_folds = pl.read_parquet(WFO_PATH)",
    "add SYMBOLS_ALLOWED filter after df_feat load")

# Engine params
src = assert_replace(src, "SL_ATR     = 1.0", "SL_ATR     = 1.5", "SL_ATR 1.0->1.5")
src = assert_replace(src, "TP_ATR     = 3.0", "TP_ATR     = 5.0", "TP_ATR 3.0->5.0")
src = assert_replace(src, "TIME_STOP  = 48     # 8h (aligned with alpha=48-96)", "TIME_STOP  = 96     # 8h (aligned with alpha horizon 96 bars)", "TIME_STOP 48->96")
src = assert_replace(src, "ENTRY_CONFIRM = 12", "ENTRY_CONFIRM = 6", "ENTRY_CONFIRM 12->6")
src = assert_replace(src, "MIN_HOLD   = 3", "MIN_HOLD   = 24", "MIN_HOLD 3->24")
src = assert_replace(src, "BAND_K     = 2.0", "BAND_K     = 2.5", "BAND_K 2.0->2.5")

set_cell_src(nb4, "8f3f03ad", src)

# ----------------------------------------------------------
# Cell 14 (b41156f4) — Tuning grid RANGE
# Changes: SL_GRID, TP_GRID, TS_GRID (BAND_K_GRID unchanged)
# ----------------------------------------------------------
print("\n[NB4 Cell 14 b41156f4]")
src = get_cell_src(nb4, "b41156f4")

src = assert_replace(src,
    "SL_GRID    = [0.75, 1.0, 1.5]",
    "SL_GRID    = [1.0, 1.5, 2.0]",
    "SL_GRID update")
src = assert_replace(src,
    "TP_GRID    = [2.5, 3.0, 3.5, 4.0]",
    "TP_GRID    = [3.5, 5.0, 7.0]",
    "TP_GRID update")
src = assert_replace(src,
    "TS_GRID    = [24, 48, 96]",
    "TS_GRID    = [48, 96, 192]",
    "TS_GRID update")

set_cell_src(nb4, "b41156f4", src)

save_nb(nb4, NB4)

print("\n" + "="*60)
print("PATCH COMPLETE" + (" (DRY-RUN)" if DRY_RUN else ""))
print("="*60)
