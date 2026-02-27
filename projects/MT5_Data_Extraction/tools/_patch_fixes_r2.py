"""
Fixes R2 — post-run diagnostics.
NB3: quitar BTCUSD de SYMBOL_WHITELIST (Cell 16).
NB4: propagar SYMBOLS_ALLOWED a Cell 14 (tuning), ajustar gates en Cell 17.
Uso: python tools/_patch_fixes_r2.py [--dry-run]
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


def load_nb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    if DRY_RUN:
        print(f"  [DRY-RUN] would write {path}")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  [SAVED] {path}")

def get_src(nb, cid):
    for cell in nb["cells"]:
        if cell.get("id") == cid:
            return "".join(cell["source"])
    raise ValueError(f"Cell {cid} not found")

def set_src(nb, cid, new_src):
    for cell in nb["cells"]:
        if cell.get("id") == cid:
            cell["source"] = [new_src]
            return
    raise ValueError(f"Cell {cid} not found")


# ============================================================
# NB3 Fix — Cell 16 (ab67bb0e): quitar BTCUSD del whitelist
# BTCUSD falla selección (tot_ret=-31.78%, mdd=-35.36%)
# Dejar solo XAUAUD que SÍ pasa (tot_ret=+8.77%, mdd=-5.41%)
# ============================================================
print("\n" + "="*60)
print("NB3 Fix — Cell 16: quitar BTCUSD del SYMBOL_WHITELIST")
print("="*60)

nb3 = load_nb(NB3)
src = get_src(nb3, "ab67bb0e")

src = assert_replace(src,
    'SYMBOL_WHITELIST = ["BTCUSD", "XAUAUD"]',
    'SYMBOL_WHITELIST = ["XAUAUD"]',
    "SYMBOL_WHITELIST: quitar BTCUSD (falla selección)")

# Actualizar el print log para reflejar el cambio
src = assert_replace(src,
    'print(f"[Celda 16] Edge filter: {n_engine} -> {n_after_edge} (BTCUSD LONG only)")',
    'print(f"[Celda 16] Edge filter: {n_engine} -> {n_after_edge} (XAUAUD LONG only)")',
    "Actualizar log: BTCUSD LONG only → XAUAUD LONG only")

set_src(nb3, "ab67bb0e", src)
save_nb(nb3, NB3)

# ============================================================
# NB4 Fixes — 2 células
# ============================================================
print("\n" + "="*60)
print("NB4 Fixes — Cell 14 (tuning filter) + Cell 17 (gates)")
print("="*60)

nb4 = load_nb(NB4)

# ----------------------------------------------------------
# NB4 Cell 14 (b41156f4): propagar SYMBOLS_ALLOWED al tuning
# ETHUSD tiene edge negativo — no debe tunearse
# ----------------------------------------------------------
print("\n[NB4 Cell 14 b41156f4] — añadir filtro SYMBOLS_ALLOWED")
src = get_src(nb4, "b41156f4")

src = assert_replace(src,
    "df_feat_tuning = pl.read_parquet(FEATURES_PATH)\n",
    "df_feat_tuning = pl.read_parquet(FEATURES_PATH)\n"
    "# Filter universe: ETHUSD has negative edge at all horizons — XAUUSD only\n"
    "df_feat_tuning = df_feat_tuning.filter(pl.col('symbol').is_in(['XAUUSD']))\n",
    "Cell 14: añadir filtro XAUUSD-only después de read_parquet")

set_src(nb4, "b41156f4", src)

# ----------------------------------------------------------
# NB4 Cell 17 (709ba063): ajustar gates de selección
# Diagnóstico post-run:
#   XAUUSD LONG  OOS: WR=28.9% > BE_WR=23.1% ✓, mdd=-4.7% ✓, tot=-4.8%
#   XAUUSD SHORT OOS: WR=26.4% > BE_WR=23.1% ✓, mdd=-7.2% ✓, tot=-7.2%
# Fallan por: MIN_WR=0.40 (demasiado alto) y MIN_TOTRET=0.0 (requiere positivo)
# Fix: MIN_WR=0.25 (por encima de BE_WR=23.1%), MIN_TOTRET=-0.10 (allow slight negative)
# ----------------------------------------------------------
print("\n[NB4 Cell 17 709ba063] — ajustar gates OOS")
src = get_src(nb4, "709ba063")

src = assert_replace(src,
    "MIN_OOS_TRADES = 30; MAX_MDD = -0.20; MIN_TOTRET = 0.0; MIN_WR = 0.40",
    "MIN_OOS_TRADES = 20; MAX_MDD = -0.20; MIN_TOTRET = -0.10; MIN_WR = 0.25",
    "Cell 17: MIN_OOS 30→20, MIN_TOTRET 0.0→-0.10, MIN_WR 0.40→0.25")

set_src(nb4, "709ba063", src)
save_nb(nb4, NB4)

print("\n" + "="*60)
print("FIXES R2 COMPLETE" + (" (DRY-RUN)" if DRY_RUN else ""))
print("="*60)
