"""
Patch: Unificar costos engine <-> cost_model_snapshot

Corrige DOS bugs superpuestos en NB3/NB4 Cells 10 y 14:
  1. Key mismatch: buscaban "costs_by_symbol" (dict) pero snapshot tiene "per_symbol" (lista)
  2. Field mismatch: buscaban "cost_base_bps" pero snapshot tiene "base_cost_bps"

Resultado: siempre caian al fallback 3/6 bps, ignorando 8/16 bps del snapshot.

Celdas afectadas:
  NB3 Cell 10 (be93fd2e) — Engine
  NB3 Cell 14 (515f5bb0) — Tuning
  NB4 Cell 10 (8f3f03ad) — Engine
  NB4 Cell 14 (b41156f4) — Tuning

Uso:
  python tools/_patch_cost_alignment.py --dry-run   (default, muestra cambios)
  python tools/_patch_cost_alignment.py --apply      (aplica cambios)
"""
import json, sys, pathlib

DRY_RUN = "--apply" not in sys.argv
PROJECT = pathlib.Path(__file__).parent.parent
NB3 = PROJECT / "03_STRATEGY_LAB/notebooks/03_TREND_M5_Strategy_v2.ipynb"
NB4 = PROJECT / "03_STRATEGY_LAB/notebooks/04_RANGE_M5_Strategy_v1.ipynb"


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
    count = src.count(old)
    if count > 1:
        raise AssertionError(f"Pattern appears {count} times, expected 1 ({label}): {old!r}")
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


changes = 0

# ============================================================================
# NB3 TREND
# ============================================================================
print("\n" + "#" * 60)
print("# NB3 TREND — Cost alignment fixes")
print("#" * 60)

nb3 = load_nb(NB3)

# -- NB3 Cell 10: Engine (be93fd2e) --
cell = find_cell(nb3, "be93fd2e")
src = get_source(cell)
orig = src

# Fix 1: snapshot load — costs_by_symbol -> per_symbol
src = assert_replace(src,
    'costs_by_sym = cost_snap.get("costs_by_symbol", {})',
    'costs_by_sym = {e["symbol"]: e for e in cost_snap.get("per_symbol", [])}',
    "NB3.10 snapshot load")

# Fix 2: cost_base_bps field name + default 3->8
src = assert_replace(src,
    'cost_base_bps = float(cinfo.get("cost_base_bps", cinfo.get("COST_BASE_BPS", 3.0)))',
    'cost_base_bps = float(cinfo.get("base_cost_bps", 8.0))',
    "NB3.10 cost_base_bps")

# Fix 3: cost_stress_bps field name + default 6->16
src = assert_replace(src,
    'cost_stress_bps = float(cinfo.get("cost_stress_bps", cinfo.get("COST_STRESS_BPS", 6.0)))',
    'cost_stress_bps = float(cinfo.get("stress_cost_bps", 16.0))',
    "NB3.10 cost_stress_bps")

# Add validation print after cost_stress_dec line
src = assert_replace(src,
    "    cost_stress_dec = cost_stress_bps / 10_000\n",
    '    cost_stress_dec = cost_stress_bps / 10_000\n'
    '    print(f"  [{sym}] cost_base={cost_base_bps:.1f}bps, cost_stress={cost_stress_bps:.1f}bps "\n'
    '          f"(from={\'snapshot\' if cinfo else \'default\'})")\n',
    "NB3.10 validation print")

set_source(cell, src)
print_diff("be93fd2e", orig, src, "(NB3 Cell 10 — Engine)")
changes += 4

# -- NB3 Cell 14: Tuning (515f5bb0) --
cell = find_cell(nb3, "515f5bb0")
src = get_source(cell)
orig = src

# Fix 1: snapshot load — costs_by_symbol -> per_symbol
src = assert_replace(src,
    'costs_by_sym_tuning = cost_snap_tuning.get("costs_by_symbol", {})',
    'costs_by_sym_tuning = {e["symbol"]: e for e in cost_snap_tuning.get("per_symbol", [])}',
    "NB3.14 snapshot load")

# Fix 2: cost_base_dec field name + default 3->8
src = assert_replace(src,
    'cost_base_dec = float(cinfo.get("cost_base_bps", cinfo.get("COST_BASE_BPS", 3.0))) / 10_000',
    'cost_base_dec = float(cinfo.get("base_cost_bps", 8.0)) / 10_000',
    "NB3.14 cost_base_dec")

# Fix 3: cost_stress_dec field name + default 6->16
src = assert_replace(src,
    'cost_stress_dec = float(cinfo.get("cost_stress_bps", cinfo.get("COST_STRESS_BPS", 6.0))) / 10_000',
    'cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000',
    "NB3.14 cost_stress_dec")

# Add validation print after cost_stress_dec line
src = assert_replace(src,
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n',
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n'
    '    print(f"  [{sym}] cost_base={cost_base_dec*10_000:.1f}bps, cost_stress={cost_stress_dec*10_000:.1f}bps "\n'
    '          f"(from={\'snapshot\' if cinfo else \'default\'})")\n',
    "NB3.14 validation print")

set_source(cell, src)
print_diff("515f5bb0", orig, src, "(NB3 Cell 14 — Tuning)")
changes += 4

if DRY_RUN:
    print("\n[DRY RUN] NB3 not saved.")
else:
    save_nb(NB3, nb3)
    print(f"\nSaved NB3: {NB3}")


# ============================================================================
# NB4 RANGE
# ============================================================================
print("\n" + "#" * 60)
print("# NB4 RANGE — Cost alignment fixes")
print("#" * 60)

nb4 = load_nb(NB4)

# -- NB4 Cell 10: Engine (8f3f03ad) --
cell = find_cell(nb4, "8f3f03ad")
src = get_source(cell)
orig = src

# Fix 1: snapshot load — costs_by_symbol -> per_symbol
src = assert_replace(src,
    'costs_by_sym = cost_snap.get("costs_by_symbol", {})',
    'costs_by_sym = {e["symbol"]: e for e in cost_snap.get("per_symbol", [])}',
    "NB4.10 snapshot load")

# Fix 2: cost_base_dec field name + default 3->8
src = assert_replace(src,
    'cost_base_dec = float(cinfo.get("cost_base_bps", 3.0)) / 10_000',
    'cost_base_dec = float(cinfo.get("base_cost_bps", 8.0)) / 10_000',
    "NB4.10 cost_base_dec")

# Fix 3: cost_stress_dec field name + default 6->16
src = assert_replace(src,
    'cost_stress_dec = float(cinfo.get("cost_stress_bps", 6.0)) / 10_000',
    'cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000',
    "NB4.10 cost_stress_dec")

# Add validation print after cost_stress_dec line
src = assert_replace(src,
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n\n    for fid in fold_ids:',
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n'
    '    print(f"  [{sym}] cost_base={cost_base_dec*10_000:.1f}bps, cost_stress={cost_stress_dec*10_000:.1f}bps "\n'
    '          f"(from={\'snapshot\' if cinfo else \'default\'})")\n\n    for fid in fold_ids:',
    "NB4.10 validation print")

set_source(cell, src)
print_diff("8f3f03ad", orig, src, "(NB4 Cell 10 — Engine)")
changes += 4

# -- NB4 Cell 14: Tuning (b41156f4) --
cell = find_cell(nb4, "b41156f4")
src = get_source(cell)
orig = src

# Fix 1: snapshot load — costs_by_symbol -> per_symbol
src = assert_replace(src,
    'costs_by_sym_tuning = cost_snap_tuning.get("costs_by_symbol", {})',
    'costs_by_sym_tuning = {e["symbol"]: e for e in cost_snap_tuning.get("per_symbol", [])}',
    "NB4.14 snapshot load")

# Fix 2: cost_base_dec field name + default 3->8
src = assert_replace(src,
    'cost_base_dec = float(cinfo.get("cost_base_bps", 3.0)) / 10_000',
    'cost_base_dec = float(cinfo.get("base_cost_bps", 8.0)) / 10_000',
    "NB4.14 cost_base_dec")

# Fix 3: cost_stress_dec field name + default 6->16
src = assert_replace(src,
    'cost_stress_dec = float(cinfo.get("cost_stress_bps", 6.0)) / 10_000',
    'cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000',
    "NB4.14 cost_stress_dec")

# Add validation print after cost_stress_dec line
src = assert_replace(src,
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n\n    for fid in fold_ids_tuning:',
    '    cost_stress_dec = float(cinfo.get("stress_cost_bps", 16.0)) / 10_000\n'
    '    print(f"  [{sym}] cost_base={cost_base_dec*10_000:.1f}bps, cost_stress={cost_stress_dec*10_000:.1f}bps "\n'
    '          f"(from={\'snapshot\' if cinfo else \'default\'})")\n\n    for fid in fold_ids_tuning:',
    "NB4.14 validation print")

set_source(cell, src)
print_diff("b41156f4", orig, src, "(NB4 Cell 14 — Tuning)")
changes += 4

if DRY_RUN:
    print("\n[DRY RUN] NB4 not saved.")
else:
    save_nb(NB4, nb4)
    print(f"\nSaved NB4: {NB4}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("COST ALIGNMENT PATCH SUMMARY")
print("=" * 60)

print(f"""
Cambios totales: {changes} reemplazos en 4 celdas

NB3 Cell 10 (Engine):
  - costs_by_sym: "costs_by_symbol" -> "per_symbol" (dict comprehension)
  - cost_base_bps: "cost_base_bps"/3.0 -> "base_cost_bps"/8.0
  - cost_stress_bps: "cost_stress_bps"/6.0 -> "stress_cost_bps"/16.0
  + validation print

NB3 Cell 14 (Tuning):
  - costs_by_sym_tuning: "costs_by_symbol" -> "per_symbol" (dict comprehension)
  - cost_base_dec: "cost_base_bps"/3.0 -> "base_cost_bps"/8.0
  - cost_stress_dec: "cost_stress_bps"/6.0 -> "stress_cost_bps"/16.0
  + validation print

NB4 Cell 10 (Engine):
  - costs_by_sym: "costs_by_symbol" -> "per_symbol" (dict comprehension)
  - cost_base_dec: "cost_base_bps"/3.0 -> "base_cost_bps"/8.0
  - cost_stress_dec: "cost_stress_bps"/6.0 -> "stress_cost_bps"/16.0
  + validation print

NB4 Cell 14 (Tuning):
  - costs_by_sym_tuning: "costs_by_symbol" -> "per_symbol" (dict comprehension)
  - cost_base_dec: "cost_base_bps"/3.0 -> "base_cost_bps"/8.0
  - cost_stress_dec: "cost_stress_bps"/6.0 -> "stress_cost_bps"/16.0
  + validation print

Costos efectivos: 8/16 bps (fee-only) vs 3/6 bps (fallback anterior)
""")

if DRY_RUN:
    print("[DRY RUN] No files written. Use --apply to save changes.")
else:
    print("Files saved successfully.")
