"""Fix cost model en NB4 Cells 10 y 14: per_symbol→costs_by_symbol + field names."""
import json
from pathlib import Path

NB4 = Path(r"C:\Quant\projects\MT5_Data_Extraction\03_STRATEGY_LAB\notebooks\04_RANGE_M5_Strategy_v1.ipynb")

def assert_replace(src: str, old: str, new: str, label: str) -> str:
    assert old in src, f"ASSERT FAIL [{label}]: no encontré:\n  {old!r}"
    return src.replace(old, new)

nb = json.load(open(NB4, encoding="utf-8"))

checks = 0
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid not in ("8f3f03ad", "b41156f4"):
        continue

    src = "".join(cell.get("source", []))

    if cid == "8f3f03ad":
        # Bug 1: estructura del snapshot
        src = assert_replace(
            src,
            '{e["symbol"]: e for e in cost_snap.get("per_symbol", [])}',
            'cost_snap.get("costs_by_symbol", {})',
            "Cell10-struct"
        )
        # Bug 2: field name base
        src = assert_replace(
            src,
            'cinfo.get("base_cost_bps", 8.0)',
            'cinfo.get("cost_base_bps", 8.0)',
            "Cell10-base"
        )
        # Bug 3: field name stress
        src = assert_replace(
            src,
            'cinfo.get("stress_cost_bps", 16.0)',
            'cinfo.get("cost_stress_bps", 16.0)',
            "Cell10-stress"
        )
        checks += 3
        print("Cell 10 (8f3f03ad): 3 fixes OK")

    elif cid == "b41156f4":
        # Bug 1: estructura del snapshot
        src = assert_replace(
            src,
            '{e["symbol"]: e for e in cost_snap_tuning.get("per_symbol", [])}',
            'cost_snap_tuning.get("costs_by_symbol", {})',
            "Cell14-struct"
        )
        # Bug 2: field name base
        src = assert_replace(
            src,
            'cinfo.get("base_cost_bps", 8.0)',
            'cinfo.get("cost_base_bps", 8.0)',
            "Cell14-base"
        )
        # Bug 3: field name stress
        src = assert_replace(
            src,
            'cinfo.get("stress_cost_bps", 16.0)',
            'cinfo.get("cost_stress_bps", 16.0)',
            "Cell14-stress"
        )
        checks += 3
        print("Cell 14 (b41156f4): 3 fixes OK")

    cell["source"] = [src]

assert checks == 6, f"ASSERT FAIL: esperaba 6 checks, hice {checks}"

json.dump(nb, open(NB4, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nTotal: {checks}/6 checks OK — NB4 guardado.")
