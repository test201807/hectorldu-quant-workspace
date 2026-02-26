"""Extract specific cell source from notebooks."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r"C:\Quant\projects\MT5_Data_Extraction")

nb_path = sys.argv[1]
cell_indices = [int(x) for x in sys.argv[2].split(",")]

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

for idx in cell_indices:
    cell = nb['cells'][idx]
    src = ''.join(cell['source'])
    print(f"\n{'='*80}")
    print(f"CELL {idx} ({cell['cell_type']})")
    print(f"{'='*80}")
    print(src)
