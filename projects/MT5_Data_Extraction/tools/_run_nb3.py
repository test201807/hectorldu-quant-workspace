"""Execute NB3 only via nbconvert with extended timeout."""
import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8')

os.chdir(r"C:\Quant\projects\MT5_Data_Extraction\03_STRATEGY_LAB\notebooks")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_file = "03_TREND_M5_Strategy_v2.ipynb"
print(f"Executing NB3: {nb_file}")
print("="*60)

ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')

try:
    with open(nb_file, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and cell.outputs:
            for out in cell.outputs:
                if out.output_type == 'stream':
                    text = out.text.strip()
                    if text:
                        lines = text.split('\n')
                        if len(lines) > 5:
                            print(f"  Cell [{i:2d}] OK: {lines[0]}")
                            print(f"           ... ({len(lines)} lines)")
                            print(f"           {lines[-1]}")
                        else:
                            for line in lines:
                                print(f"  Cell [{i:2d}] {line}")
                elif out.output_type == 'error':
                    print(f"  Cell [{i:2d}] ERROR: {out.ename}: {out.evalue}")

    print(f"\nNB3: PASS ({len(nb.cells)} cells executed)")

except Exception as e:
    print(f"\nNB3: FAIL")
    if hasattr(e, 'cell_index'):
        print(f"  Failed at cell {e.cell_index}")
    traceback.print_exc()
