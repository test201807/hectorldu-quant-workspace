"""Execute NB3 and NB4 via nbconvert and report results."""
import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8')

os.chdir(r"C:\Quant\projects\MT5_Data_Extraction\03_STRATEGY_LAB\notebooks")

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebooks = [
    ("NB3", "03_TREND_M5_Strategy_v2.ipynb"),
    ("NB4", "04_RANGE_M5_Strategy_v1.ipynb"),
]

ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')

for label, nb_file in notebooks:
    print(f"\n{'='*60}")
    print(f"Executing {label}: {nb_file}")
    print(f"{'='*60}")
    try:
        with open(nb_file, encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {'metadata': {'path': os.getcwd()}})

        # Print cell outputs
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code' and cell.outputs:
                for out in cell.outputs:
                    if out.output_type == 'stream':
                        text = out.text.strip()
                        if text:
                            # Just print first and last lines of long outputs
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

        print(f"\n{label}: PASS ({len(nb.cells)} cells executed)")

    except Exception as e:
        print(f"\n{label}: FAIL")
        # Find the failing cell
        if hasattr(e, 'cell_index'):
            print(f"  Failed at cell {e.cell_index}")
        traceback.print_exc()
        print()
