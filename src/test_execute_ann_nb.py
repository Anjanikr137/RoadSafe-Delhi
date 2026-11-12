"""
src/test_execute_ann_nb.py
Executes the code cells of models/ann/notebooks/ANN_Model_Development.ipynb.
"""

import os
import json

nb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/ann/notebooks/ANN_Model_Development.ipynb"))
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Testing execution of: {nb_path}")
ctx = {
    "__file__": nb_path,
    "__name__": "__main__"
}

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, cell in enumerate(code_cells):
    code_text = "".join(cell["source"])
    lines = []
    for line in code_text.split("\n"):
        if "display(" in line or "plt.show()" in line:
            continue
        lines.append(line)
    exec_code = "\n".join(lines)
    try:
        exec(exec_code, ctx)
        print(f"  Cell {i+1} OK")
    except Exception as e:
        print(f"  [ERROR] Cell {i+1} failed: {e}")
        raise e

print("\nANN MODEL DEVELOPMENT NOTEBOOK EXECUTED CLEANLY!")

