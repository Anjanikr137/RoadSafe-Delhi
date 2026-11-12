"""
src/test_execute_notebooks.py
Executes all code cells from the 11 pipeline notebooks to verify end-to-end execution.
"""

import os
import json

nb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../notebooks"))
nbs = sorted([f for f in os.listdir(nb_dir) if f.endswith(".ipynb")])

print(f"Testing execution of {len(nbs)} pipeline notebooks...")
for nb_name in nbs:
    fp = os.path.join(nb_dir, nb_name)
    with open(fp, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"\n---> Executing: {nb_name}")
    # Context dictionary for execution
    ctx = {
        "__file__": fp,
        "__name__": "__main__"
    }

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(code_cells):
        code_text = "".join(cell["source"])
        # Skip IPython display commands during headless execution
        lines = []
        for line in code_text.split("\n"):
            if "display(" in line or "plt.show()" in line:
                continue
            lines.append(line)
        exec_code = "\n".join(lines)
        try:
            exec(exec_code, ctx)
        except Exception as e:
            print(f"  [ERROR] Cell {i+1} failed in {nb_name}: {e}")
            raise e
    print(f"  [SUCCESS] {nb_name} executed all {len(code_cells)} code cells successfully.")

print("\nALL 11 PIPELINE NOTEBOOKS EXECUTED CLEANLY!")

