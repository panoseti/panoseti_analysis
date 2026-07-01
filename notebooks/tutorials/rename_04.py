import json
import os
from pathlib import Path

old_path = Path(
    "/Users/nico/panoseti/panoseti_analysis/notebooks/tutorials/05_python_dev_loop.ipynb"
)
new_path = Path(
    "/Users/nico/panoseti/panoseti_analysis/notebooks/tutorials/04_python_dev_loop.ipynb"
)

if old_path.exists():
    with open(old_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if (
            cell["cell_type"] == "markdown"
            and cell["source"]
            and cell["source"][0].startswith("# Tutorial 5: Pure-Python Dev Loop")
        ):
            cell["source"][0] = "# Tutorial 4: Pure-Python Dev Loop\\n"

    with open(new_path, "w") as f:
        json.dump(nb, f, indent=1)

    os.remove(old_path)
    print("Renamed 05 to 04")
