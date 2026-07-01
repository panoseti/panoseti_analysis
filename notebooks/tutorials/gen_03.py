import json
from pathlib import Path

nb2_path = Path(
    "/Users/nico/panoseti/panoseti_analysis/notebooks/tutorials/02_interacting_with_zarr.ipynb"
)
nb3_path = Path(
    "/Users/nico/panoseti/panoseti_analysis/notebooks/tutorials/03_cloud_detector_tutorial.ipynb"
)
out_path = Path("/Users/nico/panoseti/panoseti_analysis/notebooks/tutorials/03_data_analysis.ipynb")

with open(nb2_path) as f:
    nb2 = json.load(f)

with open(nb3_path) as f:
    nb3 = json.load(f)

# Change the title in nb2
for cell in nb2["cells"]:
    if (
        cell["cell_type"] == "markdown"
        and cell["source"]
        and cell["source"][0].startswith("# Tutorial 2: Interacting")
    ):
        cell["source"][0] = "# Tutorial 3: Data Analysis & Products\\n"

nb3_cells = []
for cell in nb3["cells"]:
    if (
        cell["cell_type"] == "markdown"
        and cell["source"]
        and cell["source"][0].startswith("# Tutorial 3: Cloud")
    ):
        cell["source"][0] = "## Cloud Detector Step-by-Step\\n"
    nb3_cells.append(cell)

nb2["cells"].extend(nb3_cells)

with open(out_path, "w") as f:
    json.dump(nb2, f, indent=1)

print("Generated 03_data_analysis.ipynb")
