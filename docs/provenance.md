# Provenance Specification

Scientific provenance in `panoseti_analysis` uses **two complementary layers** that serve different
audiences. Do not conflate them.

## Layer A — Nextflow 26.04 Native Lineage (execution record)

Enabled via `nextflow.config`:

```nextflow
lineage {
    enabled = true
}
```

Content-addressed `lid://` URIs identify tasks, inputs, outputs, params, container, and resume
state. This is the **execution-side** record: convenient for debugging runs and resuming partial
pipelines. It is a young, still-stabilizing feature — treat it as a complement, not the
durable archive. When a producing task's `lid://` is available, adapters capture it in
`ProcessingStep.nextflow_lineage_id`.

## Layer B — `processing_history` in Zarr Store Attrs (portable scientific record)

An ordered list of `ProcessingStep` records written into every Zarr store's **root attrs**
under the key `processing_history`. These records travel with the store wherever it goes
(BeeGFS, Cylon, email attachment) and are readable with `xr.open_zarr`:

```python
import xarray as xr
ds = xr.open_zarr("obs_run.dp_img16.module_1.L2.zarr")
for step in ds.attrs["processing_history"]:
    print(step["step_name"], step["timestamp_utc"])
```

### `ProcessingStep` schema (IVOA Provenance DM-aligned)

```python
class ProcessingStep:
    step_name: str             # IVOA Activity.name ("convert","calibrate_ph","classify_cloud",…)
    step_version: str          # software/kernel version
    params: dict[str, Any]     # Activity Parameters (kernel call params)
    recipe_name: str | None    # recipe file name (e.g. "cloud_v1")
    recipe_hash: str | None    # "sha256:<hex>" of the recipe YAML bytes
    input_checksums: list[str] # "sha256:<hex>" of upstream store(s) used
    output_checksum: str|None  # "sha256:<hex>" of this store (circular at write time; set post-write)
    timestamp_utc: str         # ISO 8601, set by the adapter at runtime
    software: dict[str,str]    # {"git_sha": …, "package_version": …, "container": …}
    nextflow_lineage_id: str|None  # lid:// from NF 26.04 lineage; None outside NF
```

IVOA alignment: `step_name` → `Activity.name`, `input_checksums` → `used` (Entity refs),
`output_checksum` → `wasGeneratedBy`, `software` → `wasAssociatedWith` Agent,
`timestamp_utc` → `Activity.startTime`. A future VO export is a rename, not a redesign.

### Chain example: L0 → L1 → L2

```json
// L2 store root attrs
"processing_history": [
  { "step_name": "convert",         "input_checksums": [],              "output_checksum": "sha256:aaa…", "params": {"pff_version": "1.2"} },
  { "step_name": "calibrate_img16", "input_checksums": ["sha256:aaa…"], "output_checksum": null,         "params": {"kind":"img16","sigma_threshold":5.0} },
  { "step_name": "classify_cloud",  "input_checksums": ["sha256:bbb…"], "output_checksum": null,         "params": {"cadence_s":60,"threshold":0.5,"model_checksum":"sha256:ccc…"} }
]
```

Each level reads the upstream store's history from attrs and appends its own step
(`input_checksum = upstream output_checksum`), so the L2 store carries the full 3-step chain.

### Cross-boundary threading (ML, outside Nextflow)

```
L1 store attrs: [convert, calibrate]                       (Nextflow, BeeGFS)
       │ pa-features-cloud reads L1 history + checksum
       ▼
feature cache attrs: [convert, calibrate, cloud_features]  (BeeGFS, references L1 checksum)
       │ pa-train-cloud reads feature-cache checksum
       ▼
model bundle _provenance.json: TrainingProvenance          (step_name="train_cloud",
       input_checksums=[feature-cache, L1], recipe, hyperparams, metrics, git_sha, wandb_run_id)
       │ pa-classify-cloud / pa-ray-classify-cloud records model checksum
       ▼
L2 store attrs: [convert, calibrate, classify_cloud(model=sha256:ccc…)]
```

This closes the loop across the Nextflow/Ray boundary: the L2 store names the model;
the model names its training inputs; the inputs carry their calibration history.

### `TrainingProvenance` schema

Written alongside each trained model bundle (sidecar `_provenance.json`):

```python
class TrainingProvenance:
    step_name: str             # "train_cloud" or "train_vae"
    step_version: str
    recipe_name: str | None
    recipe_hash: str | None
    params: dict[str, Any]     # all hyperparams
    input_checksums: list[str] # feature-cache + L1 checksums
    output_checksum: str|None  # sha256 of the .pt weights file
    timestamp_utc: str
    software: dict[str,str]    # git_sha, ray_version, torch_version, …
    metrics: dict[str, float]  # {"val_accuracy": 0.96, …}
    wandb_run_id: str|None
```

## Provenance schema version

`PROVENANCE_SCHEMA_VERSION = "1.0"` (`config/versions.py`).
Breaking changes to `ProcessingStep` or `TrainingProvenance` bump this constant.

## Recipes and recipe_hash

Every `ProcessingStep` carries `recipe_name` and `recipe_hash` (content-hash of the YAML
bytes). This lets you reconstruct exactly which science parameters were used even if the
recipe file is later edited.

```python
from panoseti_analysis.config.recipes import load_recipe
params, name, recipe_hash = load_recipe("recipes/cloud_v1.yml")
# recipe_hash == "sha256:<hex>"
```

## Reading provenance programmatically

```python
from panoseti_analysis.io.provenance import read_history
import zarr

root = zarr.open_group("obs.dp_img16.module_1.L2.zarr", mode="r", zarr_format=3)
history = read_history(dict(root.attrs))
for step in history:
    print(f"{step.step_name} @ {step.timestamp_utc}")
    if step.recipe_hash:
        print(f"  recipe: {step.recipe_name} ({step.recipe_hash})")
    if step.input_checksums:
        print(f"  consumed: {step.input_checksums}")
```

## Storage version

`PANOSETI_ANALYSIS_STORAGE_VERSION = "2.0"` and `MANIFEST_SCHEMA_VERSION = "2.0"`
(`config/versions.py`). Bumped from 1.0 when `processing_history` was added to `StoreLineage`
(breaking schema change — older JSON without the field still loads, defaulting to `[]`).
