# CLAUDE.md

Guidance for Claude Code (and humans) working in `panoseti_analysis`.

## What this repo is

The PANOSETI data-reduction monorepo. PANOSETI is an optical/near-IR observatory
searching for SETI laser pulses and ultra-high-energy gamma rays. It produces two
data-product families across many ns-synchronized telescope **modules**:

- **Pulse-height** events (`dp_ph256` 16×16, `dp_ph1024` 32×32) — `int16` ADC intensities.
- **Movie-mode** imaging (`dp_img8`, `dp_img16` 32×32) — `uint8`/`uint16` above-threshold counts.

**Raw PFF is the immutable archive; Zarr is a re-derivable view** computed on SDSC Expanse.

## Submodule boundary — read, never edit here

- `pypff/` is a **git submodule** (separate repo). It owns the mmap PFF reader, the
  PFF→Zarr v3 converter, the L0 Zarr layout (`pypff/docs/zarr_v3_spec.md`), and Pydantic
  models. **Do not edit `pypff/` from this repo.** Changes go to the pypff repo via its
  own PR; collect proposals under "Proposed pypff changes" in the plan/PR description.
- The superseded `zarr-seqera-v0/` prototype has been removed (its calibration math was
  ported into the Layer A kernels; `img16` L1 output is bit-identical, verified at retirement).

## Three-layer architecture (strict)

| Layer                 | Location                                                         | Rule                                                                                                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A — pure kernels**  | `src/panoseti_analysis/algorithms/`                              | `xarray`/`numpy`/`pydantic` in & out. **No** I/O, **no** framework imports (`ray\|nextflow\|grpc\|slurm\|typer\|zarr\|pypff`), **no** `.open_zarr`/`.to_zarr`. CI lint (`tests/algorithms/test_layer_boundary.py`) enforces this. |
| **B — adapters**      | `src/panoseti_analysis/adapters/` + `bin/pa-*`                   | Thin Typer CLIs: read paths → open via `io/` → call **one** kernel → write via `io/` → emit `*.lineage.json`. The **only** code Nextflow invokes.                                                                                 |
| **C — orchestration** | `main.nf`, `workflows/`, `subworkflows/local/`, `modules/local/` | Nextflow. Calls only Layer B CLIs (on `$PATH`), never imports kernels.                                                                                                                                                            |

Supporting: `io/` (filesystem boundary: open/write/checksum/pack/pff/quicklook),
`config/` (versions, `data_level` registry, Pydantic models). The `ray` adapter is a
sibling of Layer B — same kernel call, different transport — so Layer A never changes.

### Design principles

These apply across all layers and cut through YAGNI, readability, and correctness:

**Fail loudly.** Boundaries validate immediately with precise error messages. Use
`ds.pano.validate(level=..., kind=...)` at every adapter entry point — never let a
missing variable surface as an obscure downstream crash.

**Single chokepoints.** One place owns each cross-cutting concern:

- `ds.pano.stamp()` is the sole writer of `data_level`, `storage_version`, `calibration`,
  `timestamp_qc`, and `qc` attrs. Never write these keys directly.
- `adapters/_common.build_provenance_step()` is the sole constructor of `ProcessingStep`
  chains. Do not repeat the `read_history → now_utc → capture_software` pattern.
- `CalibrationResolver.resolve()` is the sole source of calibration params. Do not
  construct `ImgCalibParams` / `PhCalibParams` with hard-coded values outside of this
  seam and the default fallbacks in `io/calibration_source.py`.

**QC metrics alongside products.** Every L1 store carries a `qc` attrs block (stamped
by `io/qc.stamp_qc`) and an optional `.qc.json` sidecar written by the calibrate adapter.
New calibration kernels must call `run_qc` before writing. QC checks belong in
`algorithms/qc.py` (Layer A); writers belong in `io/qc.py` (Layer B).

**Anti-framework guardrail for the dev driver.** `adapters/recipe_driver.py` is a flat
~130-line function (`run_pipeline`) that calls `run_convert → run_hk → run_calibrate →
run_classify → run_manifest` in order. It is a readable statement of step sequence —
**not** a second DAG engine. No scheduling, no resume, no parallelism. Nextflow
remains the only Layer C and the production orchestrator.

**Simplest possible implementation.** Three similar lines beat a premature abstraction.
Do not add features, fallbacks, or design patterns for hypothetical future requirements.
Explicit over clever.

### Ray Integration Principle

**Ray is a payload, not a substrate.** The default execution model is Nextflow-process-with-typer-CLI.
Ray is opt-in per process via a `gpu_ray` label. Processes that don't need distributed memory/GPUs stay non-Ray.

**Three launcher modes** (`adapters/ray/launcher.py::init_ray(launcher)`):

| Mode         | Cluster ownership                                 | When                                                |
| ------------ | ------------------------------------------------- | --------------------------------------------------- |
| `attach`     | **Persistent, user-owned** (user ran `ray start`) | RAL (default), dev                                  |
| `slurm`      | Transient — == SLURM allocation                   | Expanse; cluster brought up via `ray symmetric-run` |
| `standalone` | Process-local, real multi-process                 | CI / laptop                                         |

**The transient-cluster invariant is SLURM-only.** On RAL the cluster outlives any job; on `standalone` it is process-local. `attach` and `slurm` both call `ray.init(address="auto")` — the difference is _who owns the cluster_.

## Toolchain

- **uv workspace**; `pypff` is a workspace member (`[tool.uv.workspace] members=["pypff"]`).
- **Python ≥3.11**, **ruff** + **mypy --strict** (mypy skip-follows `zarr`/`pypff`).
- **Nextflow ≥26.04**, strict DSL2 (`nextflow.enable.strict = true`), nf-core 4.0.2 template
  (branding off), publishing via the **`output {}` block** (no `publishDir`).
- Build **Docker** images, run as **Apptainer** on Expanse.
  `containers/ingest.Dockerfile` is for CPU tasks; `containers/ml.Dockerfile` brings Ray + PyTorch + CUDA for ML/Ray workloads.

## How to run

_Quick start: see `README.md`. Full dev setup: see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)._

## Real-time streaming

See [`docs/streaming_ral.md`](docs/streaming_ral.md) — setup guide for the RAL attach-mode streaming pipeline.

## Training on RAL

See [`docs/training_ral.md`](docs/training_ral.md) — cluster topology, startup commands, and full training workflow.

## Recipes vs Profiles

- `recipes/ml/*.yml` — ML training recipes (WHAT SCIENCE: feature params, split ratios, hyperparams, scaling). Passed via `--recipe` to training CLIs (`pa-train-cloud`, `pa-tune-cloud`, etc.). Dev variants (notebook-friendly hyperparams): `*_dev.yml`. Template for new models: `my_model_v1_template.yml`.
- `recipes/*.yml` — non-ML pipeline recipes (calibration defaults, ingest subsets). Passed via `-params-file` to Nextflow.
- `-profile` — WHERE/HOW (executor, container, resource limits). Never embed science params here.

The `recipe_hash` (sha256 of the YAML bytes) is stamped into every `ProcessingStep` and `TrainingProvenance` record, so every produced artifact is traceable to an exact science configuration.

## Storage conventions (see `docs/storage_spec.md`)

- **Level-major** output: `L0/`, `L1/`, each with per-`(dp,module)` `.zarr` stores +
  `manifest.json` + (propagated) `.panoseti-meta/`.
- **One store per `(data_product, module)`** — never consolidate (time axes differ).
  Cross-module science is a **temporal join** (the coincidence-finder kernel), not physical
  consolidation.
- **L0** mirrors native PFF frame order and records `timestamp_qc`. **L1+ guarantees
  monotonic-non-decreasing `unix_t_ns`** (the coincidence-finder depends on it); L0→L1
  stable-sorts to repair, never drops frames (`pkt_num` rides along for reversibility).
- **HK** telemetry → per-run, per-hashset `hk.<hashset>.zarr` (head-node redis snapshots).
- Two version keys: `panoseti_pff_zarr_version` (pypff, L0 layout) and
  `panoseti_analysis_storage_version` (level structure, manifest, HK, QC, PACK).
- **PACK** (transfer): `tar` for the compute hop, Zarr **ZipStore** (STORED+ZIP64) for the
  archive hop. Never Globus raw chunk directories.

## Where new code goes

```
src/panoseti_analysis/algorithms/  ← pure kernels (TDD; add a test first)
src/panoseti_analysis/adapters/    ← new CLI? add bin/pa-<name> shim + [project.scripts]
src/panoseti_analysis/io/          ← new filesystem op
src/panoseti_analysis/config/      ← new registry/version/model
modules/local/ + subworkflows/local/  ← new Nextflow process / subworkflow
docs/storage_spec.md               ← any storage-format change (bump the version key)
```
