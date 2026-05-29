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

| Layer | Location | Rule |
|---|---|---|
| **A — pure kernels** | `src/panoseti_analysis/algorithms/` | `xarray`/`numpy`/`pydantic` in & out. **No** I/O, **no** framework imports (`ray\|nextflow\|grpc\|slurm\|typer\|zarr\|pypff`), **no** `.open_zarr`/`.to_zarr`. CI lint (`tests/algorithms/test_layer_boundary.py`) enforces this. |
| **B — adapters** | `src/panoseti_analysis/adapters/` + `bin/pa-*` | Thin Typer CLIs: read paths → open via `io/` → call **one** kernel → write via `io/` → emit `*.lineage.json`. The **only** code Nextflow invokes. |
| **C — orchestration** | `main.nf`, `workflows/`, `subworkflows/local/`, `modules/local/` | Nextflow. Calls only Layer B CLIs (on `$PATH`), never imports kernels. |

Supporting: `io/` (filesystem boundary: open/write/checksum/pack/pff/quicklook),
`config/` (versions, `data_level` registry, Pydantic models). A Ray or gRPC adapter is a
future *sibling* of Layer B — same kernel call, different transport — so Layer A never changes.

## Toolchain

- **uv workspace**; `pypff` is a workspace member (`[tool.uv.workspace] members=["pypff"]`).
- **Python ≥3.11**, **ruff** + **mypy --strict** (mypy skip-follows `zarr`/`pypff`).
- **Nextflow ≥26.04**, strict DSL2 (`nextflow.enable.strict = true`), nf-core 4.0.2 template
  (branding off), publishing via the **`output {}` block** (no `publishDir`).
- Build **Docker** images, run as **Apptainer** on Expanse.

## How to run

```bash
uv sync                                   # create .venv, install workspace
uv run pytest                             # Python unit + integration tests
uv run ruff check src tests && uv run mypy src/panoseti_analysis

# Ingest pipeline (laptop, bundled test data):
nextflow run . -profile test,laptop --outdir results_smoke
# Real run (samplesheet of run_id,obs_dir) or single-run convenience:
nextflow run . -profile laptop --input samplesheet.csv --outdir <OUT>
nextflow run . -profile laptop --input_obs_dir /path/obs.pffd --outdir <OUT>
# Choose steps (ingest implemented; reconstruct/ml are stubs):
nextflow run . -profile laptop --steps ingest --input_obs_dir … --outdir …
# HPC: -profile hpc_slurm  (SLURM + Apptainer; --slurm_account/--slurm_queue)
```

CLIs (also on `$PATH` inside Nextflow): `pa-convert`, `pa-calibrate`, `pa-hk`,
`pa-manifest`, `pa-pack`.

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
