# panoseti/analysis: Usage

> Parameter reference is generated from `nextflow_schema.json` (`nextflow run . --help`).

## Introduction

The ingest pipeline converts PANOSETI `.pffd` observation runs into calibrated Zarr v3
stores: `PFF → L0 (per-(dp,module)) → L1 (calibrated)`, with lineage manifests and
housekeeping stores. See [`storage_spec.md`](storage_spec.md) for the output format.

## Input

Two ways to specify input:

### 1. Samplesheet (recommended; scales to many runs)

A CSV with a header row and two columns — one row per observation run:

```csv title="samplesheet.csv"
run_id,obs_dir
obs_2024-07-25T04_34_46Z,/expanse/lustre/scratch/$USER/panoseti/inputs/obs_2024-07-25T04_34_46Z.pffd
obs_2024-07-26T05_10_00Z,/expanse/lustre/scratch/$USER/panoseti/inputs/obs_2024-07-26T05_10_00Z.pffd
```

| Column | Description |
|---|---|
| `run_id` | Unique run identifier (becomes `meta.id`/`meta.run_id`; scopes the per-run manifest). |
| `obs_dir` | Path to an existing `.pffd` observation directory. |

```bash
nextflow run . -profile laptop --input samplesheet.csv --outdir <OUTDIR>
```

### 2. Single-run convenience

Skip the samplesheet for a one-off run; the `.pffd` is auto-wrapped into a single run
(`run_id` derived from the directory name):

```bash
nextflow run . -profile laptop --input_obs_dir /path/to/obs.pffd --outdir <OUTDIR>
```

## Running the pipeline

```bash
# bundled truncated test data (CI smoke):
nextflow run . -profile test,laptop --outdir results_smoke

# choose steps (only `ingest` is implemented; reconstruct/ml are stubs):
nextflow run . -profile laptop --steps ingest --input_obs_dir … --outdir …
```

Use `-resume` to reuse cached tasks. Outputs are published level-major under `--outdir`
(`L0/`, `L1/`, each with stores + `manifest.json`).

## Profiles

| Profile | Executor | Container | Use |
|---|---|---|---|
| `laptop` | local | none (uses uv `.venv`) | development / smoke tests |
| `docker` | local | Docker | reproducible local runs |
| `hpc_slurm` | SLURM | Apptainer | SDSC Expanse / any SLURM cluster |

Combine with `test` for the bundled dataset, e.g. `-profile test,laptop`.

### HPC (SDSC Expanse)

```bash
nextflow run . -profile hpc_slurm \
  --input_obs_dir /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
  --outdir        /expanse/lustre/scratch/$USER/panoseti/results \
  --slurm_account <account> --slurm_queue shared -resume
```

Or use the launcher: `bash scripts/launchers/expanse.sh <obs.pffd> <results_dir>`.
Build the container per [`../containers/README.md`](../containers/README.md) and pin it via
`--container` / `conf/hpc_slurm.config`.

## Key parameters

| Param | Default | Description |
|---|---|---|
| `--steps` | `ingest` | Comma-separated steps (`ingest,reconstruct,ml`). |
| `--codec` / `--level` | `zstd` / `5` | Zarr compression. |
| `--ph_sigma` / `--ph_offset` / `--ph_stride` | `5.0` / `800` / `200` | Pulse-height calibration. |
| `--img_stride` / `--img_block` / `--img_adc_to_pe` | `200` / `8` / `1.5` | Movie-mode calibration. |

Full list: `nextflow run . --help`.
