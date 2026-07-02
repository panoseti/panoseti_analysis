# panoseti-analysis

PANOSETI data-reduction monorepo: PFF → Zarr ingest/calibration pipelines for HPC,
real-time streaming ML, and cloud/anomaly detection. Built on a strict three-layer
architecture with an **nf-core**-templated Nextflow pipeline.

> Orientation for contributors and Claude Code: see [`CLAUDE.md`](CLAUDE.md).
> Storage conventions: see [`docs/storage_spec.md`](docs/storage_spec.md).

## Where to find things

| Goal                               | Start here                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Run the ingest pipeline (PFF → L1) | [Quick start](#quick-start) below                                                                              |
| Understand the data format         | [`docs/storage_spec.md`](docs/storage_spec.md)                                                                 |
| Cloud detection inference demo     | [`ml/cloud-detection/notebooks/01_inference_demo.ipynb`](ml/cloud-detection/notebooks/01_inference_demo.ipynb) |
| Reproduce cloud detector training  | [`ml/cloud-detection/README.md`](ml/cloud-detection/README.md)                                                 |
| ML architecture & execution models | [`docs/ml_architecture.md`](docs/ml_architecture.md)                                                           |
| Real-time streaming pipeline (RAL) | [`docs/streaming_ral.md`](docs/streaming_ral.md)                                                               |
| Train models on RAL                | [`docs/training_ral.md`](docs/training_ral.md)                                                                 |
| VAE anomaly detection              | [`ml/anomaly-detection/README.md`](ml/anomaly-detection/README.md)                                             |
| Provenance & reproducibility       | [`docs/provenance.md`](docs/provenance.md)                                                                     |
| gRPC services (DAQ, ML inference)  | [`grpc/README.md`](grpc/README.md)                                                                             |
| W&B experiment tracking setup      | [`docs/wandb.md`](docs/wandb.md)                                                                               |
| All ML projects index              | [`ml/README.md`](ml/README.md)                                                                                 |

## Layout

```
src/panoseti_analysis/
  algorithms/   Layer A — pure kernels (xarray/numpy/pydantic; no I/O, no frameworks)
  adapters/     Layer B — thin Typer CLIs + Ray/Serve adapters
  io/           filesystem boundary (open/write/checksum/pack; wraps pypff)
  config/       versions, data_level registry, Pydantic models
bin/            pa-* shims (on $PATH inside Nextflow processes)
ml/             ML project directories (cloud-detection/, anomaly-detection/)
  cloud-detection/   notebooks, recipes, model card for the cloud detector
  anomaly-detection/ notebooks, recipes for β-VAE
main.nf, workflows/, subworkflows/local/, modules/local/, conf/   Layer C — Nextflow
pypff/          git submodule (PFF reader + PFF→Zarr v3 converter) — do not edit here
grpc/           git submodule (panoseti_grpc: DAQ/ML gRPC services)
tests/          Layer A unit tests + io/adapter tests + boundary lint + e2e data
docs/           storage_spec.md, ml_architecture.md, provenance.md, wandb.md
```

## Quick start

```bash
uv sync                                              # workspace venv (incl. pypff submodule)
uv run pytest                                        # 57 tests: kernels, io, adapters
uv run ruff check src tests && uv run mypy src/panoseti_analysis
```

## Demos (Pipeline Execution Paths)

The pipeline supports multiple execution paths tailored for different environments. Below are demonstrations for the five primary use cases:

### 1. In-memory pure Python (Algorithm dev loop)

For fast iteration without orchestration overhead. Uses pure `xarray` and avoids Nextflow/Ray entirely.

```bash
uv run python src/panoseti_analysis/adapters/recipe_driver.py tests/data/obs_TEST.pffd results_recipe_driver
```

_Produces L0 and L1 `.zarr` stores, manifests, and `.png` quicklooks sequentially in-memory._

### 2. Local laptop / RAL GPU server (Nextflow CPU fallback)

The standard smoke test running via Nextflow on a laptop (no containers) or a single node. Includes ML L2 classification using a bundled PyTorch model.

```bash
nextflow run . -profile test,laptop --outdir results_smoke
ls results_smoke/L0 results_smoke/L1 results_smoke/L2
```

_For RAL, you can also connect to the persistent Ray cluster natively for distributed tasks using the `--ray_launcher attach` parameter._

### 3. HPC / SLURM Cluster (Offline processing)

For large-scale offline processing (e.g., SDSC Expanse). Nextflow translates tasks into individual `sbatch` jobs and uses Apptainer containers.

```bash
nextflow run . -profile hpc_slurm --steps ingest,ml --input_obs_dir <OBS_DIR> --outdir <OUT_DIR>
```

_To avoid PyTorch cold starts during ML inference, enable Ray distribution over SLURM via `--use_ray true --ray_launcher slurm`, which dynamically boots a transient cluster using `srun ray symmetric-run`._

### 4. Real-time streaming (Ray Serve + gRPC)

Run the pipeline in real-time on live DAQ data without writing intermediate Zarr files to disk.

```bash
# Start the gRPC broker and Serve deployment (e.g., on digilab-transmit GPU node)
pa-stream-cloud --model-path assets/models/cloud_detector_v1.pt --grpc-host localhost
```

_Produces live per-minute cloud scores via a gRPC stream._

### 5. ML training & tuning loops (Ray Train)

Train models directly on BeeGFS data staged to local SSDs, logging seamlessly to Weights & Biases or TensorBoard.

```bash
# Train on the RAL persistent cluster
pa-train-cloud --launcher attach --recipe recipes/ml/cloud_v1.yml --out-dir models/
```

_On an HPC, you can run this via SLURM to utilize the transient Ray cluster approach._

## Pipeline

```
.pffd ──pa-convert──▶ L0/ (per-(dp,module) Zarr + lineage)
                          │  fan-out by (dp,module)
              ┌───────────┴───────────┐
        pa-calibrate ph         pa-calibrate img      pa-hk ──▶ hk.<hashset>.zarr
     (pedestal + n-σ)        (block+temporal median)
              └───────────┬───────────┘
                          ▼
                       L1/ (calibrated Zarr) + manifest.json (lineage)
```

L0 mirrors native PFF order; **L1 guarantees monotonic `unix_t_ns`** (stable-sorted,
frames never dropped). Profiles: `laptop` (no container), `docker`, `hpc_slurm`
(SLURM + Apptainer, e.g. SDSC Expanse). Select steps with `--steps ingest`
(`reconstruct`/`ml` are stubs). Input is a `run_id,obs_dir` samplesheet (`--input`) or
the single-run convenience `--input_obs_dir`.

## License

MIT © 2026 PANOSETI.
