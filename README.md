# panoseti-analysis

PANOSETI data-reduction monorepo: PFF → Zarr ingest/calibration pipelines for HPC,
real-time streaming ML, and cloud/anomaly detection. Built on a strict three-layer
architecture with an **nf-core**-templated Nextflow pipeline.

> Orientation for contributors and Claude Code: see [`CLAUDE.md`](CLAUDE.md).
> Storage conventions: see [`docs/storage_spec.md`](docs/storage_spec.md).

## Where to find things

| Goal | Start here |
|------|------------|
| Run the ingest pipeline (PFF → L1) | [Quick start](#quick-start) below |
| Understand the data format | [`docs/storage_spec.md`](docs/storage_spec.md) |
| Cloud detection inference demo | [`ml/cloud-detection/notebooks/01_inference_demo.ipynb`](ml/cloud-detection/notebooks/01_inference_demo.ipynb) |
| Reproduce cloud detector training | [`ml/cloud-detection/README.md`](ml/cloud-detection/README.md) |
| ML architecture & execution models | [`docs/ml_architecture.md`](docs/ml_architecture.md) |
| Real-time streaming pipeline | [`CLAUDE.md` § Real-time streaming](CLAUDE.md) |
| VAE anomaly detection | [`ml/anomaly-detection/README.md`](ml/anomaly-detection/README.md) |
| Provenance & reproducibility | [`docs/provenance.md`](docs/provenance.md) |
| gRPC services (DAQ, ML inference) | [`grpc/README.md`](grpc/README.md) |
| W&B experiment tracking setup | [`docs/wandb.md`](docs/wandb.md) |
| All ML projects index | [`ml/README.md`](ml/README.md) |

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

# Ingest pipeline on bundled test data (PFF -> L0 -> L1, level-major output):
nextflow run . -profile test,laptop --outdir results_smoke
ls results_smoke/L0 results_smoke/L1                 # per-(dp,module) .zarr + manifest.json
```

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
