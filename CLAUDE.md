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

```bash
uv sync                                   # create .venv, install workspace
uv run pytest                             # Python unit + integration tests
uv run ruff check src tests && uv run mypy src/panoseti_analysis

# Ingest pipeline (laptop, bundled test data):
nextflow run . -profile test,laptop --outdir results_smoke
# Real run (samplesheet of run_id,obs_dir) or single-run convenience:
nextflow run . -profile laptop --input samplesheet.csv --outdir <OUT>
nextflow run . -profile laptop --input_obs_dir /path/obs.pffd --outdir <OUT>
# Choose steps (ingest implemented; reconstruct is stub; ml runs cloud classification):
nextflow run . -profile laptop --steps ingest,ml --input_obs_dir … --outdir …
# HPC: -profile hpc_slurm  (SLURM + Apptainer; --slurm_account/--slurm_queue)
# To use Ray for ML workloads: add --use_ray true
```

CLIs (also on `$PATH` inside Nextflow): `pa-convert`, `pa-calibrate`, `pa-hk`,
`pa-manifest`, `pa-pack`.

## Real-time streaming (RAL only — attach mode)

The streaming pipeline consumes the live DaqData.StreamImages gRPC feed and emits
cloud-detection scores in real time.  It runs **alongside** the batch pipeline on the
same Ray cluster (attach mode, externally owned).  Ray Serve is a **persistent substrate**
here (the scoped exception to the transient-cluster rule — the cluster is owned externally).

```bash
# 1. Start the panoseti_grpc unified server with DaqData + MLInference enabled:
#    Edit grpc/src/panoseti_grpc/config/server.toml:  ml_inference = true
pseti-grpc server                                   # or: pseti-grpc server --config custom.toml

# 2. (Optional) For replay from archived PFF, configure simulate_daq_cfg in server.toml
#    and point movie_pff_path to a file under /mnt/beegfs/data/L0/

# 3. Run the streaming pipeline (gaming GPU, 60s cadence):
pa-stream-cloud \
    --model-path assets/models/cloud_detector_v1.pt \
    --recipe recipes/stream_cloud_v1.yml \
    --grpc-host localhost \
    --archive-dir /mnt/beegfs/streams/

# 4. Subscribe to live predictions (from another terminal):
python -c "
from panoseti_grpc.ml_inference.client import MLInferenceClient
with MLInferenceClient() as c:
    for p in c.stream_predictions():
        print(p.module_id, p.cloud_score, p.cloud_label)
"
```

**Ray Serve persistence note:** `pa-stream-cloud` deploys `CloudInferDeployment` onto the
externally-owned cluster and **shuts it down** on exit (Ctrl-C).  It does NOT shut down the
Ray cluster itself — the cluster remains for training jobs.  GPU placement: serving replica
runs on `digilab-transmit` (`accelerator_type:G`); A6000s on `digilab-receiver` are reserved
for training.

## Training on RAL

RAL is bare-metal, no SLURM. 5 nodes:
- **`digilab-receiver`** (head, 2× RTX A6000 48 GB + 1 TB SSD NVMe, `accelerator_type:RTX` — training; runs dashboard docker-compose)
- **`digilab-transmit`** (gaming GPU, `accelerator_type:G` — Ray Serve inference)
- **`panoseti-dfs0`**, **`panoseti-dfs1`**, **`panoseti-dfs2`** (BeeGFS storage nodes, CPU-only)

BeeGFS at `/mnt/beegfs`.

**Starting the cluster** (replaces manual `ray start` in tmux):
```bash
ray up conf/ray/ral_cluster.yaml           # start / reconnect all 5 nodes
ray up conf/ray/ral_cluster.yaml --no-restart  # attach without restarting Ray
ray status                                 # verify workers connected
cd ~/ray-test && docker compose up -d      # start prometheus/grafana (if not running)
# Dashboard: http://digilab-receiver:8265   Grafana: http://digilab-receiver:3000
```

The pipeline **attaches** to this running cluster — it never provisions RAL.

```bash
# 1. Ingest the label-covered subset to L1 (Nextflow handles PFF → L0 → L1):
nextflow run . -profile ral --steps ingest -params-file recipes/ingest_subset.yml --outdir /mnt/beegfs/runs/

# 2. Materialize features (runs standalone, reads L1 from BeeGFS):
pa-features-cloud --stores /mnt/beegfs/runs/*/L1/*.zarr \
  --out /mnt/beegfs/features/ --recipe recipes/cloud_v1.yml \
  --label-csv /mnt/beegfs/labels/cloud_labels.csv

# 3. Train cloud detector (attaches to user's running Ray cluster):
pa-train-cloud --launcher attach --recipe recipes/cloud_v1.yml \
  --feature-cache /mnt/beegfs/features/features.<hash>.zarr \
  --out /mnt/beegfs/models/ \
  --local-cache-dir /local/scratch

# 4. Train BetaVAE:
pa-prep-ph --stores /mnt/beegfs/runs/*/L1/*.dp_ph256.*.zarr \
  --out /mnt/beegfs/features/ --recipe recipes/vae_train_v1.yml
pa-train-vae --launcher attach --recipe recipes/vae_train_v1.yml \
  --feature-cache /mnt/beegfs/features/features.<hash>.zarr \
  --out /mnt/beegfs/models/ --local-cache-dir /local/scratch

# 5. Run inference with the new model bundle:
nextflow run . -profile laptop --steps ml \
  --cloud_model_pt /mnt/beegfs/models/cloud_detector_v2.pt \
  --cloud_model_json /mnt/beegfs/models/cloud_detector_v2.CloudDetection.json \
  --input_obs_dir /path/obs.pffd --outdir results/
```

**Recipes vs Profiles:**

- `recipes/*.yml` — WHAT SCIENCE (feature params, split ratios, hyperparams, scaling, model_out, label_csv). Passed via `--recipe` to training CLIs or `-params-file` to Nextflow.
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
