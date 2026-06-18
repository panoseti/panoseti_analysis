# Training Models on RAL

RAL (Remote Astrophysics Lab) is a 5-node bare-metal cluster — no SLURM. Models are
trained with Ray Train in `attach` mode against a user-managed persistent Ray cluster.

See [ML Architecture](ml_architecture.md) for training system design, data flow,
checkpointing, and SSD-staging details.

## Cluster topology

| Node                | Specs                                                         | Role                                         |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------- |
| `digilab-receiver`  | 1× RTX A6000 48 GB + 1 TB NVMe SSD (`accelerator_type:A6000`) | Head node; training; Ray + Grafana dashboard |
| `digilab-transmit`  | 1× RTX A6000 48 GB (`accelerator_type:A6000`)                 | Ray Serve inference; 2-node DDP partner      |
| `panoseti-dfs0/1/2` | CPU-only                                                      | BeeGFS storage nodes                         |

Both GPU nodes carry the same `accelerator_type:A6000` label, set explicitly by
`cluster/ral_up.sh` — not Ray's auto-detection. The two nodes are connected by a 400G
RDMA fabric (Mellanox ConnectX-7, `mlx5_0`, RoCE v2), enabling multi-node DDP.

**GPU placement strategy:**

- Cloud-detector training → `num_workers: 1, accelerator_type: "A6000"` (single node; comms overhead dominates for small models)
- Large-model training (VAE, DINO) → `num_workers: 2, accelerator_type: "A6000"` with `allow_multinode: true` (2-node DDP over 400G RDMA)
- Ray Serve inference → `accelerator_type: "A6000"` (either node; use `--gpu-node-ip 10.0.1.34` to pin to digilab-transmit and keep digilab-receiver free for training)

BeeGFS at `/mnt/beegfs`. ML artifacts: `/mnt/beegfs/models/`, `/mnt/beegfs/features/`,
`/mnt/beegfs/runs/`.

## Starting the cluster

```bash
bash cluster/ral_up.sh             # start all 5 nodes with NCCL/RDMA env + GPU labels
bash cluster/ral_up.sh --no-sync   # restart Ray only (skip source rsync)
bash cluster/ral_down.sh           # ray stop on every node
ray status                         # verify all 5 nodes connected
cd cluster/monitoring && docker compose up -d   # prometheus/grafana (if not running)
# Dashboard: http://digilab-receiver:8265   Grafana: http://digilab-receiver:3000
```

Edit node IPs / GPU counts / env names / labels in [`cluster/ral_nodes.conf`](../cluster/ral_nodes.conf)
— never in the scripts. `ral_up.sh` also runs a payload guard that aborts if a large file
(> 50 MB) would be rsynced to the workers (override with `--allow-large`).

**Why a script instead of `ray up`:** Ray's local provider gives the head and all workers one
shared node type and caps its worker quota at `len(worker_ips)`, so the head consumes a slot and
only `len(worker_ips) − 1` workers ever launch — `digilab-transmit` is the perennial casualty.
This fixed-size cluster needs no autoscaler, so every node is started explicitly with deterministic
GPU labels.

The pipeline **attaches** to this running cluster — it never provisions RAL.

## Full training workflow

```bash
# 1. Ingest the label-covered subset to L1 (Nextflow handles PFF → L0 → L1):
nextflow run . -profile ral --steps ingest -params-file recipes/ingest_subset.yml --outdir /mnt/beegfs/runs/

# 2. Materialize features (runs standalone, reads L1 from BeeGFS):
pa-features-cloud --stores /mnt/beegfs/runs/*/L1/*.zarr \
  --out /mnt/beegfs/features/ --recipe recipes/ml/cloud_v1.yml \
  --label-csv /mnt/beegfs/labels/cloud_labels.csv

# 3. Train cloud detector (attaches to user's running Ray cluster):
pa-train-cloud --launcher attach --recipe recipes/ml/cloud_v1.yml \
  --feature-cache /mnt/beegfs/features/features.<hash>.zarr \
  --out /mnt/beegfs/models/ \
  --local-cache-dir /local/scratch

# 4. Train BetaVAE:
pa-prep-ph --stores /mnt/beegfs/runs/*/L1/*.dp_ph256.*.zarr \
  --out /mnt/beegfs/features/ --recipe recipes/ml/vae_train_v1.yml
pa-train-vae --launcher attach --recipe recipes/ml/vae_train_v1.yml \
  --feature-cache /mnt/beegfs/features/features.<hash>.zarr \
  --out /mnt/beegfs/models/ --local-cache-dir /local/scratch

# 5. Run inference with the new model bundle:
nextflow run . -profile laptop --steps ml \
  --cloud_model_pt /mnt/beegfs/models/cloud_detector_v2.pt \
  --cloud_model_json /mnt/beegfs/models/cloud_detector_v2.CloudDetection.json \
  --input_obs_dir /path/obs.pffd --outdir results/
```

## Hyperparameter sweep (Ray Tune)

Use `pa-tune-cloud` for ASHA sweeps:

```bash
pa-tune-cloud --launcher attach --recipe recipes/ml/cloud_v1.yml \
  --feature-cache /mnt/beegfs/features/features.<hash>.zarr \
  --out /mnt/beegfs/models/
```

Add a `tune:` block to the recipe to define the search space. See
[`recipes/ml/my_model_v1_template.yml`](../recipes/ml/my_model_v1_template.yml) for the
format (`uniform`, `loguniform`, `choice`, `grid`, `randint`).

## Environment notes

- `cluster/ral_up.sh` does **not** copy the conda env — each node already has its env
  (`drp` on receiver/transmit, `ray_env` on the dfs nodes) and an editable install of the
  workspace. It only rsyncs source (editable picks up code changes; `--reinstall` re-runs
  `pip install --no-deps -e .` when deps change).
- The `drp` env may carry `transformers`/`datasets`; these are **not** project dependencies
  (absent from `pyproject.toml` / `uv.lock`), are imported nowhere, and never ship in the
  Docker images (`uv sync --frozen`). They do not affect cluster launch or build time — leave
  them, or `pip uninstall` for a leaner dev env. Do not add them to `pyproject` until used.
