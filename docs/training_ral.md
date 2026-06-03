# Training Models on RAL

RAL (Remote Astrophysics Lab) is a 5-node bare-metal cluster — no SLURM. Models are
trained with Ray Train in `attach` mode against a user-managed persistent Ray cluster.

See [ML Architecture](ml_architecture.md) for training system design, data flow,
checkpointing, and SSD-staging details.

## Cluster topology

| Node                | Specs                                                     | Role                                         |
| ------------------- | --------------------------------------------------------- | -------------------------------------------- |
| `digilab-receiver`  | 2× RTX A6000 48 GB + 1 TB NVMe SSD (`accelerator_type:G`) | Head node; training; Ray + Grafana dashboard |
| `digilab-transmit`  | 2× consumer RTX (`accelerator_type:RTX`)                  | Ray Serve inference                          |
| `panoseti-dfs0/1/2` | CPU-only                                                  | BeeGFS storage nodes                         |

BeeGFS at `/mnt/beegfs`. ML artifacts: `/mnt/beegfs/models/`, `/mnt/beegfs/features/`,
`/mnt/beegfs/runs/`.

## Starting the cluster

```bash
ray up conf/ray/ral_cluster.yaml             # start / reconnect all 5 nodes
ray up conf/ray/ral_cluster.yaml --no-restart  # attach without restarting Ray
ray status                                   # verify workers connected
cd ~/ray-test && docker compose up -d        # start prometheus/grafana (if not running)
# Dashboard: http://digilab-receiver:8265   Grafana: http://digilab-receiver:3000
```

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
