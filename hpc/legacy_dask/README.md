# Legacy Dask Cluster Scripts

These scripts were written by Andrea Zonca to orchestrate a Dask distributed cluster for
multi-node parallel processing on SDSC Expanse. They are preserved here for reference.

## Why these are no longer the main path

The new pipeline (v0.3.0) delegates parallelism to **Nextflow** instead of Dask:
- Each `(data_product, module)` L1 calibration job runs as an independent Nextflow process.
- On HPC, Nextflow submits one SLURM job per product and manages the DAG.
- No scheduler process, no worker pre-loading, no SSH cluster to tear down.

This is simpler, more portable, and integrates natively with Seqera Platform / Tower.

## When these might still be useful

- If you need to process data faster than per-product parallelism allows (e.g., very long
  observations where the calibration step itself should be parallelized across time chunks).
- As a reference for future work if the calibration step needs distributed computation.

## Files

| File | Purpose |
|---|---|
| `cluster_manager.py` | Factory for SSH / Local Dask clusters; reads config from `config.toml` |
| `cluster_lifecycle_manager.py` | Async lifecycle wrapper (start → wait → stop) |
| `cleanup_dask_cluster.sh` | Kills stray Dask worker/scheduler processes on remote nodes |
| `step0_setup_cluster.py` | Entry point: starts the cluster and prints the scheduler address |
| `worker_preload.py` | Imported by each Dask worker at startup to pre-load libraries |

## Original usage

```bash
# 1. Start cluster, get scheduler address
python step0_setup_cluster.py --config config.toml

# 2. Pass scheduler address to step2
python step2_dask_baseline.py input.zarr output.zarr --dask-scheduler tcp://host:8786

# 3. Tear down
bash cleanup_dask_cluster.sh
```
