# Anomaly Detection — β-VAE for Pulse-Height Data

## Overview

Unsupervised anomaly detection for PANOSETI pulse-height (PH) data using a β-Variational
Autoencoder (β-VAE). The model learns a compact latent representation of normal sky
backgrounds; high reconstruction error or unusual latent positions indicate anomalies
(potential optical transients, instrument artifacts, or cosmic rays).

## Model: `ph_vae_v1` (prototype)

| Field        | Value                                                                           |
| ------------ | ------------------------------------------------------------------------------- |
| Architecture | β-VAE with 2D convolutional encoder/decoder                                     |
| Input        | `(1, 16, 16)` float32 — log-normalized pedestal-subtracted PH frames (dp_ph256) |
| Latent dim   | 32 (configurable)                                                               |
| Hidden dim   | 64 (configurable)                                                               |
| Loss         | MSE reconstruction + β·KL + sparsity·L1(μ)                                      |

## Feature Specification

| Step          | Description                                                                |
| ------------- | -------------------------------------------------------------------------- |
| Input         | L1 `pedestal_subtracted` (16×16 int16) from `pa-calibrate ph`              |
| Preprocessing | log-norm to unit range: `log(x - min + 1) / log(max - min + 1)`            |
| Output        | Latent code μ (32-dim), reconstruction, anomaly score (reconstruction MSE) |

## Recipes

| Recipe                                                 | Purpose                  |
| ------------------------------------------------------ | ------------------------ |
| [`recipes/vae_train_v1.yml`](recipes/vae_train_v1.yml) | Training hyperparameters |

## Notebooks

| Notebook                                                           | Purpose                                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| [`notebooks/01_ph_vae_demo.ipynb`](notebooks/01_ph_vae_demo.ipynb) | Latent space exploration + anomaly scoring (coming soon) |

Original prototype notebooks: `~/panoseti/anomaly-detection/ph_vae_model.ipynb`,
`ph_eda.ipynb`, `imaging_vae_model.ipynb`.

## Training

```bash
# Step 1: Prepare features from L1 ph256 stores
pa-prep-ph \
  --stores /mnt/beegfs/runs/*/L1/*.dp_ph256.*.zarr \
  --out ml/anomaly-detection/cache/ \
  --recipe ml/anomaly-detection/recipes/vae_train_v1.yml

# Step 2: Train
pa-train-vae \
  --launcher attach \
  --feature-cache ml/anomaly-detection/cache/features.<hash>.zarr \
  --out ml/anomaly-detection/models/ \
  --recipe ml/anomaly-detection/recipes/vae_train_v1.yml \
  --local-cache-dir /local/scratch
```
