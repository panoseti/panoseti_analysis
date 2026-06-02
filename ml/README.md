# PANOSETI ML Projects

Each subdirectory is a self-contained ML project with its own notebooks, recipes,
and model documentation. Shared kernels and adapters live in
[`src/panoseti_analysis/`](../src/panoseti_analysis/).

## Projects

| Project | Description | Status |
|---------|-------------|--------|
| [cloud-detection/](cloud-detection/) | Binary sky-condition classifier (clear vs. cloudy) from 32×32 img16 frames | Production (v1 model deployed) |
| [anomaly-detection/](anomaly-detection/) | β-VAE for pulse-height anomaly detection and latent-space exploration | Prototype |

## Shared library layer

| Layer | Path | Purpose |
|-------|------|---------|
| Kernels (Layer A) | `src/panoseti_analysis/algorithms/` | `calibrate_img`, `extract_cloud_features`, `predict_cloud_score`, `BetaVAE` |
| Adapters (Layer B) | `src/panoseti_analysis/adapters/ray/` | `pa-train-cloud`, `pa-train-vae`, `pa-features-cloud`, `pa-classify-cloud` |
| Streaming | `src/panoseti_analysis/adapters/stream/` | `pa-stream-cloud` (real-time inference via Ray Serve) |
| I/O | `src/panoseti_analysis/io/` | `load_classifier`, `save_classifier`, `write_store`, `checksum_store` |

## Migrating notebooks from `~/panoseti/`

The original ML code lived in `~/panoseti/cloud-detection/` and `~/panoseti/anomaly-detection/`.
To preserve git history when migrating a subdirectory:

```bash
# 1. In the source repo, extract history for the subdir:
cd ~/panoseti
git filter-repo --path cloud-detection/model_training/ --force
# This rewrites the repo in-place — do this on a copy!

# 2. Back in panoseti_analysis, add as a subtree:
git subtree add --prefix=ml/cloud-detection/legacy \
    ~/panoseti/cloud-detection master --squash
```

For reference during development, create a local symlink (gitignored):
```bash
ln -s ~/panoseti/cloud-detection ml/cloud-detection/legacy-src
ln -s ~/panoseti/anomaly-detection ml/anomaly-detection/legacy-src
```
