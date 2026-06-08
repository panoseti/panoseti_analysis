# Cloud Detection — Model Card & Training Guide

## Model: `cloud_detector_v1`

Binary sky-condition classifier that scores each 60-second observing window as
**clear** (0) or **cloudy** (1) from PANOSETI 32×32 img16 movie-mode frames.

| Field | Value |
|-------|-------|
| Model file | `assets/models/cloud_detector_v1.pt` |
| Architecture | 2-channel CNN (see [CNN Architecture](#cnn-architecture)) |
| Input | `(2, 32, 32)` float32 — [deriv-fft, raw-fft] |
| Output | `cloud_score` ∈ [0, 1] per 60-second window |
| Threshold | 0.5 (configurable via `CloudInferParams.threshold`) |
| Training data | 8 observing runs, 8,804 labeled samples (50/50 clear/cloudy) |
| Val accuracy | ~96% (from original training; see `model_summary.txt` in legacy repo) |
| Trained | 2024-03-01, original `~/panoseti/cloud-detection/` repo |

## Feature Specification

Cloud detection uses **two feature channels** computed over a 60-second rolling window:

| Channel | Name | Description |
|---------|------|-------------|
| 0 | `feature_deriv_fft` | 2D FFT of the 60-second time derivative (current − 60s-ago frame), with Hann window |
| 1 | `feature_raw_fft` | 2D FFT of the current integrated frame (10 × 100µs frames = 1ms integration), with Hann window |

Both channels use log-magnitude scaling: `log(|FFT| + ε)`.

**Original training scaling** (from `~/panoseti/cloud-detection/`): square-root scaling
`x * (1 / |x|^0.5)`. This differs from the log-magnitude used in the new pipeline.
See [Stage 2b notes](#stage-2b-features-from-pff) for reconciliation.

## CNN Architecture

```
Input: (2, 32, 32)
  │
  ├─ conv_block_1: Conv2d(2→28, k=3, groups=2) × 4 + MaxPool2d(3) + Dropout2d(0.5)
  ├─ conv_block_2: Conv2d(28→27, k=5) × 3 + MaxPool2d(3) + Dropout2d(0.5)
  ├─ Flatten
  ├─ LazyLinear → 128 → ReLU + BN + Dropout
  ├─ LazyLinear → 84 → ReLU + BN + Dropout
  └─ LazyLinear → 2 (binary logits)

Parameters: see assets/models/cloud_detector_v1.json (ClassifierBundle)
```

## Training Data

Pre-computed features and labels are packaged in
`assets/models/cloud-detection-training/` (not extracted; gitignored when extracted).

| Batch | Run | Conditions | Samples |
|-------|-----|-----------|---------|
| 0 | 2023-08-15 | Partially cloudy, Moon 0% | 2,051 |
| 1 | 2023-08-01 | Mostly cloudy, Moon 100% | 3,758 |
| 2 | 2023-09-13 | Likely clear, Moon 0% | 753 |
| 3 | 2023-10-12 | Likely clear, Moon 11% | 423 |
| 4 | 2023-09-09 | Likely clear, Moon 30% | 483 |
| 5 | 2023-10-05 | Likely clear, Moon 70% | 418 |
| 6 | 2023-08-29 | Clear, Moon 90% | 522 |
| 7 | 2023-09-28 | Likely clear, Moon 95% | 396 |
| **Total** | | | **8,804** |

Label distribution: `clear_night_sky` 4,357 (49.5%), `not_clear_cloudy` 4,447 (50.5%),
`unsure` 5 (skipped during training).

## Recipes

| Recipe | Purpose |
|--------|---------|
| [`recipes/cloud_v1.yml`](recipes/cloud_v1.yml) | Batch training hyperparameters |
| [`recipes/stream_cloud_v1.yml`](recipes/stream_cloud_v1.yml) | Real-time streaming cadence/threshold |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/01_inference_demo.ipynb`](notebooks/01_inference_demo.ipynb) | End-to-end demo: L1 Zarr → cloud scores + visualizations |
| [`notebooks/02_train_legacy_features.ipynb`](notebooks/02_train_legacy_features.ipynb) | Replicate training from pre-packaged features + labels |

## Reproducing Training

### Stage 2a — Pre-packaged features + original labels (implemented)

```bash
# Step 1: Build feature cache from the packaged numpy arrays
pa-prep-cloud-legacy \
  --labels-zip assets/models/cloud-detection-training/training-labels.zip \
  --data-zip assets/models/cloud-detection-training/training-data-batch.zip \
  --extract-dir ml/cloud-detection/cache/ \
  --out ml/cloud-detection/cache/cloud_features_legacy.zarr

# Step 2: Train
pa-train-cloud \
  --launcher standalone \
  --feature-cache ml/cloud-detection/cache/cloud_features_legacy.zarr \
  --out ml/cloud-detection/models/ \
  --recipe ml/cloud-detection/recipes/cloud_v1.yml \
  --local-cache-dir /tmp/

# Or use the notebook for interactive training:
# ml/cloud-detection/notebooks/02_train_legacy_features.ipynb
```

### Stage 2b — New features from PFF + original labels (future)

Key challenge: the original labels are keyed by `feature_uid` (SHA1 hash computed by
the original batch builder), while our pipeline keys frames by `unix_t_ns`. Mapping
requires the original `pano_df` DataFrames from the batch builder output.

Once the mapping is established, `pa-features-cloud` output is already compatible with
`pa-train-cloud` — no trainer changes needed.

### Stage 2c — New features + new label format (future)

Requires a labeling UI that works with the Zarr-based `(time, module, dp)` data format.
Labels would be emitted as an `xr.Dataset` with `(unix_t_ns, module_id, dp)` coordinates.

## Deferred improvements

- Moving baseline to compensate for moon brightness changes
- AdamW instead of Adam optimizer
- Ray Data wrappers for distributed out-of-core feature loading
- `prepare_data_loader()` wrapping for multi-node distributed training
