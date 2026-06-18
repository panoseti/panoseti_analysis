# PANOSETI Analysis Documentation

This directory contains the documentation for `panoseti_analysis` — the PANOSETI
data-reduction pipeline (PFF → Zarr ingest, calibration, ML inference, real-time streaming).

For a quick orientation and the router table, see the [top-level README](../README.md).

## Documentation Index

| Document                                      | Description                                                               |
| --------------------------------------------- | ------------------------------------------------------------------------- |
| [Usage](usage.md)                             | How to run the pipeline, input formats, and available Nextflow profiles   |
| [Output](output.md)                           | Output directory structure and generated file specifications              |
| [Storage Specification](storage_spec.md)      | Zarr v3 L0/L1/L2 data product layouts, manifest schema, PACK conventions  |
| [ML Architecture](ml_architecture.md)         | ML execution models (CPU/Ray/Serve), training system, real-time streaming |
| [Real-time Streaming (RAL)](streaming_ral.md) | Step-by-step setup for the RAL attach-mode streaming pipeline             |
| [Training on RAL](training_ral.md)            | Cluster topology, startup, full training + sweep workflow                 |
| [Provenance](provenance.md)                   | ProcessingStep schema, TrainingProvenance, recipe_hash, streaming tags    |
| [W&B Experiment Tracking](wandb.md)           | Setting up Weights & Biases API key, .env file, offline mode              |
| [Contributing](CONTRIBUTING.md)               | Developer guidelines, three-layer architecture rules, boundary lint       |

## Related documentation

- [`ml/README.md`](../ml/README.md) — ML projects index (cloud detection, anomaly detection)
- [`grpc/README.md`](../grpc/README.md) — gRPC services (DAQ, ML inference)
- [`pypff/docs/zarr_v3_spec.md`](../pypff/docs/zarr_v3_spec.md) — L0 Zarr array layout spec (pypff submodule)
