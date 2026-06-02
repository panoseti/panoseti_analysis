# panoseti/analysis

[![AWS CI](https://img.shields.io/badge/CI%20tests-full%20size-FF9900?labelColor=000000&logo=Amazon%20AWS)](https://nf-co.re/analysis/results)[![Get help on Slack](http://img.shields.io/badge/slack-nf--core%20%23analysis-4A154B?labelColor=000000&logo=slack)](https://nfcore.slack.com/channels/analysis)[![Follow on Bluesky](https://img.shields.io/badge/bluesky-%40nf__core-1185fe?labelColor=000000&logo=bluesky)](https://bsky.app/profile/nf-co.re)[![Follow on Mastodon](https://img.shields.io/badge/mastodon-nf__core-6364ff?labelColor=FFFFFF&logo=mastodon)](https://mstdn.science/@nf_core)[![Watch on YouTube](http://img.shields.io/badge/youtube-nf--core-FF0000?labelColor=000000&logo=youtube)](https://www.youtube.com/c/nf-core) Documentation

Welcome to the `panoseti/analysis` documentation.

This directory contains the documentation for the pipeline, describing how to use it and the specifications of its outputs.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Usage](usage.md) | How to run the pipeline, input formats, and available Nextflow profiles |
| [Output](output.md) | Output directory structure and generated file specifications |
| [Storage Specification](storage_spec.md) | Zarr v3 L0/L1/L2 data product layouts, manifest schema, PACK conventions |
| [ML Architecture](ml_architecture.md) | ML execution models (CPU/Ray/Serve), training system, real-time streaming |
| [Provenance](provenance.md) | ProcessingStep schema, TrainingProvenance, recipe_hash, streaming tags |
| [W&B Experiment Tracking](wandb.md) | Setting up Weights & Biases API key, .env file, offline mode |
| [Contributing](CONTRIBUTING.md) | Developer guidelines, three-layer architecture rules, boundary lint |

## ML Projects

See [`ml/README.md`](../ml/README.md) for the ML project index.  
See [`grpc/README.md`](../grpc/README.md) for gRPC service documentation (DAQ, ML inference).
