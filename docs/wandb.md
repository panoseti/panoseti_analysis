# W&B Experiment Tracking

`panoseti_analysis` uses [Weights & Biases](https://wandb.ai) for ML experiment tracking.
The integration is opt-in: if `WANDB_API_KEY` is not set, training falls back to a local
TensorBoard log automatically.

## Setup

### 1. Get your API key

Log in at [wandb.ai](https://wandb.ai) → **User settings** → **API keys** → copy the key.

### 2. Create a `.env` file in the repo root

```bash
# panoseti_analysis/.env  — never commit this file (already in .gitignore)
WANDB_API_KEY=your_key_here
WANDB_ENTITY=your_wandb_username_or_team
```

The training notebooks and CLIs load `.env` automatically:
```python
# Pattern used in notebooks:
from pathlib import Path
for line in Path('.env').read_text().splitlines():
    if line.strip() and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())
```

For CLI use, export the variables in your shell:
```bash
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2)
```

### 3. Configure the recipe

Add W&B project and entity to your recipe YAML:

```yaml
# ml/cloud-detection/recipes/cloud_v1.yml
name: cloud_v1
wandb_project: panoseti-cloud-detection
wandb_entity: your_username   # optional; defaults to WANDB_ENTITY env var
# ... rest of hyperparameters
```

The training adapters read `config["wandb_project"]` and `config["wandb_entity"]`
from the recipe (see `adapters/ray/_tracking.py::make_tracker()`).

## How it works

```
pa-train-cloud → run_train_cloud() → TorchTrainer (Ray Train)
    │
    └─ train_loop_per_worker()
            │
            └─ make_tracker(config) → _WandbTracker  (if WANDB_API_KEY set)
                                   → _TensorBoardTracker  (fallback)
                                   → _NoOpTracker  (if both unavailable)
```

The tracker logs metrics each epoch:
- `train_loss`, `val_loss`
- `train_accuracy`, `val_accuracy`
- `learning_rate`

The `TrainingProvenance` record (written to the model bundle's `_provenance.json`)
includes `wandb_run_id` when W&B is active, so you can trace any model bundle back
to its exact training run.

## Offline mode

For air-gapped nodes (e.g., on SDSC Expanse with no internet):

```bash
export WANDB_MODE=offline
```

Metrics are stored locally. Sync when you have connectivity:

```bash
wandb sync wandb/offline-run-*/
```

## Viewing runs

```bash
# Open in browser
wandb login  # if not already logged in
wandb runs list --project panoseti-cloud-detection
```

Or visit `https://wandb.ai/<your_entity>/panoseti-cloud-detection`.

## TensorBoard fallback

When W&B is not configured, logs are written to `logs/` in the training output directory.
View them with:

```bash
tensorboard --logdir ml/cloud-detection/models/
```
