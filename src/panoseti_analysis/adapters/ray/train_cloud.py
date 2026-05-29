"""pa-train-cloud — retrain the cloud detector on a pre-staged feature cache (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import typer
from panoseti_analysis.adapters.ray._staging import stage_to_local
from panoseti_analysis.adapters.ray._tracking import make_tracker
from panoseti_analysis.adapters.ray.launcher import init_ray
from panoseti_analysis.algorithms.cloud_detector import CloudDetection
from panoseti_analysis.config.models import (
    ClassifierBundle,
    TrainingProvenance,
)
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.models import save_classifier
from panoseti_analysis.io.provenance import capture_software, now_utc, read_history
from panoseti_analysis.io.stores import open_store

app = typer.Typer(add_completion=False, help="Retrain the cloud detector via Ray Train.")


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Ray Train worker function — runs on each allocated worker process."""
    import ray.train
    import ray.train.torch

    # Device: follow Ray's assignment (never hard-code cuda)
    device = ray.train.torch.get_device()

    # Load feature cache from the staged local path
    local_path = Path(config["local_feature_path"])
    ds = open_store(local_path)
    X_all = ds["X"].values.astype(np.float32)  # (N, 2, H, W)
    y_all = ds["label"].values.astype(np.int64)
    split_arr = np.array(ds["split"].values, dtype=str)

    train_mask = split_arr == "train"
    val_mask = split_arr == "val"

    X_train = torch.from_numpy(X_all[train_mask])
    y_train = torch.from_numpy(y_all[train_mask])
    X_val = torch.from_numpy(X_all[val_mask])
    y_val = torch.from_numpy(y_all[val_mask])

    batch_size = int(config.get("batch_size", 128))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-5))
    gamma = float(config.get("gamma", 0.9))
    epochs = int(config.get("epochs", 50))

    model = CloudDetection()
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion = torch.nn.CrossEntropyLoss()

    tracker = make_tracker(config, run_name=config.get("recipe_name", "cloud_train"))

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = criterion(val_logits, y_val.to(device)).item()
            val_preds = val_logits.argmax(1).cpu()
            val_acc = float((val_preds == y_val).float().mean().item())

        metrics = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
        tracker.log(metrics, step=epoch)

        scheduler_exp.step()
        scheduler_plateau.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        ray.train.report(metrics)

    tracker.finish()

    # Report the best checkpoint from rank-0 worker.
    # Ray 2.x uses directory-based checkpoints — serialize the state dict to a
    # temporary file and wrap it in a Checkpoint.from_directory.
    if best_state is not None and ray.train.get_context().get_local_rank() == 0:
        import tempfile

        with tempfile.TemporaryDirectory() as ckpt_dir:
            torch.save(best_state, Path(ckpt_dir) / "model.pt")
            checkpoint = ray.train.Checkpoint.from_directory(ckpt_dir)
            ray.train.report(
                {"val_loss": best_val_loss, "val_acc": val_acc},
                checkpoint=checkpoint,
            )


def run_train_cloud(
    feature_cache: Path,
    out_dir: Path,
    recipe_path: Path,
    launcher: str = "standalone",
    local_cache_dir: Path | None = None,
    lineage_out: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Retrain the cloud detector on a pre-materialized feature cache.

    Args:
        feature_cache: Path to the feature cache zarr on shared FS.
        out_dir: Output directory for the model bundle.
        recipe_path: YAML recipe (must contain hyperparams, scaling sections).
        launcher: Ray init mode (attach/slurm/standalone).
        local_cache_dir: If set, pre-stages the feature cache to local SSD first.
        lineage_out: If set, writes a StoreLineage JSON record.

    Returns:
        (pt_path, json_path, provenance_path)
    """
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    hp = params_dict.get("hyperparams", {})
    scaling_cfg = params_dict.get("scaling", {})

    # Stage to SSD if requested
    effective_path = feature_cache
    if local_cache_dir is not None:
        effective_path = stage_to_local(feature_cache, local_cache_dir)

    # Read upstream provenance from feature cache
    ds_feat = open_store(effective_path)
    feat_history = read_history(dict(ds_feat.attrs))
    feat_cksum = checksum_store(feature_cache)

    train_config: dict[str, Any] = {
        "local_feature_path": str(effective_path),
        "recipe_name": recipe_name,
        **hp,
    }
    # W&B tracking config passthrough
    for k in ("wandb_project", "wandb_entity"):
        if k in params_dict:
            train_config[k] = params_dict[k]

    accelerator_type = scaling_cfg.get("accelerator_type", "A6000")
    num_workers = int(scaling_cfg.get("num_workers", 2))
    use_gpu = num_workers > 0 and torch.cuda.is_available()

    scaling = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        **({"accelerator_type": accelerator_type} if accelerator_type else {}),
    )

    storage_path = str(out_dir / "ray_results")
    run_cfg = RunConfig(storage_path=storage_path)

    init_ray(launcher)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling,
        run_config=run_cfg,
    )
    result = trainer.fit()

    # Restore best checkpoint.
    # Ray 2.x uses directory-based checkpoints; load model.pt from the checkpoint dir.
    checkpoint = result.checkpoint
    assert checkpoint is not None, "Training produced no checkpoint"
    with checkpoint.as_directory() as ckpt_dir:
        best_state = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=True)

    model = CloudDetection()
    model.load_state_dict(best_state)

    # Build ClassifierBundle with placeholder checksum (save_classifier will fix it)
    bundle = ClassifierBundle(
        model_name="cloud_detector_retrained",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={
            "channels": 2,
            "height": 32,
            "width": 32,
            "dtype": "float32",
        },
    )

    started_at = now_utc()
    software = capture_software()
    raw_metrics: dict[str, Any] = result.metrics or {}
    metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}

    # Suppress unused variable warning — feat_history is read for provenance context
    # but not directly embedded in TrainingProvenance (kept as ProcessingStep chain).
    _ = feat_history

    prov = TrainingProvenance(
        step_name="train_cloud",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        params={**hp, "scaling": scaling_cfg},
        input_checksums=[feat_cksum],
        timestamp_utc=started_at,
        software=software,
        metrics=metrics,
    )

    pt_path, json_path, provenance_path = save_classifier(model, bundle, prov, out_dir)

    if lineage_out is not None:
        record = {
            "step_name": "train_cloud",
            "recipe_name": recipe_name,
            "recipe_hash": recipe_hash,
            "model_bundle": json_path.name,
            "metrics": metrics,
            "input_checksums": [feat_cksum],
        }
        Path(lineage_out).write_text(json.dumps(record, indent=2))

    return pt_path, json_path, provenance_path


@app.command()
def main(
    feature_cache: Path = typer.Argument(..., help="Feature cache .zarr on BeeGFS"),
    out_dir: Path = typer.Argument(...),
    recipe: Path = typer.Option(...),
    launcher: str = typer.Option("standalone", help="attach|slurm|standalone"),
    local_cache_dir: Path | None = typer.Option(None, help="SSD scratch dir for staging"),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    run_train_cloud(
        feature_cache,
        out_dir,
        recipe,
        launcher=launcher,
        local_cache_dir=local_cache_dir,
        lineage_out=lineage_out,
    )


if __name__ == "__main__":
    app()
