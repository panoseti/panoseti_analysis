"""pa-train-cloud — retrain the cloud detector on a pre-staged feature cache (Layer B).

Thin wiring on top of the shared ML utilities: the epoch loop is
:func:`panoseti_analysis.algorithms.training.fit` (also used by notebooks), the data load is
:func:`panoseti_analysis.adapters.ml.data.load_labeled_feature_cache`, and the Ray
``TorchTrainer`` boilerplate is :func:`panoseti_analysis.adapters.ml.runner.run_torch_trainer`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

import panoseti_analysis.algorithms.cloud_detector  # noqa: F401 — register models
from panoseti_analysis.adapters.ml.data import load_labeled_feature_cache
from panoseti_analysis.adapters.ml.runner import report_final_checkpoint, run_torch_trainer
from panoseti_analysis.adapters.ml.tracking import make_tracker
from panoseti_analysis.adapters.ray._staging import stage_to_local
from panoseti_analysis.algorithms.cloud_train import (
    cloud_loss_fn,
    make_cloud_val_eval,
)
from panoseti_analysis.algorithms.optim import build_optimizer, build_scheduler
from panoseti_analysis.algorithms.registry import build_model
from panoseti_analysis.algorithms.training import fit
from panoseti_analysis.config.models import (
    ClassifierBundle,
    TrainConfig,
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
    """Ray Train worker function — runs on each allocated worker process.

    Loads the feature cache, DDP-wraps the model, then runs the *same* ``fit`` loop and
    cloud hooks that notebooks use via ``algorithms/cloud_train.py``.
    """
    import ray.train
    import ray.train.torch
    from torch.utils.data import DataLoader, TensorDataset

    device = ray.train.torch.get_device()  # follow Ray's assignment (never hard-code cuda)

    x_train, y_train, x_val, y_val = load_labeled_feature_cache(Path(config["local_feature_path"]))

    # Build model from registry — arch id comes from the recipe via train_config["arch"].
    train_cfg = TrainConfig.model_validate(config)
    model = ray.train.torch.prepare_model(build_model(train_cfg.arch))
    optimizer = build_optimizer(model.parameters(), train_cfg.model_dump())
    step_schedulers = build_scheduler(optimizer, train_cfg.model_dump())
    train_loader: DataLoader[Any] = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_eval = make_cloud_val_eval(x_val, y_val, device)

    # Only rank 0 logs to W&B — DDP runs the same loop on every worker, so without this
    # guard every worker creates its own run and metrics appear doubled.
    is_chief = ray.train.get_context().get_local_rank() == 0
    tracker_config = config if is_chief else {**config, "wandb_project": None}
    tracker = make_tracker(tracker_config, run_name=config.get("recipe_name", "cloud_train"))

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        if is_chief:
            tracker.log(metrics, step=epoch)

    # Workstream 2: set_epoch required each epoch for proper shuffle when world_size > 1.
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    def on_epoch_start(epoch: int) -> None:
        sampler = getattr(train_loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if ray.train.get_context().get_world_size() > 1 and callable(set_epoch):
            set_epoch(epoch)

    result = fit(
        model,
        train_loader,
        loss_fn=cloud_loss_fn,
        optimizer=optimizer,
        epochs=train_cfg.epochs,
        device=device,
        val_eval=val_eval,
        step_schedulers=step_schedulers,
        on_epoch_start=on_epoch_start,
        on_epoch=on_epoch,
        monitor=train_cfg.monitor,
    )

    # Rank 0 logs end-of-training media (confusion matrix + PR curve over the best
    # checkpoint) and uploads the model weights as a W&B artifact. Best-effort: the
    # reporting helpers swallow rendering errors so they can never fail a training run.
    if is_chief:
        from panoseti_analysis.adapters.ml.reporting import (
            log_confusion_matrix,
            log_pr_curve,
            log_state_dict_artifact,
        )
        from panoseti_analysis.algorithms.cloud_train import (
            cloud_val_predictions,
            confusion_counts,
            pr_curve_points,
        )

        y_true, y_pred, y_score = cloud_val_predictions(result.best_state, x_val, y_val, device)
        log_confusion_matrix(tracker, confusion_counts(y_true, y_pred))
        log_pr_curve(tracker, pr_curve_points(y_true, y_score))
        log_state_dict_artifact(tracker, result.best_state, name="cloud_detector")
    tracker.finish()

    # Report the best checkpoint to Ray (rank 0 writes it; all workers call report once).
    final_metrics = result.history[-1] if result.history else {"val_loss": result.best_value}
    report_final_checkpoint(result.best_state, final_metrics)


def run_train_cloud(
    feature_cache: Path,
    out_dir: Path,
    recipe_path: Path,
    launcher: str = "standalone",
    local_cache_dir: Path | None = None,
    lineage_out: Path | None = None,
    wandb_project: str | None = None,
    epochs: int | None = None,
) -> tuple[Path, Path, Path]:
    """Retrain the cloud detector on a pre-materialized feature cache.

    Args:
        feature_cache: Path to the feature cache zarr on shared FS.
        out_dir: Output directory for the model bundle.
        recipe_path: YAML recipe (must contain hyperparams, scaling sections).
        launcher: Ray init mode (attach/slurm/standalone).
        local_cache_dir: If set, pre-stages the feature cache to local SSD first.
        lineage_out: If set, writes a StoreLineage JSON record.
        wandb_project: W&B project name override.
        epochs: Epochs override.

    Returns:
        (pt_path, json_path, provenance_path)
    """
    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    hp = params_dict.get("hyperparams", {})
    scaling_cfg = params_dict.get("scaling", {})

    # Stage to SSD if requested. Resolve to an absolute path: Ray workers run from a
    # different CWD (the session artifacts dir), so a relative cache path won't be found.
    effective_path = feature_cache
    if local_cache_dir is not None:
        effective_path = stage_to_local(feature_cache, local_cache_dir)
    effective_path = Path(effective_path).resolve()

    # Read upstream provenance from feature cache (kept as ProcessingStep context).
    ds_feat = open_store(effective_path)
    _ = read_history(dict(ds_feat.attrs))
    feat_cksum = checksum_store(feature_cache)

    train_config: dict[str, Any] = {
        "local_feature_path": str(effective_path),
        "recipe_name": recipe_name,
        **hp,
    }
    if epochs is not None:
        train_config["epochs"] = epochs
    # W&B tracking config passthrough
    for k in ("wandb_project", "wandb_entity"):
        if k in params_dict:
            train_config[k] = params_dict[k]
    if wandb_project is not None:
        train_config["wandb_project"] = wandb_project

    best_state, metrics = run_torch_trainer(
        train_loop_per_worker,
        train_config,
        scaling_cfg=scaling_cfg,
        launcher=launcher,
        out_dir=out_dir,
    )
    print(f"{metrics=}")

    # Resolve arch from the config that was passed to workers (comes from recipe hyperparams).
    train_cfg = TrainConfig.model_validate(train_config)
    model = build_model(train_cfg.arch)
    model.load_state_dict(best_state)

    # Build ClassifierBundle with placeholder checksum (save_classifier will fix it).
    # model_name is derived from the recipe name so artifacts are traceable.
    model_name = f"{recipe_name}_{train_cfg.arch}".replace("/", "_")
    bundle = ClassifierBundle(
        model_name=model_name,
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={
            "channels": 2,
            "height": 32,
            "width": 32,
            "dtype": "float32",
        },
        arch=train_cfg.arch,
    )

    prov = TrainingProvenance(
        step_name="train_cloud",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        params={**hp, "scaling": scaling_cfg},
        input_checksums=[feat_cksum],
        timestamp_utc=now_utc(),
        software=capture_software(),
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
