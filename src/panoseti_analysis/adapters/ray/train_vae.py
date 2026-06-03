"""pa-train-vae — train BetaVAE on a PH feature cache via Ray Train (Layer B).

Thin wiring on top of the shared ML utilities, identical in shape to ``train_cloud.py``: the
epoch loop is :func:`panoseti_analysis.algorithms.training.fit` (also used by notebooks), the
data load is :func:`panoseti_analysis.adapters.ml.data.load_unlabeled_feature_cache`, and the
Ray ``TorchTrainer`` boilerplate is :func:`panoseti_analysis.adapters.ml.runner.run_torch_trainer`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from panoseti_analysis.adapters.ml.data import load_unlabeled_feature_cache
from panoseti_analysis.adapters.ml.runner import report_final_checkpoint, run_torch_trainer
from panoseti_analysis.adapters.ml.tracking import make_tracker
from panoseti_analysis.adapters.ray._staging import stage_to_local
from panoseti_analysis.algorithms.ph_vae import BetaVAE
from panoseti_analysis.algorithms.training import fit
from panoseti_analysis.algorithms.vae_train import (
    build_vae_optimizer,
    make_vae_loss_fn,
    make_vae_val_eval,
)
from panoseti_analysis.config.models import ClassifierBundle, TrainingProvenance
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.models import save_classifier
from panoseti_analysis.io.provenance import capture_software, now_utc, read_history
from panoseti_analysis.io.stores import open_store

app = typer.Typer(add_completion=False, help="Train BetaVAE on PH feature cache via Ray Train.")


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Ray Train worker function — runs on each allocated worker process.

    Loads the (unlabeled) feature cache, DDP-wraps the BetaVAE, then runs the *same* ``fit``
    loop and VAE hooks that notebooks use via ``algorithms/vae_train.py``.
    """
    import ray.train
    import ray.train.torch

    device = ray.train.torch.get_device()  # follow Ray's assignment (never hard-code cuda)

    latent_dim = int(config.get("latent_dim", 32))
    hidden_dim = int(config.get("hidden_dim", 64))
    beta = float(config.get("beta", 4e-9))
    sparsity_weight = float(config.get("sparsity_weight", 0.0))

    x_train, x_val = load_unlabeled_feature_cache(
        Path(config["local_feature_path"]),
        val_prop=float(config.get("val_prop", 0.1)),
        seed=int(config.get("seed", 1984)),
    )

    model = ray.train.torch.prepare_model(BetaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim))
    optimizer = build_vae_optimizer(model, config)
    from torch.utils.data import DataLoader, TensorDataset

    train_loader: DataLoader[Any] = DataLoader(
        TensorDataset(x_train),
        batch_size=int(config.get("batch_size", 256)),
        shuffle=True,
        drop_last=True,
    )
    val_eval = make_vae_val_eval(x_val, beta, sparsity_weight, device)

    # Only rank 0 logs to W&B (DDP runs the same loop on every worker — see train_cloud.py).
    is_chief = ray.train.get_context().get_local_rank() == 0
    tracker_config = config if is_chief else {**config, "wandb_project": None}
    tracker = make_tracker(tracker_config, run_name=config.get("recipe_name", "vae_train"))

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        if is_chief:
            tracker.log(metrics, step=epoch)

    result = fit(
        model,
        train_loader,
        loss_fn=make_vae_loss_fn(beta, sparsity_weight),
        optimizer=optimizer,
        epochs=int(config.get("epochs", 100)),
        device=device,
        val_eval=val_eval,
        on_epoch=on_epoch,
        monitor="val_loss",
    )

    # Rank 0 logs the validation reconstruction-error distribution (the anomaly score) and
    # uploads the model weights as a W&B artifact. Best-effort: helpers swallow render errors.
    if is_chief:
        from panoseti_analysis.adapters.ml.reporting import log_histogram, log_state_dict_artifact
        from panoseti_analysis.algorithms.vae_train import vae_reconstruction_errors

        errors = vae_reconstruction_errors(
            result.best_state, x_val, device, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        log_histogram(
            tracker,
            errors,
            key="val/recon_error",
            title="Val reconstruction error",
            xlabel="per-sample MSE",
        )
        log_state_dict_artifact(tracker, result.best_state, name="ph_vae")
    tracker.finish()

    final_metrics = result.history[-1] if result.history else {"val_loss": result.best_value}
    report_final_checkpoint(result.best_state, final_metrics)


def run_train_vae(
    feature_cache: Path,
    out_dir: Path,
    recipe_path: Path,
    launcher: str = "standalone",
    local_cache_dir: Path | None = None,
    lineage_out: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Train BetaVAE on a pre-materialized PH feature cache.

    Args:
        feature_cache: Path to the feature cache zarr on shared FS.
        out_dir: Output directory for the model bundle.
        recipe_path: YAML recipe (latent_dim/hidden_dim/beta + hyperparams, scaling sections).
        launcher: Ray init mode (attach/slurm/standalone).
        local_cache_dir: If set, pre-stages the feature cache to local SSD first.
        lineage_out: If set, writes a lineage JSON record.

    Returns:
        (pt_path, json_path, provenance_path)
    """
    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    hp: dict[str, Any] = dict(params_dict.get("hyperparams", {}))
    scaling_cfg: dict[str, Any] = dict(params_dict.get("scaling", {}))
    split_cfg: dict[str, Any] = dict(params_dict.get("split", {}))

    # Resolve to absolute: Ray workers run from a different CWD, so a relative path fails.
    effective_path = feature_cache
    if local_cache_dir is not None:
        effective_path = stage_to_local(feature_cache, local_cache_dir)
    effective_path = Path(effective_path).resolve()

    ds_feat = open_store(effective_path)
    _ = read_history(dict(ds_feat.attrs))  # validates the store has provenance
    feat_cksum = checksum_store(feature_cache)

    latent_dim = int(params_dict.get("latent_dim", 32))
    hidden_dim = int(params_dict.get("hidden_dim", 64))
    train_config: dict[str, Any] = {
        "local_feature_path": str(effective_path),
        "recipe_name": recipe_name,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "beta": float(params_dict.get("beta", 4e-9)),
        "sparsity_weight": float(params_dict.get("sparsity_weight", 0.0)),
        "val_prop": float(split_cfg.get("val_prop", 0.1)),
        "seed": int(split_cfg.get("seed", 1984)),
        **hp,
    }
    for k in ("wandb_project", "wandb_entity"):
        if k in params_dict:
            train_config[k] = params_dict[k]

    best_state, metrics = run_torch_trainer(
        train_loop_per_worker,
        train_config,
        scaling_cfg=scaling_cfg,
        launcher=launcher,
        out_dir=out_dir,
    )

    model = BetaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(best_state)

    bundle = ClassifierBundle(
        model_name="ph_vae",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={
            "channels": 1,
            "height": 16,
            "width": 16,
            "dtype": "float32",
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
        },
    )

    prov = TrainingProvenance(
        step_name="train_vae",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        params={
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "beta": float(train_config["beta"]),
            **hp,
            "scaling": scaling_cfg,
        },
        input_checksums=[feat_cksum],
        timestamp_utc=now_utc(),
        software=capture_software(),
        metrics=metrics,
    )

    pt_path, json_path, provenance_path = save_classifier(model, bundle, prov, out_dir)

    if lineage_out is not None:
        record = {
            "step_name": "train_vae",
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
    feature_cache: Path = typer.Argument(..., help="PH feature cache .zarr on BeeGFS"),
    out_dir: Path = typer.Argument(...),
    recipe: Path = typer.Option(...),
    launcher: str = typer.Option("standalone", help="attach|slurm|standalone"),
    local_cache_dir: Path | None = typer.Option(None, help="SSD scratch dir for staging"),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    run_train_vae(
        feature_cache,
        out_dir,
        recipe,
        launcher=launcher,
        local_cache_dir=local_cache_dir,
        lineage_out=lineage_out,
    )


if __name__ == "__main__":
    app()
