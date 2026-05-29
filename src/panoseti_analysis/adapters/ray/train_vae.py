"""pa-train-vae — train BetaVAE on PH feature cache via Ray Train (Layer B)."""

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
from panoseti_analysis.algorithms.ph_vae import BetaVAE, beta_vae_loss_function
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

app = typer.Typer(add_completion=False, help="Train BetaVAE on PH feature cache via Ray Train.")

#: Filename used to persist the best model state_dict inside a Ray checkpoint directory.
_CKPT_FILENAME = "model.pt"


def train_loop_per_worker(config: dict[str, Any]) -> None:
    import ray.train
    import ray.train.torch

    device = ray.train.torch.get_device()

    local_path = Path(config["local_feature_path"])
    ds = open_store(local_path)
    X_all = ds["X"].values.astype(np.float32)  # (N, 1, H, W)

    latent_dim = int(config.get("latent_dim", 32))
    hidden_dim = int(config.get("hidden_dim", 64))
    beta = float(config.get("beta", 4e-9))
    sparsity_weight = float(config.get("sparsity_weight", 0.0))
    batch_size = int(config.get("batch_size", 256))
    lr = float(config.get("lr", 1e-3))
    epochs = int(config.get("epochs", 100))

    # prepare_model returns nn.Module; annotate as such to keep mypy happy
    _model = BetaVAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model: torch.nn.Module = ray.train.torch.prepare_model(_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.from_numpy(X_all)

    tracker = make_tracker(config, run_name=config.get("recipe_name", "vae_train"))
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss = beta_vae_loss_function(recon, xb, mu, logvar, beta, sparsity_weight)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        metrics: dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss}
        tracker.log(metrics, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        ray.train.report(metrics)

    tracker.finish()
    if best_state_dict is not None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = Path(tmpdir) / _CKPT_FILENAME
            torch.save(best_state_dict, str(ckpt_file))
            checkpoint = ray.train.Checkpoint.from_directory(tmpdir)
            ray.train.report({"train_loss": best_loss}, checkpoint=checkpoint)


def run_train_vae(
    feature_cache: Path,
    out_dir: Path,
    recipe_path: Path,
    launcher: str = "standalone",
    local_cache_dir: Path | None = None,
    lineage_out: Path | None = None,
) -> tuple[Path, Path, Path]:
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    hp: dict[str, Any] = dict(params_dict.get("hyperparams", {}))
    scaling_cfg: dict[str, Any] = dict(params_dict.get("scaling", {}))

    effective_path = feature_cache
    if local_cache_dir is not None:
        effective_path = stage_to_local(feature_cache, local_cache_dir)

    ds_feat = open_store(effective_path)
    _feat_history = read_history(dict(ds_feat.attrs))  # unused but validates the store
    feat_cksum = checksum_store(feature_cache)

    train_config: dict[str, Any] = {
        "local_feature_path": str(effective_path),
        "recipe_name": recipe_name,
        "latent_dim": int(params_dict.get("latent_dim", 32)),
        "hidden_dim": int(params_dict.get("hidden_dim", 64)),
        "beta": float(params_dict.get("beta", 4e-9)),
        "sparsity_weight": float(params_dict.get("sparsity_weight", 0.0)),
        **hp,
    }
    for k in ("wandb_project", "wandb_entity"):
        if k in params_dict:
            train_config[k] = params_dict[k]

    accelerator_type: str | None = scaling_cfg.get("accelerator_type")
    num_workers = int(scaling_cfg.get("num_workers", 2))
    use_gpu = num_workers > 0 and torch.cuda.is_available()

    scaling_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "use_gpu": use_gpu,
    }
    if accelerator_type:
        scaling_kwargs["resources_per_worker"] = {"accelerator_type:" + accelerator_type: 0.001}

    scaling = ScalingConfig(**scaling_kwargs)

    storage_path = str(out_dir / "ray_results_vae")
    run_cfg = RunConfig(storage_path=storage_path)

    init_ray(launcher)  # type: ignore[arg-type]

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling,
        run_config=run_cfg,
    )
    result = trainer.fit()

    checkpoint = result.checkpoint
    assert checkpoint is not None, "Training produced no checkpoint"

    # Load state_dict from the file-based checkpoint directory
    with checkpoint.as_directory() as ckpt_dir:
        best_state: dict[str, torch.Tensor] = torch.load(
            str(Path(ckpt_dir) / _CKPT_FILENAME),
            map_location="cpu",
            weights_only=True,
        )

    latent_dim = int(train_config.get("latent_dim", 32))
    hidden_dim = int(train_config.get("hidden_dim", 64))
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

    started_at = now_utc()
    software = capture_software()

    raw_metrics: dict[str, Any] = result.metrics or {}
    metrics: dict[str, float] = {
        k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))
    }

    prov = TrainingProvenance(
        step_name="train_vae",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        params={
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "beta": float(train_config.get("beta", 4e-9)),
            **hp,
            "scaling": scaling_cfg,
        },
        input_checksums=[feat_cksum],
        timestamp_utc=started_at,
        software=software,
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
    feature_cache: Path = typer.Argument(...),
    out_dir: Path = typer.Argument(...),
    recipe: Path = typer.Option(...),
    launcher: str = typer.Option("standalone"),
    local_cache_dir: Path | None = typer.Option(None),
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
