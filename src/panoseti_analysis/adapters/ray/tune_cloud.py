"""pa-tune-cloud — hyperparameter sweep for the cloud detector via Ray Tune (Layer B).

Reuses the **unchanged** ``train_loop_per_worker`` from ``train_cloud.py``; the sweep
space is defined in the recipe ``tune:`` block.  The winning trial's weights are saved
with the same ``save_classifier`` path as a normal ``pa-train-cloud`` run so the output
is identical in structure (traceable to recipe_hash via TrainingProvenance).

Recipe ``tune:`` block format:

    tune:
      num_samples: 8          # Ray Tune trials (ignored for pure grid_search)
      use_asha: true          # ASHA early-stopping (default true)
      lr:
        loguniform: [1e-4, 1e-2]
      batch_size:
        choice: [64, 128, 256]
      gamma:
        grid: [0.85, 0.9, 0.95]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

import panoseti_analysis.algorithms.cloud_detector  # noqa: F401 — register models
from panoseti_analysis.adapters.ml.runner import run_torch_tuner
from panoseti_analysis.adapters.ray._staging import stage_to_local
from panoseti_analysis.adapters.ray.train_cloud import train_loop_per_worker
from panoseti_analysis.algorithms.registry import build_model
from panoseti_analysis.config.models import ClassifierBundle, TrainConfig, TrainingProvenance
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.models import save_classifier
from panoseti_analysis.io.provenance import capture_software, now_utc, read_history
from panoseti_analysis.io.stores import open_store

app = typer.Typer(add_completion=False, help="Sweep cloud-detector hyperparams via Ray Tune.")


def run_tune_cloud(
    feature_cache: Path,
    out_dir: Path,
    recipe_path: Path,
    launcher: str = "standalone",
    local_cache_dir: Path | None = None,
    lineage_out: Path | None = None,
    wandb_project: str | None = None,
) -> tuple[Path, Path, Path]:
    """Run a Ray Tune sweep over the cloud detector and save the best model.

    Args:
        feature_cache: Path to the feature cache zarr on shared FS.
        out_dir: Output directory for the best model bundle.
        recipe_path: YAML recipe (must contain ``hyperparams`` + ``scaling`` + ``tune``).
        launcher: Ray init mode (attach/slurm/standalone).
        local_cache_dir: If set, pre-stages the feature cache to local SSD first.
        lineage_out: If set, writes a lineage JSON record.
        wandb_project: W&B project name override.

    Returns:
        (pt_path, json_path, provenance_path)
    """
    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    hp = params_dict.get("hyperparams", {})
    scaling_cfg = params_dict.get("scaling", {})
    tune_block = params_dict.get("tune", {})
    if not tune_block:
        raise ValueError(
            f"Recipe {recipe_path} has no 'tune:' block. "
            "Use pa-train-cloud for single-config training."
        )
    num_samples = int(tune_block.pop("num_samples", 1))
    use_asha = bool(tune_block.pop("use_asha", True))
    metric = str(tune_block.pop("metric", "val_loss"))
    tune_mode = str(tune_block.pop("mode", "min"))
    # Remaining keys are the search space parameters.

    effective_path = feature_cache
    if local_cache_dir is not None:
        effective_path = stage_to_local(feature_cache, local_cache_dir)
    effective_path = Path(effective_path).resolve()

    ds_feat = open_store(effective_path)
    _ = read_history(dict(ds_feat.attrs))
    feat_cksum = checksum_store(feature_cache)

    base_config: dict[str, Any] = {
        "local_feature_path": str(effective_path),
        "recipe_name": recipe_name,
        **hp,
    }
    for k in ("wandb_project", "wandb_entity"):
        if k in params_dict:
            base_config[k] = params_dict[k]
    if wandb_project is not None:
        base_config["wandb_project"] = wandb_project

    best_state, best_cfg, metrics = run_torch_tuner(
        train_loop_per_worker,
        base_config,
        tune_block=tune_block,
        scaling_cfg=scaling_cfg,
        launcher=launcher,
        out_dir=out_dir,
        num_samples=num_samples,
        metric=metric,
        mode=tune_mode,
        use_asha=use_asha,
    )

    # Merge the winning hyperparams back into base_config for provenance.
    final_hp = {**hp, **best_cfg}
    train_cfg = TrainConfig.model_validate(final_hp)
    model = build_model(train_cfg.arch)
    model.load_state_dict(best_state)

    model_name = f"{recipe_name}_tune_{train_cfg.arch}".replace("/", "_")
    bundle = ClassifierBundle(
        model_name=model_name,
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={"channels": 2, "height": 32, "width": 32, "dtype": "float32"},
        arch=train_cfg.arch,
    )

    prov = TrainingProvenance(
        step_name="tune_cloud",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        params={**final_hp, "scaling": scaling_cfg, "tune_winner": best_cfg},
        input_checksums=[feat_cksum],
        timestamp_utc=now_utc(),
        software=capture_software(),
        metrics=metrics,
    )

    pt_path, json_path, provenance_path = save_classifier(model, bundle, prov, out_dir)

    if lineage_out is not None:
        record = {
            "step_name": "tune_cloud",
            "recipe_name": recipe_name,
            "recipe_hash": recipe_hash,
            "model_bundle": json_path.name,
            "best_config": best_cfg,
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
    run_tune_cloud(
        feature_cache,
        out_dir,
        recipe,
        launcher=launcher,
        local_cache_dir=local_cache_dir,
        lineage_out=lineage_out,
    )


if __name__ == "__main__":
    app()
