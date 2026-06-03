"""Generic Ray ``TorchTrainer`` / ``Tuner`` runner shared by the model trainers (Layer B).

Owns the boilerplate that was duplicated across ``train_cloud.py`` and ``train_vae.py``:
build the ``ScalingConfig`` / ``RunConfig`` / ``runtime_env``, init Ray, run the trainer,
and restore the best checkpoint. Model-specific concerns (bundle, provenance, save) stay in
the per-model adapter, which calls this and then persists the returned ``best_state``.

Two public entry points:
- ``run_torch_trainer`` — single training run (existing behaviour, unchanged).
- ``run_torch_tuner``   — hyperparameter sweep via Ray Tune; search space comes from the
  recipe ``tune:`` block and is translated by ``_search_space_from_recipe``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch

# ── internal helpers ──────────────────────────────────────────────────────────


def _build_scaling_and_run_cfg(scaling_cfg: dict[str, Any], out_dir: Path) -> tuple[Any, Any]:
    """Build ScalingConfig + RunConfig from the recipe ``scaling`` block."""
    from ray.train import RunConfig, ScalingConfig

    accelerator_type = scaling_cfg.get("accelerator_type", "G")
    num_workers = int(scaling_cfg.get("num_workers", 2))
    use_gpu = num_workers > 0 and torch.cuda.is_available()
    scaling = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        **({"accelerator_type": accelerator_type} if accelerator_type else {}),
    )
    # Ray Train v2 requires an absolute storage path (a bare relative path is read as a URI
    # with an empty scheme and rejected by pyarrow). resolve() so a relative --out still works.
    run_cfg = RunConfig(storage_path=str((out_dir / "ray_results").resolve()))
    return scaling, run_cfg


def _init_ray_with_env(launcher: str) -> None:
    """Init Ray with W&B key forwarded; no working_dir needed (editable install on each node)."""
    from panoseti_analysis.adapters.ray.launcher import init_ray

    env_vars: dict[str, str] = {}
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    init_ray(
        cast(Literal["attach", "slurm", "standalone"], launcher),
        runtime_env={"env_vars": env_vars},
    )


def _load_best_state(checkpoint: Any, checkpoint_filename: str) -> dict[str, torch.Tensor]:
    with checkpoint.as_directory() as ckpt_dir:
        return torch.load(
            Path(ckpt_dir) / checkpoint_filename, map_location="cpu", weights_only=True
        )


def _parse_search_space(tune_block: dict[str, Any]) -> dict[str, Any]:
    """Translate the recipe ``tune:`` block into a Ray Tune search space dict.

    Format understood:
        lr:
          uniform: [1e-4, 1e-2]   # tune.uniform(lo, hi)
        batch_size:
          choice: [64, 128, 256]  # tune.choice([...])
        gamma:
          grid: [0.85, 0.9, 0.95] # tune.grid_search([...])
    """
    from ray import tune

    space: dict[str, Any] = {}
    for key, spec in tune_block.items():
        if not isinstance(spec, dict):
            space[key] = spec  # literal / constant
            continue
        if "uniform" in spec:
            lo, hi = spec["uniform"]
            space[key] = tune.uniform(lo, hi)
        elif "choice" in spec:
            space[key] = tune.choice(spec["choice"])
        elif "grid" in spec:
            space[key] = tune.grid_search(spec["grid"])
        elif "loguniform" in spec:
            lo, hi = spec["loguniform"]
            space[key] = tune.loguniform(lo, hi)
        elif "randint" in spec:
            lo, hi = spec["randint"]
            space[key] = tune.randint(lo, hi)
        else:
            raise ValueError(
                f"Unknown tune spec for key {key!r}: {spec}. "
                "Supported: uniform, loguniform, choice, grid, randint"
            )
    return space


# ── public entry points ───────────────────────────────────────────────────────


def run_torch_trainer(
    train_loop_per_worker: Callable[[dict[str, Any]], None],
    train_config: dict[str, Any],
    *,
    scaling_cfg: dict[str, Any],
    launcher: str,
    out_dir: Path,
    checkpoint_filename: str = "model.pt",
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run a Ray Train ``TorchTrainer`` and return ``(best_state, metrics)``.

    Args:
        train_loop_per_worker: The per-worker training function (runs on each worker).
        train_config: Config dict passed to every worker (hyperparams + data path + W&B).
        scaling_cfg: ``num_workers`` / ``accelerator_type`` from the recipe ``scaling`` block.
        launcher: Ray init mode (``attach`` / ``slurm`` / ``standalone``).
        out_dir: Output dir; Ray run artifacts go under ``out_dir/ray_results``.
        checkpoint_filename: Name of the model file the worker saves into its checkpoint dir.

    Returns:
        ``best_state`` (a CPU state dict) and the scalar metrics from the final report.
    """
    from ray.train.torch import TorchTrainer

    scaling, run_cfg = _build_scaling_and_run_cfg(scaling_cfg, out_dir)
    _init_ray_with_env(launcher)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling,
        run_config=run_cfg,
    )
    result = trainer.fit()

    checkpoint = result.checkpoint
    assert checkpoint is not None, "Training produced no checkpoint"
    best_state = _load_best_state(checkpoint, checkpoint_filename)

    raw_metrics: dict[str, Any] = result.metrics or {}
    metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
    return best_state, metrics


def run_torch_tuner(
    train_loop_per_worker: Callable[[dict[str, Any]], None],
    base_config: dict[str, Any],
    *,
    tune_block: dict[str, Any],
    scaling_cfg: dict[str, Any],
    launcher: str,
    out_dir: Path,
    num_samples: int = 1,
    metric: str = "val_loss",
    mode: str = "min",
    checkpoint_filename: str = "model.pt",
    use_asha: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
    """Hyperparameter sweep via Ray Tune + TorchTrainer.

    Wraps the **unchanged** ``train_loop_per_worker`` in a ``Tuner`` — no changes to
    the training function are needed.  The recipe ``tune:`` block defines the search
    space (see ``_parse_search_space`` for the format).

    Args:
        train_loop_per_worker: Same per-worker function used by ``run_torch_trainer``.
        base_config: Fixed config merged under each trial's ``train_loop_config`` (e.g.
            feature-cache path, W&B project, recipe_name).
        tune_block: The YAML ``tune:`` section parsed from the recipe.
        scaling_cfg: ``num_workers`` / ``accelerator_type`` from the recipe.
        launcher: Ray init mode.
        out_dir: Output dir for Ray results.
        num_samples: Number of trials (ignored for pure grid search — Ray counts grids).
        metric: Metric to optimise (default ``"val_loss"``).
        mode: ``"min"`` or ``"max"``.
        checkpoint_filename: Checkpoint file name written by the worker.
        use_asha: Whether to apply ASHA early stopping (recommended for large sweeps).

    Returns:
        ``(best_state, best_config, best_metrics)`` from the winning trial.
    """
    from ray.train.torch import TorchTrainer
    from ray.tune import TuneConfig, Tuner

    scaling, run_cfg = _build_scaling_and_run_cfg(scaling_cfg, out_dir)
    _init_ray_with_env(launcher)

    search_space = _parse_search_space(tune_block)

    scheduler = None
    if use_asha:
        from ray.tune.schedulers import ASHAScheduler

        scheduler = ASHAScheduler(metric=metric, mode=mode)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=base_config,
        scaling_config=scaling,
        run_config=run_cfg,
    )

    tuner = Tuner(
        trainer,
        param_space={"train_loop_config": search_space},
        tune_config=TuneConfig(
            num_samples=num_samples,
            metric=metric,
            mode=mode,
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=metric, mode=mode)

    assert best_result.checkpoint is not None, "Tune produced no checkpoint"
    best_state = _load_best_state(best_result.checkpoint, checkpoint_filename)
    cfg: dict[str, Any] = best_result.config or {}
    best_config: dict[str, Any] = cfg.get("train_loop_config", {})
    raw_metrics: dict[str, Any] = best_result.metrics or {}
    best_metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
    return best_state, best_config, best_metrics


def report_final_checkpoint(best_state: dict[str, torch.Tensor], metrics: dict[str, float]) -> None:
    """From inside a worker: report final metrics, with rank-0 writing the best checkpoint.

    All workers must call this exactly once (it is a collective barrier). Only local rank 0
    materialises the checkpoint, so Ray uploads a single copy of the best weights.
    """
    import shutil
    import tempfile

    import ray.train

    checkpoint: ray.train.Checkpoint | None = None
    ckpt_dir: str | None = None
    if ray.train.get_context().get_local_rank() == 0:
        ckpt_dir = tempfile.mkdtemp(prefix="ray_ckpt_")
        torch.save(best_state, Path(ckpt_dir) / "model.pt")
        checkpoint = ray.train.Checkpoint.from_directory(ckpt_dir)
    ray.train.report(metrics, checkpoint=checkpoint)
    if ckpt_dir is not None:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
