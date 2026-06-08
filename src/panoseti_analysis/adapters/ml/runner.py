"""Generic Ray ``TorchTrainer`` runner shared by the model trainers (Layer B).

Owns the boilerplate that was duplicated across ``train_cloud.py`` and ``train_vae.py``:
build the ``ScalingConfig`` / ``RunConfig`` / ``runtime_env``, init Ray, run the trainer,
and restore the best checkpoint. Model-specific concerns (bundle, provenance, save) stay in
the per-model adapter, which calls this and then persists the returned ``best_state``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import torch


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
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from panoseti_analysis.adapters.ray.launcher import init_ray

    accelerator_type = scaling_cfg.get("accelerator_type", "G")
    num_workers = int(scaling_cfg.get("num_workers", 2))
    use_gpu = num_workers > 0 and torch.cuda.is_available()
    scaling = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        **({"accelerator_type": accelerator_type} if accelerator_type else {}),
    )
    run_cfg = RunConfig(storage_path=str(out_dir / "ray_results"))

    # Workers import panoseti_analysis from the editable install on each node; only the
    # W&B key needs forwarding (no working_dir — see train_cloud.py for the rationale).
    env_vars: dict[str, str] = {}
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    init_ray(
        cast(Literal["attach", "slurm", "standalone"], launcher),
        runtime_env={"env_vars": env_vars},
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling,
        run_config=run_cfg,
    )
    result = trainer.fit()

    checkpoint = result.checkpoint
    assert checkpoint is not None, "Training produced no checkpoint"
    with checkpoint.as_directory() as ckpt_dir:
        best_state = torch.load(
            Path(ckpt_dir) / checkpoint_filename, map_location="cpu", weights_only=True
        )

    raw_metrics: dict[str, Any] = result.metrics or {}
    metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}
    return best_state, metrics


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
