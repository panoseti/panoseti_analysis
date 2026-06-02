"""Experiment tracking abstraction for Ray Train jobs.

W&B is the primary tracker (online, with offline fallback via WANDB_MODE=offline).
TensorBoard is always available as a local fallback.

Usage (inside a Ray train_loop_per_worker):
    tracker = make_tracker(config)
    tracker.log({"loss": 0.5, "epoch": 1})
    tracker.finish()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tracker(ABC):
    """Minimal experiment-tracking interface."""

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class _NoOpTracker(Tracker):
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


class _WandbTracker(Tracker):
    def __init__(
        self,
        run_name: str,
        project: str,
        entity: str | None,
        config: dict[str, Any],
    ) -> None:
        import wandb

        self._run = wandb.init(
            name=run_name,
            project=project,
            entity=entity,
            config=config,
            reinit="finish_previous",
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        import wandb

        wandb.finish()


class _TensorBoardTracker(Tracker):
    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)
        self._step = 0

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        s = step if step is not None else self._step
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(k, v, global_step=s)
        self._step += 1

    def finish(self) -> None:
        self._writer.close()


def make_tracker(
    config: dict[str, Any],
    *,
    run_name: str = "unnamed",
    log_dir: str = "/tmp/tb_logs",
) -> Tracker:
    """Create a tracker from config.

    If ``config`` contains ``wandb_project``, a W&B tracker is returned.
    W&B respects the ``WANDB_MODE`` environment variable — set to ``"offline"``
    for air-gapped nodes; sync later with ``wandb sync``.
    If W&B is unavailable or ``wandb_project`` is absent, falls back to TensorBoard.
    Falls back to NoOp if TensorBoard is also unavailable.
    """
    project = config.get("wandb_project")
    if project:
        try:
            return _WandbTracker(
                run_name=run_name,
                project=str(project),
                entity=config.get("wandb_entity"),
                config=config,
            )
        except Exception:
            pass  # W&B unavailable; fall through to TensorBoard

    try:
        return _TensorBoardTracker(log_dir=log_dir)
    except Exception:
        return _NoOpTracker()
