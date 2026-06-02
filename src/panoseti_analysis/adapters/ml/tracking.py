"""Experiment tracking shared across all model trainers (Layer B).

W&B is the primary tracker (online, with offline fallback via ``WANDB_MODE=offline``).
TensorBoard is the local fallback; a no-op tracker is the last resort. The interface adds
``log_image`` / ``log_table`` / ``log_artifact`` so trainers and notebooks can log richer
W&B panels (confusion matrices, PR tables, model artifacts) through one indirection.

Usage::

    tracker = make_tracker(config, run_name="cloud_train")
    tracker.log({"val_loss": 0.5, "val_acc": 0.9}, step=epoch)
    tracker.log_artifact(model_path, name="cloud_detector", type_="model")
    tracker.finish()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    """Minimal experiment-tracking interface.

    Scalar logging is required; the media/artifact methods default to no-ops so every
    backend (and the no-op tracker) satisfies the interface without extra code.
    """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    def log_image(
        self, key: str, image: Any, *, step: int | None = None, caption: str | None = None
    ) -> None:
        """Log a single image (numpy array / matplotlib figure / path). No-op by default."""

    def log_table(
        self, key: str, columns: list[str], rows: list[list[Any]], *, step: int | None = None
    ) -> None:
        """Log a small table (e.g. a confusion matrix or PR points). No-op by default."""

    def log_artifact(self, path: Path, *, name: str | None = None, type_: str = "model") -> None:
        """Upload a file artifact (e.g. the trained model bundle). No-op by default."""


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

    def log_image(
        self, key: str, image: Any, *, step: int | None = None, caption: str | None = None
    ) -> None:
        import wandb

        wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_table(
        self, key: str, columns: list[str], rows: list[list[Any]], *, step: int | None = None
    ) -> None:
        import wandb

        wandb.log({key: wandb.Table(columns=columns, data=rows)}, step=step)

    def log_artifact(self, path: Path, *, name: str | None = None, type_: str = "model") -> None:
        import wandb

        artifact = wandb.Artifact(name=name or Path(path).stem, type=type_)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)

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

    If ``config`` contains ``wandb_project``, a W&B tracker is returned (respecting
    ``WANDB_MODE``). Otherwise falls back to TensorBoard, then to a no-op tracker.
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
