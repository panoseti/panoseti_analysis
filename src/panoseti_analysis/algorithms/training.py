"""Framework-free training engine shared by every model (Layer A, pure torch).

One flat ``fit`` loop; model-specific behaviour is injected as callbacks (``loss_fn``,
``val_eval``, ``step_schedulers``). No ray / zarr / typer imports — the Ray training
adapter (``adapters/ml/runner.py``) and notebooks both call this exact loop, so the epoch
logic lives in a single editable place.

Device is always passed in (never hard-coded), so the same loop runs on CPU in a notebook
and on a Ray-assigned GPU in a worker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

# A batch is whatever the DataLoader yields — a tensor or a sequence of tensors.
Batch = Any
#: ``loss_fn(model, batch) -> (loss_tensor, {name: scalar})``. The scalars are averaged
#: across batches and surfaced as ``train_<name>`` in the per-epoch metrics.
LossFn = Callable[[torch.nn.Module, Batch], "tuple[torch.Tensor, dict[str, float]]"]
#: ``val_eval(model) -> {name: scalar}``. Called under ``model.eval()`` + ``no_grad``.
ValEval = Callable[[torch.nn.Module], "dict[str, float]"]


@dataclass
class TrainResult:
    """Outcome of a :func:`fit` run."""

    best_state: dict[str, torch.Tensor]
    best_value: float
    monitor: str
    history: list[dict[str, float]] = field(default_factory=list)


def _move(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Unwrap DDP (torch wraps the model and prefixes keys with "module.").
    raw = getattr(model, "module", model)
    # Move to CPU so the checkpoint is device-agnostic and doesn't pin GPU memory.
    return {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader[Any],
    *,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    val_eval: ValEval | None = None,
    step_schedulers: Callable[[float], None] | None = None,
    on_epoch_start: Callable[[int], None] | None = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
    monitor: str = "val_loss",
    minimize: bool = True,
) -> TrainResult:
    """Train ``model`` for ``epochs``, tracking the best checkpoint by ``monitor``.

    Args:
        model: The network. May be a bare ``nn.Module`` or a DDP-wrapped one.
        train_loader: Yields training batches (moved to ``device`` automatically).
        loss_fn: Computes the per-batch loss and a dict of scalar logs.
        optimizer: Pre-built optimizer (the caller chooses AdamW/Adam/etc.).
        epochs: Number of epochs.
        device: Target device (CPU or an accelerator); never hard-coded here.
        val_eval: Optional per-epoch validation; its dict is merged into the metrics
            and should contain ``monitor`` (e.g. ``"val_loss"``).
        step_schedulers: Optional callback stepping LR schedulers with the monitored value.
        on_epoch_start: Optional callback ``(epoch)`` invoked at the *start* of each epoch
            **before** iterating the loader.  The Ray adapter uses this to call
            ``train_loader.sampler.set_epoch(epoch)`` for correct multi-GPU shuffle
            (see ``adapters/ray/train_cloud.py``).  Framework-free: the callback is a
            plain Python callable, so Layer A stays pure.
        on_epoch: Optional callback ``(epoch, metrics)`` for logging / ``ray.train.report``.
        monitor: Metric key used to select the best checkpoint.
        minimize: Whether lower ``monitor`` is better.

    Returns:
        A :class:`TrainResult` with the best (DDP-unwrapped, CPU) state dict and full history.
    """
    best_value = float("inf") if minimize else float("-inf")
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []

    def is_better(value: float) -> bool:
        return value < best_value if minimize else value > best_value

    for epoch in range(epochs):
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad()
            loss, logs = loss_fn(model, batch)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            for key, value in logs.items():
                sums[key] = sums.get(key, 0.0) + float(value)
            n_batches += 1

        metrics: dict[str, float] = {"epoch": float(epoch)}
        for key, total in sums.items():
            metrics[f"train_{key}"] = total / max(n_batches, 1)
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        if val_eval is not None:
            model.eval()
            with torch.no_grad():
                metrics.update(val_eval(model))

        monitored = metrics.get(monitor)
        if step_schedulers is not None:
            step_schedulers(monitored if monitored is not None else metrics.get("train_loss", 0.0))

        if monitored is not None and is_better(monitored):
            best_value = monitored
            best_state = _clone_state(model)

        if on_epoch is not None:
            on_epoch(epoch, metrics)
        history.append(metrics)

    # Fall back to the final weights if nothing was ever monitored (e.g. no val_eval).
    if not best_state:
        best_state = _clone_state(model)

    return TrainResult(
        best_state=best_state, best_value=best_value, monitor=monitor, history=history
    )
