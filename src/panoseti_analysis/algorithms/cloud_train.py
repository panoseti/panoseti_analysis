"""Cloud-detector training: model-specific hooks + a notebook-friendly wrapper.

The epoch loop itself lives in :mod:`panoseti_analysis.algorithms.training`. Here we supply
the cloud-specific pieces — loss, validation metrics, optimizer/scheduler config — plus
:func:`fit_cloud_detector` for plain (non-Ray) training in notebooks. The Ray training
adapter reuses the *same* hooks (``cloud_loss_fn`` / ``make_cloud_val_eval`` /
``build_cloud_optimizer``) so there is one source of truth for how this model trains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import panoseti_analysis.algorithms.cloud_detector  # noqa: F401 — register CloudDetection + V2
from panoseti_analysis.algorithms.classification_metrics import (
    binary_classification_metrics,
    confusion_counts,
    pr_curve_points,
)
from panoseti_analysis.algorithms.optim import build_optimizer, build_scheduler
from panoseti_analysis.algorithms.registry import build_model
from panoseti_analysis.algorithms.training import TrainResult, fit

# Re-export for back-compat (callers that imported these from cloud_train still work).
__all__ = [
    "binary_classification_metrics",
    "build_cloud_optimizer",
    "cloud_loss_fn",
    "cloud_val_predictions",
    "confusion_counts",
    "fit_cloud_detector",
    "make_cloud_val_eval",
    "pr_curve_points",
]


def cloud_loss_fn(model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy on (features, label) batches."""
    xb, yb = batch
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    return loss, {"loss": loss.item()}


def make_cloud_val_eval(
    X_val: torch.Tensor, y_val: torch.Tensor, device: torch.device
) -> Callable[[torch.nn.Module], dict[str, float]]:
    """Build a validation closure returning ``val_loss`` + classification metrics."""

    def val_eval(model: torch.nn.Module) -> dict[str, float]:
        logits = model(X_val.to(device))
        loss = float(F.cross_entropy(logits, y_val.to(device)).item())
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        y_true = y_val.cpu().numpy()
        metrics = {"val_loss": loss, "val_acc": float((preds == y_true).mean())}
        metrics.update(binary_classification_metrics(y_true, preds, probs))
        return metrics

    return val_eval


def cloud_val_predictions(
    state_dict: dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> tuple[Any, Any, Any]:
    """Run a saved state_dict over the val set → ``(y_true, y_pred, y_score)`` numpy arrays.

    Used for end-of-training reporting (confusion matrix / PR curve), so the media reflect
    the *best* checkpoint rather than the last epoch's (possibly DDP-wrapped) live weights.
    """
    import numpy as np

    # Infer arch from the state_dict keys: V2 has "conv_lift" prefix; legacy CNN does not.
    arch = "cloud_detector_v2" if any("conv_lift" in k for k in state_dict) else "cloud_detector"
    model = build_model(arch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
    y_score = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    y_pred = logits.argmax(1).cpu().numpy()
    y_true = np.asarray(y_val.cpu().numpy())
    return y_true, y_pred, y_score


def build_cloud_optimizer(
    model: torch.nn.Module, hp: dict[str, Any]
) -> tuple[torch.optim.Optimizer, Callable[[float], None]]:
    """AdamW + ExponentialLR + ReduceLROnPlateau (backward-compat wrapper).

    Delegates to the generic ``algorithms.optim`` registry so behaviour is identical.
    New callers should use ``build_optimizer`` / ``build_scheduler`` directly.
    """
    optimizer = build_optimizer(model.parameters(), hp)
    step_schedulers = build_scheduler(optimizer, hp)
    return optimizer, step_schedulers


def fit_cloud_detector(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hp: dict[str, Any],
    *,
    device: torch.device,
    model: torch.nn.Module | None = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainResult:
    """Train a :class:`CloudDetection` model with plain PyTorch (no Ray).

    This is the notebook-native entry point; it composes the cloud hooks with the shared
    :func:`~panoseti_analysis.algorithms.training.fit` engine. The Ray worker uses the same
    hooks but wires ``fit`` itself (so it can DDP-wrap the model first).
    """
    if model is None:
        arch = str(hp.get("arch", "cloud_detector_v2"))
        model = build_model(arch).to(device)
    batch_size = int(hp.get("batch_size", 128))
    epochs = int(hp.get("epochs", 50))
    train_loader: DataLoader[Any] = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True
    )
    optimizer, step_schedulers = build_cloud_optimizer(model, hp)
    val_eval = make_cloud_val_eval(X_val, y_val, device)
    return fit(
        model,
        train_loader,
        loss_fn=cloud_loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        val_eval=val_eval,
        step_schedulers=step_schedulers,
        on_epoch=on_epoch,
        monitor="val_loss",
    )
