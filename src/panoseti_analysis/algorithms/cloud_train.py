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

from panoseti_analysis.algorithms.cloud_detector import CloudDetectionV2
from panoseti_analysis.algorithms.training import TrainResult, fit


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


def binary_classification_metrics(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float]:
    """Precision/recall/F1 + average precision for the positive (cloudy) class.

    Pure-numpy so it stays Layer-A-clean (no sklearn). ``y_score`` is P(cloudy).
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Average precision = area under the precision-recall curve (step interpolation).
    order = np.argsort(-np.asarray(y_score))
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    ap = 0.0
    if total_pos:
        cum_tp = np.cumsum(y_sorted)
        cum_fp = np.cumsum(1 - y_sorted)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        rec = cum_tp / total_pos
        rec_prev = np.concatenate([[0.0], rec[:-1]])
        ap = float(np.sum((rec - rec_prev) * prec))
    return {"val_precision": precision, "val_recall": recall, "val_f1": f1, "val_ap": ap}


def confusion_counts(y_true: Any, y_pred: Any) -> list[list[int]]:
    """2x2 confusion matrix ``[[tn, fp], [fn, tp]]`` for binary (clear/cloudy) labels."""
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return [[tn, fp], [fn, tp]]


def pr_curve_points(y_true: Any, y_score: Any, *, n_points: int = 50) -> dict[str, list[float]]:
    """Precision/recall points along descending-score thresholds, down-sampled to n_points.

    Pure numpy (no sklearn) so it stays Layer-A-clean. ``y_score`` is P(cloudy).
    """
    import numpy as np

    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-np.asarray(y_score))
    y_sorted = y_true[order]
    total_pos = int(y_sorted.sum())
    if total_pos == 0:
        return {"recall": [], "precision": []}
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / total_pos
    idx = np.linspace(0, len(recall) - 1, num=min(n_points, len(recall))).astype(int)
    return {"recall": recall[idx].tolist(), "precision": precision[idx].tolist()}


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

    model = CloudDetectionV2().to(device)
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
    """AdamW + ExponentialLR + ReduceLROnPlateau, matching the production schedule."""
    lr = float(hp.get("lr", 1e-3))
    weight_decay = float(hp.get("weight_decay", 1e-5))
    gamma = float(hp.get("gamma", 0.9))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    def step_schedulers(val_loss: float) -> None:
        scheduler_exp.step()
        scheduler_plateau.step(val_loss)

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
        model = CloudDetectionV2().to(device)
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
