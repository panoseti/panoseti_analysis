"""Cloud-detection research bench — the *volatile* layer.

This module is meant to be edited freely from the notebooks (run with ``%autoreload 2``):
quick training wrappers, plotting, device selection. It composes the *trusted, tested*
package layer and never reimplements it:

- epoch loop ......... ``panoseti_analysis.algorithms.training.fit``
- model + hooks ...... ``panoseti_analysis.algorithms.cloud_train`` / ``.cloud_detector``
- data loading ....... ``panoseti_analysis.adapters.ml.data.load_labeled_feature_cache``
- repo paths ......... ``panoseti_analysis.paths``

Keep durable, reusable logic in the package (guarded by tests); keep experiments here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from panoseti_analysis.adapters.ml.data import load_labeled_feature_cache
from panoseti_analysis.algorithms.cloud_detector import CloudDetection
from panoseti_analysis.algorithms.cloud_train import (
    cloud_val_predictions,
    confusion_counts,
    fit_cloud_detector,
    pr_curve_points,
)
from panoseti_analysis.algorithms.training import TrainResult


def get_device() -> torch.device:
    """Pick the best available device (never hard-coded cuda)."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def load_features(feature_cache: str | Path) -> tuple[torch.Tensor, ...]:
    """``(X_train, y_train, X_val, y_val)`` from a labelled feature cache."""
    return load_labeled_feature_cache(Path(feature_cache))


def quick_train(
    feature_cache: str | Path,
    hp: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    on_epoch: Any = None,
) -> tuple[CloudDetection, TrainResult]:
    """Load a feature cache and train a CloudDetection model — returns (best model, result).

    ``hp`` overrides defaults (lr/batch_size/epochs/weight_decay/gamma). The returned model
    has the *best* checkpoint loaded, ready for evaluation/plotting.
    """
    device = device or get_device()
    defaults = {"lr": 1e-3, "batch_size": 128, "epochs": 20, "weight_decay": 1e-5, "gamma": 0.9}
    defaults.update(hp or {})

    x_train, y_train, x_val, y_val = load_features(feature_cache)
    result = fit_cloud_detector(
        x_train, y_train, x_val, y_val, defaults, device=device, on_epoch=on_epoch
    )
    model = CloudDetection().to(device)
    model.load_state_dict(result.best_state)
    model.eval()
    return model, result


def plot_history(result: TrainResult, keys: tuple[str, ...] = ("train_loss", "val_loss")) -> None:
    """Line-plot selected metrics across epochs from a TrainResult.history."""
    epochs = [int(h["epoch"]) for h in result.history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for key in keys:
        ys = [h.get(key) for h in result.history]
        if any(y is not None for y in ys):
            ax.plot(epochs, ys, marker=".", label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_eval(
    model: CloudDetection,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device | None = None,
) -> None:
    """Confusion matrix + PR curve for a trained model on the validation set."""
    device = device or get_device()
    y_true, y_pred, y_score = cloud_val_predictions(model.state_dict(), x_val, y_val, device)
    cm = np.asarray(confusion_counts(y_true, y_pred))
    pr = pr_curve_points(y_true, y_score)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 3.5))
    im = ax0.imshow(cm, cmap="Blues")
    ax0.set_xticks([0, 1], labels=["pred clear", "pred cloudy"])
    ax0.set_yticks([0, 1], labels=["true clear", "true cloudy"])
    for i in range(2):
        for j in range(2):
            ax0.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax0.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax0, fraction=0.046)

    if pr["recall"]:
        ax1.plot(pr["recall"], pr["precision"], marker=".", color="steelblue")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Precision–Recall")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
