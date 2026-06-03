"""Shared ML research bench — batteries-included helpers for notebook R&D (Layer B).

This is the **single definition** of the helper functions that used to be copy-pasted
into every ``ml/<model>/notebooks/lab.py``.  Per-model ``lab.py`` files import from
here and only add genuinely model-specific experiment code.

Public API:
    ``get_device()``               — auto-select best available torch device
    ``quick_train(cache, cfg)``    — load feature cache, train, return (model, result)
    ``plot_history(result, keys)`` — display training-curve plot in a notebook
    ``plot_eval(model, x, y)``     — display confusion matrix + PR curve in a notebook

All heavy imports (matplotlib, torch) are deferred or guarded so this module imports
quickly on worker nodes where display is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    """Pick the best available torch device (never hard-coded cuda)."""
    if torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            return acc
    return torch.device("cpu")


def quick_train(
    feature_cache: str | Path,
    hp: dict[str, Any] | None = None,
    *,
    device: torch.device | None = None,
    on_epoch: Any = None,
) -> tuple[torch.nn.Module, Any]:
    """Load a labeled feature cache, train a model, return (best model, TrainResult).

    ``hp`` overrides defaults.  ``hp["arch"]`` selects the model from the registry
    (default ``"cloud_detector_v2"``).  The returned model has the best checkpoint
    loaded and is set to eval mode.

    Example::

        model, result = quick_train(FEATURE_CACHE, {"epochs": 5, "arch": "cloud_detector_v2"})
        plot_history(result)
        plot_eval(model, x_val, y_val)
    """
    from panoseti_analysis.adapters.ml.data import load_labeled_feature_cache
    from panoseti_analysis.algorithms.cloud_train import fit_cloud_detector
    from panoseti_analysis.algorithms.registry import build_model

    device = device or get_device()
    defaults: dict[str, Any] = {
        "arch": "cloud_detector_v2",
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 20,
        "weight_decay": 1e-5,
        "gamma": 0.9,
    }
    defaults.update(hp or {})

    x_train, y_train, x_val, y_val = load_labeled_feature_cache(Path(feature_cache))
    result = fit_cloud_detector(
        x_train, y_train, x_val, y_val, defaults, device=device, on_epoch=on_epoch
    )
    model = build_model(defaults["arch"]).to(device)
    model.load_state_dict(result.best_state)
    model.eval()
    return model, result


def plot_history(result: Any, keys: tuple[str, ...] = ("train_loss", "val_loss")) -> None:
    """Display a training-history curve in a notebook (calls plt.show())."""
    import matplotlib.pyplot as plt

    from panoseti_analysis.adapters.ml.viz import history_curves

    fig = history_curves(result.history, keys=keys)
    if fig is not None:
        plt.show()
        plt.close(fig)


def plot_eval(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device | None = None,
) -> None:
    """Display a confusion matrix + PR curve for a trained model on the validation set."""
    import matplotlib.pyplot as plt

    from panoseti_analysis.adapters.ml.viz import confusion_matrix_fig, pr_curve_fig
    from panoseti_analysis.algorithms.cloud_train import (
        cloud_val_predictions,
        confusion_counts,
        pr_curve_points,
    )

    device = device or get_device()
    y_true, y_pred, y_score = cloud_val_predictions(model.state_dict(), x_val, y_val, device)
    for fig in (
        confusion_matrix_fig(confusion_counts(y_true, y_pred), class_names=("clear", "cloudy")),
        pr_curve_fig(pr_curve_points(y_true, y_score)),
    ):
        if fig is not None:
            plt.show()
            plt.close(fig)
