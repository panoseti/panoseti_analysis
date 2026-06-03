"""Cloud-detection research bench — the *volatile* layer.

This module is meant to be edited freely from the notebooks (run with ``%autoreload 2``):
quick training wrappers, plotting, device selection, and cloud-specific experiments.

The batteries-included helpers (get_device, quick_train, plot_history, plot_eval) live in
the shared ``panoseti_analysis.adapters.ml.bench`` module so every model's lab.py composes
the same tested implementation.  Add only cloud-specific experiment code here.

Package layer (do NOT reimplement here):
- epoch loop ......... ``panoseti_analysis.algorithms.training.fit``
- model + hooks ...... ``panoseti_analysis.algorithms.cloud_train`` / ``.cloud_detector``
- data loading ....... ``panoseti_analysis.adapters.ml.data.load_labeled_feature_cache``
- figure builders .... ``panoseti_analysis.adapters.ml.viz``
- shared bench ....... ``panoseti_analysis.adapters.ml.bench``  ← new shared home
- repo paths ......... ``panoseti_analysis.paths``
"""

from __future__ import annotations

import panoseti_analysis.algorithms.cloud_detector  # noqa: F401 — register models

# ── shared batteries-included helpers ─────────────────────────────────────────
from panoseti_analysis.adapters.ml.bench import get_device, plot_eval, plot_history, quick_train

# ── cloud-specific imports ─────────────────────────────────────────────────────
from panoseti_analysis.adapters.ml.data import load_labeled_feature_cache
from panoseti_analysis.algorithms.cloud_train import (
    cloud_val_predictions,
    confusion_counts,
    fit_cloud_detector,
    pr_curve_points,
)
from panoseti_analysis.algorithms.registry import build_model, registered_ids
from panoseti_analysis.algorithms.training import TrainResult

__all__ = [
    "TrainResult",
    "build_model",
    "cloud_val_predictions",
    "confusion_counts",
    "fit_cloud_detector",
    "get_device",
    "load_labeled_feature_cache",
    "plot_eval",
    "plot_history",
    "pr_curve_points",
    "quick_train",
    "registered_ids",
]
