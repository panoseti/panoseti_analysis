"""Shared W&B media logging for model trainers (Layer B).

These helpers render figures with matplotlib and log them through the *generic*
:class:`~panoseti_analysis.adapters.ml.tracking.Tracker` plumbing (``log_image`` /
``log_artifact``), so they work for any model and degrade gracefully on non-W&B backends.
The pure report data (confusion-matrix counts, PR-curve points) is built by Layer-A kernels
(e.g. ``algorithms/cloud_train.py``); this module only turns that data into pictures.

Everything is best-effort: a missing matplotlib (or any rendering hiccup) must never fail a
training run, so callers should treat these as fire-and-forget.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from panoseti_analysis.adapters.ml.tracking import Tracker

logger = logging.getLogger(__name__)


def log_confusion_matrix(
    tracker: Tracker,
    cm: Sequence[Sequence[int]],
    *,
    class_names: Sequence[str] = ("clear", "cloudy"),
    key: str = "val/confusion_matrix",
    step: int | None = None,
) -> None:
    """Render a confusion-matrix heatmap (counts) and log it as an image."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display on Ray workers / CI
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.asarray(cm)
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks(range(len(class_names)), labels=[f"pred {c}" for c in class_names])
        ax.set_yticks(range(len(class_names)), labels=[f"true {c}" for c in class_names])
        thresh = arr.max() / 2 if arr.size else 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(
                    j,
                    i,
                    str(int(arr[i, j])),
                    ha="center",
                    va="center",
                    color="white" if arr[i, j] > thresh else "black",
                )
        ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        tracker.log_image(key, fig, step=step)
        plt.close(fig)
    except Exception:  # pragma: no cover - best-effort media logging
        logger.warning("Skipped confusion-matrix logging", exc_info=True)


def log_pr_curve(
    tracker: Tracker,
    pr: dict[str, list[float]],
    *,
    key: str = "val/pr_curve",
    step: int | None = None,
) -> None:
    """Render a precision–recall curve and log it as an image."""
    if not pr.get("recall"):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(pr["recall"], pr["precision"], marker=".", linewidth=1, color="steelblue")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_title("Precision–Recall")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        tracker.log_image(key, fig, step=step)
        plt.close(fig)
    except Exception:  # pragma: no cover - best-effort media logging
        logger.warning("Skipped PR-curve logging", exc_info=True)


def log_histogram(
    tracker: Tracker,
    values: Any,
    *,
    key: str = "val/hist",
    title: str = "",
    xlabel: str = "value",
    bins: int = 50,
    step: int | None = None,
) -> None:
    """Render a 1-D histogram (e.g. VAE reconstruction errors) and log it as an image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.asarray(values).ravel()
        if arr.size == 0:
            return
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.hist(arr, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        tracker.log_image(key, fig, step=step)
        plt.close(fig)
    except Exception:  # pragma: no cover - best-effort media logging
        logger.warning("Skipped histogram logging", exc_info=True)


def log_state_dict_artifact(
    tracker: Tracker,
    state_dict: dict[str, Any],
    *,
    name: str = "model",
    type_: str = "model",
) -> None:
    """Persist a state_dict to a temp ``.pt`` and upload it as a tracked artifact."""
    try:
        import torch

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{name}.pt"
            torch.save(state_dict, path)
            tracker.log_artifact(path, name=name, type_=type_)
    except Exception:  # pragma: no cover - best-effort artifact logging
        logger.warning("Skipped model-artifact logging", exc_info=True)
