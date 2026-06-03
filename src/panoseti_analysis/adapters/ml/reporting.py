"""Shared W&B media logging for model trainers (Layer B).

Delegates figure rendering to :mod:`panoseti_analysis.adapters.ml.viz` (the
single source of truth for figure layout), then calls ``tracker.log_image`` /
``tracker.log_artifact`` to push the result to W&B.

Everything is best-effort: a missing matplotlib (or any rendering hiccup) must
never fail a training run, so callers should treat these as fire-and-forget.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from panoseti_analysis.adapters.ml.tracking import Tracker
from panoseti_analysis.adapters.ml.viz import (
    confusion_matrix_fig,
    histogram_fig,
    pr_curve_fig,
)

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
        fig = confusion_matrix_fig(cm, class_names=class_names)
        if fig is not None:
            tracker.log_image(key, fig, step=step)
            import matplotlib.pyplot as plt

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
    try:
        fig = pr_curve_fig(pr)
        if fig is not None:
            tracker.log_image(key, fig, step=step)
            import matplotlib.pyplot as plt

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
        fig = histogram_fig(values, title=title, xlabel=xlabel, bins=bins)
        if fig is not None:
            tracker.log_image(key, fig, step=step)
            import matplotlib.pyplot as plt

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
