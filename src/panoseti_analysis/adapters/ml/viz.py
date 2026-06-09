"""Model-agnostic matplotlib figure builders (Layer B).

Returns ``matplotlib.figure.Figure`` objects — the caller decides whether to
``plt.show()`` (notebooks) or ``tracker.log_image(...)`` (W&B / Ray workers).

Everything is best-effort: callers should treat figures as optional.  Import
matplotlib lazily so the module loads on headless workers without a display.

This consolidates:
- ``adapters/ml/reporting.py`` — W&B logging side (calls these builders)
- ``ml/.../lab.py::plot_history`` / ``plot_eval`` — notebook display side
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def history_curves(
    history: list[dict[str, float]],
    keys: tuple[str, ...] = ("train_loss", "val_loss"),
    *,
    title: str = "Training history",
) -> Any:
    """Line-plot selected metric keys from a ``TrainResult.history`` list.

    Returns a ``matplotlib.figure.Figure``, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(h.get("epoch", i)) for i, h in enumerate(history)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for key in keys:
            ys_raw = [h.get(key) for h in history]
            if any(y is not None for y in ys_raw):
                ys: list[float] = [float(y) if y is not None else float("nan") for y in ys_raw]
                ax.plot(epochs, ys, marker=".", label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:  # pragma: no cover
        logger.warning("history_curves: skipped", exc_info=True)
        return None


def confusion_matrix_fig(
    cm: Sequence[Sequence[int]],
    *,
    class_names: Sequence[str] = ("negative", "positive"),
    title: str = "Confusion matrix",
) -> Any:
    """Render a confusion-matrix heatmap (counts).

    Returns a ``matplotlib.figure.Figure``, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
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
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig
    except Exception:  # pragma: no cover
        logger.warning("confusion_matrix_fig: skipped", exc_info=True)
        return None


def pr_curve_fig(
    pr: dict[str, list[float]],
    *,
    title: str = "Precision–Recall",
) -> Any:
    """Render a precision–recall curve.

    Returns a ``matplotlib.figure.Figure``, or ``None`` if unavailable / empty.
    """
    if not pr.get("recall"):
        return None
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
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:  # pragma: no cover
        logger.warning("pr_curve_fig: skipped", exc_info=True)
        return None


def histogram_fig(
    values: Any,
    *,
    title: str = "",
    xlabel: str = "value",
    bins: int = 50,
) -> Any:
    """Render a 1-D histogram (e.g. VAE reconstruction errors).

    Returns a ``matplotlib.figure.Figure``, or ``None`` if unavailable / empty.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.asarray(values).ravel()
        if arr.size == 0:
            return None
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.hist(arr, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig
    except Exception:  # pragma: no cover
        logger.warning("histogram_fig: skipped", exc_info=True)
        return None
