"""Generic binary-classification metrics (Layer A, pure numpy).

These helpers were originally embedded in ``algorithms/cloud_train.py`` and are
cloud-specific in name only.  Any binary classifier can use them.

Re-exported from ``cloud_train`` for back-compat — existing callers that import
from there continue to work unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_classification_metrics(y_true: Any, y_pred: Any, y_score: Any) -> dict[str, float]:
    """Precision/recall/F1 + average precision for the positive class.

    Pure-numpy so it stays Layer-A-clean (no sklearn). ``y_score`` is P(positive).
    """
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
    """2×2 confusion matrix ``[[tn, fp], [fn, tp]]`` for binary labels."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return [[tn, fp], [fn, tp]]


def pr_curve_points(y_true: Any, y_score: Any, *, n_points: int = 50) -> dict[str, list[float]]:
    """Precision/recall points along descending-score thresholds, down-sampled to n_points.

    Pure numpy (no sklearn). ``y_score`` is P(positive).
    """
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
