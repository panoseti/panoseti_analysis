"""Tests for the shared training engine (algorithms/training.py) and cloud hooks."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from panoseti_analysis.algorithms.cloud_detector import (  # noqa: E402
    CloudDetection,
    CloudDetectionV2,
)
from panoseti_analysis.algorithms.cloud_train import (  # noqa: E402
    binary_classification_metrics,
    cloud_val_predictions,
    confusion_counts,
    fit_cloud_detector,
    pr_curve_points,
)
from panoseti_analysis.algorithms.training import TrainResult, fit  # noqa: E402

CPU = torch.device("cpu")


def test_fit_reduces_loss_on_separable_data() -> None:
    """The shared `fit` loop must drive training loss down on a separable problem."""
    torch.manual_seed(0)
    n = 256
    x = torch.randn(n, 4)
    w = torch.tensor([2.0, -2.0, 1.0, -1.0])
    y = (x @ w > 0).long()
    model = torch.nn.Linear(4, 2)
    loader: DataLoader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    def loss_fn(m: torch.nn.Module, batch: object) -> tuple[torch.Tensor, dict[str, float]]:
        xb, yb = batch  # type: ignore[misc]
        loss = F.cross_entropy(m(xb), yb)
        return loss, {"loss": loss.item()}

    def val_eval(m: torch.nn.Module) -> dict[str, float]:
        return {"val_loss": float(F.cross_entropy(m(x), y).item())}

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    result = fit(
        model,
        loader,
        loss_fn=loss_fn,
        optimizer=opt,
        epochs=25,
        device=CPU,
        val_eval=val_eval,
        monitor="val_loss",
    )

    assert isinstance(result, TrainResult)
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.best_value <= result.history[0]["val_loss"]
    # best_state must hold the (unwrapped) model's parameters.
    assert set(result.best_state) == set(model.state_dict())


def test_fit_falls_back_to_final_state_without_val() -> None:
    """With no val_eval / monitor, fit still returns a usable final state dict."""
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    model = torch.nn.Linear(4, 2)
    loader: DataLoader = DataLoader(TensorDataset(x, y), batch_size=16)

    def loss_fn(m: torch.nn.Module, batch: object) -> tuple[torch.Tensor, dict[str, float]]:
        xb, yb = batch  # type: ignore[misc]
        loss = F.cross_entropy(m(xb), yb)
        return loss, {"loss": loss.item()}

    result = fit(
        model,
        loader,
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=3,
        device=CPU,
        monitor="val_loss",
    )
    assert set(result.best_state) == set(model.state_dict())
    assert len(result.history) == 3


def test_fit_cloud_detector_smoke() -> None:
    """fit_cloud_detector runs end-to-end and yields a loadable CloudDetection state."""
    torch.manual_seed(0)
    x_train = torch.randn(48, 2, 32, 32)
    y_train = torch.randint(0, 2, (48,))
    x_val = torch.randn(16, 2, 32, 32)
    y_val = torch.randint(0, 2, (16,))
    hp = {"lr": 1e-3, "batch_size": 8, "epochs": 2, "weight_decay": 1e-5, "gamma": 0.9}

    result = fit_cloud_detector(x_train, y_train, x_val, y_val, hp, device=CPU)

    assert result.history, "expected per-epoch history"
    last = result.history[-1]
    for key in ("val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "val_ap"):
        assert key in last, f"missing metric {key}"
    # best_state must load cleanly into a fresh model (default arch is cloud_detector_v2).
    CloudDetectionV2().load_state_dict(result.best_state)


def test_binary_classification_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.8, 0.9])
    m = binary_classification_metrics(y_true, y_pred, y_score)
    assert m["val_recall"] == 1.0
    assert m["val_precision"] == pytest.approx(2 / 3)
    assert m["val_f1"] == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))
    assert 0.0 <= m["val_ap"] <= 1.0


def test_confusion_counts() -> None:
    cm = confusion_counts(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    # [[tn, fp], [fn, tp]] = [[1, 1], [0, 2]]
    assert cm == [[1, 1], [0, 2]]


def test_pr_curve_points() -> None:
    pr = pr_curve_points(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.8, 0.9]))
    # Sorted by descending score: labels [1, 1, 0, 0] → recall climbs to 1, precision starts at 1.
    assert pr["recall"][0] == pytest.approx(0.5)
    assert pr["precision"][0] == pytest.approx(1.0)
    assert pr["recall"][-1] == pytest.approx(1.0)
    assert len(pr["recall"]) == len(pr["precision"])


def test_pr_curve_points_no_positives() -> None:
    pr = pr_curve_points(np.array([0, 0, 0]), np.array([0.2, 0.5, 0.9]))
    assert pr == {"recall": [], "precision": []}


def test_cloud_val_predictions_shapes() -> None:
    torch.manual_seed(0)
    x_val = torch.randn(8, 2, 32, 32)
    y_val = torch.randint(0, 2, (8,))
    state = CloudDetection().state_dict()
    y_true, y_pred, y_score = cloud_val_predictions(state, x_val, y_val, CPU)
    assert y_true.shape == (8,)
    assert y_pred.shape == (8,)
    assert y_score.shape == (8,)
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert ((y_score >= 0) & (y_score <= 1)).all()
