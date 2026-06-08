"""Tests for adapters/ml/reporting.py — W&B media logging glue.

A fake Tracker records calls so we can assert the helpers route confusion matrices / PR
curves through ``log_image`` and model weights through ``log_artifact``, and that they stay
best-effort (never raise) when something goes wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from panoseti_analysis.adapters.ml.reporting import (
    log_confusion_matrix,
    log_pr_curve,
    log_state_dict_artifact,
)
from panoseti_analysis.adapters.ml.tracking import Tracker


class _FakeTracker(Tracker):
    def __init__(self) -> None:
        self.images: list[tuple[str, int | None]] = []
        self.artifacts: list[tuple[str, str | None, str]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        pass

    def finish(self) -> None:
        pass

    def log_image(
        self, key: str, image: Any, *, step: int | None = None, caption: str | None = None
    ) -> None:
        self.images.append((key, step))

    def log_artifact(self, path: Path, *, name: str | None = None, type_: str = "model") -> None:
        self.artifacts.append((Path(path).name, name, type_))


def test_log_confusion_matrix_logs_one_image() -> None:
    pytest.importorskip("matplotlib")
    tracker = _FakeTracker()
    log_confusion_matrix(tracker, [[5, 1], [2, 4]])
    assert len(tracker.images) == 1
    assert tracker.images[0][0] == "val/confusion_matrix"


def test_log_pr_curve_logs_one_image() -> None:
    pytest.importorskip("matplotlib")
    tracker = _FakeTracker()
    log_pr_curve(tracker, {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.8, 0.6]})
    assert len(tracker.images) == 1
    assert tracker.images[0][0] == "val/pr_curve"


def test_log_pr_curve_skips_empty() -> None:
    tracker = _FakeTracker()
    log_pr_curve(tracker, {"recall": [], "precision": []})
    assert tracker.images == []


def test_log_state_dict_artifact() -> None:
    torch = pytest.importorskip("torch")
    tracker = _FakeTracker()
    log_state_dict_artifact(tracker, {"w": torch.zeros(3)}, name="cloud_detector")
    assert tracker.artifacts == [("cloud_detector.pt", "cloud_detector", "model")]


def test_reporting_is_best_effort_on_bad_input() -> None:
    """A rendering error must be swallowed, not raised, so it never fails a training run."""
    pytest.importorskip("matplotlib")
    tracker = _FakeTracker()
    # A ragged confusion matrix would blow up np.asarray / imshow; helper must not raise.
    log_confusion_matrix(tracker, [[1, 2], [3]])  # type: ignore[list-item]
    assert tracker.images == []  # nothing logged, but no exception
