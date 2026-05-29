"""Tests for the experiment-tracking abstraction (adapters/ray/_tracking.py)."""
from __future__ import annotations

from panoseti_analysis.adapters.ray._tracking import (
    Tracker,
    _NoOpTracker,
    make_tracker,
)


def test_make_tracker_returns_noop_without_config() -> None:
    """make_tracker({}) should return some Tracker instance (W&B and TB both absent)."""
    tracker = make_tracker({})
    assert isinstance(tracker, Tracker)


def test_tracker_log_does_not_raise() -> None:
    """Calling log() on any tracker returned by make_tracker should not raise."""
    tracker = make_tracker({})
    tracker.log({"loss": 0.1})  # no step
    tracker.log({"loss": 0.05}, step=1)


def test_tracker_finish_does_not_raise() -> None:
    """Calling finish() on any tracker returned by make_tracker should not raise."""
    tracker = make_tracker({})
    tracker.finish()


def test_noop_tracker_works() -> None:
    """_NoOpTracker.log and .finish should be silent no-ops."""
    tracker = _NoOpTracker()
    tracker.log({"accuracy": 0.99, "epoch": 5})
    tracker.log({"accuracy": 0.999}, step=10)
    tracker.finish()
    # If we got here without exception the test passes
