"""Tests for algorithms/splits.py — time-windowed train/val/test split."""

from __future__ import annotations

import numpy as np
import pytest

from panoseti_analysis.algorithms.splits import data_split


@pytest.fixture
def timestamps() -> np.ndarray:
    """100 monotonically increasing nanosecond timestamps."""
    rng = np.random.default_rng(0)
    base = np.arange(100, dtype=np.int64) * 1_000_000_000
    # shuffle so we test that argsort works on unsorted input
    idx = rng.permutation(100)
    return base[idx]


def test_no_index_overlap(timestamps: np.ndarray) -> None:
    """test / val / train index sets must be pairwise disjoint."""
    splits = data_split(timestamps)
    test_set = set(splits["test"].tolist())
    val_set = set(splits["val"].tolist())
    train_set = set(splits["train"].tolist())
    assert test_set.isdisjoint(val_set), "test and val overlap"
    assert test_set.isdisjoint(train_set), "test and train overlap"
    assert val_set.isdisjoint(train_set), "val and train overlap"


def test_temporal_order_within_splits(timestamps: np.ndarray) -> None:
    """Temporal ordering: max(test) <= min(val) <= min(train)."""
    splits = data_split(timestamps)
    test_ts = timestamps[splits["test"]]
    val_ts = timestamps[splits["val"]]
    train_ts = timestamps[splits["train"]]
    assert test_ts.max() <= val_ts.min(), "test bleeds into val temporally"
    assert val_ts.min() <= train_ts.min(), "val bleeds into train temporally"


def test_boundary_gap_discarded(timestamps: np.ndarray) -> None:
    """Total assigned < n when boundary_prop > 0; gap count matches formula."""
    n = len(timestamps)
    splits = data_split(timestamps, boundary_prop=0.20)
    total_assigned = len(splits["test"]) + len(splits["val"]) + len(splits["train"])
    assert total_assigned < n, "expected some indices discarded as boundary gap"

    # Verify the gap count exactly matches the formula
    test_cnt = int(n * 0.15)
    val_cnt = int(n * 0.10)
    remaining = n - test_cnt - val_cnt
    boundary_cnt = int(remaining * 0.20)
    assert total_assigned == n - boundary_cnt, (
        f"gap discarded {n - total_assigned} indices, expected {boundary_cnt}"
    )


def test_empty_input() -> None:
    """data_split on an empty array returns empty arrays without error."""
    result = data_split(np.array([], dtype=np.int64))
    assert result["test"].shape == (0,)
    assert result["val"].shape == (0,)
    assert result["train"].shape == (0,)


def test_seed_parameter_accepted() -> None:
    """Function accepts the seed parameter without error."""
    ts = np.arange(50, dtype=np.int64) * 1_000_000_000
    result = data_split(ts, seed=42)
    assert "train" in result


def test_all_indices_in_range(timestamps: np.ndarray) -> None:
    """All returned indices are valid positions into the input array."""
    n = len(timestamps)
    splits = data_split(timestamps)
    for key, idx in splits.items():
        assert np.all(idx >= 0) and np.all(idx < n), f"Split '{key}' contains out-of-range indices"
