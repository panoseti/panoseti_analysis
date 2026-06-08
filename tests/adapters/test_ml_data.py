"""Tests for the shared feature-cache loaders (adapters/ml/data.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from panoseti_analysis.adapters.ml.data import (  # noqa: E402
    load_labeled_feature_cache,
    load_unlabeled_feature_cache,
)


def _write_feature_cache(
    path: Path, *, n_train: int = 6, n_val: int = 4, channels: int = 2
) -> None:
    n = n_train + n_val
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, channels, 32, 32)).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.int64)
    split = np.array(["train"] * n_train + ["val"] * n_val)
    xr.Dataset(
        {
            "X": (["sample", "channel", "H", "W"], x),
            "label": (["sample"], y),
            "split": (["sample"], split),
        }
    ).to_zarr(str(path), zarr_format=3)


def test_load_labeled_feature_cache(tmp_path: Path) -> None:
    path = tmp_path / "feat.zarr"
    _write_feature_cache(path, n_train=6, n_val=4)

    x_train, y_train, x_val, y_val = load_labeled_feature_cache(path)

    assert x_train.shape == (6, 2, 32, 32)
    assert x_val.shape == (4, 2, 32, 32)
    assert y_train.shape == (6,)
    assert y_val.shape == (4,)
    assert x_train.dtype == torch.float32
    assert y_train.dtype == torch.int64


def test_load_unlabeled_feature_cache(tmp_path: Path) -> None:
    path = tmp_path / "feat.zarr"
    _write_feature_cache(path, n_train=5, n_val=3, channels=1)

    x_train, x_val = load_unlabeled_feature_cache(path)

    assert x_train.shape == (5, 1, 32, 32)
    assert x_val.shape == (3, 1, 32, 32)
    assert x_train.dtype == torch.float32


def test_load_unlabeled_feature_cache_no_split_column(tmp_path: Path) -> None:
    """The PH cache (pa-prep-ph) has no split column → deterministic seeded carve-out."""
    path = tmp_path / "ph_feat.zarr"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 1, 16, 16)).astype(np.float32)
    xr.Dataset({"X": (["sample", "channel", "H", "W"], x)}).to_zarr(str(path), zarr_format=3)

    x_train, x_val = load_unlabeled_feature_cache(path, val_prop=0.25, seed=7)
    assert x_train.shape == (15, 1, 16, 16)
    assert x_val.shape == (5, 1, 16, 16)

    # Deterministic: same seed → same split (disjoint, exhaustive partition).
    x_train2, x_val2 = load_unlabeled_feature_cache(path, val_prop=0.25, seed=7)
    assert torch.equal(x_train, x_train2)
    assert torch.equal(x_val, x_val2)
