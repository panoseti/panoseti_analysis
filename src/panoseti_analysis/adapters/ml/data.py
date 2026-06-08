"""Feature-cache → tensors glue, shared by the model trainers (Layer B).

Opening a feature-cache Zarr store and splitting it into train/val tensors by the ``split``
column is identical across models; it lived inline in each ``train_*.py``. Centralised here
so every trainer (and notebook) loads data the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from panoseti_analysis.io.stores import open_store


def load_labeled_feature_cache(
    path: Path, *, split_col: str = "split"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a labeled feature cache into ``(X_train, y_train, X_val, y_val)`` tensors.

    Expects an ``open_store``-readable Zarr with ``X`` (sample, channel, H, W), an integer
    ``label`` (sample,), and a string ``split`` (sample,) column of ``"train"``/``"val"``/...
    """
    ds = open_store(Path(path))
    x = ds["X"].values.astype(np.float32)
    y = ds["label"].values.astype(np.int64)
    split = np.asarray(ds[split_col].values, dtype=str)
    train, val = split == "train", split == "val"
    return (
        torch.from_numpy(x[train]),
        torch.from_numpy(y[train]),
        torch.from_numpy(x[val]),
        torch.from_numpy(y[val]),
    )


def load_unlabeled_feature_cache(
    path: Path,
    *,
    x_var: str = "X",
    split_col: str = "split",
    val_prop: float = 0.1,
    seed: int = 1984,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an unlabeled feature cache into ``(X_train, X_val)`` tensors (e.g. for the VAE).

    If a ``split`` column is present it is honoured. The PH feature cache (``pa-prep-ph``)
    does not write one, so when it's absent we carve out a deterministic random validation
    subset of size ``val_prop`` (seeded by ``seed``), keeping training reproducible without
    changing the on-disk cache format.
    """
    ds = open_store(Path(path))
    x = ds[x_var].values.astype(np.float32)
    if split_col in ds:
        split = np.asarray(ds[split_col].values, dtype=str)
        return torch.from_numpy(x[split == "train"]), torch.from_numpy(x[split == "val"])

    n = len(x)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = round(n * val_prop)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return torch.from_numpy(x[train_idx]), torch.from_numpy(x[val_idx])
