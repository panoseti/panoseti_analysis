"""Time-windowed split with a boundary gap between val and train.

Fixes the original training leakage bug where train_test_split(shuffle=True) was
used, causing near-duplicate adjacent frames to leak across the split boundary.
"""

from __future__ import annotations

import numpy as np


def data_split(
    unix_t_ns: np.ndarray,
    *,
    test_prop: float = 0.15,
    val_prop: float = 0.10,
    boundary_prop: float = 0.20,
    seed: int = 35,
) -> dict[str, np.ndarray]:
    """Time-windowed split with boundary gap.

    Sort indices by time, then allocate:
    - test:     earliest  ``test_prop`` fraction
    - val:      next      ``val_prop`` fraction
    - boundary: discarded ``boundary_prop`` fraction of the *remaining* data
    - train:    the rest

    Returns dict with keys ``"train"``, ``"val"``, ``"test"`` — each an int64
    array of indices into ``unix_t_ns`` (NOT the sorted-order positions).

    Args:
        unix_t_ns: 1-D array of nanosecond timestamps (need not be sorted).
        test_prop: fraction of total for test.
        val_prop: fraction of total for validation.
        boundary_prop: fraction of (n - test_cnt - val_cnt) to discard as gap.
        seed: not used for the split itself (deterministic temporal ordering);
              kept as a parameter for future reproducible augmentation.
    """
    n = len(unix_t_ns)
    sorted_idx = np.argsort(unix_t_ns, kind="stable")

    test_cnt = int(n * test_prop)
    val_cnt = int(n * val_prop)
    remaining = n - test_cnt - val_cnt
    boundary_cnt = int(remaining * boundary_prop)

    test_idx = sorted_idx[:test_cnt]
    val_idx = sorted_idx[test_cnt : test_cnt + val_cnt]
    # boundary = sorted_idx[test_cnt + val_cnt : test_cnt + val_cnt + boundary_cnt]  # discarded
    train_idx = sorted_idx[test_cnt + val_cnt + boundary_cnt :]

    return {"test": test_idx, "val": val_idx, "train": train_idx}
