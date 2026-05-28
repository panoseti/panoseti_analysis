"""Pure quick-look statistics (Layer A). PNG/JSON writers live in ``io/quicklook.py``."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Stats:
    n_frames: int
    n_pixels: int
    mean: float
    std: float
    p01: float
    p99: float
    nonzero_frac: float
    n_hot: int  # pixels with time-mean > 10x the global mean
    n_dead: int  # pixels with time-mean == 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def summarize(da: xr.DataArray) -> Stats:
    """Compute summary statistics over a ``(T, H, W)`` DataArray (loads into memory)."""
    arr = np.asarray(da.compute())
    finite = arr[np.isfinite(arr)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pixel_means = np.nanmean(arr, axis=0).ravel()
        global_mean = float(np.nanmean(pixel_means))
    hot_thresh = 10.0 * global_mean if global_mean > 0 else np.inf
    return Stats(
        n_frames=int(arr.shape[0]),
        n_pixels=int(arr.shape[1] * arr.shape[2]),
        mean=global_mean,
        std=float(np.nanstd(finite)) if finite.size else float("nan"),
        p01=float(np.nanpercentile(finite, 1)) if finite.size else float("nan"),
        p99=float(np.nanpercentile(finite, 99)) if finite.size else float("nan"),
        nonzero_frac=float(np.count_nonzero(finite) / finite.size) if finite.size else 0.0,
        n_hot=int(np.sum(pixel_means > hot_thresh)),
        n_dead=int(np.sum(pixel_means == 0)),
    )


def frame_rate_hz(ds: xr.Dataset) -> float:
    """Estimate frame rate from the median positive inter-frame interval in ``unix_t_ns``."""
    ts = np.asarray(ds["unix_t_ns"].values)
    if ts.size < 2:
        return float("nan")
    diffs = np.diff(ts)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return float("nan")
    median_ns = float(np.median(positive))
    return 1e9 / median_ns if median_ns > 0 else float("nan")
