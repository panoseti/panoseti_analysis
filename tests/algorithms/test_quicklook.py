"""TDD for the pure quick-look statistics kernel."""

from __future__ import annotations

import numpy as np
import xarray as xr

from panoseti_analysis.algorithms.quicklook import Stats, frame_rate_hz, summarize


def test_summarize_constant_array() -> None:
    da = xr.DataArray(np.ones((10, 4, 4), dtype="float32"), dims=("time", "y", "x"))
    stats = summarize(da)
    assert isinstance(stats, Stats)
    assert stats.n_frames == 10
    assert stats.n_pixels == 16
    assert stats.mean == 1.0
    assert stats.std == 0.0
    assert stats.nonzero_frac == 1.0
    assert stats.n_dead == 0
    assert stats.n_hot == 0
    assert set(stats.to_json()) >= {"n_frames", "n_pixels", "mean", "std", "n_hot", "n_dead"}


def test_summarize_flags_dead_pixels() -> None:
    arr = np.ones((5, 2, 2), dtype="float32")
    arr[:, 0, 0] = 0.0  # one always-zero (dead) pixel
    stats = summarize(xr.DataArray(arr, dims=("time", "y", "x")))
    assert stats.n_dead == 1


def test_frame_rate_hz_from_cadence(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph(n=50, cadence_ns=1_000_000)  # 1 ms -> 1000 Hz
    assert abs(frame_rate_hz(ds) - 1000.0) < 1e-6
