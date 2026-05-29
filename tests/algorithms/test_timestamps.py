"""TDD for the timestamp-QC + monotonic-repair kernel."""

from __future__ import annotations

import numpy as np
import xarray as xr

from panoseti_analysis.algorithms.timestamps import compute_timestamp_qc, repair_timestamps
from panoseti_analysis.config.models import TimestampQCStatus

_HUGE = 10**18  # suspect threshold high enough to never trip in benign cases


def _shuffle_time(ds: xr.Dataset, seed: int = 1) -> xr.Dataset:
    perm = np.random.default_rng(seed).permutation(ds.sizes["time"])
    return ds.isel(time=perm)


def test_monotonic_input_is_clean(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph()
    qc = compute_timestamp_qc(
        ds["unix_t_ns"].values, cadence_ns=None, suspect_displacement_ns=_HUGE
    )
    assert qc.status is TimestampQCStatus.CLEAN
    assert qc.monotonic is True
    assert qc.n_nonmonotonic == 0
    assert qc.n_duplicates == 0
    assert qc.n_frames == ds.sizes["time"]


def test_repair_sorts_and_preserves_all_frames(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph()
    n = ds.sizes["time"]
    shuffled = _shuffle_time(ds)
    out, qc = repair_timestamps(shuffled, cadence_ns=None, suspect_displacement_ns=_HUGE)
    t = out["unix_t_ns"].values
    assert np.all(np.diff(t) >= 0)  # monotonic non-decreasing after repair
    assert qc.status is TimestampQCStatus.REPAIRED
    assert qc.monotonic is True
    assert out.sizes["time"] == n  # no frames dropped
    assert set(t.tolist()) == set(ds["unix_t_ns"].values.tolist())  # none invented
    # pkt_num rides along: original frame order (monotonic in index) is recovered
    np.testing.assert_array_equal(out["pkt_num"].values, np.arange(n, dtype="uint32"))


def test_duplicates_are_flagged_not_dropped(make_img) -> None:  # type: ignore[no-untyped-def]
    ds = make_img()
    n = ds.sizes["time"]
    t = ds["unix_t_ns"].values.copy()
    t[5] = t[4]  # duplicate timestamp
    ds = ds.assign(unix_t_ns=("time", t))
    out, qc = repair_timestamps(ds, cadence_ns=20_000, suspect_displacement_ns=_HUGE)
    assert qc.n_duplicates == 1
    assert qc.status is TimestampQCStatus.FLAGGED
    assert out.sizes["time"] == n  # duplicate flagged, not removed


def test_large_gap_is_flagged(make_img) -> None:  # type: ignore[no-untyped-def]
    ds = make_img(cadence_ns=20_000)
    t = ds["unix_t_ns"].values.copy()
    t[30:] += 50 * 20_000  # gap of 50 cadences >> 4x threshold
    qc = compute_timestamp_qc(t, cadence_ns=20_000, suspect_displacement_ns=_HUGE)
    assert qc.n_gaps_over_threshold >= 1
    assert qc.max_gap_ns >= 50 * 20_000
    assert qc.status is TimestampQCStatus.FLAGGED


def test_corruption_grade_backward_jump_is_suspect(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph()
    t = ds["unix_t_ns"].values.copy()
    t[20] -= 10**15  # huge backward displacement
    ds = ds.assign(unix_t_ns=("time", t))
    _out, qc = repair_timestamps(ds, cadence_ns=None, suspect_displacement_ns=10**12)
    assert qc.status is TimestampQCStatus.SUSPECT
