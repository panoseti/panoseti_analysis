"""Timestamp QC and monotonic repair (Layer A, pure).

Contract (storage_spec.md §4): L0 mirrors native PFF order and only *records* QC;
L1+ must guarantee monotonic-non-decreasing ``unix_t_ns``. ``repair_timestamps``
stable-sorts L0 into L1 order, never drops frames (duplicates flagged), and is
reversible because ``pkt_num`` rides along.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from panoseti_analysis.config.models import TimestampQC, TimestampQCStatus


def _gap_threshold_ns(
    sorted_diffs: np.ndarray, cadence_ns: int | None, gap_factor: float, mad_k: float
) -> int:
    """Data-product-aware gap threshold: k*cadence (img) or median+k*MAD (event ph)."""
    if cadence_ns is not None and cadence_ns > 0:
        return int(gap_factor * cadence_ns)
    positive = sorted_diffs[sorted_diffs > 0]
    if positive.size == 0:
        return 0
    med = float(np.median(positive))
    mad = float(np.median(np.abs(positive - med)))
    return int(med + mad_k * mad)


def compute_timestamp_qc(
    unix_t_ns: np.ndarray,
    *,
    cadence_ns: int | None,
    suspect_displacement_ns: int,
    gap_factor: float = 4.0,
    mad_k: float = 10.0,
) -> TimestampQC:
    """Inspect an int64 time vector in NATIVE order and assign a QC record.

    Status is ``clean``/``flagged``/``suspect`` (never ``repaired`` — this function
    does not reorder). ``suspect`` when any single backward jump exceeds
    ``suspect_displacement_ns`` (corruption-grade). Gaps use the data-product-aware
    threshold (``gap_factor*cadence_ns`` for img; ``median+mad_k*MAD`` for event ph).
    """
    t = np.asarray(unix_t_ns)
    n = int(t.size)
    if n == 0:
        return TimestampQC(status=TimestampQCStatus.CLEAN, monotonic=True, n_frames=0)

    diffs = np.diff(t)
    monotonic = bool(np.all(diffs >= 0)) if n > 1 else True
    n_nonmonotonic = int(np.count_nonzero(diffs < 0))
    n_duplicates = int(n - np.unique(t).size)

    sorted_t = np.sort(t)
    sorted_diffs = np.diff(sorted_t) if n > 1 else np.empty(0, dtype=t.dtype)
    max_gap_ns = int(sorted_diffs.max()) if sorted_diffs.size else 0
    gap_threshold = _gap_threshold_ns(sorted_diffs, cadence_ns, gap_factor, mad_k)
    n_gaps_over = int(np.count_nonzero(sorted_diffs > gap_threshold)) if gap_threshold > 0 else 0
    max_backward = int(-diffs.min()) if n_nonmonotonic else 0

    if max_backward > suspect_displacement_ns:
        status = TimestampQCStatus.SUSPECT
    elif n_nonmonotonic or n_duplicates or n_gaps_over:
        status = TimestampQCStatus.FLAGGED
    else:
        status = TimestampQCStatus.CLEAN

    return TimestampQC(
        status=status,
        monotonic=monotonic,
        n_frames=n,
        n_nonmonotonic=n_nonmonotonic,
        n_duplicates=n_duplicates,
        max_gap_ns=max_gap_ns,
        n_gaps_over_threshold=n_gaps_over,
        gap_threshold_ns=gap_threshold,
        t_start_ns=int(sorted_t[0]),
        t_end_ns=int(sorted_t[-1]),
    )


def repair_timestamps(
    ds: xr.Dataset,
    *,
    cadence_ns: int | None,
    suspect_displacement_ns: int,
    gap_factor: float = 4.0,
    mad_k: float = 10.0,
) -> tuple[xr.Dataset, TimestampQC]:
    """Stable-sort ``ds`` by ``unix_t_ns`` (L0->L1) and return ``(sorted_ds, qc)``.

    QC is computed on the PRE-sort order so the suspect/displacement signal survives.
    The returned status reflects the L1 outcome: ``suspect`` (kept; adapter may abort),
    ``repaired`` (reordering occurred), ``flagged`` (gaps/duplicates but in order), or
    ``clean``. All time-indexed variables are reordered together; nothing is dropped.
    """
    t = ds["unix_t_ns"].values
    native = compute_timestamp_qc(
        t,
        cadence_ns=cadence_ns,
        suspect_displacement_ns=suspect_displacement_ns,
        gap_factor=gap_factor,
        mad_k=mad_k,
    )

    order = np.argsort(t, kind="stable")
    reordered = bool(not np.array_equal(order, np.arange(t.size)))
    out = ds.isel(time=order) if reordered else ds

    if native.status is TimestampQCStatus.SUSPECT:
        status = TimestampQCStatus.SUSPECT
    elif reordered:
        status = TimestampQCStatus.REPAIRED
    elif native.n_duplicates or native.n_gaps_over_threshold:
        status = TimestampQCStatus.FLAGGED
    else:
        status = TimestampQCStatus.CLEAN

    qc = native.model_copy(update={"status": status, "monotonic": True})
    return out, qc
