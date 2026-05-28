"""Shared helpers for Layer B adapters (cadence inference, lineage emission)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import StoreLineage


class SuspectTimestamps(RuntimeError):
    """Raised when a store's timestamps are corruption-grade and ``--fail-on-suspect`` is set."""


def infer_cadence_ns(ds: xr.Dataset) -> int | None:
    """Sampling cadence: ``None`` for event-based ph; median positive Δt for img."""
    if infer_kind(str(ds.attrs.get("data_product", ""))) == "ph":
        return None
    t = np.asarray(ds["unix_t_ns"].values)
    if t.size < 2:
        return None
    diffs = np.diff(np.sort(t))
    positive = diffs[diffs > 0]
    return int(np.median(positive)) if positive.size else None


def derive_suspect_displacement_ns(cadence_ns: int | None) -> int:
    """Corruption-grade backward-jump threshold (tunable; calibrate on real data).

    img: 1000x cadence (a 1000-frame backward jump is implausible buffering);
    event ph: 1 second.
    """
    if cadence_ns is not None and cadence_ns > 0:
        return 1000 * cadence_ns
    return 1_000_000_000


def write_lineage_json(record: StoreLineage, path: str | Path) -> None:
    Path(path).write_text(record.model_dump_json(indent=2))
