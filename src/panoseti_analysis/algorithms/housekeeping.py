"""Housekeeping (HK) telemetry -> per-hashset Datasets — Layer A, pure.

``hk.pff`` is a single observatory-wide head-node redis snapshot file, so HK is
per-run, per-hashset (device-type) — not per-module. Within a run each hashset has a
fixed schema, so each becomes a clean typed table on its own ``hk_t_ns`` (~1 Hz) axis.
The adapter writes each to its own ``hk.<hashset>.zarr`` store.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: Field names treated as the snapshot time axis (consumed into the hk_t_ns coordinate).
_TIME_FIELD_CANDIDATES = ("unix_t_ns", "timestamp", "time", "computer_time", "tv_sec")


def build_hk_datasets(hk: dict[str, dict[str, np.ndarray]]) -> dict[str, xr.Dataset]:
    """Build one Dataset per hashset from pypff's parsed ``{hashset: {field: ndarray}}``."""
    datasets: dict[str, xr.Dataset] = {}
    for hashset, fields in hk.items():
        if not fields:
            continue
        lengths = {name: int(np.asarray(arr).shape[0]) for name, arr in fields.items()}
        n = next(iter(lengths.values()))
        if any(length != n for length in lengths.values()):
            raise ValueError(f"hashset {hashset!r} has fields of differing length: {lengths}")

        time_field = next((c for c in _TIME_FIELD_CANDIDATES if c in fields), None)
        if time_field is not None:
            hk_t_ns = np.asarray(fields[time_field]).astype("int64")
        else:
            hk_t_ns = np.arange(n, dtype="int64")

        data_vars = {
            name: ("hk_time", np.asarray(arr)) for name, arr in fields.items() if name != time_field
        }
        datasets[hashset] = xr.Dataset(
            data_vars=data_vars,
            coords={"hk_t_ns": ("hk_time", hk_t_ns)},
            attrs={"hashset": hashset},
        )
    return datasets
