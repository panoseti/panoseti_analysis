"""Open and write Zarr v3 stores — the read/write half of the I/O boundary.

Generalizes the prototype's ``_common.open_l0`` / ``write_l1``. Kernels never call
these; only adapters do.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from zarr.codecs import ZstdCodec

from panoseti_analysis.config.models import ProcessingStep


def open_store(store: str | Path, *, chunks: Any = "auto") -> xr.Dataset:
    """Open any panoseti_analysis/pypff Zarr v3 store (consolidated metadata never written)."""
    return xr.open_zarr(str(store), consolidated=False, chunks=chunks)


def open_l0(store: str | Path) -> xr.Dataset:
    """Open an L0 store produced by ``pypff.zarr.convert_run`` and assert the time dtype."""
    ds = xr.open_zarr(str(store), consolidated=False, chunks={})
    if ds["unix_t_ns"].dtype != np.int64:
        raise ValueError(f"unix_t_ns must be int64, got {ds['unix_t_ns'].dtype}")
    return ds


#: Time-like dimensions rechunked to a uniform size before writing.
_TIME_DIMS = ("time", "hk_time")
#: Uniform time chunk (frames) — keeps last chunk <= first (a Zarr v3 requirement).
_TIME_CHUNK = 16384


def _compressors(codec: str, level: int) -> list[Any]:
    if codec == "zstd":
        return [ZstdCodec(level=level)]
    if codec == "none":
        return []
    raise ValueError(f"unsupported codec {codec!r} (expected 'zstd' or 'none')")


def write_store(
    ds: xr.Dataset,
    out_path: str | Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    processing_history: list[ProcessingStep] | None = None,
) -> int:
    """Write a Dataset to a Zarr v3 directory store; return total bytes on disk.

    Idempotent (removes any existing store first). Attributes are taken from ``ds.attrs``
    — kernels stamp ``data_level`` / version / ``calibration`` / ``timestamp_qc`` before
    this is called. Compression is applied per variable.

    If *processing_history* is a non-empty list, it is serialized into the Zarr root
    attrs under the ``"processing_history"`` key (list of dicts via ``model_dump``).
    ``None`` or ``[]`` leaves the key absent — preserving backward compat with stores
    written before schema v2.0.  The caller's Dataset is never mutated.
    """
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    # Drop any inherited (L0) chunk encoding, then rechunk time-like dims uniformly so
    # the final chunk is never larger than the first (a Zarr v3 write requirement).
    ds = ds.drop_encoding()
    time_chunks = {
        str(d): min(int(ds.sizes[d]), _TIME_CHUNK) for d in ds.dims if d in _TIME_DIMS
    }
    if time_chunks:
        ds = ds.chunk(time_chunks)

    # Stamp processing_history into a shallow-copied attrs dict without mutating ds.
    if processing_history:
        new_attrs = dict(ds.attrs)
        new_attrs["processing_history"] = [s.model_dump() for s in processing_history]
        ds = ds.assign_attrs(new_attrs)

    compressors = _compressors(codec, level)
    names = list(ds.data_vars) + list(ds.coords)
    encoding = {str(name): {"compressors": compressors} for name in names}

    ds.to_zarr(str(out_path), mode="w", zarr_format=3, consolidated=False, encoding=encoding)
    return sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
