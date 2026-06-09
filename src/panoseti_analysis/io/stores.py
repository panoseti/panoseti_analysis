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
#: Uniform inner time chunk (frames) — the addressable unit within a shard when
#: shard_factor > 0, and the physical zarr chunk when shard_factor = 0.
#: 16 384 frames x 2 048 B/frame (img16) = 32 MB per inner chunk.
_TIME_CHUNK = 16384


def _compressors(codec: str, level: int) -> list[Any]:
    if codec == "zstd":
        return [ZstdCodec(level=level)]
    if codec == "none":
        return []
    raise ValueError(f"unsupported codec {codec!r} (expected 'zstd' or 'none')")


def stamp_history(store: str | Path, history: list[ProcessingStep]) -> None:
    """Stamp processing_history into root attrs of an existing Zarr v3 store in-place.

    Used when the store was written by an external writer (pypff) rather than write_store.
    """
    import zarr

    root = zarr.open_group(str(store), mode="r+", zarr_format=3)
    attrs = dict(root.attrs)
    attrs["processing_history"] = [s.model_dump() for s in history]
    root.attrs.update(attrs)


def write_store(
    ds: xr.Dataset,
    out_path: str | Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    shard_factor: int = 0,
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

    If *shard_factor* > 0, each variable's time-like dimension is sharded:
    ``shard_factor`` dask chunks are packed into one physical shard file.
    Default 0 disables sharding. Recommended value: 8 for L1 on BeeGFS/Expanse.
    """
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    ds = ds.drop_encoding()

    # Rechunk time-like dims. When sharding, use the full shard size (_TIME_CHUNK *
    # shard_factor) so each dask task maps to exactly one shard file — this prevents
    # concurrent writes into the same shard (ShardingCodec is not thread-safe across
    # tasks targeting the same shard). When not sharding, rechunk to _TIME_CHUNK.
    if shard_factor > 0:
        shard_frames = _TIME_CHUNK * shard_factor
        time_chunks: dict[str, int] = {
            str(d): min(int(ds.sizes[d]), shard_frames) for d in ds.dims if d in _TIME_DIMS
        }
    else:
        time_chunks = {
            str(d): min(int(ds.sizes[d]), _TIME_CHUNK) for d in ds.dims if d in _TIME_DIMS
        }
    if time_chunks:
        ds = ds.chunk(time_chunks)

    if processing_history:
        new_attrs = dict(ds.attrs)
        new_attrs["processing_history"] = [s.model_dump() for s in processing_history]
        ds = ds.assign_attrs(new_attrs)

    compressors = _compressors(codec, level)
    names = list(ds.data_vars) + list(ds.coords)
    encoding: dict[str, Any] = {str(name): {"compressors": compressors} for name in names}

    if shard_factor > 0:
        for name in names:
            try:
                var = ds[name] if name in ds.data_vars else ds.coords[name]
            except KeyError:
                continue
            if not hasattr(var.data, "chunks"):
                continue
            # shard shape == dask chunk shape: one dask task writes one shard file.
            dask_chunk = tuple(c[0] for c in var.data.chunks)
            encoding[str(name)]["shards"] = dask_chunk

    ds.to_zarr(str(out_path), mode="w", zarr_format=3, consolidated=False, encoding=encoding)
    return sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
