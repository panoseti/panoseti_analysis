"""Re-pack a Zarr directory store into a single transfer file (PACK stage).

Hop-aware (see storage_spec.md §8):

- ``pack_tar``   — compute hop (UCSD→Expanse): fast unpack to Lustre directory stores.
- ``pack_zipstore`` — archive hop (Expanse→Cylon): readable in place as a Zarr ZipStore.
  Uses ``ZIP_STORED`` (chunks are already zstd — no double compression) and ZIP64
  (a whole store can exceed 4 GB even though individual chunks won't).
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def pack_tar(store: str | Path, out_path: str | Path) -> Path:
    """Tar a store uncompressed; the store dir name is preserved as the archive root."""
    store = Path(store)
    out_path = Path(out_path)
    with tarfile.open(out_path, "w") as tar:  # no compression: chunks are already zstd
        tar.add(store, arcname=store.name)
    return out_path


def pack_zipstore(store: str | Path, out_path: str | Path) -> Path:
    """Pack a store into a Zarr ZipStore (STORED + ZIP64); store root = zip root.

    The result opens in place via ``zarr.storage.ZipStore(out_path)`` /
    ``xr.open_zarr("zip://" + out_path)`` without unpacking.
    """
    store = Path(store)
    out_path = Path(out_path)
    with zipfile.ZipFile(
        out_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for path in sorted(p for p in store.rglob("*") if p.is_file()):
            zf.write(path, arcname=path.relative_to(store).as_posix())
    return out_path
