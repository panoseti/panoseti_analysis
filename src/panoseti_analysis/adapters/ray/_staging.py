"""SSD pre-staging for Ray training jobs.

Copies a compact feature cache from BeeGFS (slow 1 GbE) to the local SSD
(fast local storage) before training begins. Idempotent: skips if the local
copy's checksum matches the source.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from panoseti_analysis.io.checksum import checksum_store


def stage_to_local(
    source: str | Path,
    local_cache_dir: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Stage a Zarr store (or directory) from *source* to *local_cache_dir*.

    If the destination already exists and its checksum matches the source,
    returns the existing destination path without copying (idempotent).

    Args:
        source: Path to the feature cache store on BeeGFS (or any shared FS).
        local_cache_dir: Directory on the local SSD (e.g. /local/scratch or
            $RAY_TMPDIR). Created if it does not exist.
        name: Subdirectory name inside *local_cache_dir*. Defaults to
            ``source.name``.

    Returns:
        Path to the staged copy on local storage.
    """
    source = Path(source)
    local_cache_dir = Path(local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    dest = local_cache_dir / (name or source.name)

    if dest.exists():
        src_cksum = checksum_store(source)
        dst_cksum = checksum_store(dest)
        if src_cksum == dst_cksum:
            return dest
        shutil.rmtree(dest)

    shutil.copytree(str(source), str(dest))
    return dest
