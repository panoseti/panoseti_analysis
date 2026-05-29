"""Tests for the SSD pre-staging helper (adapters/ray/_staging.py)."""
from __future__ import annotations

from pathlib import Path

import zarr

from panoseti_analysis.adapters.ray._staging import stage_to_local
from panoseti_analysis.io.checksum import checksum_store


def _make_zarr_store(path: Path) -> Path:
    """Create a minimal zarr store with one array at *path* (zarr v3 compatible)."""
    import numpy as np

    store = zarr.open_group(str(path), mode="w")
    store["data"] = np.array([1, 2, 3])
    return path


def test_stage_to_local_copies_store(tmp_path: Path) -> None:
    """stage_to_local should copy the source store to local_cache_dir."""
    src = _make_zarr_store(tmp_path / "src.zarr")
    local_dir = tmp_path / "local"

    dest = stage_to_local(src, local_dir)

    assert dest.exists(), f"dest not found: {dest}"
    assert dest == local_dir / "src.zarr"


def test_stage_to_local_is_idempotent(tmp_path: Path) -> None:
    """Staging the same store twice should return the same path without error."""
    src = _make_zarr_store(tmp_path / "src.zarr")
    local_dir = tmp_path / "local"

    dest1 = stage_to_local(src, local_dir)
    dest2 = stage_to_local(src, local_dir)

    assert dest1 == dest2
    assert dest2.exists()


def test_stage_to_local_restages_on_content_change(tmp_path: Path) -> None:
    """If source content changes, the destination should be refreshed."""
    src = _make_zarr_store(tmp_path / "src.zarr")
    local_dir = tmp_path / "local"

    dest = stage_to_local(src, local_dir)
    cksum_before = checksum_store(dest)

    # Modify the source store
    import numpy as np

    store = zarr.open_group(str(src), mode="a")
    store["extra"] = np.array([99, 100])

    dest2 = stage_to_local(src, local_dir)
    cksum_after = checksum_store(dest2)

    assert dest2 == dest
    assert cksum_before != cksum_after, "Destination should have been refreshed"


def test_stage_custom_name(tmp_path: Path) -> None:
    """name= keyword should override the destination directory name."""
    src = _make_zarr_store(tmp_path / "src.zarr")
    local_dir = tmp_path / "local"

    dest = stage_to_local(src, local_dir, name="custom")

    assert dest == local_dir / "custom"
    assert dest.exists()
