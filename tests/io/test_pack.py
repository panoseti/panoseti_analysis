"""io.pack tests: tar round-trip and ZipStore (STORED + readable-in-place)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

from panoseti_analysis.io.pack import pack_tar, pack_zipstore
from panoseti_analysis.io.stores import write_store


def _make_store(ds: xr.Dataset, tmp_path: Path) -> Path:
    store = tmp_path / "run.dp_ph256.module_1.zarr"
    write_store(ds, store)
    return store


def test_pack_tar_round_trips(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    store = _make_store(l0_ph_ds, tmp_path)
    tar_path = pack_tar(store, tmp_path / "store.tar")
    assert tar_path.is_file()
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
    # store dir name is preserved as the archive root
    assert any(n == store.name or n.startswith(store.name + "/") for n in names)
    assert f"{store.name}/zarr.json" in names


def test_pack_zipstore_is_stored_and_readable(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    store = _make_store(l0_ph_ds, tmp_path)
    zip_path = pack_zipstore(store, tmp_path / "store.zarr.zip")
    assert zip_path.is_file()

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        # store root is the zip root (zarr.json at top level, no store-name prefix)
        assert "zarr.json" in zf.namelist()
        # chunks are already zstd: must be STORED, never DEFLATE (no double compression)
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)

    # readable in place as a Zarr ZipStore, without unpacking
    import zarr

    zs = zarr.storage.ZipStore(str(zip_path), mode="r")
    back = xr.open_zarr(zs, consolidated=False)
    np.testing.assert_array_equal(back["images"].values, l0_ph_ds["images"].values)
