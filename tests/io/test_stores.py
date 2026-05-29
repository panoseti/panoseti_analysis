"""io.stores round-trip tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from panoseti_analysis.io.stores import open_l0, open_store, write_store


def test_write_then_open_round_trips(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    nbytes = write_store(l0_ph_ds, out)
    assert nbytes > 0
    assert out.is_dir()

    back = open_store(out)
    np.testing.assert_array_equal(back["images"].values, l0_ph_ds["images"].values)
    np.testing.assert_array_equal(back["unix_t_ns"].values, l0_ph_ds["unix_t_ns"].values)
    assert back.attrs["data_product"] == "ph256"


def test_write_store_is_idempotent(l0_img_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    write_store(l0_img_ds, out)
    # second write over an existing store must not raise and must still round-trip
    write_store(l0_img_ds, out)
    back = open_store(out)
    assert back.sizes["time"] == l0_img_ds.sizes["time"]


def test_open_l0_requires_int64_time(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    out = tmp_path / "s.zarr"
    write_store(l0_ph_ds, out)
    ds = open_l0(out)
    assert ds["unix_t_ns"].dtype == np.int64


def test_write_store_rejects_unknown_codec(l0_ph_ds: xr.Dataset, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported codec"):
        write_store(l0_ph_ds, tmp_path / "s.zarr", codec="lz4")


def test_write_store_handles_nonuniform_dask_chunks(tmp_path: Path) -> None:
    # last time-chunk (60) larger than the first (40) — invalid for Zarr unless rechunked
    import dask.array as da

    arr = da.zeros((100, 4, 4), chunks=((40, 60), 4, 4), dtype="float32")
    ds = xr.Dataset(
        {"images": (("time", "y", "x"), arr), "unix_t_ns": ("time", np.arange(100, dtype="int64"))}
    )
    write_store(ds, tmp_path / "s.zarr")  # must not raise
    assert open_store(tmp_path / "s.zarr").sizes["time"] == 100
