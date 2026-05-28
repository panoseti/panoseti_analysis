"""Unit tests for pulse-height calibration."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import zarr

from panoseti_zarr_pipeline.calibrate_ph import calibrate_ph


def _make_l0_ph(tmp_path, shape=(50, 16, 16), dp="ph256"):
    """Create a minimal synthetic L0 ph store."""
    store_path = tmp_path / f"{dp}.zarr"
    T, H, W = shape

    rng = np.random.default_rng(0)
    # Simulate int16 ADC counts: pedestal ~-500 with noise ±100
    pedestal = rng.integers(-600, -400, size=(H, W), dtype=np.int16)
    noise = rng.integers(-100, 100, size=(T, H, W), dtype=np.int16)
    images = np.broadcast_to(pedestal, (T, H, W)).astype(np.int16) + noise

    ds = xr.Dataset(
        {
            "images": xr.DataArray(images, dims=["time", "y", "x"]),
            "unix_t_ns": xr.DataArray(
                np.arange(T, dtype=np.int64) * int(1e6), dims=["time"]
            ),
            "pkt_num": xr.DataArray(np.arange(T, dtype=np.uint32), dims=["time"]),
            "quabo_num": xr.DataArray(np.zeros(T, dtype=np.uint8), dims=["time"]),
        },
        attrs={
            "data_product": dp,
            "panoseti_pff_zarr_version": "1.0",
            "bytes_per_pixel": 2,
            "module": "1",
            "header_format": "single",
            "header_fields": ["pkt_num", "quabo_num"],
            "quabo_fields": [],
        },
    )
    ds.to_zarr(str(store_path), mode="w", zarr_format=3, consolidated=False)
    return store_path


def test_calibrate_ph_creates_l1(tmp_path):
    l0 = _make_l0_ph(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_ph(l0, l1, sigma_threshold=5.0, baseline_offset=800, frame_stride=10)
    assert l1.exists()
    assert (l1 / "summary.json").exists()
    assert (l1 / "preview.png").exists()


def test_calibrate_ph_output_variables(tmp_path):
    l0 = _make_l0_ph(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_ph(l0, l1, sigma_threshold=5.0, baseline_offset=800, frame_stride=10)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert "pedestal_subtracted" in ds
    assert "hot_pixel_mask" in ds
    assert "dead_pixel_mask" in ds
    assert "unix_t_ns" in ds
    assert "pkt_num" in ds


def test_calibrate_ph_dtype(tmp_path):
    l0 = _make_l0_ph(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_ph(l0, l1, sigma_threshold=5.0, baseline_offset=800, frame_stride=10)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds["pedestal_subtracted"].dtype == np.float32
    assert ds["unix_t_ns"].dtype == np.int64
    assert ds["hot_pixel_mask"].dtype == np.uint8


def test_calibrate_ph_shape_preserved(tmp_path):
    shape = (50, 16, 16)
    l0 = _make_l0_ph(tmp_path, shape=shape)
    l1 = tmp_path / "l1.zarr"
    calibrate_ph(l0, l1, sigma_threshold=5.0, baseline_offset=800, frame_stride=10)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds["pedestal_subtracted"].shape == shape


def test_calibrate_ph_l1_attrs(tmp_path):
    l0 = _make_l0_ph(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_ph(l0, l1)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds.attrs["panoseti_pff_zarr_l1_version"] == "0.1"
    calib = ds.attrs["calibration"]
    assert calib["kind"] == "ph"
    assert "baseline_offset" in calib
    assert "sigma_threshold" in calib
