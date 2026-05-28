"""Unit tests for image-mode calibration."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from panoseti_zarr_pipeline.calibrate_img import calibrate_img


def _make_l0_img(tmp_path, shape=(50, 32, 32), dp="img16", dtype="uint16"):
    """Create a minimal synthetic L0 img store."""
    store_path = tmp_path / f"{dp}.zarr"
    T, H, W = shape

    rng = np.random.default_rng(1)
    # Simulate uint16 threshold-crossing counts: low dark rate ~3 cts + Poisson
    images = rng.poisson(lam=3, size=(T, H, W)).astype(dtype)

    ds = xr.Dataset(
        {
            "images": xr.DataArray(images, dims=["time", "y", "x"]),
            "unix_t_ns": xr.DataArray(
                np.arange(T, dtype=np.int64) * int(2e6), dims=["time"]
            ),
            "quabo_0_pkt_num": xr.DataArray(np.arange(T, dtype=np.uint32), dims=["time"]),
        },
        attrs={
            "data_product": dp,
            "panoseti_pff_zarr_version": "1.0",
            "bytes_per_pixel": 2,
            "module": "1",
            "header_format": "module",
            "header_fields": [],
            "quabo_fields": ["quabo_0_pkt_num"],
        },
    )
    ds.to_zarr(str(store_path), mode="w", zarr_format=3, consolidated=False)
    return store_path


def test_calibrate_img_creates_l1(tmp_path):
    l0 = _make_l0_img(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=10, block_size=8, adc_to_pe=1.5)
    assert l1.exists()
    assert (l1 / "summary.json").exists()
    assert (l1 / "preview.png").exists()


def test_calibrate_img_output_variables(tmp_path):
    l0 = _make_l0_img(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=10, block_size=8, adc_to_pe=1.5)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert "median_subtracted" in ds
    assert "hot_pixel_mask" in ds
    assert "dead_pixel_mask" in ds
    assert "unix_t_ns" in ds


def test_calibrate_img_dtype(tmp_path):
    l0 = _make_l0_img(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=10, block_size=8, adc_to_pe=1.5)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds["median_subtracted"].dtype == np.float32
    assert ds["unix_t_ns"].dtype == np.int64


def test_calibrate_img_shape_preserved(tmp_path):
    shape = (50, 32, 32)
    l0 = _make_l0_img(tmp_path, shape=shape)
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=10, block_size=8, adc_to_pe=1.5)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds["median_subtracted"].shape == shape


def test_calibrate_img_l1_attrs(tmp_path):
    l0 = _make_l0_img(tmp_path)
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=10)
    ds = xr.open_zarr(str(l1), consolidated=False)
    assert ds.attrs["panoseti_pff_zarr_l1_version"] == "0.1"
    calib = ds.attrs["calibration"]
    assert calib["kind"] == "img"
    assert "frame_stride" in calib
    assert "adc_to_pe" in calib


def test_calibrate_img_adc_to_pe_scaling(tmp_path):
    """Dividing by adc_to_pe=1 should yield float32 equal to integer input (near-zero median)."""
    l0 = _make_l0_img(tmp_path)
    # With adc_to_pe=1 and large frame_stride=49, pedestal ~= median of 2 frames
    l1 = tmp_path / "l1.zarr"
    calibrate_img(l0, l1, frame_stride=49, block_size=8, adc_to_pe=1.0)
    ds = xr.open_zarr(str(l1), consolidated=False)
    result = ds["median_subtracted"].values
    # After subtraction the mean should be near zero (< 3σ of Poisson noise)
    assert abs(float(np.nanmean(result))) < 5.0
