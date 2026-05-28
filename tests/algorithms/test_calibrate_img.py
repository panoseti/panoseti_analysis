"""TDD for the movie-mode calibration kernel (pure ds -> ds)."""

from __future__ import annotations

import numpy as np

from panoseti_analysis.algorithms.calibrate_img import calibrate_img
from panoseti_analysis.config.models import ImgCalibParams


def test_outputs_shapes_dtypes_and_attrs(make_img) -> None:  # type: ignore[no-untyped-def]
    ds = make_img(n=60, size=32)
    out = calibrate_img(ds, ImgCalibParams(frame_stride=5, block_size=8))

    assert out["median_subtracted"].dims == ("time", "y", "x")
    assert out["median_subtracted"].dtype == np.float32
    assert out["median_subtracted"].shape == (60, 32, 32)
    assert out["hot_pixel_mask"].dims == ("y", "x")
    assert out["hot_pixel_mask"].dtype == np.uint8
    assert out["dead_pixel_mask"].dtype == np.uint8

    np.testing.assert_array_equal(out["unix_t_ns"].values, ds["unix_t_ns"].values)
    assert "quabo_0_pkt_num" in out.data_vars

    assert out.attrs["data_level"] == "L1"
    assert out.attrs["calibration"]["kind"] == "img"
    assert out.attrs["calibration"]["block_size"] == 8


def test_adc_to_pe_scales_linearly(make_img) -> None:  # type: ignore[no-untyped-def]
    ds = make_img(n=60, size=32)
    out1 = calibrate_img(ds, ImgCalibParams(frame_stride=5, block_size=8, adc_to_pe=1.0))
    out2 = calibrate_img(ds, ImgCalibParams(frame_stride=5, block_size=8, adc_to_pe=2.0))
    a = out1["median_subtracted"].values
    b = out2["median_subtracted"].values
    np.testing.assert_allclose(b, a / 2.0, rtol=1e-5, atol=1e-4)


def test_does_not_mutate_input(make_img) -> None:  # type: ignore[no-untyped-def]
    ds = make_img(n=40, size=32)
    before = ds["images"].values.copy()
    calibrate_img(ds, ImgCalibParams(frame_stride=5, block_size=8))
    np.testing.assert_array_equal(ds["images"].values, before)
