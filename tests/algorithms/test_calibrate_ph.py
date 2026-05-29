"""TDD for the pulse-height calibration kernel (pure ds -> ds)."""

from __future__ import annotations

import numpy as np

from panoseti_analysis.algorithms.calibrate_ph import calibrate_ph
from panoseti_analysis.config.models import PhCalibParams


def test_outputs_shapes_dtypes_and_attrs(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph(n=60, size=16)
    out = calibrate_ph(ds, PhCalibParams(frame_stride=5))

    assert out["pedestal_subtracted"].dims == ("time", "y", "x")
    assert out["pedestal_subtracted"].dtype == np.float32
    assert out["hot_pixel_mask"].dims == ("y", "x")
    assert out["hot_pixel_mask"].dtype == np.uint8
    assert out["dead_pixel_mask"].dtype == np.uint8

    # timing/header arrays carried forward unchanged
    np.testing.assert_array_equal(out["unix_t_ns"].values, ds["unix_t_ns"].values)
    np.testing.assert_array_equal(out["pkt_num"].values, ds["pkt_num"].values)

    assert out.attrs["data_level"] == "L1"
    assert "panoseti_analysis_storage_version" in out.attrs
    assert out.attrs["calibration"]["kind"] == "ph"
    assert out.attrs["calibration"]["sigma_threshold"] == 5.0


def test_does_not_mutate_input(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph(n=40, size=16)
    before = ds["images"].values.copy()
    calibrate_ph(ds, PhCalibParams(frame_stride=5))
    np.testing.assert_array_equal(ds["images"].values, before)
    assert "data_level" not in ds.attrs


def test_ph1024_runs_and_keeps_shape(make_ph) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph(n=40, size=32, data_product="ph1024")
    out = calibrate_ph(ds, PhCalibParams(frame_stride=5))
    assert out["pedestal_subtracted"].shape == (40, 32, 32)
