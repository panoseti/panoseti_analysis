"""Tests for the QC-alongside-products layer (Layer A pure)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from panoseti_analysis.algorithms.qc import (
    QCCheckResult,
    QCReport,
    qc_check,
    registered_checks,
    run_qc,
)


def _make_l1_img(n: int = 30, n_hot_pixels: int = 0, n_dead_pixels: int = 0) -> xr.Dataset:
    rng = np.random.default_rng(0)
    img = rng.normal(10.0, 2.0, (n, 32, 32)).astype(np.float32)
    t = 1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 20_000
    hot_mask = np.zeros((32, 32), dtype=np.uint8)
    dead_mask = np.zeros((32, 32), dtype=np.uint8)
    if n_hot_pixels:
        hot_mask.flat[:n_hot_pixels] = 1
    if n_dead_pixels:
        dead_mask.flat[:n_dead_pixels] = 1
    return xr.Dataset(
        {
            "median_subtracted": (["T", "H", "W"], img),
            "unix_t_ns": (["T"], t),
            "hot_pixel_mask": (["H", "W"], hot_mask),
            "dead_pixel_mask": (["H", "W"], dead_mask),
        },
        attrs={"data_product": "img16"},
    )


def _make_l1_ph(n: int = 30) -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 2.0, (n, 16, 16)).astype(np.float32)
    t = 1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 1_000_000
    mask = np.zeros((16, 16), dtype=np.uint8)
    return xr.Dataset(
        {
            "pedestal_subtracted": (["T", "H", "W"], data),
            "unix_t_ns": (["T"], t),
            "hot_pixel_mask": (["H", "W"], mask),
            "dead_pixel_mask": (["H", "W"], mask),
        },
        attrs={"data_product": "ph256"},
    )


class TestBuiltinChecksRegistered:
    def test_img_l1_checks_registered(self) -> None:
        keys = registered_checks("L1", "img")
        assert "hot_pixel_frac" in keys
        assert "dead_pixel_frac" in keys
        assert "frame_rate_hz" in keys
        assert "finite_frac" in keys

    def test_ph_l1_checks_registered(self) -> None:
        keys = registered_checks("L1", "ph")
        assert "hot_pixel_frac" in keys
        assert "dead_pixel_frac" in keys
        assert "frame_rate_hz" in keys


class TestRunQcImg:
    def test_clean_data_all_pass(self) -> None:
        ds = _make_l1_img(n=30)
        report = run_qc(ds, level="L1", kind="img")
        assert isinstance(report, QCReport)
        assert report.isgood is True
        assert all(c.passed for c in report.checks)

    def test_report_fields(self) -> None:
        ds = _make_l1_img(n=30, n_hot_pixels=0)
        report = run_qc(ds, level="L1", kind="img")
        assert report.level == "L1"
        assert report.kind == "img"
        assert "hot_pixel_frac" in report.metrics
        assert "frame_rate_hz" in report.metrics

    def test_excessive_hot_pixels_fail(self) -> None:
        # 128 hot pixels > 10% of 32*32=1024 total → check must fail
        ds = _make_l1_img(n=30, n_hot_pixels=128)
        report = run_qc(ds, level="L1", kind="img")
        hot_check = next(c for c in report.checks if c.key == "hot_pixel_frac")
        assert not hot_check.passed
        assert not report.isgood

    def test_nan_data_fails_finite_check(self) -> None:
        ds = _make_l1_img(n=30)
        arr = ds["median_subtracted"].values.copy()
        # >5% of 30*32*32=30720 total values: set >1537 to NaN → all frames, one column (30*32=960 NaN each)
        arr[:, :, :2] = np.nan  # 30*32*2 = 1920 > 5%
        ds = ds.assign(median_subtracted=(["T", "H", "W"], arr))
        report = run_qc(ds, level="L1", kind="img")
        finite_check = next(c for c in report.checks if c.key == "finite_frac")
        assert not finite_check.passed


class TestRunQcPh:
    def test_clean_ph_data_all_pass(self) -> None:
        ds = _make_l1_ph(n=30)
        report = run_qc(ds, level="L1", kind="ph")
        assert report.isgood is True

    def test_unknown_level_kind_returns_empty(self) -> None:
        ds = _make_l1_img(n=10)
        report = run_qc(ds, level="L99", kind="img")
        assert report.checks == []
        assert report.metrics == {}
        assert report.isgood is True  # vacuously


class TestQCIsNonMutating:
    def test_run_qc_does_not_mutate_dataset(self) -> None:
        ds = _make_l1_img(n=10)
        original_attrs = dict(ds.attrs)
        run_qc(ds, level="L1", kind="img")
        assert ds.attrs == original_attrs


class TestCustomCheckRegistration:
    def test_custom_check_registered_and_run(self) -> None:
        @qc_check("L1", "test_custom", "my_check")
        def _always_pass(ds: xr.Dataset) -> QCCheckResult:
            return QCCheckResult(key="my_check", value=1.0, threshold=0.5, passed=True)

        ds = _make_l1_img(n=5)
        report = run_qc(ds, level="L1", kind="test_custom")
        assert "my_check" in report.metrics
        assert report.isgood is True
