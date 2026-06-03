"""Tests for io/qc.py — sidecar writer + stamp_qc."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from panoseti_analysis.algorithms.qc import run_qc
from panoseti_analysis.io.qc import stamp_qc, write_qc_sidecar


def _make_l1_img(n: int = 20) -> xr.Dataset:
    rng = np.random.default_rng(0)
    img = rng.normal(10.0, 2.0, (n, 32, 32)).astype(np.float32)
    t = 1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 20_000
    mask = np.zeros((32, 32), dtype=np.uint8)
    return xr.Dataset(
        {
            "median_subtracted": (["T", "H", "W"], img),
            "unix_t_ns": (["T"], t),
            "hot_pixel_mask": (["H", "W"], mask),
            "dead_pixel_mask": (["H", "W"], mask),
        },
        attrs={"data_product": "img16", "data_level": "L1"},
    )


class TestWriteQcSidecar:
    def test_sidecar_is_valid_json(self, tmp_path: Path) -> None:
        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        sidecar = tmp_path / "store.qc.json"
        write_qc_sidecar(report, sidecar)
        data = json.loads(sidecar.read_text())
        assert "isgood" in data
        assert "metrics" in data
        assert "checks" in data

    def test_sidecar_round_trips(self, tmp_path: Path) -> None:
        from panoseti_analysis.algorithms.qc import QCReport

        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        sidecar = tmp_path / "store.qc.json"
        write_qc_sidecar(report, sidecar)
        loaded = QCReport.model_validate_json(sidecar.read_text())
        assert loaded.isgood == report.isgood
        assert loaded.metrics == report.metrics


class TestStampQc:
    def test_qc_attr_present_after_stamp(self) -> None:
        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        stamped = stamp_qc(ds, report)
        assert "qc" in stamped.attrs
        assert isinstance(stamped.attrs["qc"], dict)

    def test_stamp_is_non_mutating(self) -> None:
        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        _ = stamp_qc(ds, report)
        assert "qc" not in ds.attrs

    def test_qc_attr_json_clean(self) -> None:
        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        stamped = stamp_qc(ds, report)
        qc_val = stamped.attrs["qc"]
        # Must be JSON-serializable (no pydantic objects, no numpy scalars)
        json.dumps(qc_val)

    def test_data_level_preserved(self) -> None:
        ds = _make_l1_img()
        report = run_qc(ds, level="L1", kind="img")
        stamped = stamp_qc(ds, report)
        assert stamped.attrs.get("data_level") == "L1"


class TestQcIntegrationWithCalibrate:
    def test_calibrate_stamps_qc_into_l1(
        self,
        make_img: object,
        tmp_path: Path,  # type: ignore[no-untyped-def]
    ) -> None:
        from panoseti_analysis.adapters.calibrate import run_calibrate
        from panoseti_analysis.io.stores import open_store, write_store

        l0 = tmp_path / "run.dp_img16.module_1.zarr"
        write_store(make_img(n=60, size=32), l0)  # type: ignore[call-arg]
        l1 = tmp_path / "run.dp_img16.module_1.L1.zarr"
        qc_sidecar = tmp_path / "run.qc.json"

        run_calibrate(l0, l1, img_stride=5, qc_out=qc_sidecar)

        ds = open_store(l1)
        assert "qc" in ds.attrs, "QC report must be stamped into L1 attrs"
        assert "isgood" in ds.attrs["qc"]
        assert qc_sidecar.exists(), "QC sidecar JSON must be written"
        data = json.loads(qc_sidecar.read_text())
        assert "metrics" in data
