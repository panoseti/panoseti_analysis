"""Adapter tests for pa-calibrate (L0 -> L1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from panoseti_analysis.adapters._common import SuspectTimestamps
from panoseti_analysis.adapters.calibrate import run_calibrate
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.stores import open_store, write_store


def test_calibrate_ph_writes_l1_and_lineage(make_ph, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    l0 = tmp_path / "run.dp_ph256.module_1.zarr"
    write_store(make_ph(n=60, size=16), l0)
    l1 = tmp_path / "run.dp_ph256.module_1.L1.zarr"
    lineage = tmp_path / "ph.lineage.json"

    rec = run_calibrate(l0, l1, ph_stride=5, lineage_out=lineage)

    out = open_store(l1)
    assert "pedestal_subtracted" in out.data_vars
    assert out.attrs["data_level"] == "L1"
    assert "timestamp_qc" in out.attrs
    assert rec.kind == "ph"
    assert rec.source_store == l0.name
    StoreLineage.model_validate_json(lineage.read_text())  # round-trips


def test_calibrate_img_kind_inferred(make_img, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    l0 = tmp_path / "run.dp_img16.module_1.zarr"
    write_store(make_img(n=60, size=32), l0)
    rec = run_calibrate(l0, tmp_path / "l1.zarr", img_stride=5, block=8)
    assert rec.kind == "img"
    assert "median_subtracted" in open_store(tmp_path / "l1.zarr").data_vars


def test_suspect_timestamps_abort_unless_overridden(make_ph, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ds = make_ph(n=60, size=16)
    t = ds["unix_t_ns"].values.copy()
    t[20] -= 10**15  # corruption-grade backward jump (ph suspect threshold = 1e9)
    ds = ds.assign(unix_t_ns=("time", t))
    l0 = tmp_path / "run.dp_ph256.module_1.zarr"
    write_store(ds, l0)

    with pytest.raises(SuspectTimestamps):
        run_calibrate(l0, tmp_path / "l1.zarr", ph_stride=5, fail_on_suspect=True)

    rec = run_calibrate(l0, tmp_path / "l1b.zarr", ph_stride=5, fail_on_suspect=False)
    assert rec.timestamp_qc is not None
    assert rec.timestamp_qc.status.value == "suspect"
    assert isinstance(open_store(tmp_path / "l1b.zarr"), xr.Dataset)


def test_calibrate_with_shard_factor(make_img, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """run_calibrate with shard_factor=4 must produce a sharded L1 store."""
    import zarr as _zarr

    l0 = tmp_path / "l0.zarr"
    l1 = tmp_path / "l1.zarr"
    write_store(make_img(n=200, size=32), l0)

    run_calibrate(l0, l1, kind="img", shard_factor=4)

    z = _zarr.open(str(l1), mode="r", zarr_format=3)
    # whichever array the img calibration outputs
    arr_name = next(k for k in z if hasattr(z[k], "shards"))
    assert z[arr_name].shards is not None, "shard_factor=4 must produce a sharded store"
