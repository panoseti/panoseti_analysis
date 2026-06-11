"""Tests for run_calibrate_inmem and run_pipeline skip_l0_materialize mode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from panoseti_analysis.adapters.calibrate import run_calibrate, run_calibrate_inmem
from panoseti_analysis.adapters.recipe_driver import run_pipeline
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.stores import open_store, write_store

_OBS_DIR = Path(__file__).parent.parent / "data" / "obs_TEST.pffd"


# ── Test 1: run_calibrate_inmem produces a valid L1 store ─────────────────────


def test_calibrate_inmem_img_produces_l1(make_img, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """run_calibrate_inmem with a synthetic L0 img16 Dataset produces L1 with median_subtracted."""
    ds_l0 = make_img(n=60, size=32, data_product="img16")
    l1_path = tmp_path / "run.dp_img16.module_1.L1.zarr"
    lineage_file = tmp_path / "run.dp_img16.module_1.L1.lineage.json"

    rec = run_calibrate_inmem(ds_l0, l1_path, img_stride=5, lineage_out=lineage_file)

    out = open_store(l1_path)
    assert "median_subtracted" in out.data_vars
    assert out.attrs["data_level"] == "L1"
    assert "timestamp_qc" in out.attrs
    assert rec.kind == "img"
    assert rec.level == "L1"
    assert rec.store == l1_path.name
    # No source_store for in-memory path
    assert rec.source_store is None
    StoreLineage.model_validate_json(lineage_file.read_text())  # round-trips cleanly


def test_calibrate_inmem_ph_produces_l1(make_ph, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """run_calibrate_inmem with a synthetic L0 ph256 Dataset produces L1 with pedestal_subtracted."""
    ds_l0 = make_ph(n=60, size=16, data_product="ph256")
    l1_path = tmp_path / "run.dp_ph256.module_1.L1.zarr"

    rec = run_calibrate_inmem(ds_l0, l1_path, ph_stride=5)

    out = open_store(l1_path)
    assert "pedestal_subtracted" in out.data_vars
    assert out.attrs["data_level"] == "L1"
    assert rec.kind == "ph"


# ── Test 2: run_calibrate_inmem output matches run_calibrate for the same L0 ──


def test_calibrate_inmem_matches_run_calibrate(make_img, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """run_calibrate_inmem and run_calibrate produce identical L1 data for the same L0."""
    ds_l0 = make_img(n=60, size=32, data_product="img16", seed=42)

    # Path-based: write L0 to disk first, then calibrate
    l0_path = tmp_path / "l0.zarr"
    write_store(ds_l0, l0_path)
    l1_path_disk = tmp_path / "l1_disk.zarr"
    run_calibrate(l0_path, l1_path_disk, img_stride=5, block=8)

    # In-memory: feed the same Dataset directly
    l1_path_inmem = tmp_path / "l1_inmem.zarr"
    run_calibrate_inmem(ds_l0, l1_path_inmem, img_stride=5, block=8)

    disk_ds = open_store(l1_path_disk)
    inmem_ds = open_store(l1_path_inmem)

    # Core data variables must be numerically identical
    for var in ("median_subtracted",):
        np.testing.assert_array_equal(
            disk_ds[var].values,
            inmem_ds[var].values,
            err_msg=f"Variable {var!r} differs between disk and in-memory paths",
        )

    # Both must be L1
    assert disk_ds.attrs["data_level"] == inmem_ds.attrs["data_level"] == "L1"


# ── Test 3: run_pipeline with skip_l0_materialize=True ────────────────────────


def test_skip_l0_materialize_produces_l1_no_l0_zarr(tmp_path: Path) -> None:
    """run_pipeline(skip_l0_materialize=True) produces L1 stores with no L0 Zarr on disk."""
    out_dir = tmp_path / "out"
    outputs = run_pipeline(_OBS_DIR, out_dir, skip_l0_materialize=True)

    # L1 stores must be produced
    assert len(outputs.l1_stores) > 0
    for rec in outputs.l1_stores:
        assert isinstance(rec, StoreLineage)
        assert rec.level == "L1"
        assert "L1" in rec.store

    # L0 stores list must be empty (no L0 written)
    assert outputs.l0_stores == []

    # L0 directory must exist (created by mkdir) but contain no .zarr stores
    l0_dir = out_dir / "L0"
    assert l0_dir.exists()
    l0_zarr_stores = list(l0_dir.glob("*.zarr"))
    assert l0_zarr_stores == [], (
        f"Expected no L0 .zarr stores when skip_l0_materialize=True, found: {l0_zarr_stores}"
    )

    # L1 Zarr stores must be on disk
    l1_dir = out_dir / "L1"
    for rec in outputs.l1_stores:
        l1_path = l1_dir / rec.store
        assert l1_path.exists(), f"Expected L1 store on disk: {l1_path}"


def test_skip_l0_materialize_with_mock(make_img, tmp_path: Path) -> None:
    """run_pipeline skip_l0_materialize=True with mocked sequence_to_dataset + PanosetiRun."""
    ds_l0 = make_img(n=60, size=32, data_product="img16")

    class _MockSeq:
        pass

    class _MockRun:
        def list_products(self) -> list[str]:
            return ["dp_img16.bpp_2.module_1"]

        def get_product(self, dp: str) -> _MockSeq:
            return _MockSeq()

    out_dir = tmp_path / "out"

    with (
        patch(
            "panoseti_analysis.adapters.recipe_driver.read_pff_run",
            return_value=_MockRun(),
        ),
        patch("pypff.zarr.sequence_to_dataset", return_value=ds_l0),
    ):
        outputs = run_pipeline(
            tmp_path / "fake.pffd",
            out_dir,
            skip_l0_materialize=True,
        )

    assert outputs.l0_stores == []
    assert len(outputs.l1_stores) == 1
    assert outputs.l1_stores[0].level == "L1"
    assert outputs.l1_stores[0].kind == "img"

    l0_dir = out_dir / "L0"
    assert not list(l0_dir.glob("*.zarr")), "No L0 .zarr stores expected"
