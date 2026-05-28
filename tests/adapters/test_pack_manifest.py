"""Adapter tests for pa-pack and pa-manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoseti_analysis.adapters.calibrate import run_calibrate
from panoseti_analysis.adapters.manifest import run_manifest
from panoseti_analysis.adapters.pack import run_pack
from panoseti_analysis.io.stores import write_store


def test_pack_tar_zip_and_reject_unknown(l0_ph_ds, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store = tmp_path / "s.zarr"
    write_store(l0_ph_ds, store)
    assert run_pack(store, tmp_path / "s.tar", fmt="tar").is_file()
    assert run_pack(store, tmp_path / "s.zarr.zip", fmt="zip").is_file()
    with pytest.raises(ValueError, match="unknown pack format"):
        run_pack(store, tmp_path / "s.7z", fmt="7z")


def test_manifest_aggregates_lineage_with_checksums(make_ph, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # produce two L1 stores + their lineage fragments
    lineage_files = []
    for dp in ("ph256", "ph1024"):
        size = 16 if dp == "ph256" else 32
        l0 = tmp_path / f"run.dp_{dp}.module_1.zarr"
        write_store(make_ph(n=40, size=size, data_product=dp), l0)
        l1 = tmp_path / f"run.dp_{dp}.module_1.L1.zarr"
        lf = tmp_path / f"{dp}.lineage.json"
        run_calibrate(l0, l1, ph_stride=5, lineage_out=lf)
        lineage_files.append(lf)

    out = tmp_path / "manifest.json"
    man = run_manifest(lineage_files, "L1", "run", out, stores_dir=tmp_path)

    assert out.is_file()
    assert man.level == "L1"
    assert len(man.stores) == 2
    assert all(s.checksum and s.checksum.startswith("sha256:") for s in man.stores)
