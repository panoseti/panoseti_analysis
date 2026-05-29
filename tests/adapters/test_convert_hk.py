"""Integration adapter tests for pa-convert and pa-hk against obs_TEST.pffd."""

from __future__ import annotations

import json
from pathlib import Path

from panoseti_analysis.adapters.convert import run_convert
from panoseti_analysis.adapters.hk import run_hk

_OBS = Path(__file__).parents[1] / "data" / "obs_TEST.pffd"


def test_convert_obs_test_produces_two_l0_stores(tmp_path: Path) -> None:
    lineage = tmp_path / "l0_lineage.json"
    records = run_convert(_OBS, tmp_path, lineage_out=lineage)

    assert len(records) == 2
    assert {r.kind for r in records} == {"ph", "img"}
    for rec in records:
        assert rec.level == "L0"
        assert (tmp_path / rec.store).is_dir()
        assert rec.n_frames > 0

    arr = json.loads(lineage.read_text())
    assert len(arr) == 2
    assert all("store" in r and "kind" in r for r in arr)


def test_hk_without_hk_file_returns_empty(tmp_path: Path) -> None:
    # obs_TEST has no hk.pff -> no HK stores written
    assert run_hk(_OBS, tmp_path / "hk") == []
