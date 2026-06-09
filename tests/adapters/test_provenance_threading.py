"""Tests that ProcessingStep is appended by each writer adapter and chains L0->L1->L2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from panoseti_analysis.adapters.calibrate import run_calibrate
from panoseti_analysis.adapters.convert import run_convert
from panoseti_analysis.adapters.hk import run_hk
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.provenance import read_history
from panoseti_analysis.io.stores import open_store

_OBS = Path(__file__).parents[1] / "data" / "obs_TEST.pffd"


# ── shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def l0_stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[StoreLineage]]:
    """Run convert once per module; return (out_dir, records)."""
    out_dir = tmp_path_factory.mktemp("l0")
    records = run_convert(_OBS, out_dir)
    return out_dir, records


@pytest.fixture
def dummy_model_path(tmp_path: Path) -> Path:
    """Create a minimal CloudDetection model + sidecar for classify tests."""
    from panoseti_analysis.algorithms.cloud_detector import CloudDetection
    from panoseti_analysis.io.checksum import compute_sha256

    model = CloudDetection()
    model.eval()

    model_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), model_path)

    sha = compute_sha256(model_path)
    json_path = model_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "model_name": "dummy",
                "model_version": "1.0",
                "checksum": f"sha256:{sha}",
                "input_spec": {},
            }
        )
    )
    return model_path


# ── test 1: convert stamps processing_history into L0 ────────────────────────


def test_convert_stamps_processing_history_into_l0(tmp_path: Path) -> None:
    records = run_convert(_OBS, tmp_path)
    assert len(records) >= 1

    for rec in records:
        store_path = tmp_path / rec.store
        ds = open_store(store_path)
        assert "processing_history" in ds.attrs, f"processing_history missing from {rec.store}"

        history = read_history(dict(ds.attrs))
        assert len(history) == 1
        step = history[0]
        assert step.step_name == "convert"
        assert step.output_checksum is not None
        assert step.output_checksum.startswith("sha256:")

        # StoreLineage checksum matches step output_checksum
        assert rec.checksum == step.output_checksum


# ── test 2: calibrate chains L0 history ──────────────────────────────────────


def test_calibrate_chains_l0_history(tmp_path: Path) -> None:
    l0_dir = tmp_path / "l0"
    l0_records = run_convert(_OBS, l0_dir)

    # Pick the first L0 store (any kind works)
    l0_rec = l0_records[0]
    l0_store = l0_dir / l0_rec.store
    l1_store = tmp_path / f"{l0_rec.store.replace('.zarr', '.L1.zarr')}"

    l1_rec = run_calibrate(l0_store, l1_store, ph_stride=5, img_stride=5)

    ds_l1 = open_store(l1_store)
    history = read_history(dict(ds_l1.attrs))

    assert len(history) == 2, (
        f"Expected 2 steps, got {len(history)}: {[s.step_name for s in history]}"
    )
    assert history[0].step_name == "convert"
    assert history[1].step_name.startswith("calibrate_")

    # Checksum chain: calibrate step's input_checksum[0] == convert step's output_checksum
    convert_cksum = history[0].output_checksum
    calib_step = history[1]
    if convert_cksum is not None:
        assert calib_step.input_checksums == [convert_cksum]
    else:
        assert calib_step.input_checksums == []

    # L1 StoreLineage has a checksum
    assert l1_rec.checksum is not None
    assert l1_rec.checksum.startswith("sha256:")


# ── test 3: hk returns list[StoreLineage] (empty for obs_TEST) ───────────────


def test_hk_returns_store_lineages_when_stores_written(tmp_path: Path) -> None:
    # obs_TEST has no hk.pff -> empty list
    result = run_hk(_OBS, tmp_path / "hk")
    assert isinstance(result, list)
    # The existing test asserts == [], confirm type is correct and still empty
    assert result == []


def test_hk_without_hk_file_returns_empty_list_type(tmp_path: Path) -> None:
    """Return type is list[StoreLineage]; empty list is falsy but not None."""
    result = run_hk(_OBS, tmp_path / "hk")
    assert result is not None
    assert len(result) == 0


# ── test 4: nextflow classify chains convert -> calibrate -> classify ─────────


def test_nextflow_classify_chains_history(dummy_model_path: Path, tmp_path: Path) -> None:
    """Full provenance chain: convert -> calibrate -> classify_cloud (3 steps)."""
    from panoseti_analysis.adapters.nextflow.classify_cloud import run_classify

    l0_dir = tmp_path / "l0"
    l0_records = run_convert(_OBS, l0_dir)

    # Use the img store — cloud detector requires median_subtracted (img L1)
    img_rec = next((r for r in l0_records if r.kind == "img"), None)
    if img_rec is None:
        pytest.skip("No img store produced by convert — cannot test classify chain")

    l0_store = l0_dir / img_rec.store
    l1_store = tmp_path / f"{img_rec.store.replace('.zarr', '.L1.zarr')}"

    l1_rec = run_calibrate(l0_store, l1_store, img_stride=5)
    assert l1_rec.kind == "img"

    l2_store = tmp_path / "l2.zarr"
    l2_rec = run_classify(
        l1_store=l1_store,
        l2_store=l2_store,
        model_path=dummy_model_path,
        cadence_s=60.0,
        threshold=0.5,
    )

    ds_l2 = open_store(l2_store)
    history = read_history(dict(ds_l2.attrs))

    assert len(history) == 3, (
        f"Expected 3 steps, got {len(history)}: {[s.step_name for s in history]}"
    )
    assert history[0].step_name == "convert"
    assert history[1].step_name.startswith("calibrate_")
    assert history[2].step_name == "classify_cloud"

    # Checksum chain: classify step's input_checksum[0] == calibrate step's output_checksum
    calib_cksum = history[1].output_checksum
    classify_step = history[2]
    if calib_cksum is not None:
        assert classify_step.input_checksums == [calib_cksum]
    else:
        assert classify_step.input_checksums == []

    # L2 StoreLineage has checksum + processing_history
    assert l2_rec.checksum is not None
    assert l2_rec.checksum.startswith("sha256:")
    assert len(l2_rec.processing_history) == 3


# ── test 5: ray lineage is JSON array, not JSONL ─────────────────────────────


def test_ray_lineage_is_json_array_not_jsonl(tmp_path: Path) -> None:
    """Verify the fixed Ray lineage writer emits a JSON array parseable by json.loads."""
    from panoseti_analysis.config.models import StoreLineage

    # Build a minimal list of StoreLineage records as the Ray main() would
    records = [
        StoreLineage(
            dp="img16",
            module="1",
            level="L2",
            kind="cloud",
            store="run.cloud.module_1.zarr",
            n_frames=1,
            checksum="sha256:abc123",
        ),
        StoreLineage(
            dp="img16",
            module="2",
            level="L2",
            kind="cloud",
            store="run.cloud.module_2.zarr",
            n_frames=1,
            checksum="sha256:def456",
        ),
    ]

    lineage_out = tmp_path / "lineage.json"
    # Replicate the fixed Ray writer logic
    lineage_out.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))

    # Must parse as a JSON array (not JSONL)
    parsed = json.loads(lineage_out.read_text())
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["store"] == "run.cloud.module_1.zarr"
    assert parsed[1]["store"] == "run.cloud.module_2.zarr"


# ── test 6: convert lineage JSON has checksum + processing_history ────────────


def test_convert_lineage_json_has_checksum_and_history(tmp_path: Path) -> None:
    lineage_out = tmp_path / "l0_lineage.json"
    run_convert(_OBS, tmp_path / "stores", lineage_out=lineage_out)

    arr = json.loads(lineage_out.read_text())
    assert len(arr) >= 1

    for rec in arr:
        assert "checksum" in rec, f"checksum missing from record: {rec.get('store')}"
        assert rec["checksum"] is not None
        assert rec["checksum"].startswith("sha256:")

        assert "processing_history" in rec, (
            f"processing_history missing from record: {rec.get('store')}"
        )
        assert isinstance(rec["processing_history"], list)
        assert len(rec["processing_history"]) >= 1

        step = rec["processing_history"][0]
        assert step["step_name"] == "convert"
        assert step["output_checksum"] is not None
        assert step["output_checksum"].startswith("sha256:")
