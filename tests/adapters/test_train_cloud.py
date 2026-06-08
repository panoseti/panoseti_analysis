"""Tests for adapters/ray/train_cloud.py — pa-train-cloud adapter (T8).

Fast unit tests run without Ray (test 2).
Slow integration test (test 1) spawns a real Ray Train cluster — skip with -m 'not slow'.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# The integration test (test_train_cloud_checkpoint_round_trip) is marked slow.
# The unit test (test_train_cloud_save_classifier_load_classifier_roundtrip) is fast.

# ---------------------------------------------------------------------------
# Helpers shared between tests
# ---------------------------------------------------------------------------

_EPOCH_NS = 1_700_000_000_000_000_000


def _make_l1_img_store(tmp_path: Path, *, n_frames: int = 200, module: str = "1") -> Path:
    """Create a minimal L1 img store with median_subtracted, suitable for feature extraction."""
    cadence_ns = 1_000_000_000  # 1 s per frame
    t_arr = _EPOCH_NS + np.arange(n_frames, dtype=np.int64) * cadence_ns
    rng = np.random.default_rng(42)
    img = rng.standard_normal((n_frames, 32, 32)).astype(np.float32) * 10.0

    store_path = tmp_path / f"obs_TEST.dp_img16.module_{module}.L1.zarr"
    ds = xr.Dataset(
        data_vars={
            "median_subtracted": (["time", "y", "x"], img),
            "unix_t_ns": (["time"], t_arr),
        },
        attrs={
            "data_product": "img16",
            "module": module,
            "data_level": "L1",
        },
    )
    ds.to_zarr(str(store_path), zarr_format=3)
    return store_path


def _write_test_recipe(
    path: Path, *, epochs: int = 2, batch_size: int = 8, num_workers: int = 1
) -> None:
    """Write a minimal cloud_train recipe YAML to *path*."""
    content = f"""\
name: cloud_train_test
feature_cadence_s: 60.0
threshold: 0.5
split:
  test_prop: 0.15
  val_prop: 0.10
  boundary_prop: 0.20
  seed: 35
hyperparams:
  lr: 0.01
  batch_size: {batch_size}
  epochs: {epochs}
  weight_decay: 0.00001
  gamma: 0.9
scaling:
  num_workers: {num_workers}
  accelerator_type: null
"""
    path.write_text(content)


def _write_label_csv(
    path: Path,
    *,
    module: str,
    t_start_ns: int,
    t_end_ns: int,
    block_ns: int = 120_000_000_000,
    width_ns: int = 60_000_000_000,
) -> None:
    """Write a (module, t_start_ns, t_end_ns, label) CSV that labels every other 60 s
    block as cloudy (1); unlabelled windows default to clear (0). Spreading both classes
    across the whole timeline keeps the temporal train/val split non-degenerate.
    """
    rows = ["module,t_start_ns,t_end_ns,label"]
    t = t_start_ns
    while t < t_end_ns:
        rows.append(f"{module},{t},{t + width_ns},1")
        t += block_ns
    path.write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# Test 2 (fast, no Ray) — save_classifier / load_classifier round-trip
# ---------------------------------------------------------------------------


def test_train_cloud_save_classifier_load_classifier_roundtrip(tmp_path: Path) -> None:
    """Unit test: save_classifier + load_classifier must round-trip a CloudDetection model."""
    pytest.importorskip("torch")

    from panoseti_analysis.algorithms.cloud_detector import CloudDetection
    from panoseti_analysis.config.models import ClassifierBundle, TrainingProvenance
    from panoseti_analysis.io.models import load_classifier, save_classifier
    from panoseti_analysis.io.provenance import now_utc

    model = CloudDetection()

    bundle = ClassifierBundle(
        model_name="cloud_detector_retrained",
        model_version="2.0",
        checksum="sha256:placeholder",
        input_spec={
            "channels": 2,
            "height": 32,
            "width": 32,
            "dtype": "float32",
        },
    )

    prov = TrainingProvenance(
        step_name="train_cloud",
        step_version="2.0",
        recipe_name="cloud_train_test",
        recipe_hash="sha256:abc123",
        params={"lr": 0.01, "epochs": 2, "batch_size": 8},
        input_checksums=["sha256:deadbeef"],
        timestamp_utc=now_utc(),
        software={"git_sha": "abcdef0"},
        metrics={"val_loss": 0.5, "val_acc": 0.8},
    )

    pt_path, json_path, provenance_path = save_classifier(model, bundle, prov, tmp_path)

    # Verify files exist
    assert pt_path.exists(), f"pt_path missing: {pt_path}"
    assert json_path.exists(), f"json_path missing: {json_path}"
    assert provenance_path.exists(), f"provenance_path missing: {provenance_path}"

    # load_classifier must succeed (checksum must match)
    loaded_model, loaded_bundle = load_classifier(pt_path)

    assert isinstance(loaded_model, CloudDetection), (
        f"Expected CloudDetection, got {type(loaded_model)}"
    )
    assert loaded_bundle.model_name == bundle.model_name
    assert loaded_bundle.model_version == bundle.model_version
    assert loaded_bundle.input_spec == bundle.input_spec
    assert loaded_bundle.checksum.startswith("sha256:")
    assert loaded_bundle.checksum != "sha256:placeholder"

    # provenance must parse as TrainingProvenance
    loaded_prov = TrainingProvenance.model_validate_json(provenance_path.read_text())
    assert loaded_prov.step_name == "train_cloud"
    assert loaded_prov.output_checksum is not None
    assert loaded_prov.output_checksum.startswith("sha256:")


# ---------------------------------------------------------------------------
# Test 1 (slow, requires Ray Train) — end-to-end checkpoint round-trip
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_cloud_checkpoint_round_trip(tmp_path: Path) -> None:
    """Integration test: run_train_cloud must produce a valid loadable checkpoint.

    Uses the standalone Ray launcher (real multi-process, num_workers=1) with a
    minimal recipe (2 epochs, batch_size=8) to keep CI runtime manageable.
    """
    pytest.importorskip("ray")
    pytest.importorskip("torch")

    import ray

    from panoseti_analysis.adapters.features import run_features_cloud
    from panoseti_analysis.adapters.ray.train_cloud import run_train_cloud
    from panoseti_analysis.algorithms.cloud_detector import CloudDetection
    from panoseti_analysis.config.models import TrainingProvenance
    from panoseti_analysis.io.models import load_classifier

    # Shut down any pre-existing Ray instance to get a clean slate
    if ray.is_initialized():
        ray.shutdown()

    try:
        # 1. Build a synthetic L1 img store (3600 s at 1 Hz → ~60 features at 60 s cadence,
        #    enough for a non-degenerate train/val/test split with both classes present).
        l1_dir = tmp_path / "l1"
        l1_dir.mkdir()
        n_frames = 3600
        l1_store = _make_l1_img_store(l1_dir, n_frames=n_frames)

        # 2. Write a test recipe pointing to the features adapter
        features_recipe = tmp_path / "cloud_v1.yml"
        _write_test_recipe(features_recipe, epochs=2, batch_size=2, num_workers=1)

        # 3. Materialise a feature cache WITH labels (a label CSV — without one every sample
        #    is labelled -1 and cross-entropy rejects it).
        label_csv = tmp_path / "labels.csv"
        _write_label_csv(
            label_csv,
            module="1",
            t_start_ns=_EPOCH_NS,
            t_end_ns=_EPOCH_NS + (n_frames - 1) * 1_000_000_000,
        )
        feature_cache = tmp_path / "features.zarr"
        run_features_cloud([l1_store], feature_cache, features_recipe, label_csv=label_csv)

        # 4. Write the train recipe (same file, num_workers=1, few epochs)
        train_recipe = tmp_path / "cloud_train.yml"
        _write_test_recipe(train_recipe, epochs=2, batch_size=2, num_workers=1)

        out_dir = tmp_path / "model_out"
        lineage_out = tmp_path / "train.lineage.json"

        # 5. Run training
        pt_path, json_path, provenance_path = run_train_cloud(
            feature_cache=feature_cache,
            out_dir=out_dir,
            recipe_path=train_recipe,
            launcher="standalone",
            lineage_out=lineage_out,
        )

        # 6. Assertions: files exist
        assert pt_path.exists(), f"pt_path missing: {pt_path}"
        assert json_path.exists(), f"json_path missing: {json_path}"
        assert provenance_path.exists(), f"provenance_path missing: {provenance_path}"

        # 7. load_classifier must succeed (checksum must match)
        loaded_model, _loaded_bundle = load_classifier(pt_path)
        assert isinstance(loaded_model, CloudDetection), (
            f"Expected CloudDetection, got {type(loaded_model)}"
        )

        # 8. Provenance must parse
        loaded_prov = TrainingProvenance.model_validate_json(provenance_path.read_text())
        assert loaded_prov.step_name == "train_cloud"
        assert loaded_prov.output_checksum is not None
        assert loaded_prov.output_checksum.startswith("sha256:")
        assert "val_loss" in loaded_prov.metrics or "val_acc" in loaded_prov.metrics

        # 9. Lineage JSON must exist and have expected keys
        assert lineage_out.exists()
        lineage_data = json.loads(lineage_out.read_text())
        assert lineage_data["step_name"] == "train_cloud"
        assert lineage_data["model_bundle"] == json_path.name
        assert lineage_data["input_checksums"][0].startswith("sha256:")

    finally:
        if ray.is_initialized():
            ray.shutdown()
