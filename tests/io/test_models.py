"""Tests for io/models.py — save_classifier / load_classifier round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from panoseti_analysis.algorithms.cloud_detector import CloudDetection
from panoseti_analysis.config.models import ClassifierBundle, TrainingProvenance
from panoseti_analysis.io.checksum import compute_sha256
from panoseti_analysis.io.models import load_classifier, save_classifier
from panoseti_analysis.io.provenance import now_utc


def _make_bundle(checksum: str = "sha256:placeholder") -> ClassifierBundle:
    return ClassifierBundle(
        model_name="cloud_v1",
        model_version="0.1.0",
        checksum=checksum,
        input_spec={"shape": [2, 32, 32], "dtype": "float32"},
    )


def _make_provenance(output_checksum: str | None = None) -> TrainingProvenance:
    return TrainingProvenance(
        step_name="train_cloud",
        step_version="0.1.0",
        recipe_name="cloud_v1",
        recipe_hash="sha256:abc123",
        params={"lr": 0.001, "epochs": 50},
        input_checksums=["sha256:deadbeef"],
        output_checksum=output_checksum,
        timestamp_utc=now_utc(),
        software={"git_sha": "abcdef0"},
        metrics={"val_accuracy": 0.96},
    )


def test_save_then_load_classifier(tmp_path: Path) -> None:
    """save_classifier followed by load_classifier must not raise and bundle must round-trip."""
    model = CloudDetection()
    bundle = _make_bundle()  # placeholder checksum — will be replaced by save_classifier
    prov = _make_provenance()

    pt_path, _json_path, _prov_path = save_classifier(model, bundle, prov, tmp_path)

    loaded_model, loaded_bundle = load_classifier(pt_path)

    assert isinstance(loaded_model, CloudDetection)
    assert loaded_bundle.model_name == bundle.model_name
    assert loaded_bundle.model_version == bundle.model_version
    assert loaded_bundle.input_spec == bundle.input_spec


def test_save_classifier_checksum_is_correct(tmp_path: Path) -> None:
    """The .json sidecar checksum must equal 'sha256:' + compute_sha256(pt_path)."""
    model = CloudDetection()
    bundle = _make_bundle()
    prov = _make_provenance()

    pt_path, json_path, _ = save_classifier(model, bundle, prov, tmp_path)

    data = json.loads(json_path.read_text())
    expected_checksum = f"sha256:{compute_sha256(pt_path)}"
    assert data["checksum"] == expected_checksum


def test_save_classifier_provenance_written(tmp_path: Path) -> None:
    """The _provenance.json sidecar must exist and parse as TrainingProvenance."""
    model = CloudDetection()
    bundle = _make_bundle()
    prov = _make_provenance()

    _pt_path, _json_path, provenance_path = save_classifier(model, bundle, prov, tmp_path)

    assert provenance_path.exists()
    loaded = TrainingProvenance.model_validate_json(provenance_path.read_text())
    assert loaded.step_name == prov.step_name
    assert loaded.recipe_name == prov.recipe_name


def test_save_classifier_output_checksum_stamped(tmp_path: Path) -> None:
    """If TrainingProvenance.output_checksum is None on input, it must be set in the saved file."""
    model = CloudDetection()
    bundle = _make_bundle()
    prov = _make_provenance(output_checksum=None)

    assert prov.output_checksum is None

    pt_path, _json_path, provenance_path = save_classifier(model, bundle, prov, tmp_path)

    loaded_prov = TrainingProvenance.model_validate_json(provenance_path.read_text())
    expected_checksum = f"sha256:{compute_sha256(pt_path)}"
    assert loaded_prov.output_checksum == expected_checksum


def test_save_classifier_preserves_existing_checksum(tmp_path: Path) -> None:
    """If bundle.checksum already matches the written .pt, save_classifier must keep it."""
    model = CloudDetection()
    # First save to get a real checksum, then re-save with that checksum pre-set.
    bundle = _make_bundle()
    prov = _make_provenance()
    pt_path, _json_path, _ = save_classifier(model, bundle, prov, tmp_path)
    real_checksum = f"sha256:{compute_sha256(pt_path)}"

    # Build a new bundle with the real checksum already set
    bundle_with_cksum = _make_bundle(checksum=real_checksum)
    out2 = tmp_path / "second"
    _pt2, json2, _ = save_classifier(model, bundle_with_cksum, prov, out2)

    data = json.loads(json2.read_text())
    assert data["checksum"] == real_checksum
