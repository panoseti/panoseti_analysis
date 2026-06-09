"""Tests for registry-driven load_classifier (fixes state-load bug)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from panoseti_analysis.algorithms.cloud_detector import (  # noqa: E402
    CloudDetection,
    CloudDetectionV2,
)
from panoseti_analysis.config.models import (  # noqa: E402
    ClassifierBundle,
    TrainingProvenance,
)
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION  # noqa: E402
from panoseti_analysis.io.models import load_classifier, save_classifier  # noqa: E402
from panoseti_analysis.io.provenance import now_utc  # noqa: E402


def _make_provenance() -> TrainingProvenance:
    return TrainingProvenance(
        step_name="test",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params={},
        timestamp_utc=now_utc(),
    )


def test_load_classifier_v2_round_trip() -> None:
    """save_classifier(V2) then load_classifier returns a CloudDetectionV2 — not CloudDetection."""
    model_v2 = CloudDetectionV2()
    bundle = ClassifierBundle(
        model_name="test_v2",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={"channels": 2, "height": 32, "width": 32},
        arch="cloud_detector_v2",
    )
    with tempfile.TemporaryDirectory() as td:
        pt_path, _, _ = save_classifier(model_v2, bundle, _make_provenance(), Path(td))
        loaded_model, loaded_bundle = load_classifier(pt_path)

    assert isinstance(loaded_model, CloudDetectionV2), (
        f"Expected CloudDetectionV2 but got {type(loaded_model).__name__}. "
        "State-load bug: load_classifier resolved the wrong arch."
    )
    assert loaded_bundle.arch == "cloud_detector_v2"


def test_load_classifier_legacy_round_trip() -> None:
    """Legacy sidecar (no arch field → default 'cloud_detector') loads CloudDetection."""
    model_v1 = CloudDetection()
    bundle = ClassifierBundle(
        model_name="test_v1_legacy",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={"channels": 2, "height": 32, "width": 32},
        # arch defaults to "cloud_detector" — mimics an old sidecar
    )
    with tempfile.TemporaryDirectory() as td:
        pt_path, _, _ = save_classifier(model_v1, bundle, _make_provenance(), Path(td))
        loaded_model, loaded_bundle = load_classifier(pt_path)

    assert isinstance(loaded_model, CloudDetection)
    assert loaded_bundle.arch == "cloud_detector"


def test_arch_persisted_in_json() -> None:
    """The arch field is serialised into the .json sidecar and round-trips."""
    import json

    model_v2 = CloudDetectionV2()
    bundle = ClassifierBundle(
        model_name="test_arch_json",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={},
        arch="cloud_detector_v2",
    )
    with tempfile.TemporaryDirectory() as td:
        _, json_path, _ = save_classifier(model_v2, bundle, _make_provenance(), Path(td))
        data = json.loads(json_path.read_text())

    assert data["arch"] == "cloud_detector_v2"


def test_checksum_mismatch_raises() -> None:
    """load_classifier raises ValueError if the checksum in the sidecar doesn't match the .pt."""
    import json

    model_v2 = CloudDetectionV2()
    bundle = ClassifierBundle(
        model_name="test_cksum",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={},
        arch="cloud_detector_v2",
    )
    with tempfile.TemporaryDirectory() as td:
        pt_path, json_path, _ = save_classifier(model_v2, bundle, _make_provenance(), Path(td))
        # Corrupt the checksum in the sidecar
        data = json.loads(json_path.read_text())
        data["checksum"] = "sha256:0000000000000000"
        json_path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="checksum mismatch"):
            load_classifier(pt_path)
