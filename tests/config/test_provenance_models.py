"""Tests for ProcessingStep, TrainingProvenance, and StoreLineage.processing_history.

TDD red-phase: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from panoseti_analysis.config.models import (
    Manifest,
    ProcessingStep,
    StoreLineage,
    TimestampQC,
    TimestampQCStatus,
    TrainingProvenance,
)
from panoseti_analysis.config.versions import (
    MANIFEST_SCHEMA_VERSION,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
    PROVENANCE_SCHEMA_VERSION,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_step(**overrides: object) -> ProcessingStep:
    defaults: dict[str, object] = {
        "step_name": "convert",
        "step_version": "1.2.3",
        "params": {"chunk_frames": 512},
        "timestamp_utc": "2026-05-29T00:00:00Z",
    }
    defaults.update(overrides)
    return ProcessingStep(**defaults)  # type: ignore[arg-type]


def _make_lineage(**overrides: object) -> StoreLineage:
    defaults: dict[str, object] = {
        "dp": "ph256",
        "module": "1",
        "level": "L1",
        "kind": "ph",
        "store": "run.dp_ph256.module_1.zarr",
        "n_frames": 10,
    }
    defaults.update(overrides)
    return StoreLineage(**defaults)  # type: ignore[arg-type]


# ── ProcessingStep ─────────────────────────────────────────────────────────────


class TestProcessingStep:
    def test_creates_with_required_fields(self) -> None:
        step = _make_step()
        assert step.step_name == "convert"
        assert step.step_version == "1.2.3"
        assert step.params == {"chunk_frames": 512}
        assert step.timestamp_utc == "2026-05-29T00:00:00Z"

    def test_optional_fields_default_to_none_or_empty(self) -> None:
        step = _make_step()
        assert step.recipe_name is None
        assert step.recipe_hash is None
        assert step.input_checksums == []
        assert step.output_checksum is None
        assert step.software == {}
        assert step.nextflow_lineage_id is None

    def test_optional_fields_accept_values(self) -> None:
        step = _make_step(
            recipe_name="ingest_v1",
            recipe_hash="sha256:deadbeef",
            input_checksums=["sha256:aaa", "sha256:bbb"],
            output_checksum="sha256:ccc",
            software={"git_sha": "abc123", "package_version": "0.1.0", "container": "ghcr.io/x:1"},
            nextflow_lineage_id="lid://abc",
        )
        assert step.recipe_name == "ingest_v1"
        assert step.recipe_hash == "sha256:deadbeef"
        assert step.input_checksums == ["sha256:aaa", "sha256:bbb"]
        assert step.output_checksum == "sha256:ccc"
        assert step.software["git_sha"] == "abc123"
        assert step.nextflow_lineage_id == "lid://abc"

    def test_json_round_trip(self) -> None:
        step = _make_step(
            recipe_name="ingest_v1",
            software={"git_sha": "abc123"},
        )
        blob = step.model_dump_json()
        restored = ProcessingStep.model_validate_json(blob)
        assert restored == step

    def test_minimal_json_round_trip(self) -> None:
        step = _make_step()
        blob = step.model_dump_json()
        restored = ProcessingStep.model_validate_json(blob)
        assert restored == step

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ProcessingStep(  # type: ignore[call-arg]
                step_name="x",
                step_version="1",
                params={},
                timestamp_utc="2026-01-01T00:00:00Z",
                bogus_field="bad",
            )

    def test_frozen(self) -> None:
        step = _make_step()
        with pytest.raises((ValidationError, TypeError)):
            step.step_name = "mutated"  # type: ignore[misc]

    def test_missing_required_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            # Missing params and timestamp_utc
            ProcessingStep(step_name="x", step_version="1")  # type: ignore[call-arg]

    def test_all_documented_step_names_work(self) -> None:
        for name in (
            "convert",
            "calibrate_ph",
            "cloud_features",
            "classify_cloud",
            "train_cloud",
            "train_vae",
        ):
            step = _make_step(step_name=name)
            assert step.step_name == name

    # ── duration_s field ──────────────────────────────────────────────────────

    def test_duration_s_defaults_to_none(self) -> None:
        step = _make_step()
        assert step.duration_s is None

    def test_duration_s_accepts_float(self) -> None:
        step = _make_step(duration_s=1.5)
        assert step.duration_s == pytest.approx(1.5)

    def test_duration_s_json_round_trip_with_value(self) -> None:
        step = _make_step(duration_s=1.5)
        blob = step.model_dump_json()
        restored = ProcessingStep.model_validate_json(blob)
        assert restored == step
        assert restored.duration_s == pytest.approx(1.5)

    def test_duration_s_json_round_trip_without_value(self) -> None:
        step = _make_step()
        blob = step.model_dump_json()
        restored = ProcessingStep.model_validate_json(blob)
        assert restored == step
        assert restored.duration_s is None

    def test_old_json_without_duration_s_loads(self) -> None:
        """Backwards-compat: JSON written before duration_s was added still loads."""
        old_json = """{
            "step_name": "convert",
            "step_version": "1.0",
            "params": {},
            "timestamp_utc": "2026-01-01T00:00:00Z"
        }"""
        step = ProcessingStep.model_validate_json(old_json)
        assert step.duration_s is None


# ── TrainingProvenance ─────────────────────────────────────────────────────────


class TestTrainingProvenance:
    def _make_tp(self, **overrides: object) -> TrainingProvenance:
        defaults: dict[str, object] = {
            "step_name": "train_cloud",
            "step_version": "0.5.0",
            "params": {"lr": 1e-3, "batch": 32, "epochs": 50},
            "timestamp_utc": "2026-05-29T12:00:00Z",
        }
        defaults.update(overrides)
        return TrainingProvenance(**defaults)  # type: ignore[arg-type]

    def test_creates_with_required_fields(self) -> None:
        tp = self._make_tp()
        assert tp.step_name == "train_cloud"
        assert tp.step_version == "0.5.0"
        assert tp.params == {"lr": 1e-3, "batch": 32, "epochs": 50}
        assert tp.timestamp_utc == "2026-05-29T12:00:00Z"

    def test_optional_fields_default(self) -> None:
        tp = self._make_tp()
        assert tp.recipe_name is None
        assert tp.recipe_hash is None
        assert tp.input_checksums == []
        assert tp.output_checksum is None
        assert tp.software == {}
        assert tp.metrics == {}
        assert tp.wandb_run_id is None

    def test_metrics_dict_works(self) -> None:
        tp = self._make_tp(metrics={"val_accuracy": 0.96, "val_ap": 0.98})
        assert tp.metrics["val_accuracy"] == pytest.approx(0.96)
        assert tp.metrics["val_ap"] == pytest.approx(0.98)

    def test_json_round_trip(self) -> None:
        tp = self._make_tp(
            software={"git_sha": "xyz", "ray_version": "2.9", "torch_version": "2.2"},
            metrics={"val_accuracy": 0.96},
            wandb_run_id="run-42abc",
        )
        blob = tp.model_dump_json()
        restored = TrainingProvenance.model_validate_json(blob)
        assert restored == tp

    def test_minimal_json_round_trip(self) -> None:
        tp = self._make_tp()
        blob = tp.model_dump_json()
        restored = TrainingProvenance.model_validate_json(blob)
        assert restored == tp

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            TrainingProvenance(  # type: ignore[call-arg]
                step_name="x",
                step_version="1",
                params={},
                timestamp_utc="2026-01-01T00:00:00Z",
                bogus_field="bad",
            )

    def test_frozen(self) -> None:
        tp = self._make_tp()
        with pytest.raises((ValidationError, TypeError)):
            tp.step_name = "mutated"  # type: ignore[misc]

    def test_train_vae_name(self) -> None:
        tp = self._make_tp(step_name="train_vae")
        assert tp.step_name == "train_vae"


# ── StoreLineage.processing_history ───────────────────────────────────────────


class TestStoreLineageProcessingHistory:
    def test_processing_history_defaults_to_empty_list(self) -> None:
        sl = _make_lineage()
        assert sl.processing_history == []

    def test_existing_fields_unaffected(self) -> None:
        qc = TimestampQC(
            status=TimestampQCStatus.CLEAN,
            monotonic=True,
            n_frames=10,
            t_start_ns=1000,
            t_end_ns=2000,
        )
        sl = _make_lineage(
            checksum="sha256:abc",
            source_store="L0/x.zarr",
            calibration_params={"sigma_threshold": 5.0},
            timestamp_qc=qc,
        )
        assert sl.checksum == "sha256:abc"
        assert sl.timestamp_qc is not None
        assert sl.timestamp_qc.status is TimestampQCStatus.CLEAN
        assert sl.processing_history == []

    def test_processing_history_with_steps(self) -> None:
        step1 = _make_step(step_name="convert", step_version="1.0")
        step2 = _make_step(
            step_name="calibrate_ph", step_version="2.0", params={"sigma_threshold": 5.0}
        )
        sl = _make_lineage(processing_history=[step1, step2])
        assert len(sl.processing_history) == 2
        assert sl.processing_history[0].step_name == "convert"
        assert sl.processing_history[1].step_name == "calibrate_ph"

    def test_processing_history_json_round_trip(self) -> None:
        step1 = _make_step(step_name="convert")
        step2 = _make_step(step_name="calibrate_ph", params={"sigma_threshold": 5.0})
        sl = _make_lineage(processing_history=[step1, step2])
        blob = sl.model_dump_json()
        restored = StoreLineage.model_validate_json(blob)
        assert restored == sl
        assert len(restored.processing_history) == 2
        assert restored.processing_history[0].step_name == "convert"

    def test_old_json_without_processing_history_loads(self) -> None:
        """Backwards-compat: JSON written before schema v2 has no processing_history field."""
        old_json = """{
            "dp": "ph256",
            "module": "1",
            "level": "L1",
            "kind": "ph",
            "store": "run.dp_ph256.module_1.zarr",
            "n_frames": 5
        }"""
        sl = StoreLineage.model_validate_json(old_json)
        assert sl.processing_history == []

    def test_processing_history_extra_keys_forbidden(self) -> None:
        """ProcessingStep inside StoreLineage also rejects extra keys."""
        with pytest.raises(ValidationError):
            StoreLineage(
                dp="ph256",
                module="1",
                level="L1",
                kind="ph",
                store="x.zarr",
                n_frames=1,
                processing_history=[
                    {  # type: ignore[list-item]
                        "step_name": "convert",
                        "step_version": "1",
                        "params": {},
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "bogus": "bad",
                    }
                ],
            )


# ── Manifest with processing_history ──────────────────────────────────────────


class TestManifestWithProcessingHistory:
    def test_manifest_round_trip_with_history(self) -> None:
        step = _make_step(step_name="convert", software={"git_sha": "abc"})
        sl = _make_lineage(
            checksum="sha256:xxx",
            processing_history=[step],
        )
        man = Manifest(
            panoseti_analysis_storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            run_id="obs_20260101",
            level="L1",
            created_utc="2026-05-29T00:00:00Z",
            stores=[sl],
        )
        blob = man.model_dump_json()
        restored = Manifest.model_validate_json(blob)
        assert restored == man
        assert restored.stores[0].processing_history[0].step_name == "convert"

    def test_manifest_round_trip_empty_history(self) -> None:
        sl = _make_lineage()
        man = Manifest(
            panoseti_analysis_storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            run_id="obs_20260101",
            level="L1",
            created_utc="2026-05-29T00:00:00Z",
            stores=[sl],
        )
        blob = man.model_dump_json()
        restored = Manifest.model_validate_json(blob)
        assert restored.stores[0].processing_history == []


# ── versions ──────────────────────────────────────────────────────────────────


class TestVersions:
    def test_storage_version_bumped_to_2(self) -> None:
        assert PANOSETI_ANALYSIS_STORAGE_VERSION == "2.0"

    def test_manifest_schema_version_bumped_to_2(self) -> None:
        assert MANIFEST_SCHEMA_VERSION == "2.0"

    def test_provenance_schema_version_is_1(self) -> None:
        assert PROVENANCE_SCHEMA_VERSION == "1.0"
