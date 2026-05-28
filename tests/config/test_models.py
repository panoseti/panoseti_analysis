"""Tests for the Pydantic storage-schema models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from panoseti_analysis.config.models import (
    ImgCalibParams,
    Manifest,
    PhCalibParams,
    StoreLineage,
    TimestampQC,
    TimestampQCStatus,
)
from panoseti_analysis.config.versions import (
    MANIFEST_SCHEMA_VERSION,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
)


def test_calib_param_defaults() -> None:
    assert PhCalibParams().sigma_threshold == 5.0
    assert PhCalibParams().baseline_offset == 800
    assert ImgCalibParams().block_size == 8
    assert ImgCalibParams().adc_to_pe == 1.5


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        PhCalibParams(sigma_threshold=3.0, bogus=1)  # type: ignore[call-arg]


def test_timestamp_qc_status_serializes_to_string() -> None:
    qc = TimestampQC(status=TimestampQCStatus.REPAIRED, monotonic=True, n_frames=100)
    dumped = qc.model_dump(mode="json")
    assert dumped["status"] == "repaired"
    assert dumped["monotonic"] is True


def test_manifest_round_trip() -> None:
    qc = TimestampQC(
        status=TimestampQCStatus.CLEAN, monotonic=True, n_frames=3,
        t_start_ns=1000, t_end_ns=3000,
    )
    entry = StoreLineage(
        dp="ph256", module="1", level="L1", kind="ph",
        store="run.dp_ph256.module_1.zarr", n_frames=3, time_range=(1000, 3000),
        checksum="sha256:abc", source_store="L0/run.dp_ph256.module_1.zarr",
        calibration_params={"kind": "ph", "sigma_threshold": 5.0}, timestamp_qc=qc,
    )
    man = Manifest(
        panoseti_analysis_storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        run_id="run", level="L1", created_utc="2026-05-28T00:00:00Z",
        seed_manifest_version="1.0", stores=[entry],
    )
    blob = man.model_dump_json()
    restored = Manifest.model_validate_json(blob)
    assert restored == man
    assert restored.stores[0].timestamp_qc is not None
    assert restored.stores[0].timestamp_qc.status is TimestampQCStatus.CLEAN
