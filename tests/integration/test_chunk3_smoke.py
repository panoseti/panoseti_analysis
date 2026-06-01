"""Structural CI smoke tests for the Chunk 3 training + provenance system.

These tests verify that the public API surface is wired correctly (imports,
CLI entry-points, Pydantic schemas, recipe loading) without touching a real
Ray cluster or real data. They are fast and run in CI.

The ral_only keystone test is marked skip — run it on RAL against real data:
    uv run pytest -m ral_only tests/integration/test_chunk3_smoke.py -v
"""

from __future__ import annotations

import importlib

import pytest

from panoseti_analysis.config.models import ProcessingStep, TrainingProvenance
from panoseti_analysis.config.versions import (
    MANIFEST_SCHEMA_VERSION,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
    PROVENANCE_SCHEMA_VERSION,
)

# ── 1. Schema versions ─────────────────────────────────────────────────────────


def test_storage_version_is_2():
    """Schema bump to v2.0 is in place (processing_history added to StoreLineage)."""
    assert PANOSETI_ANALYSIS_STORAGE_VERSION == "2.0"
    assert MANIFEST_SCHEMA_VERSION == "2.0"
    assert PROVENANCE_SCHEMA_VERSION == "1.0"


# ── 2. ProcessingStep + TrainingProvenance round-trip ─────────────────────────


def test_processing_step_round_trip():
    """ProcessingStep serialises / deserialises correctly."""
    step = ProcessingStep(
        step_name="test_step",
        step_version="0.1.0",
        params={"a": 1},
        timestamp_utc="2026-01-01T00:00:00Z",
    )
    dumped = step.model_dump()
    assert dumped["step_name"] == "test_step"
    assert dumped["input_checksums"] == []
    restored = ProcessingStep(**dumped)
    assert restored == step


def test_training_provenance_round_trip():
    """TrainingProvenance serialises / deserialises correctly."""
    prov = TrainingProvenance(
        step_name="train_cloud",
        step_version="0.1.0",
        params={"lr": 1e-3, "batch": 128},
        timestamp_utc="2026-01-01T00:00:00Z",
        metrics={"val_accuracy": 0.96},
    )
    dumped = prov.model_dump()
    assert dumped["metrics"]["val_accuracy"] == pytest.approx(0.96)
    restored = TrainingProvenance(**dumped)
    assert restored == prov


# ── 3. CLI entry-points exist (importable) ────────────────────────────────────


@pytest.mark.parametrize(
    "module_path",
    [
        "panoseti_analysis.adapters.features",
        "panoseti_analysis.adapters.ph_prep",
        "panoseti_analysis.adapters.ray.train_cloud",
        "panoseti_analysis.adapters.ray.train_vae",
        "panoseti_analysis.adapters.ray.launcher",
        "panoseti_analysis.adapters.ray._staging",
        "panoseti_analysis.adapters.ray._tracking",
        "panoseti_analysis.algorithms.ph_vae",
        "panoseti_analysis.algorithms.splits",
        "panoseti_analysis.config.recipes",
        "panoseti_analysis.io.provenance",
    ],
)
def test_module_importable(module_path: str) -> None:
    """All Chunk 3 modules can be imported without error."""
    importlib.import_module(module_path)


# ── 4. RAL keystone (skip in CI) ──────────────────────────────────────────────


@pytest.mark.ral_only
@pytest.mark.skip(reason="ral_only: requires RAL cluster + real data. Run manually on RAL.")
def test_ral_train_to_inference_roundtrip() -> None:
    """Keystone acceptance test: ingest → features → train → inference → L2 provenance chain.

    Steps (run on RAL against real data):
      1. pa-features-cloud --launcher attach --recipe recipes/cloud_v1.yml ...
      2. pa-train-cloud --launcher attach --recipe recipes/cloud_v1.yml ...
      3. nextflow run . --steps ml --cloud_model_pt <bundle.pt> ... --outdir results/
      4. Assert L2 processing_history == [convert, calibrate, classify_cloud(model=<checksum>)]
      5. Assert model metrics >= {"val_accuracy": 0.95, "val_ap": 0.97}
    """
    pytest.skip("ral_only")
