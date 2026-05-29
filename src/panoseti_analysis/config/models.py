"""Pydantic models for kernel params and the JSON-serializable storage schema.

These are the typed contracts that cross the Layer A / Layer B boundary and get
written into Zarr attrs and ``manifest.json``. Columnar coincidence containers hold
NumPy arrays and live with their kernel in ``algorithms/coincidence.py`` instead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Strict base: forbid unknown keys (catch typos), immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ── calibration params (Layer A kernel inputs) ───────────────────────────────
class PhCalibParams(_Base):
    """Pulse-height calibration parameters (ph256/ph1024)."""

    sigma_threshold: float = 5.0
    baseline_offset: int = 800
    frame_stride: int = 200


class ImgCalibParams(_Base):
    """Movie-mode calibration parameters (img8/img16)."""

    frame_stride: int = 200
    block_size: int = 8
    adc_to_pe: float = 1.5


# ── ML parameters (Layer A kernel inputs/configs) ─────────────────────────────
class ClassifierBundle(_Base):
    """Metadata describing an ML model checkpoint and its input expectations."""

    model_name: str
    model_version: str
    checksum: str
    input_spec: dict[str, Any]


class CloudInferParams(_Base):
    """Inference parameters for the cloud detector."""

    cadence_s: float = 60.0
    threshold: float = 0.5


# ── provenance / processing history ──────────────────────────────────────────
class ProcessingStep(_Base):
    """One step in a data-reduction pipeline, aligned to the IVOA Provenance DM.

    Field comments give the IVOA entity so a future VO export is a rename, not a
    redesign.
    """

    step_name: str                               # IVOA Activity.name (e.g. "convert", "calibrate_ph")
    step_version: str                            # IVOA Activity/Agent version (kernel/software version)
    params: dict[str, Any]                       # IVOA Activity Parameters
    recipe_name: str | None = None               # IVOA Agent/config descriptor
    recipe_hash: str | None = None               # content hash of the recipe ("sha256:…")
    input_checksums: list[str] = Field(default_factory=list)   # IVOA used (Entity refs)
    output_checksum: str | None = None           # IVOA wasGeneratedBy
    timestamp_utc: str                           # IVOA Activity.startTime (ISO 8601); ISO 8601 format required; consumers validate
    software: dict[str, str] = Field(default_factory=dict)     # IVOA wasAssociatedWith Agent
    nextflow_lineage_id: str | None = None       # external execution ref (lid://); NF 26.04 lineage


class TrainingProvenance(_Base):
    """Model-card provenance block written alongside a trained model checkpoint.

    A richer ``ProcessingStep``-shaped record for the training step; stored as a
    ``.json`` sidecar next to the ``.pt`` weights file.
    """

    step_name: str                               # e.g. "train_cloud" or "train_vae"
    step_version: str                            # software/model version
    recipe_name: str | None = None
    recipe_hash: str | None = None
    params: dict[str, Any]                       # all hyperparams (lr, batch, epochs, …)
    input_checksums: list[str] = Field(default_factory=list)   # feature-cache + L1 checksums
    output_checksum: str | None = None           # checksum of the .pt weights file
    timestamp_utc: str                                           # ISO 8601 format required; consumers validate
    software: dict[str, str] = Field(default_factory=dict)     # git_sha, ray_version, torch_version, …
    metrics: dict[str, float] = Field(default_factory=dict)    # e.g. {"val_accuracy": 0.96}; numeric-only; use dict[str, Any] if non-numeric values are needed
    wandb_run_id: str | None = None


# ── timestamp QC ──────────────────────────────────────────────────────────────
class TimestampQCStatus(StrEnum):
    CLEAN = "clean"          # already monotonic non-decreasing, no gaps over threshold
    REPAIRED = "repaired"    # was non-monotonic; stable-sorted into order
    FLAGGED = "flagged"      # monotonic/repaired but has gaps and/or duplicates
    SUSPECT = "suspect"      # corruption-grade displacement; hard-fail under --fail_on_suspect


class TimestampQC(_Base):
    """Quality record for a store's ``unix_t_ns`` axis (see storage_spec.md §4)."""

    status: TimestampQCStatus
    monotonic: bool
    n_frames: int
    n_nonmonotonic: int = 0
    n_duplicates: int = 0
    max_gap_ns: int = 0
    n_gaps_over_threshold: int = 0
    gap_threshold_ns: int = 0
    t_start_ns: int = 0
    t_end_ns: int = 0


# ── manifest / lineage ────────────────────────────────────────────────────────
class StoreLineage(_Base):
    """Provenance for one store in a per-run, per-level ``manifest.json``."""

    dp: str
    module: str
    level: str
    kind: str  # "ph" | "img" | "hk" | "cloud"
    store: str  # store filename (not a full path)
    n_frames: int
    time_range: tuple[int, int] | None = None  # (t_start_ns, t_end_ns)
    checksum: str | None = None  # e.g. "sha256:…"
    source_store: str | None = None  # the store this one derived from
    calibration_params: dict[str, Any] | None = None
    timestamp_qc: TimestampQC | None = None
    hashset: str | None = None  # set for HK stores (kind == "hk")
    cadence_ns: int | None = None  # sampling cadence (img); None/0 for event-based ph
    # L2+ fields
    model: dict[str, Any] | None = None
    inference_params: dict[str, Any] | None = None
    # schema v2.0 — added at end so older JSON without this field still loads (defaults to [])
    processing_history: list[ProcessingStep] = Field(default_factory=list)


class Manifest(_Base):
    """Canonical per-run, per-level store index carrying lineage."""

    panoseti_analysis_storage_version: str
    manifest_schema_version: str
    run_id: str
    level: str
    created_utc: str
    seed_manifest_version: str | None = None
    stores: list[StoreLineage] = Field(default_factory=list)
