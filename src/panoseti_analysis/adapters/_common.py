"""Shared helpers for Layer B adapters (cadence inference, lineage emission, provenance)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import CloudInferParams, ProcessingStep, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.quicklook import generate_cloud_quicklook
from panoseti_analysis.io.stores import write_store


class SuspectTimestamps(RuntimeError):
    """Raised when a store's timestamps are corruption-grade and ``--fail-on-suspect`` is set."""


def infer_cadence_ns(ds: xr.Dataset) -> int | None:
    """Sampling cadence: ``None`` for event-based ph; median positive Δt for img."""
    if infer_kind(str(ds.attrs.get("data_product", ""))) == "ph":
        return None
    t = np.asarray(ds["unix_t_ns"].values)
    if t.size < 2:
        return None
    diffs = np.diff(np.sort(t))
    positive = diffs[diffs > 0]
    return int(np.median(positive)) if positive.size else None


def derive_suspect_displacement_ns(cadence_ns: int | None) -> int:
    """Corruption-grade backward-jump threshold (tunable; calibrate on real data).

    img: 1000x cadence (a 1000-frame backward jump is implausible buffering);
    event ph: 1 second.
    """
    if cadence_ns is not None and cadence_ns > 0:
        return 1000 * cadence_ns
    return 1_000_000_000


def write_lineage_json(record: StoreLineage, path: str | Path) -> None:
    Path(path).write_text(record.model_dump_json(indent=2))


# ── provenance helper ─────────────────────────────────────────────────────────


def build_provenance_step(
    ds: xr.Dataset | None,
    *,
    step_name: str,
    params: dict[str, Any],
    extra_input_checksums: list[str] | None = None,
) -> tuple[ProcessingStep, list[ProcessingStep]]:
    """Build a ``ProcessingStep`` and an extended processing-history list.

    Captures the current time and software env at call time.  Does NOT compute
    the output checksum — the caller writes the output store first, then checksums.

    Args:
        ds: The input Dataset (reads its ``processing_history`` from attrs).
            Pass ``None`` when there is no upstream store (e.g. HK build).
        step_name: e.g. ``"classify_cloud"``, ``"calibrate_ph"``.
        params: JSON-serializable step parameters.
        extra_input_checksums: Additional input checksums beyond the prior
            step's ``output_checksum`` (e.g. a model file checksum).

    Returns:
        ``(step, new_history)`` — the new step and the extended history list.
    """
    prior: list[ProcessingStep] = read_history(dict(ds.attrs)) if ds is not None else []
    prior_cksum: str | None = prior[-1].output_checksum if prior else None
    input_cksums: list[str] = ([prior_cksum] if prior_cksum else []) + (extra_input_checksums or [])
    step = ProcessingStep(
        step_name=step_name,
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params=params,
        input_checksums=input_cksums,
        timestamp_utc=now_utc(),
        software=capture_software(),
    )
    return step, append_step(prior, step)


# ── classify shared body ──────────────────────────────────────────────────────


def classify_cloud_store(
    l1_store: Path,
    l2_store: Path,
    *,
    model: torch.nn.Module,
    bundle_dict: dict[str, Any],
    params: CloudInferParams,
    codec: str = "zstd",
    level: int = 5,
    quicklook_out: Path | None = None,
) -> StoreLineage:
    """Open one L1 store, run cloud detection, write L2 + lineage, return record.

    Shared body used by both the CPU (Nextflow) and Ray classify adapters; the
    only difference between them is how they derive ``l2_store`` and ``quicklook_out``.
    """
    ds_l1 = xr.open_zarr(str(l1_store), consolidated=False)
    _step, new_history = build_provenance_step(
        ds_l1,
        step_name="classify_cloud",
        params={
            "cadence_s": params.cadence_s,
            "threshold": params.threshold,
            "model_checksum": bundle_dict.get("checksum"),
        },
    )

    ds_l2 = predict_cloud_score(ds_l1, model, params)
    ds_l2 = ds_l2.pano.stamp(data_level="L2", carry_from=ds_l1)

    write_store(ds_l2, l2_store, codec=codec, level=level, processing_history=new_history)
    l2_checksum = checksum_store(l2_store)

    record = StoreLineage(
        dp=str(ds_l1.attrs.get("data_product", "?")),
        module=str(ds_l1.attrs.get("module", "?")),
        level="L2",
        kind="cloud",
        store=l2_store.name,
        n_frames=int(ds_l2.sizes.get("T_l2", 0)),
        source_store=l1_store.name,
        model=bundle_dict,
        inference_params=params.model_dump(),
        checksum=l2_checksum,
        processing_history=new_history,
    )
    if quicklook_out is not None:
        generate_cloud_quicklook(ds_l2, quicklook_out)
    return record
