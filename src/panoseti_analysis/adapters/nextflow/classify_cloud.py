"""pa-classify-cloud — Nextflow CLI adapter for cloud detection (CPU)."""

from __future__ import annotations

from pathlib import Path

import typer
import xarray as xr

from panoseti_analysis.adapters._common import write_lineage_json
from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams, ProcessingStep, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.models import load_classifier
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.quicklook import generate_cloud_quicklook
from panoseti_analysis.io.stores import write_store

app = typer.Typer(add_completion=False, help="Run cloud detector inference on one L1 store (CPU).")


def run_classify(
    l1_store: Path,
    l2_store: Path,
    model_path: Path,
    cadence_s: float = 60.0,
    threshold: float = 0.5,
    lineage_out: Path | None = None,
    quicklook_out: Path | None = None,
    codec: str = "zstd",
    level: int = 5,
) -> StoreLineage:
    """Load L1, run inference, write L2 + lineage."""
    # 1. Load Model
    model, bundle = load_classifier(model_path)

    # 2. Load Data
    ds_l1 = xr.open_zarr(l1_store)

    l1_history = read_history(dict(ds_l1.attrs))
    l1_input_cksum = l1_history[-1].output_checksum if l1_history else None
    started_at = now_utc()
    software = capture_software()

    # 3. Predict
    params = CloudInferParams(cadence_s=cadence_s, threshold=threshold)
    ds_l2 = predict_cloud_score(ds_l1, model, params)

    # 4. Write Store
    # Inherit attributes and update them
    ds_l2.attrs.update(ds_l1.attrs)
    ds_l2.attrs["data_level"] = "L2"

    classify_step = ProcessingStep(
        step_name="classify_cloud",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params={
            "cadence_s": cadence_s,
            "threshold": threshold,
            "model_checksum": bundle.checksum,
        },
        input_checksums=[l1_input_cksum] if l1_input_cksum else [],
        timestamp_utc=started_at,
        software=software,
    )
    new_history = append_step(l1_history, classify_step)

    write_store(ds_l2, l2_store, codec=codec, level=level, processing_history=new_history)

    l2_checksum = checksum_store(l2_store)

    # 5. Write Lineage
    record = StoreLineage(
        dp=str(ds_l1.attrs.get("data_product", "?")),
        module=str(ds_l1.attrs.get("module", "?")),
        level="L2",
        kind="cloud",
        store=l2_store.name,
        n_frames=int(ds_l2.sizes.get("T_l2", 0)),
        source_store=l1_store.name,
        model=bundle.model_dump(),
        inference_params=params.model_dump(),
        time_range=None,  # We could extract this if needed
        checksum=l2_checksum,
        processing_history=new_history,
    )
    if lineage_out is not None:
        write_lineage_json(record, lineage_out)

    # 6. Write Quicklook
    if quicklook_out is not None:
        generate_cloud_quicklook(ds_l2, quicklook_out)

    return record


@app.command()
def main(
    l1_store: Path = typer.Argument(..., exists=True),
    l2_store: Path = typer.Argument(...),
    model_path: Path = typer.Argument(..., exists=True),
    cadence_s: float = typer.Option(60.0),
    threshold: float = typer.Option(0.5),
    lineage_out: Path | None = typer.Option(None),
    quicklook_out: Path | None = typer.Option(None),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
) -> None:
    run_classify(
        l1_store, l2_store, model_path, cadence_s=cadence_s, threshold=threshold,
        lineage_out=lineage_out, quicklook_out=quicklook_out, codec=codec, level=level
    )


if __name__ == "__main__":
    app()
