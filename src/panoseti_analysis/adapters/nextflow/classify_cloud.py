"""pa-classify-cloud — Nextflow CLI adapter for cloud detection (CPU)."""

from __future__ import annotations

import time
from pathlib import Path

import typer
import xarray as xr

from panoseti_analysis.adapters._common import write_lineage_json
from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams, StoreLineage
from panoseti_analysis.io.models import load_classifier
from panoseti_analysis.io.stores import write_store
from panoseti_analysis.io.quicklook import generate_cloud_quicklook

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
    
    # 3. Predict
    params = CloudInferParams(cadence_s=cadence_s, threshold=threshold)
    ds_l2 = predict_cloud_score(ds_l1, model, params)
    
    # 4. Write Store
    # Inherit attributes and update them
    ds_l2.attrs.update(ds_l1.attrs)
    ds_l2.attrs["data_level"] = "L2"
    
    write_store(ds_l2, l2_store, codec=codec, level=level)
    
    # 5. Write Lineage
    record = StoreLineage(
        dp=str(ds_l1.attrs.get("data_product", "?")),
        module=str(ds_l1.attrs.get("module", "?")),
        level="L2",
        kind="cloud",
        store=l2_store.name,
        n_frames=int(ds_l2.sizes["T_l2"] if "T_l2" in ds_l2.sizes else 0),
        source_store=l1_store.name,
        model=bundle.model_dump(),
        inference_params=params.model_dump(),
        time_range=None,  # We could extract this if needed
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
