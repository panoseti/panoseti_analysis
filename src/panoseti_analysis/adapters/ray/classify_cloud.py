"""pa-ray-classify-cloud — Ray batch adapter for cloud detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import ray
import torch
import typer
import xarray as xr

from panoseti_analysis.adapters.ray.launcher import init_ray
from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams, ProcessingStep, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.models import load_classifier
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.quicklook import generate_cloud_quicklook
from panoseti_analysis.io.stores import write_store

app = typer.Typer(add_completion=False, help="Run cloud detector inference via Ray.")


@ray.remote
def process_store(
    l1_store: Path,
    out_dir: Path,
    model: torch.nn.Module,
    bundle_dict: dict[str, Any],
    params: CloudInferParams,
    codec: str,
    level: int,
    quicklook_dir: Path | None = None,
) -> tuple[Path, StoreLineage]:
    """Remote task to process one store."""
    ds_l1 = xr.open_zarr(l1_store)

    l1_history = read_history(dict(ds_l1.attrs))
    l1_input_cksum = l1_history[-1].output_checksum if l1_history else None
    started_at = now_utc()
    software = capture_software()

    ds_l2 = predict_cloud_score(ds_l1, model, params)

    ds_l2.attrs.update(ds_l1.attrs)
    ds_l2.attrs["data_level"] = "L2"

    module = str(ds_l1.attrs.get("module", "?"))
    run_id = str(ds_l1.attrs.get("run_id", l1_store.stem.split(".")[0]))
    l2_store_name = f"{run_id}.cloud.module_{module}.zarr"
    l2_store = out_dir / l2_store_name

    classify_step = ProcessingStep(
        step_name="classify_cloud",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params={
            "cadence_s": params.cadence_s,
            "threshold": params.threshold,
            "model_checksum": bundle_dict.get("checksum"),
        },
        input_checksums=[l1_input_cksum] if l1_input_cksum else [],
        timestamp_utc=started_at,
        software=software,
    )
    new_history = append_step(l1_history, classify_step)

    write_store(ds_l2, l2_store, codec=codec, level=level, processing_history=new_history)

    l2_checksum = checksum_store(l2_store)

    record = StoreLineage(
        dp=str(ds_l1.attrs.get("data_product", "?")),
        module=str(module),
        level="L2",
        kind="cloud",
        store=l2_store_name,
        n_frames=int(ds_l2.sizes.get("T_l2", 0)),
        source_store=l1_store.name,
        model=bundle_dict,
        inference_params=params.model_dump(),
        checksum=l2_checksum,
        processing_history=new_history,
    )

    if quicklook_dir is not None:
        quicklook_out = quicklook_dir / f"{run_id}.cloud.module_{module}.quicklook.png"
        generate_cloud_quicklook(ds_l2, quicklook_out)

    return l2_store, record


@app.command()
def main(
    stores_list_file: Path = typer.Argument(..., help="File containing paths to L1 stores (one per line)"),
    out_dir: Path = typer.Argument(...),
    model_path: Path = typer.Argument(..., exists=True),
    launcher: str = typer.Option("attach", help="attach|slurm|standalone"),
    cadence_s: float = typer.Option(60.0),
    threshold: float = typer.Option(0.5),
    lineage_out: Path | None = typer.Option(None),
    quicklook_dir: Path | None = typer.Option(None),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
) -> None:
    init_ray(cast(Literal["attach", "slurm", "standalone"], launcher))

    out_dir.mkdir(parents=True, exist_ok=True)
    if quicklook_dir is not None:
        quicklook_dir.mkdir(parents=True, exist_ok=True)

    with stores_list_file.open() as f:
        l1_stores = [Path(line.strip()) for line in f if line.strip()]

    model, bundle = load_classifier(model_path)
    model_ref = ray.put(model)
    bundle_dict = bundle.model_dump()
    params = CloudInferParams(cadence_s=cadence_s, threshold=threshold)

    futures = [
        process_store.remote(s, out_dir, model_ref, bundle_dict, params, codec, level, quicklook_dir)
        for s in l1_stores
    ]

    results = ray.get(futures)

    records = [r for _, r in results]
    if lineage_out is not None:
        Path(lineage_out).write_text(
            json.dumps([r.model_dump(mode="json") for r in records], indent=2)
        )


if __name__ == "__main__":
    app()
