"""pa-ray-classify-cloud — Ray batch adapter for cloud detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import ray
import typer
import xarray as xr
import torch

from panoseti_analysis.adapters._common import write_lineage_json
from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
from panoseti_analysis.config.models import CloudInferParams, StoreLineage
from panoseti_analysis.io.models import load_classifier
from panoseti_analysis.io.stores import write_store
from panoseti_analysis.io.quicklook import generate_cloud_quicklook

app = typer.Typer(add_completion=False, help="Run cloud detector inference via Ray.")


@ray.remote
def process_store(
    l1_store: Path,
    out_dir: Path,
    model: torch.nn.Module,
    bundle_dict: dict,
    params: CloudInferParams,
    codec: str,
    level: int,
    quicklook_dir: Path | None = None,
) -> tuple[Path, StoreLineage]:
    """Remote task to process one store."""
    ds_l1 = xr.open_zarr(l1_store)

    ds_l2 = predict_cloud_score(ds_l1, model, params)

    ds_l2.attrs.update(ds_l1.attrs)
    ds_l2.attrs["data_level"] = "L2"

    module = str(ds_l1.attrs.get("module", "?"))
    run_id = str(ds_l1.attrs.get("run_id", l1_store.stem.split(".")[0]))
    l2_store_name = f"{run_id}.cloud.module_{module}.zarr"
    l2_store = out_dir / l2_store_name

    write_store(ds_l2, l2_store, codec=codec, level=level)

    record = StoreLineage(
        dp=str(ds_l1.attrs.get("data_product", "?")),
        module=str(module),
        level="L2",
        kind="cloud",
        store=l2_store_name,
        n_frames=int(ds_l2.sizes["T_l2"] if "T_l2" in ds_l2.sizes else 0),
        source_store=l1_store.name,
        model=bundle_dict,
        inference_params=params.model_dump(),
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
    cadence_s: float = typer.Option(60.0),
    threshold: float = typer.Option(0.5),
    lineage_out: Path | None = typer.Option(None),
    quicklook_dir: Path | None = typer.Option(None),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
) -> None:
    ray.init()

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
        with lineage_out.open("w") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")


if __name__ == "__main__":
    app()
