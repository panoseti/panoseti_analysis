"""pa-ray-classify-cloud — Ray batch adapter for cloud detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import ray
import typer
import xarray as xr

from panoseti_analysis.adapters._common import classify_cloud_store
from panoseti_analysis.adapters.ray.launcher import init_ray
from panoseti_analysis.config.models import CloudInferParams, StoreLineage
from panoseti_analysis.io.models import load_classifier

app = typer.Typer(add_completion=False, help="Run cloud detector inference via Ray.")


@ray.remote
def process_store(
    l1_store: Path,
    out_dir: Path,
    model: Any,
    bundle_dict: dict[str, Any],
    params: CloudInferParams,
    codec: str,
    level: int,
    quicklook_dir: Path | None = None,
) -> tuple[Path, StoreLineage]:
    """Remote task: derive the L2 store path from L1 attrs, then classify."""
    attrs = xr.open_zarr(str(l1_store), consolidated=False).attrs
    module = str(attrs.get("module", "?"))
    run_id = str(attrs.get("run_id", l1_store.stem.split(".")[0]))
    l2_store_name = f"{run_id}.cloud.module_{module}.zarr"
    l2_store = out_dir / l2_store_name

    quicklook_out = (
        quicklook_dir / f"{run_id}.cloud.module_{module}.quicklook.png"
        if quicklook_dir is not None
        else None
    )
    record = classify_cloud_store(
        l1_store,
        l2_store,
        model=model,
        bundle_dict=bundle_dict,
        params=params,
        codec=codec,
        level=level,
        quicklook_out=quicklook_out,
    )
    return l2_store, record


@app.command()
def main(
    stores_list_file: Path = typer.Argument(
        ..., help="File containing paths to L1 stores (one per line)"
    ),
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
        process_store.remote(
            s, out_dir, model_ref, bundle_dict, params, codec, level, quicklook_dir
        )
        for s in l1_stores
    ]

    results = ray.get(futures)

    records: list[StoreLineage] = [r for _, r in results]
    if lineage_out is not None:
        Path(lineage_out).write_text(
            json.dumps([r.model_dump(mode="json") for r in records], indent=2)
        )


if __name__ == "__main__":
    app()
