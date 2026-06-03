"""pa-classify-cloud — Nextflow CLI adapter for cloud detection (CPU)."""

from __future__ import annotations

from pathlib import Path

import typer

from panoseti_analysis.adapters._common import classify_cloud_store, write_lineage_json
from panoseti_analysis.config.models import CloudInferParams, StoreLineage
from panoseti_analysis.io.models import load_classifier

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
    model, bundle = load_classifier(model_path)
    params = CloudInferParams(cadence_s=cadence_s, threshold=threshold)
    record = classify_cloud_store(
        l1_store,
        l2_store,
        model=model,
        bundle_dict=bundle.model_dump(),
        params=params,
        codec=codec,
        level=level,
        quicklook_out=quicklook_out,
    )
    if lineage_out is not None:
        write_lineage_json(record, lineage_out)
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
        l1_store,
        l2_store,
        model_path,
        cadence_s=cadence_s,
        threshold=threshold,
        lineage_out=lineage_out,
        quicklook_out=quicklook_out,
        codec=codec,
        level=level,
    )


if __name__ == "__main__":
    app()
