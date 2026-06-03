"""pa-hk — hk.pff -> one hk.<hashset>.zarr store per device-type (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from panoseti_analysis.algorithms.housekeeping import build_hk_datasets
from panoseti_analysis.config.models import ProcessingStep, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.pff import read_hk
from panoseti_analysis.io.provenance import capture_software, now_utc
from panoseti_analysis.io.stores import write_store

app = typer.Typer(add_completion=False, help="Build per-hashset HK Zarr stores from hk.pff.")


def run_hk(
    obs_dir: Path,
    out_dir: Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    lineage_out: Path | None = None,
) -> list[StoreLineage]:
    """Parse hk.pff and write one store per hashset. Returns [] when there is no HK."""
    datasets = build_hk_datasets(read_hk(obs_dir))
    records: list[StoreLineage] = []
    if datasets:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    started_at = now_utc()
    software = capture_software()

    for hashset, ds in datasets.items():
        path = Path(out_dir) / f"hk.{hashset}.zarr"
        step = ProcessingStep(
            step_name="build_hk",
            step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            params={"codec": codec, "level": level, "hashset": hashset},
            timestamp_utc=started_at,
            software=software,
        )
        write_store(ds, path, codec=codec, level=level, processing_history=[step])
        hk_checksum = checksum_store(path)
        record = StoreLineage(
            dp="hk",
            module="?",
            level="L0",
            kind="hk",
            store=path.name,
            n_frames=int(ds.sizes.get("hk_time", len(next(iter(ds.data_vars.values()))))),
            checksum=hk_checksum,
            hashset=hashset,
            processing_history=[step],
        )
        records.append(record)

    if lineage_out is not None:
        Path(lineage_out).write_text(
            json.dumps([r.model_dump(mode="json") for r in records], indent=2)
        )

    return records


@app.command()
def main(
    obs_dir: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Argument(...),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    records = run_hk(obs_dir, out_dir, codec=codec, level=level, lineage_out=lineage_out)
    typer.echo(f"wrote {len(records)} HK store(s)")


if __name__ == "__main__":
    app()
