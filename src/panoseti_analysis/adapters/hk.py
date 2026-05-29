"""pa-hk — hk.pff -> one hk.<hashset>.zarr store per device-type (Layer B)."""

from __future__ import annotations

from pathlib import Path

import typer

from panoseti_analysis.algorithms.housekeeping import build_hk_datasets
from panoseti_analysis.io.pff import read_hk
from panoseti_analysis.io.stores import write_store

app = typer.Typer(add_completion=False, help="Build per-hashset HK Zarr stores from hk.pff.")


def run_hk(
    obs_dir: Path,
    out_dir: Path,
    *,
    codec: str = "zstd",
    level: int = 5,
) -> list[Path]:
    """Parse hk.pff and write one store per hashset. Returns [] when there is no HK."""
    datasets = build_hk_datasets(read_hk(obs_dir))
    written: list[Path] = []
    if datasets:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    for hashset, ds in datasets.items():
        path = Path(out_dir) / f"hk.{hashset}.zarr"
        write_store(ds, path, codec=codec, level=level)
        written.append(path)
    return written


@app.command()
def main(
    obs_dir: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Argument(...),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
) -> None:
    written = run_hk(obs_dir, out_dir, codec=codec, level=level)
    typer.echo(f"wrote {len(written)} HK store(s)")


if __name__ == "__main__":
    app()
