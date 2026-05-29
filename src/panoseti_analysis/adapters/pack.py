"""pa-pack — re-pack a Zarr store into a transfer file (Layer B)."""

from __future__ import annotations

from pathlib import Path

import typer

from panoseti_analysis.io.pack import pack_tar, pack_zipstore

app = typer.Typer(add_completion=False, help="Pack a Zarr store (tar | zipstore).")


def run_pack(store: Path, out_path: Path, *, fmt: str = "zip") -> Path:
    """Pack ``store`` into ``out_path`` as ``tar`` (compute hop) or ``zip`` (archive hop)."""
    if fmt == "tar":
        return pack_tar(store, out_path)
    if fmt == "zip":
        return pack_zipstore(store, out_path)
    raise ValueError(f"unknown pack format {fmt!r} (expected 'tar' or 'zip')")


@app.command()
def main(
    store: Path = typer.Argument(..., exists=True),
    out_path: Path = typer.Argument(...),
    fmt: str = typer.Option("zip", "--format", help="tar | zip"),
) -> None:
    run_pack(store, out_path, fmt=fmt)


if __name__ == "__main__":
    app()
