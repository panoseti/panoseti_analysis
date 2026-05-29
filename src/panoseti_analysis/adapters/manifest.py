"""pa-manifest — assemble a per-run/per-level manifest.json from lineage fragments (Layer B).

The adapter does the I/O (reads ``*.lineage.json``, computes per-store checksums); the pure
``build_level_manifest`` kernel only merges/validates.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from panoseti_analysis.algorithms.manifest import build_level_manifest
from panoseti_analysis.config.models import Manifest, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store

app = typer.Typer(add_completion=False, help="Build a per-level manifest.json.")


def run_manifest(
    lineage_files: list[Path],
    level: str,
    run_id: str,
    out_path: Path,
    *,
    stores_dir: Path | None = None,
    seed: Manifest | None = None,
) -> Manifest:
    """Read lineage fragments, checksum stores (if found), build + write the manifest."""
    entries: list[StoreLineage] = []
    for lf in lineage_files:
        data = json.loads(Path(lf).read_text())
        items = data if isinstance(data, list) else [data]  # L0 = array; L1 = one object
        for item in items:
            entry = StoreLineage.model_validate(item)
            if stores_dir is not None:
                store_path = Path(stores_dir) / entry.store
                if store_path.is_dir():
                    entry = entry.model_copy(update={"checksum": checksum_store(store_path)})
            entries.append(entry)

    manifest = build_level_manifest(
        run_id,
        level,
        entries,
        storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        created_utc=datetime.now(UTC).isoformat(),
        seed=seed,
    )
    Path(out_path).write_text(manifest.model_dump_json(indent=2))
    return manifest


@app.command()
def main(
    out_path: Path = typer.Argument(...),
    level: str = typer.Option(...),
    run_id: str = typer.Option(...),
    lineage: list[Path] = typer.Option(..., help="One or more *.lineage.json files"),
    stores_dir: Path | None = typer.Option(None, help="Dir holding the stores, for checksums"),
) -> None:
    run_manifest(lineage, level, run_id, out_path, stores_dir=stores_dir)


if __name__ == "__main__":
    app()
