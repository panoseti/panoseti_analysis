"""pa-convert — PFF .pffd -> per-(dp,module) L0 Zarr stores + lineage array (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from panoseti_analysis.adapters._common import infer_cadence_ns
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import StoreLineage
from panoseti_analysis.io.pff import read_pff_run
from panoseti_analysis.io.stores import open_store

app = typer.Typer(add_completion=False, help="Convert a .pffd run to L0 Zarr stores.")


def run_convert(
    obs_dir: Path,
    out_dir: Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    time_chunk: int = 0,
    lineage_out: Path | None = None,
) -> list[StoreLineage]:
    """Convert via pypff, then enumerate the emitted L0 stores into lineage records."""
    from pypff.zarr import convert_run  # local import: keeps Layer B free of import-time pypff cost

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stores are written flat; the level-major L0/ dir is created at publish time.
    convert_run(read_pff_run(obs_dir), out_dir, codec=codec, level=level, time_chunk=time_chunk or None)

    records: list[StoreLineage] = []
    for store_path in sorted(out_dir.glob("*.zarr")):
        ds = open_store(store_path)
        data_product = str(ds.attrs["data_product"])
        t = ds["unix_t_ns"].values
        time_range = (int(t.min()), int(t.max())) if t.size else None
        records.append(
            StoreLineage(
                dp=data_product,
                module=str(ds.attrs.get("module", "?")),
                level="L0",
                kind=infer_kind(data_product),
                store=store_path.name,
                n_frames=int(ds.sizes["time"]),
                time_range=time_range,
                cadence_ns=infer_cadence_ns(ds),
            )
        )

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
    time_chunk: int = typer.Option(0, help="0 = auto-sized by pypff"),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    run_convert(
        obs_dir, out_dir, codec=codec, level=level, time_chunk=time_chunk, lineage_out=lineage_out
    )


if __name__ == "__main__":
    app()
