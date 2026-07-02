"""pa-convert — PFF .pffd -> per-(dp,module) L0 Zarr stores + lineage array (Layer B)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from panoseti_analysis.adapters._common import infer_cadence_ns
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import ProcessingStep, StoreLineage
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.pff import read_pff_run
from panoseti_analysis.io.provenance import capture_software, now_utc
from panoseti_analysis.io.stores import open_store, stamp_history

app = typer.Typer(add_completion=False, help="Convert a .pffd run to L0 Zarr stores.")


def run_convert(
    obs_dir: Path,
    out_dir: Path,
    *,
    codec: str = "zstd",
    level: int = 5,
    time_chunk: int = 0,
    shard_factor: int = 0,
    lineage_out: Path | None = None,
    use_tensorstore: bool = False,
    max_workers: int | None = None,
    checksum: bool = False,
) -> list[StoreLineage]:
    """Convert via pypff, then enumerate the emitted L0 stores into lineage records."""
    from pypff.zarr import convert_run  # local import: keeps Layer B free of import-time pypff cost

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stores are written flat; the level-major L0/ dir is created at publish time.
    started_at = now_utc()
    software = capture_software()
    _t0 = time.perf_counter()
    convert_run(
        read_pff_run(obs_dir),
        out_dir,
        codec=codec,
        level=level,
        time_chunk=time_chunk or None,
        shard_factor=shard_factor,
    )
    duration_s = time.perf_counter() - _t0

    records: list[StoreLineage] = []
    for store_path in sorted(out_dir.glob("*.zarr")):
        ds = open_store(store_path)
        data_product = str(ds.attrs["data_product"])

        # Compute checksum of the pypff-written store (before stamping provenance).
        # Skipped by default (R&D mode) to avoid a full second I/O pass over the store.
        output_cksum = checksum_store(store_path) if checksum else None

        # time_range is omitted when checksumming is off (R&D mode) to avoid a redundant
        # disk read of the full unix_t_ns array; the calibrate step can populate it from L0.
        time_range: tuple[int, int] | None = None
        if checksum:
            t = ds["unix_t_ns"].values
            time_range = (int(t.min()), int(t.max())) if t.size else None

        step = ProcessingStep(
            step_name="convert",
            step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            params={
                "codec": codec,
                "level": level,
                "time_chunk": time_chunk,
                "shard_factor": shard_factor,
            },
            output_checksum=output_cksum,
            timestamp_utc=started_at,
            software=software,
            duration_s=duration_s,
        )
        stamp_history(store_path, [step])

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
                checksum=output_cksum,
                processing_history=[step],
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
    shard_factor: Annotated[
        int,
        typer.Option(
            "--shard-factor",
            help="Inner chunks per shard (0 = no sharding). Use 16 for BeeGFS/Expanse.",
        ),
    ] = 0,
    lineage_out: Path | None = typer.Option(None),
    use_tensorstore: bool = typer.Option(
        False, "--use-tensorstore", help="Use tensorstore backend for faster conversion"
    ),
    max_workers: int | None = typer.Option(
        None, "--max-workers", help="Max parallel workers for data products"
    ),
    checksum: bool = typer.Option(
        False,
        "--checksum/--no-checksum",
        help="Compute sha256 checksum of each L0 store after writing (adds a full I/O pass). "
        "Enable for production; leave off for R&D.",
    ),
) -> None:
    run_convert(
        obs_dir,
        out_dir,
        codec=codec,
        level=level,
        time_chunk=time_chunk,
        shard_factor=shard_factor,
        lineage_out=lineage_out,
        use_tensorstore=use_tensorstore,
        max_workers=max_workers,
        checksum=checksum,
    )


if __name__ == "__main__":
    app()
