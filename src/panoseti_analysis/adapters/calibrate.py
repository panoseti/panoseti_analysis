"""pa-calibrate — L0 -> L1 (Layer B). Repairs timestamps, calibrates, writes lineage."""

from __future__ import annotations

from pathlib import Path

import typer

from panoseti_analysis.adapters._common import (
    SuspectTimestamps,
    derive_suspect_displacement_ns,
    infer_cadence_ns,
    write_lineage_json,
)
from panoseti_analysis.algorithms.calibrate_img import calibrate_img
from panoseti_analysis.algorithms.calibrate_ph import calibrate_ph
from panoseti_analysis.algorithms.timestamps import repair_timestamps
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import (
    ImgCalibParams,
    PhCalibParams,
    StoreLineage,
    TimestampQCStatus,
)
from panoseti_analysis.config.versions import TIMESTAMP_QC_KEY
from panoseti_analysis.io.stores import open_l0, write_store

app = typer.Typer(add_completion=False, help="Calibrate one L0 store to L1.")


def run_calibrate(
    l0_store: Path,
    l1_store: Path,
    *,
    kind: str | None = None,
    sigma: float = 5.0,
    offset: int = 800,
    ph_stride: int = 200,
    img_stride: int = 200,
    block: int = 8,
    adc_to_pe: float = 1.5,
    fail_on_suspect: bool = True,
    lineage_out: Path | None = None,
    codec: str = "zstd",
    level: int = 5,
) -> StoreLineage:
    """Open L0, repair timestamps (single QC call), calibrate, write L1 + lineage."""
    ds = open_l0(l0_store)
    data_product = str(ds.attrs["data_product"])
    resolved_kind = kind or infer_kind(data_product)

    cadence_ns = infer_cadence_ns(ds)
    suspect_ns = derive_suspect_displacement_ns(cadence_ns)
    ds_sorted, qc = repair_timestamps(
        ds, cadence_ns=cadence_ns, suspect_displacement_ns=suspect_ns
    )
    if qc.status is TimestampQCStatus.SUSPECT and fail_on_suspect:
        raise SuspectTimestamps(
            f"{l0_store.name}: corruption-grade timestamps (status=suspect); "
            "re-run with --no-fail-on-suspect to override"
        )

    if resolved_kind == "ph":
        out = calibrate_ph(
            ds_sorted,
            PhCalibParams(sigma_threshold=sigma, baseline_offset=offset, frame_stride=ph_stride),
        )
    else:
        out = calibrate_img(
            ds_sorted, ImgCalibParams(frame_stride=img_stride, block_size=block, adc_to_pe=adc_to_pe)
        )

    out.attrs[TIMESTAMP_QC_KEY] = qc.model_dump(mode="json")
    write_store(out, l1_store, codec=codec, level=level)

    record = StoreLineage(
        dp=data_product,
        module=str(ds.attrs.get("module", "?")),
        level="L1",
        kind=resolved_kind,
        store=l1_store.name,
        n_frames=int(out.sizes["time"]),
        time_range=(qc.t_start_ns, qc.t_end_ns),
        source_store=l0_store.name,
        calibration_params=dict(out.attrs["calibration"]),
        timestamp_qc=qc,
        cadence_ns=cadence_ns,
    )
    if lineage_out is not None:
        write_lineage_json(record, lineage_out)
    return record


@app.command()
def main(
    l0_store: Path = typer.Argument(..., exists=True),
    l1_store: Path = typer.Argument(...),
    kind: str | None = typer.Option(None, help="ph|img (inferred from data_product if omitted)"),
    sigma: float = typer.Option(5.0),
    offset: int = typer.Option(800),
    ph_stride: int = typer.Option(200),
    img_stride: int = typer.Option(200),
    block: int = typer.Option(8),
    adc_to_pe: float = typer.Option(1.5),
    fail_on_suspect: bool = typer.Option(True),
    lineage_out: Path | None = typer.Option(None),
    codec: str = typer.Option("zstd"),
    level: int = typer.Option(5),
) -> None:
    try:
        run_calibrate(
            l0_store, l1_store, kind=kind, sigma=sigma, offset=offset, ph_stride=ph_stride,
            img_stride=img_stride, block=block, adc_to_pe=adc_to_pe,
            fail_on_suspect=fail_on_suspect, lineage_out=lineage_out, codec=codec, level=level,
        )
    except SuspectTimestamps as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
