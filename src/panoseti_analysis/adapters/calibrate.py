"""pa-calibrate — L0 -> L1 (Layer B). Repairs timestamps, calibrates, writes lineage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
import xarray as xr

from panoseti_analysis.adapters._common import (
    SuspectTimestamps,
    derive_suspect_displacement_ns,
    infer_cadence_ns,
    write_lineage_json,
)
from panoseti_analysis.algorithms.calibrate_img import calibrate_img
from panoseti_analysis.algorithms.calibrate_ph import calibrate_ph
from panoseti_analysis.algorithms.qc import QCReport, run_qc
from panoseti_analysis.algorithms.timestamps import repair_timestamps
from panoseti_analysis.config.calibration import CalibrationResolution
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import (
    ImgCalibParams,
    PhCalibParams,
    ProcessingStep,
    StoreLineage,
    TimestampQC,
    TimestampQCStatus,
)
from panoseti_analysis.config.versions import (
    CALIBRATION_KEY,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
    TIMESTAMP_QC_KEY,
)
from panoseti_analysis.io.calibration_source import FileCalibrationResolver
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.qc import stamp_qc, write_qc_sidecar
from panoseti_analysis.io.stores import open_l0, write_store

app = typer.Typer(add_completion=False, help="Calibrate one L0 store to L1.")

# Named tuple-like type alias for _calibrate_ds return (avoids a dataclass for a private helper)
_CalibrateResult = tuple[
    xr.Dataset,  # out_ds  — calibrated, QC-stamped
    TimestampQC,  # ts_qc
    list[ProcessingStep],  # l0_history
    QCReport,  # qc_report (so callers can write sidecars without re-running QC)
    str,  # resolved_kind
    str,  # data_product
    int | None,  # cadence_ns
]


def _calibrate_ds(
    ds: xr.Dataset,
    *,
    kind: str | None = None,
    sigma: float = 5.0,
    offset: int = 800,
    ph_stride: int = 200,
    img_stride: int = 200,
    block: int = 8,
    adc_to_pe: float = 1.5,
    fail_on_suspect: bool = True,
    recipe: Path | None = None,
    store_name_for_err: str = "<in-memory>",
) -> _CalibrateResult:
    """Core calibration logic: repair timestamps, calibrate, run QC.

    Returns ``(out_ds, ts_qc, l0_history, qc_report, resolved_kind, data_product, cadence_ns)``.

    The returned ``out_ds`` has ``TIMESTAMP_QC_KEY`` stamped into attrs, has
    ``CALIBRATION_KEY`` updated when a recipe-based resolution is present, and has
    the QC report stamped via ``stamp_qc``.  Callers write to disk and emit lineage.

    Args:
        ds:                 The L0 ``xr.Dataset`` to calibrate.
        kind:               ``"ph"`` or ``"img"``; inferred from ``data_product`` attr if None.
        sigma:              Ph calibration: sigma threshold for outlier rejection.
        offset:             Ph calibration: baseline offset (ADC counts).
        ph_stride:          Ph calibration: frame stride for baseline estimation.
        img_stride:         Img calibration: frame stride for median estimation.
        block:              Img calibration: spatial block size for local median.
        adc_to_pe:          Img calibration: ADC-to-photoelectron conversion factor.
        fail_on_suspect:    Raise ``SuspectTimestamps`` when QC status is SUSPECT.
        recipe:             Optional YAML recipe file; overrides explicit calib params.
        store_name_for_err: Display name used in ``SuspectTimestamps`` error messages.
    """
    data_product = str(ds.attrs["data_product"])
    resolved_kind = kind or infer_kind(data_product)

    # Build optional resolver (None = use explicit args as before).
    resolver = FileCalibrationResolver(recipe) if recipe is not None else None
    resolution: CalibrationResolution | None = None
    context: dict[str, Any] = {
        "run_id": str(ds.attrs.get("run_id", "")),
        "data_product": data_product,
        "module": str(ds.attrs.get("module", "?")),
    }

    if resolver is not None:
        resolution = resolver.resolve(resolved_kind, context)
        p = resolution.params
        if resolved_kind == "ph":
            sigma = float(p.get("sigma_threshold", sigma))
            offset = int(p.get("baseline_offset", offset))
            ph_stride = int(p.get("frame_stride", ph_stride))
        else:
            img_stride = int(p.get("frame_stride", img_stride))
            block = int(p.get("block_size", block))
            adc_to_pe = float(p.get("adc_to_pe", adc_to_pe))

    l0_history = read_history(dict(ds.attrs))

    cadence_ns = infer_cadence_ns(ds)
    suspect_ns = derive_suspect_displacement_ns(cadence_ns)
    ds_sorted, qc = repair_timestamps(ds, cadence_ns=cadence_ns, suspect_displacement_ns=suspect_ns)
    if qc.status is TimestampQCStatus.SUSPECT and fail_on_suspect:
        raise SuspectTimestamps(
            f"{store_name_for_err}: corruption-grade timestamps (status=suspect); "
            "re-run with --no-fail-on-suspect to override"
        )

    if resolved_kind == "ph":
        out = calibrate_ph(
            ds_sorted,
            PhCalibParams(sigma_threshold=sigma, baseline_offset=offset, frame_stride=ph_stride),
        )
    else:
        out = calibrate_img(
            ds_sorted,
            ImgCalibParams(frame_stride=img_stride, block_size=block, adc_to_pe=adc_to_pe),
        )

    out.attrs[TIMESTAMP_QC_KEY] = qc.model_dump(mode="json")

    if resolution is not None:
        cal = {
            **dict(out.attrs.get(CALIBRATION_KEY, {})),
            "resolution": resolution.model_dump(mode="json"),
        }
        out = out.pano.stamp(data_level="L1", calibration=cal)

    # QC: run checks and stamp report into attrs.
    qc_report = run_qc(out, level="L1", kind=resolved_kind)
    out = stamp_qc(out, qc_report)

    return out, qc, l0_history, qc_report, resolved_kind, data_product, cadence_ns


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
    shard_factor: int = 0,
    recipe: Path | None = None,
    qc_out: Path | None = None,
) -> StoreLineage:
    """Open L0, repair timestamps (single QC call), calibrate, write L1 + lineage.

    When ``recipe`` is given, a ``FileCalibrationResolver`` derives the calibration
    params from the YAML file (overriding the CLI-arg defaults) and stamps a
    ``CalibrationResolution`` record into ``ds.attrs["calibration"]["resolution"]``.
    """
    ds = open_l0(l0_store)

    started_at = now_utc()
    software = capture_software()
    _t0 = time.perf_counter()

    out, qc, l0_history, qc_report, resolved_kind, data_product, cadence_ns = _calibrate_ds(
        ds,
        kind=kind,
        sigma=sigma,
        offset=offset,
        ph_stride=ph_stride,
        img_stride=img_stride,
        block=block,
        adc_to_pe=adc_to_pe,
        fail_on_suspect=fail_on_suspect,
        recipe=recipe,
        store_name_for_err=l0_store.name,
    )

    if qc_out is not None:
        write_qc_sidecar(qc_report, qc_out)

    # Stop timer after kernel + QC (before write); duration reflects algorithmic cost.
    duration_s = time.perf_counter() - _t0

    l0_input_cksum = l0_history[-1].output_checksum if l0_history else None
    calib_step = ProcessingStep(
        step_name=f"calibrate_{resolved_kind}",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params=dict(out.attrs.get("calibration", {})),
        input_checksums=[l0_input_cksum] if l0_input_cksum else [],
        timestamp_utc=started_at,
        software=software,
        duration_s=duration_s,
    )
    new_history = append_step(l0_history, calib_step)
    write_store(
        out,
        l1_store,
        codec=codec,
        level=level,
        processing_history=new_history,
        shard_factor=shard_factor,
    )

    l1_checksum = checksum_store(l1_store)

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
        checksum=l1_checksum,
        processing_history=new_history,
    )
    if lineage_out is not None:
        write_lineage_json(record, lineage_out)
    return record


def run_calibrate_inmem(
    ds_l0: xr.Dataset,
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
    shard_factor: int = 0,
    recipe: Path | None = None,
    qc_out: Path | None = None,
) -> StoreLineage:
    """Calibrate an already-loaded L0 Dataset to L1; write L1 to ``l1_store``.

    Identical semantics to ``run_calibrate`` but accepts an in-memory
    ``xr.Dataset`` instead of a path.  Useful for the ``skip_l0_materialize``
    path in ``run_pipeline`` where the L0 Zarr store is bypassed entirely.

    Args:
        ds_l0:        Pre-loaded L0 ``xr.Dataset`` (e.g. from ``sequence_to_dataset``).
        l1_store:     Destination path for the L1 Zarr store.
        kind:         ``"ph"`` or ``"img"``; inferred from ``data_product`` attr if None.
        sigma:        Ph calibration: sigma threshold.
        offset:       Ph calibration: baseline offset (ADC counts).
        ph_stride:    Ph calibration: frame stride for baseline estimation.
        img_stride:   Img calibration: frame stride for median estimation.
        block:        Img calibration: spatial block size for local median.
        adc_to_pe:    Img calibration: ADC-to-photoelectron conversion factor.
        fail_on_suspect: Raise ``SuspectTimestamps`` when QC status is SUSPECT.
        lineage_out:  Optional path to write the ``StoreLineage`` JSON sidecar.
        codec:        Zarr compression codec.
        level:        Compression level.
        shard_factor: Inner chunks per shard (0 = no sharding).
        recipe:       Optional YAML recipe file for calibration params.
        qc_out:       Optional path to write the QC sidecar JSON.

    Returns:
        ``StoreLineage`` record for the produced L1 store.
    """
    dp_for_err = str(ds_l0.attrs.get("data_product", "?"))
    started_at = now_utc()
    software = capture_software()
    _t0 = time.perf_counter()

    out, qc, l0_history, qc_report, resolved_kind, data_product, cadence_ns = _calibrate_ds(
        ds_l0,
        kind=kind,
        sigma=sigma,
        offset=offset,
        ph_stride=ph_stride,
        img_stride=img_stride,
        block=block,
        adc_to_pe=adc_to_pe,
        fail_on_suspect=fail_on_suspect,
        recipe=recipe,
        store_name_for_err=f"<inmem:{dp_for_err}>",
    )

    if qc_out is not None:
        write_qc_sidecar(qc_report, qc_out)

    duration_s = time.perf_counter() - _t0

    l0_input_cksum = l0_history[-1].output_checksum if l0_history else None
    calib_step = ProcessingStep(
        step_name=f"calibrate_{resolved_kind}",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params=dict(out.attrs.get("calibration", {})),
        input_checksums=[l0_input_cksum] if l0_input_cksum else [],
        timestamp_utc=started_at,
        software=software,
        duration_s=duration_s,
    )
    new_history = append_step(l0_history, calib_step)
    write_store(
        out,
        l1_store,
        codec=codec,
        level=level,
        processing_history=new_history,
        shard_factor=shard_factor,
    )

    l1_checksum = checksum_store(l1_store)

    record = StoreLineage(
        dp=data_product,
        module=str(ds_l0.attrs.get("module", "?")),
        level="L1",
        kind=resolved_kind,
        store=l1_store.name,
        n_frames=int(out.sizes["time"]),
        time_range=(qc.t_start_ns, qc.t_end_ns),
        calibration_params=dict(out.attrs["calibration"]),
        timestamp_qc=qc,
        cadence_ns=cadence_ns,
        checksum=l1_checksum,
        processing_history=new_history,
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
    shard_factor: Annotated[
        int,
        typer.Option(
            "--shard-factor",
            help="Inner chunks per shard (0 = no sharding). Use 8 for BeeGFS/Expanse.",
        ),
    ] = 0,
    recipe: Path | None = typer.Option(None, help="YAML recipe with calibration params."),
    qc_out: Path | None = typer.Option(None, "--qc-out", help="Path for QC sidecar JSON."),
) -> None:
    try:
        run_calibrate(
            l0_store,
            l1_store,
            kind=kind,
            sigma=sigma,
            offset=offset,
            ph_stride=ph_stride,
            img_stride=img_stride,
            block=block,
            adc_to_pe=adc_to_pe,
            fail_on_suspect=fail_on_suspect,
            lineage_out=lineage_out,
            codec=codec,
            level=level,
            shard_factor=shard_factor,
            recipe=recipe,
            qc_out=qc_out,
        )
    except SuspectTimestamps as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc


if __name__ == "__main__":
    app()
