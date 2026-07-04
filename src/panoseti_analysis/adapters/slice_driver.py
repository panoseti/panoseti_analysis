"""slice_driver — in-memory PFF→L1/L2 chain (Layer B, no disk I/O).

Public API
----------
slice_to_l1(seq, ...)         Build an in-memory L1 Dataset from a PFF subrange.
slice_to_l2_cloud(seq, ...)   Build an in-memory L2 cloud-detection Dataset from a PFF subrange.

Both functions are suitable for interactive R&D and notebook exploration.
For production calibration, use the pa-calibrate adapter (which writes to disk and
emits full lineage JSON).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
import xarray as xr

from panoseti_analysis.adapters._common import (
    SuspectTimestamps,
    derive_suspect_displacement_ns,
    infer_cadence_ns,
)
from panoseti_analysis.algorithms.calibrate_img import calibrate_img
from panoseti_analysis.algorithms.calibrate_ph import calibrate_ph
from panoseti_analysis.algorithms.timestamps import repair_timestamps
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import ImgCalibParams, PhCalibParams, TimestampQCStatus
from panoseti_analysis.io.calibration_source import FileCalibrationResolver

if TYPE_CHECKING:
    import torch
    from pypff.io2 import PFFSequence

    from panoseti_analysis.config.models import CloudInferParams


app = typer.Typer(add_completion=False, help="Build in-memory L1/L2 slices from PFF — no disk I/O.")


# ── public functions ──────────────────────────────────────────────────────────


def slice_to_l1(
    seq: PFFSequence,
    *,
    frame_range: tuple[int, int] | None = None,
    time_range: tuple[int, int] | None = None,
    decimate: int = 1,
    recipe: Path | None = None,
    # Explicit calibration param overrides (ignored if recipe is given):
    sigma: float = 5.0,
    offset: int = 800,
    ph_stride: int = 200,
    img_stride: int = 200,
    block: int = 8,
    adc_to_pe: float = 1.5,
    fail_on_suspect: bool = False,
) -> xr.Dataset:
    """Build an in-memory L1 Dataset from a PFF subrange — no disk I/O.

    Chains: sequence_to_dataset → repair_timestamps → calibrate_img/calibrate_ph.

    Frame selection priority (same as ``sequence_to_dataset`` from ``zarr_compat``):
    1. ``time_range=(start_ns, stop_ns)`` — time-based, stop_ns is exclusive.
    2. ``frame_range=(start, stop)`` — index-based (Python-slice semantics).
    3. If both are None: the full sequence.

    DECIMATION NOTE: with ``decimate > 1``, frames are spaced ``decimate × cadence``
    apart. Baseline estimation (frame_stride in calibration params) still subsamples
    within the decimated set. Results are scientifically approximate — suitable for
    R&D, not production calibration.

    Args:
        seq:            A ``PFFSequence`` opened via ``io.pff.open_pff_product``.
        frame_range:    ``(start_frame, stop_frame)`` index range (exclusive stop).
        time_range:     ``(start_ns, stop_ns)`` UNIX timestamp range (exclusive stop).
        decimate:       Frame step; 1 = every frame (default).
        recipe:         Optional YAML recipe file for calibration params.
        sigma:          Ph calibration: sigma threshold for outlier rejection.
        offset:         Ph calibration: baseline offset (ADC counts).
        ph_stride:      Ph calibration: frame stride for baseline estimation.
        img_stride:     Img calibration: frame stride for median estimation.
        block:          Img calibration: spatial block size for local median.
        adc_to_pe:      Img calibration: ADC-to-photoelectron conversion factor.
        fail_on_suspect: Raise ``SuspectTimestamps`` if QC status is SUSPECT.
                        Defaults to False because R&D slices often have gaps.

    Returns:
        In-memory L1 ``xr.Dataset`` (calibrated, timestamps repaired).
    """
    # Local import keeps Layer B import-time cost low (pypff.zarr is heavy).
    from panoseti_analysis.io.zarr_compat import sequence_to_dataset

    # Resolve frame-index and time-range args.
    start: int | None = None
    stop: int | None = None
    start_ns: int | None = None
    stop_ns: int | None = None

    if frame_range is not None:
        start, stop = frame_range
    if time_range is not None:
        start_ns, stop_ns = time_range

    step = decimate if decimate != 1 else None

    ds_l0 = sequence_to_dataset(
        seq,
        start=start,
        stop=stop,
        step=step,
        start_ns=start_ns,
        stop_ns=stop_ns,
    )

    data_product = str(ds_l0.attrs["data_product"])
    resolved_kind = infer_kind(data_product)

    # Resolve calibration params from recipe or explicit args.
    if recipe is not None:
        resolver = FileCalibrationResolver(recipe)
        context: dict[str, Any] = {
            "data_product": data_product,
            "module": str(ds_l0.attrs.get("module", "?")),
        }
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

    cadence_ns = infer_cadence_ns(ds_l0)
    suspect_ns = derive_suspect_displacement_ns(cadence_ns)
    ds_sorted, qc = repair_timestamps(
        ds_l0, cadence_ns=cadence_ns, suspect_displacement_ns=suspect_ns
    )

    if qc.status is TimestampQCStatus.SUSPECT and fail_on_suspect:
        raise SuspectTimestamps(
            f"slice from {data_product!r}: corruption-grade timestamps (status=suspect); "
            "re-run with fail_on_suspect=False to override"
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

    return out


def slice_to_l2_cloud(
    seq: PFFSequence,
    model: torch.nn.Module,
    *,
    frame_range: tuple[int, int] | None = None,
    time_range: tuple[int, int] | None = None,
    decimate: int = 1,
    recipe: Path | None = None,
    infer_params: CloudInferParams | None = None,
    # calibration overrides forwarded to slice_to_l1:
    sigma: float = 5.0,
    offset: int = 800,
    ph_stride: int = 200,
    img_stride: int = 200,
    block: int = 8,
    adc_to_pe: float = 1.5,
) -> xr.Dataset:
    """Build an in-memory L2 cloud-detection Dataset from a PFF subrange — no disk I/O.

    Chains: slice_to_l1 → predict_cloud_score.
    Only valid for img (dp_img8 / dp_img16) sequences.

    Args:
        seq:          A ``PFFSequence`` opened via ``io.pff.open_pff_product``.
        model:        Loaded PyTorch cloud-detection model.
        frame_range:  ``(start_frame, stop_frame)`` index range (exclusive stop).
        time_range:   ``(start_ns, stop_ns)`` UNIX timestamp range (exclusive stop).
        decimate:     Frame step; 1 = every frame (default).
        recipe:       Optional YAML recipe for calibration params.
        infer_params: ``CloudInferParams`` for the cloud detector; uses defaults if None.
        sigma, offset, ph_stride, img_stride, block, adc_to_pe:
                      Calibration overrides forwarded to ``slice_to_l1``.

    Returns:
        In-memory L2 ``xr.Dataset`` with ``cloud_score``, ``cloud_label``, etc.

    Raises:
        ValueError: if the sequence is pulse-height (ph) kind — cloud detection
                    requires img data.
    """
    # Lazy imports to keep module-load cost low.
    from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score
    from panoseti_analysis.config.models import CloudInferParams as _CloudInferParams

    ds_l1 = slice_to_l1(
        seq,
        frame_range=frame_range,
        time_range=time_range,
        decimate=decimate,
        recipe=recipe,
        sigma=sigma,
        offset=offset,
        ph_stride=ph_stride,
        img_stride=img_stride,
        block=block,
        adc_to_pe=adc_to_pe,
    )

    data_product = str(ds_l1.attrs.get("data_product", ""))
    resolved_kind = infer_kind(data_product)
    if resolved_kind != "img":
        raise ValueError(
            f"slice_to_l2_cloud requires an img (dp_img8/dp_img16) sequence; "
            f"got data_product={data_product!r} (kind={resolved_kind!r})"
        )

    params = infer_params if infer_params is not None else _CloudInferParams()
    return predict_cloud_score(ds_l1, model, params)


# ── CLI ───────────────────────────────────────────────────────────────────────


@app.command()
def main(
    obs_dir: Path = typer.Argument(..., help="Path to the .pffd observation directory."),
    dp: str = typer.Argument(..., help="Full data_product name, e.g. dp_img16.bpp_2.module_1"),
    module: str = typer.Option("1", help="Module ID (used for display; lookup is by dp)."),
    start_frame: int | None = typer.Option(None, "--start-frame", help="Start frame index."),
    stop_frame: int | None = typer.Option(
        None, "--stop-frame", help="Stop frame index (exclusive)."
    ),
    start_ns: int | None = typer.Option(None, "--start-ns", help="Start UNIX timestamp (ns)."),
    stop_ns: int | None = typer.Option(
        None, "--stop-ns", help="Stop UNIX timestamp (ns, exclusive)."
    ),
    decimate: int = typer.Option(1, help="Frame decimation step (1 = every frame)."),
    sigma: float = typer.Option(5.0),
    offset: int = typer.Option(800),
    ph_stride: int = typer.Option(200),
    img_stride: int = typer.Option(200),
    block: int = typer.Option(8),
    adc_to_pe: float = typer.Option(1.5),
    recipe: Path | None = typer.Option(None, help="YAML recipe with calibration params."),
) -> None:
    """Build an in-memory L1 slice from a PFF observation and print dataset info."""
    from panoseti_analysis.io.pff import open_pff_product

    frame_range: tuple[int, int] | None = None
    if start_frame is not None or stop_frame is not None:
        frame_range = (start_frame or 0, stop_frame or 0)
    time_range: tuple[int, int] | None = None
    if start_ns is not None or stop_ns is not None:
        time_range = (start_ns or 0, stop_ns or 0)

    seq = open_pff_product(obs_dir, dp=dp, module=module)
    ds = slice_to_l1(
        seq,
        frame_range=frame_range,
        time_range=time_range,
        decimate=decimate,
        recipe=recipe,
        sigma=sigma,
        offset=offset,
        ph_stride=ph_stride,
        img_stride=img_stride,
        block=block,
        adc_to_pe=adc_to_pe,
    )
    typer.echo(f"L1 Dataset: {dict(ds.sizes)} frames, vars={list(ds.data_vars)}")
    typer.echo(f"  data_product: {ds.attrs.get('data_product')!r}")
    typer.echo(f"  module:       {ds.attrs.get('module')!r}")
    typer.echo(f"  data_level:   {ds.attrs.get('data_level')!r}")


if __name__ == "__main__":
    app()
