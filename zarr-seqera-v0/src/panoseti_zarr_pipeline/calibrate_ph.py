"""
Pulse-height calibration (ph256, ph1024).

Pixel semantics: int16 ADC intensity counts. Negative values occur normally
(the ADC is signed; pedestal sits above zero but fluctuations go below).

Pipeline:
1. Fix up ph1024 (32×32) quabo positions: quabo 1 and 2 swap halves.
2. Add baseline_offset so all values are strictly positive before median.
3. Compute per-pixel pedestal = median over a time-strided subset.
4. Subtract pedestal; result is float32.
5. Compute per-pixel sigma = std over the same strided subset.
6. Mask frames where (pedestal_subtracted > sigma_threshold × sigma) → NaN elsewhere.
7. Compute hot/dead pixel masks from the per-pixel pedestal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ._common import open_l0, write_l1, summarize, make_preview_png, write_summary_json


def _fix_ph1024_quabo_order(images: xr.DataArray) -> xr.DataArray:
    """
    ph1024 stores 4 quabos in a 32×32 grid arranged as:
        [quabo0 | quabo2]   (top-left  | top-right)
        [quabo1 | quabo3]   (bot-left  | bot-right)
    but the DAQ writes them in a different order. Swap the top-right
    and bottom-left 16×16 sub-arrays so spatial layout is correct.
    """
    arr = images.values.copy()           # (T, 32, 32)
    tmp = arr[:, :16, 16:].copy()        # top-right quadrant
    arr[:, :16, 16:] = arr[:, 16:, :16] # ← bottom-left
    arr[:, 16:, :16] = tmp              # ← old top-right
    return xr.DataArray(arr, dims=images.dims, coords=images.coords, attrs=images.attrs)


def calibrate_ph(
    l0_store: Path,
    l1_store: Path,
    *,
    sigma_threshold: float = 5.0,
    baseline_offset: int = 800,
    frame_stride: int = 200,
    codec: str = "zstd",
    level: int = 5,
) -> None:
    """Convert one L0 ph-mode Zarr store to a calibrated L1 store."""
    ds: xr.DataArray = open_l0(l0_store)
    images: xr.DataArray = ds["images"].astype("float32")  # (T, H, W)

    # Fix ph1024 quabo ordering (no-op for ph256 since shape is (T,16,16))
    if images.shape[1] == 32:
        images = _fix_ph1024_quabo_order(images)

    # --- pedestal from strided subset ---
    subset = images[::frame_stride]                          # (T', H, W)
    shifted = subset + baseline_offset
    pedestal = shifted.median(dim="time")                    # (H, W)
    sigma = shifted.std(dim="time")                          # (H, W)

    # --- subtract ---
    calibrated = (images + baseline_offset) - pedestal       # (T, H, W) float32

    # --- sigma threshold mask: keep only >n_sigma above pedestal ---
    mask = calibrated > (sigma_threshold * sigma)
    above_threshold = calibrated.where(mask)
    above_threshold.name = "pedestal_subtracted"

    # --- hot / dead pixel masks from pedestal ---
    global_ped_mean = float(pedestal.mean())
    hot_mask = (pedestal > 3.0 * global_ped_mean).astype("uint8")
    hot_mask.name = "hot_pixel_mask"
    dead_mask = (pedestal == 0).astype("uint8")
    dead_mask.name = "dead_pixel_mask"

    # --- carry forward timing/header columns from L0 ---
    keep_vars = {
        k: ds[k]
        for k in ["unix_t_ns", "pkt_num", "quabo_num"]
        if k in ds
    }
    for prefix in ["quabo_0", "quabo_1", "quabo_2", "quabo_3"]:
        k = f"{prefix}_pkt_num"
        if k in ds:
            keep_vars[k] = ds[k]

    arrays = {
        "pedestal_subtracted": above_threshold,
        "hot_pixel_mask": hot_mask,
        "dead_pixel_mask": dead_mask,
        **keep_vars,
    }

    l0_attrs = dict(ds.attrs)
    l1_attrs = {
        **l0_attrs,
        "panoseti_pff_zarr_l1_version": "0.1",
        "calibration": {
            "kind": "ph",
            "baseline_offset": baseline_offset,
            "sigma_threshold": sigma_threshold,
            "frame_stride": frame_stride,
        },
    }

    write_l1(arrays, l1_store, attrs=l1_attrs, codec=codec, level=level)

    stats = summarize(above_threshold)
    write_summary_json(
        stats,
        l1_store / "summary.json",
        extra={"l0_store": str(l0_store), "l1_store": str(l1_store)},
    )
    make_preview_png(
        above_threshold,
        l1_store / "preview.png",
        title=f"L1 ph calibrated — {l0_store.name}",
    )
