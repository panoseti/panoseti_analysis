"""
Image-mode calibration (img8, img16).

Pixel semantics: uint8 or uint16 counts above threshold in an integration window.
All pixels are non-negative; typical dark-rate pedestal is a small integer count.

Pipeline:
1. Cast to int32 to allow signed subtraction without overflow.
2. Compute 8×8 block spatial medians on a time-strided subset, then median over time.
3. Upsample the (H/8, W/8) map back to (H, W) via np.repeat.
4. Subtract spatial medians from every frame.
5. Compute temporal supermedian (median over a second strided subset of the residuals).
6. Subtract supermedian (removes any remaining uniform temporal trend).
7. Scale by adc_to_pe to convert to approximate photoelectrons.
8. Compute hot/dead pixel masks from the spatial median map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ._common import open_l0, write_l1, summarize, make_preview_png, write_summary_json


def calibrate_img(
    l0_store: Path,
    l1_store: Path,
    *,
    frame_stride: int = 200,
    block_size: int = 8,
    adc_to_pe: float = 1.5,
    codec: str = "zstd",
    level: int = 5,
) -> None:
    """Convert one L0 img-mode Zarr store to a calibrated L1 store."""
    ds = open_l0(l0_store)
    images: xr.DataArray = ds["images"].astype("int32")    # (T, H, W) – no overflow

    H, W = images.shape[1], images.shape[2]

    # --- spatial block median on strided subset ---
    subset = images[::frame_stride]                         # (T', H, W)
    block_medians = (
        subset
        .coarsen(y=block_size, x=block_size, boundary="trim")
        .median()
        .median("time")
    )  # → (H//block_size, W//block_size)

    # Upsample back to (H, W) using numpy repeat, then wrap in DataArray
    bm_np = np.asarray(block_medians.compute())            # (bH, bW)
    upsampled_np = np.repeat(np.repeat(bm_np, block_size, axis=0), block_size, axis=1)
    # Trim in case coarsen boundary="trim" clipped a pixel
    upsampled_np = upsampled_np[:H, :W]
    upsampled = xr.DataArray(
        upsampled_np,
        dims=("y", "x"),
        coords={"y": images.y, "x": images.x},
    )

    # --- subtract spatial medians ---
    spatial_sub = images - upsampled                       # (T, H, W) int32

    # --- temporal supermedian from a second strided pass on the residuals ---
    supermedian = spatial_sub[::frame_stride].median("time")  # (H, W)
    calibrated_int = spatial_sub - supermedian             # (T, H, W) float32

    # --- convert to photoelectrons ---
    calibrated = (calibrated_int / adc_to_pe).astype("float32")
    calibrated.name = "median_subtracted"

    # --- hot / dead pixel masks from spatial median map ---
    global_med_mean = float(upsampled.mean())
    hot_mask = (upsampled > 3.0 * global_med_mean).astype("uint8")
    hot_mask.name = "hot_pixel_mask"
    dead_mask = (upsampled == 0).astype("uint8")
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
        "median_subtracted": calibrated,
        "hot_pixel_mask": hot_mask,
        "dead_pixel_mask": dead_mask,
        **keep_vars,
    }

    l0_attrs = dict(ds.attrs)
    l1_attrs = {
        **l0_attrs,
        "panoseti_pff_zarr_l1_version": "0.1",
        "calibration": {
            "kind": "img",
            "frame_stride": frame_stride,
            "block_size": block_size,
            "adc_to_pe": adc_to_pe,
        },
    }

    write_l1(arrays, l1_store, attrs=l1_attrs, codec=codec, level=level)

    stats = summarize(calibrated)
    write_summary_json(
        stats,
        l1_store / "summary.json",
        extra={"l0_store": str(l0_store), "l1_store": str(l1_store)},
    )
    make_preview_png(
        calibrated,
        l1_store / "preview.png",
        title=f"L1 img calibrated — {l0_store.name}",
    )
