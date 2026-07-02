"""quicklook — generate side-by-side PNG from L0 and L1 stores."""

from pathlib import Path

import matplotlib.pyplot as plt
import typer
import xarray as xr

app = typer.Typer(add_completion=False, help="Generate quicklook image comparing L0 and L1.")


@app.command()
def main(
    l0_store: Path = typer.Argument(..., help="Path to the L0 Zarr store."),
    l1_store: Path = typer.Argument(..., help="Path to the L1 Zarr store."),
    quicklook_out: Path = typer.Argument(..., help="Output PNG path."),
) -> None:
    """Read first frame of L0 and L1 data and plot side-by-side."""
    ds_l0 = xr.open_zarr(l0_store, consolidated=False)
    ds_l1 = xr.open_zarr(l1_store, consolidated=False)

    def get_image_var(ds: xr.Dataset) -> str:
        for v in ["images", "median_subtracted", "pedestal_subtracted", "calibrated_pe", "data"]:
            if v in ds.data_vars and ds[v].ndim >= 2:
                return v
        # Fallback to the first variable with >= 2 dimensions
        for var in ds.data_vars:
            if ds[var].ndim >= 2:
                return str(var)
        raise ValueError(f"No image-like variable found in {ds.data_vars}")

    var_l0 = get_image_var(ds_l0)
    var_l1 = get_image_var(ds_l1)

    # Get the first frame
    frame_idx = 0
    frame_l0 = ds_l0[var_l0][frame_idx].values
    frame_l1 = ds_l1[var_l1][frame_idx].values

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Header info
    t_ns = ds_l0["unix_t_ns"][frame_idx].values.item() if "unix_t_ns" in ds_l0 else "N/A"
    header_info = f"Frame: {frame_idx} | unix_t_ns: {t_ns}"
    fig.suptitle(f"Quicklook: {l1_store.name}\n{header_info}", fontsize=12)

    # Plot L0
    if frame_l0.ndim == 1:
        axes[0].plot(frame_l0)
        axes[0].set_title("L0 (Raw)")
    else:
        im0 = axes[0].imshow(frame_l0, cmap="viridis", aspect="equal")
        axes[0].set_title("L0 (Raw)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot L1
    if frame_l1.ndim == 1:
        axes[1].plot(frame_l1)
        axes[1].set_title("L1 (Calibrated)")
    else:
        im1 = axes[1].imshow(frame_l1, cmap="viridis", aspect="equal")
        axes[1].set_title("L1 (Calibrated)")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(quicklook_out, dpi=150)
    plt.close(fig)

    typer.echo(f"Successfully wrote quicklook to {quicklook_out}")


if __name__ == "__main__":
    app()
