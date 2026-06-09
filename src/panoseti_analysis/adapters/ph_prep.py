"""pa-prep-ph — standardise L1 pulse-height stores for BetaVAE training (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
import xarray as xr

from panoseti_analysis.config.models import ProcessingStep, StoreLineage
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.stores import open_store, write_store

app = typer.Typer(add_completion=False, help="Standardise L1 PH stores to BetaVAE feature cache.")


def _log_norm_to_unit(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-normalize to mean≈0, std≈1 via (log(arr + eps) - mean) / (std + eps).

    Mirrors the preprocessing in panoseti-software/anomaly-detection/ph_dataset.py.
    The result is float32 with approximately zero mean.
    """
    log_arr = np.log(arr.astype(np.float32) + eps)
    mu = log_arr.mean()
    sigma = log_arr.std() + eps
    return ((log_arr - mu) / sigma).astype(np.float32)


def run_prep_ph(
    stores_list: list[Path],
    out_path: Path,
    recipe_path: Path,
    lineage_out: Path | None = None,
) -> StoreLineage:
    """Standardise L1 PH stores and write a flat feature cache for BetaVAE training.

    Reads the calibrated PH variable from each PH L1 store
    (``pedestal_subtracted`` preferred, then ``baseline_subtracted``, then ``ph_counts``),
    applies log-normalization, and concatenates all frames into an
    ``X (sample, 1, H, W)`` float32 array.
    """
    _params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)

    all_X: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    all_module: list[str] = []
    input_checksums: list[str] = []
    upstream_history: list[ProcessingStep] = []

    started_at = now_utc()
    software = capture_software()

    for store_path in stores_list:
        ds = open_store(store_path)
        # Accept calibrated PH variable names in priority order
        if "pedestal_subtracted" in ds.data_vars:
            arr = ds["pedestal_subtracted"].values  # (T, H, W)
        elif "baseline_subtracted" in ds.data_vars:
            arr = ds["baseline_subtracted"].values
        elif "ph_counts" in ds.data_vars:
            arr = ds["ph_counts"].values
        else:
            continue  # not a calibrated PH store

        t = ds["unix_t_ns"].values
        module = str(ds.attrs.get("module", "?"))

        store_history = read_history(dict(ds.attrs))
        if store_history:
            cksum = store_history[-1].output_checksum
            if cksum:
                input_checksums.append(cksum)
            if not upstream_history:
                upstream_history = list(store_history)

        # Fill NaN (masked pixels from sigma-clip) with zero before log-normalizing
        arr_filled = np.nan_to_num(arr.astype(np.float32), nan=0.0)
        X_norm = _log_norm_to_unit(arr_filled)  # (T, H, W)
        X_norm = X_norm[:, np.newaxis, :, :]  # (T, 1, H, W)
        all_X.append(X_norm)
        all_t.append(t)
        all_module.extend([module] * len(t))

    if not all_X:
        raise ValueError("No calibrated PH L1 stores found in the provided list.")

    X_cat = np.concatenate(all_X, axis=0)  # (N, 1, H, W)
    t_cat = np.concatenate(all_t, axis=0)  # (N,)
    module_arr = np.array(all_module, dtype=str)

    step = ProcessingStep(
        step_name="prep_ph",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params={"recipe_name": recipe_name, "n_stores": len(stores_list)},
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        input_checksums=input_checksums,
        timestamp_utc=started_at,
        software=software,
    )
    history = append_step(upstream_history, step)

    ds_feat = xr.Dataset(
        data_vars={
            "X": (["sample", "channel", "H", "W"], X_cat),
            "t_ns": (["sample"], t_cat),
            "module": (["sample"], module_arr),
        },
        attrs={
            "recipe_name": recipe_name,
            "recipe_hash": recipe_hash,
            "data_level": "ph_features",
        },
    )

    write_store(ds_feat, out_path, codec="zstd", level=5, processing_history=history)
    output_cksum = checksum_store(out_path)

    record = StoreLineage(
        dp="ph_features",
        module="?",
        level="features",
        kind="ph",
        store=Path(out_path).name,
        n_frames=int(X_cat.shape[0]),
        checksum=output_cksum,
        processing_history=history,
    )
    if lineage_out is not None:
        Path(lineage_out).write_text(json.dumps([record.model_dump(mode="json")], indent=2))
    return record


@app.command()
def main(
    stores_list_file: Path = typer.Argument(..., help="File listing L1 PH store paths"),
    out_path: Path = typer.Argument(..., help="Output feature cache .zarr"),
    recipe: Path = typer.Option(..., help="Recipe YAML"),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    with stores_list_file.open() as f:
        stores = [Path(line.strip()) for line in f if line.strip()]
    run_prep_ph(stores, out_path, recipe, lineage_out=lineage_out)


if __name__ == "__main__":
    app()
