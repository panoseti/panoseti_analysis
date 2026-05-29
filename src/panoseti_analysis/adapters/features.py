"""pa-features-cloud — materialise cloud-detection features from L1 img stores (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import xarray as xr

from panoseti_analysis.algorithms.cloud_detector import extract_cloud_features
from panoseti_analysis.algorithms.splits import data_split
from panoseti_analysis.config.models import CloudInferParams, ProcessingStep, StoreLineage
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.checksum import checksum_store
from panoseti_analysis.io.provenance import append_step, capture_software, now_utc, read_history
from panoseti_analysis.io.stores import open_store, write_store

app = typer.Typer(add_completion=False, help="Materialise cloud-detection features from L1 stores.")


def _load_labels(label_csv: Path) -> pd.DataFrame:
    """Load flat (module, t_start_ns, t_end_ns, label) CSV.

    Columns required: ``module``, ``t_start_ns``, ``t_end_ns``, ``label``.
    All time values must be in nanoseconds (int64).
    """
    df = pd.read_csv(label_csv, dtype={"module": str})
    required = {"module", "t_start_ns", "t_end_ns", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label CSV missing columns: {missing}")
    df["t_start_ns"] = df["t_start_ns"].astype(np.int64)
    df["t_end_ns"] = df["t_end_ns"].astype(np.int64)
    return df


def _join_labels(
    t_centers: np.ndarray,
    module: str,
    labels_df: pd.DataFrame,
) -> np.ndarray:
    """Assign a label (0/1) to each feature timestamp by time-window overlap.

    A feature at t_center gets label=1 if any row in labels_df with matching module
    has t_start_ns <= t_center <= t_end_ns and label==1, else 0.

    Returns an int8 array of length len(t_centers).
    """
    mod_df = labels_df[labels_df["module"] == module]
    y = np.zeros(len(t_centers), dtype=np.int8)
    for _, row in mod_df.iterrows():
        mask = (t_centers >= row["t_start_ns"]) & (t_centers <= row["t_end_ns"])
        y[mask] = int(row["label"])
    return y


def run_features_cloud(
    stores_list: list[Path],
    out_path: Path,
    recipe_path: Path,
    label_csv: Path | None = None,
    lineage_out: Path | None = None,
) -> StoreLineage:
    """Extract and cache cloud features from a list of L1 img stores.

    Reads each L1 store, extracts features via ``extract_cloud_features``,
    optionally joins labels from a CSV, runs temporal split, and writes a single
    compact ``features.<recipe_hash>.zarr`` to ``out_path``.

    Args:
        stores_list: Paths to L1 img Zarr stores.
        out_path: Output path for the feature cache store.
        recipe_path: Path to a recipe YAML with at least ``feature_cadence_s``.
        label_csv: Optional path to a ``(module, t_start_ns, t_end_ns, label)`` CSV.
        lineage_out: Optional path to write a StoreLineage JSON record.
    """
    params_dict, recipe_name, recipe_hash = load_recipe(recipe_path)
    cadence_s = float(params_dict.get("feature_cadence_s", 60.0))
    split_cfg: dict[str, Any] = params_dict.get("split", {})

    infer_params = CloudInferParams(cadence_s=cadence_s)

    labels_df = _load_labels(label_csv) if label_csv is not None else None

    all_X: list[np.ndarray] = []
    all_t: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_module: list[str] = []
    input_checksums: list[str] = []
    upstream_history: list[ProcessingStep] = []

    started_at = now_utc()
    software = capture_software()

    for store_path in stores_list:
        ds = open_store(store_path)
        if "median_subtracted" not in ds.data_vars:
            continue  # skip non-img or uncalibrated stores

        module = str(ds.attrs.get("module", "?"))
        store_history = read_history(dict(ds.attrs))
        if store_history:
            cksum = store_history[-1].output_checksum
            if cksum:
                input_checksums.append(cksum)
            # Collect upstream history from the first store (they all derive from the same recipe)
            if not upstream_history:
                upstream_history = list(store_history)

        X, t_centers = extract_cloud_features(ds, infer_params)
        if len(t_centers) == 0:
            continue

        all_X.append(X)
        all_t.append(t_centers)
        all_module.extend([module] * len(t_centers))
        if labels_df is not None:
            y = _join_labels(t_centers, module, labels_df)
        else:
            y = np.full(len(t_centers), -1, dtype=np.int8)
        all_y.append(y)

    if not all_X:
        raise ValueError("No img L1 stores with 'median_subtracted' found in the provided list.")

    X_cat = np.concatenate(all_X, axis=0)         # (N_total, 2, H, W)
    t_cat = np.concatenate(all_t, axis=0)         # (N_total,)
    y_cat = np.concatenate(all_y, axis=0)         # (N_total,)
    module_arr = np.array(all_module, dtype=str)  # (N_total,)

    # Temporal split indices
    split = data_split(
        t_cat,
        test_prop=float(split_cfg.get("test_prop", 0.15)),
        val_prop=float(split_cfg.get("val_prop", 0.10)),
        boundary_prop=float(split_cfg.get("boundary_prop", 0.20)),
        seed=int(split_cfg.get("seed", 35)),
    )
    # Encode split membership as a string variable (train/val/test/unassigned for unlabeled)
    split_label = np.full(len(t_cat), "unassigned", dtype="<U10")
    for name, idx in split.items():
        split_label[idx] = name

    step = ProcessingStep(
        step_name="cloud_features",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        params={
            "feature_cadence_s": cadence_s,
            "recipe_name": recipe_name,
            "n_stores": len(stores_list),
        },
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
            "label": (["sample"], y_cat),
            "split": (["sample"], split_label),
            "module": (["sample"], module_arr),
        },
        attrs={
            "recipe_name": recipe_name,
            "recipe_hash": recipe_hash,
            "data_level": "features",
        },
    )

    write_store(ds_feat, out_path, codec="zstd", level=5, processing_history=history)
    output_cksum = checksum_store(out_path)

    record = StoreLineage(
        dp="cloud_features",
        module="?",
        level="features",
        kind="cloud",
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
    stores_list_file: Path = typer.Argument(
        ..., help="File listing L1 img store paths, one per line"
    ),
    out_path: Path = typer.Argument(..., help="Output feature cache .zarr path"),
    recipe: Path = typer.Option(..., help="Recipe YAML (must contain feature_cadence_s)"),
    label_csv: Path | None = typer.Option(None, help="Label CSV (module,t_start_ns,t_end_ns,label)"),
    lineage_out: Path | None = typer.Option(None),
) -> None:
    with stores_list_file.open() as f:
        stores = [Path(line.strip()) for line in f if line.strip()]
    run_features_cloud(stores, out_path, recipe, label_csv=label_csv, lineage_out=lineage_out)


if __name__ == "__main__":
    app()
