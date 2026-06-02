"""pa-prep-cloud-legacy — build a feature-cache Zarr from the pre-packaged
numpy features and label CSVs produced by the original cloud-detection pipeline.

The output Zarr schema matches ``pa-features-cloud`` so that ``pa-train-cloud``
can consume it without modification:

    ds["X"]      (N, 2, 32, 32)  float32  — [channel-0: deriv-fft, channel-1: raw-fft]
    ds["label"]  (N,)            int64    — 0=clear, 1=cloudy
    ds["split"]  (N,)            str      — "train" | "val" | "test"
    ds["batch_id"] (N,)          int64    — original batch index (for diagnostics)

Feature scaling matches the original ``data_loaders.py`` square-root convention:
    x_scaled = x * (1.0 / |x|^0.5)  for |x| > 0, else 0.0

Channel-0 is ``raw-derivative-fft.-60`` (FFT of the 60-second time-derivative).
The zip stores only ``raw-derivative.-60``; the FFT is applied here (same 2-D Hann
window and fftshift as the original ``pano_utils.apply_fft()``).

Channel-1 is ``raw-fft`` (pre-stored in the zip, used as-is after scaling).

Normalization (per-channel mean/std across the whole dataset) is NOT applied here —
the Ray trainer's TorchTrainer loop handles it within each worker via transforms.
"""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
import xarray as xr

from panoseti_analysis.config.models import ProcessingStep
from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.io.provenance import capture_software, now_utc
from panoseti_analysis.io.stores import write_store

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pa-prep-cloud-legacy",
    no_args_is_help=True,
    help="Build a feature-cache Zarr from pre-packaged legacy training data.",
)

# ---------------------------------------------------------------------------
# Constants matching original pipeline
# ---------------------------------------------------------------------------

_LABEL_MAP = {"clear_night_sky": 0, "not_clear_cloudy": 1}
_IMG_SIZE = 32
_HANN_2D = np.outer(np.hanning(_IMG_SIZE), np.hanning(_IMG_SIZE)).astype(np.float32)


def _apply_sqrt_scale(arr: np.ndarray) -> np.ndarray:
    """Square-root magnitude scaling from the original data_loaders.py."""
    out = np.where(
        np.abs(arr) > 0,
        arr * (1.0 / np.abs(arr) ** 0.5),
        0.0,
    )
    return out.astype(np.float32)


def _apply_fft(arr: np.ndarray) -> np.ndarray:
    """2-D FFT with Hann window + fftshift, matching original pano_utils.apply_fft()."""
    windowed = arr * _HANN_2D
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(windowed)))
    return fft_mag.astype(np.float32)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_labels_zip(labels_zip: Path, extract_dir: Path) -> Path:
    """Extract training-labels.zip → extract_dir/labels/.

    The outer zip contains one inner zip per batch (each inner zip holds the CSVs).
    This function flattens the nested structure: all CSVs end up directly under labels_dir.
    """
    labels_dir = extract_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(labels_zip) as outer:
        for entry_name in outer.namelist():
            if not entry_name.endswith(".zip"):
                continue
            inner_bytes = outer.read(entry_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                for inner_entry in inner.namelist():
                    dest = labels_dir / Path(inner_entry).name
                    if dest.exists():
                        continue
                    dest.write_bytes(inner.read(inner_entry))
    return labels_dir


def _extract_batch_tgz(data_zip: Path, batch_id: int, extract_dir: Path) -> Path:
    """Extract one batch tar.gz from training-data-batch.zip to extract_dir/batches/batch_N/."""
    batch_dir = extract_dir / "batches" / f"batch_{batch_id}"
    if batch_dir.exists() and any(batch_dir.iterdir()):
        logger.info("Batch %d already extracted at %s, skipping.", batch_id, batch_dir)
        return batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)

    tgz_pattern = f"batch-id_{batch_id}.tar.gz"
    with zipfile.ZipFile(data_zip) as outer:
        # Find the tar.gz entry for this batch_id
        matches = [n for n in outer.namelist() if tgz_pattern in n]
        if not matches:
            raise FileNotFoundError(
                f"No tar.gz entry matching '{tgz_pattern}' in {data_zip.name}. "
                f"Available: {[n for n in outer.namelist() if '.tar.gz' in n]}"
            )
        tgz_name = matches[0]
        logger.info("Extracting batch %d from %s …", batch_id, tgz_name)
        tgz_bytes = outer.read(tgz_name)

    with tarfile.open(fileobj=io.BytesIO(tgz_bytes), mode="r:gz") as tf:
        tf.extractall(batch_dir)
    return batch_dir


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _load_batch_labels(
    labels_dir: Path, batch_dir: Path, batch_id: int, skip_unsure: bool
) -> pd.DataFrame:
    """Load the labeled CSV for one batch and join with pano_uid from the feature index.

    Returns DataFrame(feature_uid, pano_uid, label_int).
    The pano_uid is the key used in the numpy filename.
    """
    # Labeled CSV (flat after extraction in labels_dir)
    label_pattern = f"*batch-id_{batch_id}.type_labeled*.csv"
    label_candidates = list(labels_dir.glob(label_pattern))
    if not label_candidates:
        raise FileNotFoundError(
            f"No labeled CSV found for batch {batch_id} in {labels_dir}. Pattern: {label_pattern}"
        )
    labels_df = pd.read_csv(label_candidates[0], index_col=0)

    if skip_unsure:
        labels_df = labels_df[labels_df["label"] != "unsure"].copy()
    unknown = set(labels_df["label"].unique()) - set(_LABEL_MAP)
    if unknown:
        raise ValueError(f"Batch {batch_id}: unknown label values {unknown}")
    labels_df["label_int"] = labels_df["label"].map(_LABEL_MAP).astype(np.int64)

    # Feature index CSV in the batch_dir: maps feature_uid → pano_uid
    feat_pattern = f"task_cloud-detection.batch-id_{batch_id}.type_feature.csv"
    feat_candidates = list(batch_dir.glob(feat_pattern))
    if not feat_candidates:
        raise FileNotFoundError(f"Feature index CSV '{feat_pattern}' not found in {batch_dir}")
    feat_df = pd.read_csv(feat_candidates[0], index_col=0)[["feature_uid", "pano_uid"]]

    merged = labels_df.merge(feat_df, on="feature_uid", how="left")
    missing_pano = merged["pano_uid"].isna().sum()
    if missing_pano > 0:
        logger.warning(
            "Batch %d: %d labels have no pano_uid mapping, skipping.", batch_id, missing_pano
        )
        merged = merged.dropna(subset=["pano_uid"])

    return merged[["feature_uid", "pano_uid", "label_int"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature loading for one sample
# ---------------------------------------------------------------------------


def _load_features(batch_dir: Path, pano_uid: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and return (channel_0_deriv_fft, channel_1_raw_fft) for one sample.

    Returns two (32, 32) float32 arrays with square-root scaling applied.
    Channel 0: FFT of raw-derivative.-60 (computed here; not pre-stored).
    Channel 1: raw-fft (pre-stored, used directly after scaling).

    Files live under pano_imgs/<run_dir>/<feature_type>/pano-uid_{pano_uid}.feature-type_*.npy
    """
    prefix = f"pano-uid_{pano_uid}"

    def _find(feature_type: str) -> Path:
        # Feature files live at: pano_imgs/<run_dir>/<feature_type>/pano-uid_*.npy
        feature_dir_pattern = batch_dir / "pano_imgs" / "*" / feature_type
        import glob as _glob

        candidates = _glob.glob(f"{feature_dir_pattern}/{prefix}.feature-type_{feature_type}.npy")
        if not candidates:
            raise FileNotFoundError(
                f"Feature '{feature_type}' not found for pano_uid={pano_uid} in {batch_dir}"
            )
        return Path(candidates[0])

    # Channel 0: FFT of raw-derivative.-60 (matching our pipeline's feature_deriv_fft)
    raw_deriv = np.load(_find("raw-derivative.-60")).astype(np.float32)
    ch0 = _apply_fft(raw_deriv)
    ch0 = _apply_sqrt_scale(ch0)

    # Channel 1: FFT of raw-original (matching our pipeline's feature_raw_fft).
    # Note: the zip stores raw-fft as (64, 64) — a zero-padded FFT from the original
    # pipeline. We re-compute the FFT on raw-original at (32, 32) to match the model's
    # input_spec and our production extract_cloud_features() output.
    raw_orig = np.load(_find("raw-original")).astype(np.float32)
    ch1 = _apply_fft(raw_orig)
    ch1 = _apply_sqrt_scale(ch1)

    return ch0, ch1


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_legacy_feature_cache(
    labels_zip: Path,
    data_zip: Path,
    extract_dir: Path,
    out_path: Path,
    batch_ids: list[int],
    skip_unsure: bool = True,
    recipe_path: Path | None = None,
) -> Path:
    """Build a feature-cache Zarr from pre-packaged legacy training data.

    Returns the path to the written Zarr store.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Load recipe for split params (optional)
    split_params: dict[str, Any] = {}
    recipe_hash = "sha256:builtin"
    recipe_name = "default"
    if recipe_path is not None:
        split_params, recipe_name, recipe_hash = load_recipe(recipe_path)

    test_prop = float(split_params.get("test_prop", 0.15))
    val_prop = float(split_params.get("val_prop", 0.10))

    # Step 1: Extract labels
    typer.echo(f"[prep] Extracting labels from {labels_zip.name} …")
    labels_dir = _extract_labels_zip(labels_zip, extract_dir)

    # Step 2: Load all samples
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_batch: list[int] = []
    all_uid: list[str] = []

    for bid in sorted(batch_ids):
        typer.echo(f"[prep] Processing batch {bid} …")
        batch_dir = _extract_batch_tgz(data_zip, bid, extract_dir)
        labels_df = _load_batch_labels(labels_dir, batch_dir, bid, skip_unsure)

        n_ok = 0
        n_err = 0
        for _, row in labels_df.iterrows():
            pano_uid = str(row["pano_uid"])
            try:
                ch0, ch1 = _load_features(batch_dir, pano_uid)
            except FileNotFoundError as e:
                logger.warning("Skipping missing feature: %s", e)
                n_err += 1
                continue
            all_X.append(np.stack([ch0, ch1], axis=0))  # (2, 32, 32)
            all_y.append(int(row["label_int"]))
            all_batch.append(bid)
            all_uid.append(pano_uid)
            n_ok += 1

        typer.echo(f"[prep]   batch {bid}: {n_ok} samples loaded, {n_err} skipped")

    if not all_X:
        raise RuntimeError("No samples loaded — check batch_ids and zip paths.")

    N = len(all_X)
    X = np.stack(all_X, axis=0)  # (N, 2, 32, 32)
    y = np.array(all_y, dtype=np.int64)
    batch_arr = np.array(all_batch, dtype=np.int64)
    typer.echo(
        f"[prep] Total samples: {N} (clear={int((y == 0).sum())}, cloudy={int((y == 1).sum())})"
    )

    # Step 3: Assign splits
    # Use shuffled stratified split (no unix_t_ns available for temporal split)
    rng = np.random.default_rng(35)
    perm = rng.permutation(N)
    n_test = max(1, int(N * test_prop))
    n_val = max(1, int(N * val_prop))

    split_arr = np.full(N, "train", dtype=object)
    split_arr[perm[:n_test]] = "test"
    split_arr[perm[n_test : n_test + n_val]] = "val"
    split_arr = split_arr.astype(str)

    typer.echo(
        f"[prep] Split: train={int((split_arr == 'train').sum())} "
        f"val={int((split_arr == 'val').sum())} test={int((split_arr == 'test').sum())}"
    )

    # Step 4: Build xr.Dataset
    ds = xr.Dataset(
        {
            "X": (("sample", "channel", "y", "x"), X),
            "label": (("sample",), y),
            "split": (("sample",), split_arr),
            "batch_id": (("sample",), batch_arr),
        }
    )

    # Step 5: Write feature cache
    software = capture_software()
    step = ProcessingStep(
        step_name="prep_cloud_legacy",
        step_version=software.get("package_version", "0.2.0"),
        params={
            "batch_ids": sorted(batch_ids),
            "skip_unsure": skip_unsure,
            "test_prop": test_prop,
            "val_prop": val_prop,
            "scaling": "sqrt",
            "channel_0": "FFT(raw-derivative.-60)",
            "channel_1": "FFT(raw-original)",
            "n_samples": N,
        },
        recipe_name=recipe_name,
        recipe_hash=recipe_hash,
        timestamp_utc=now_utc(),
        software=software,
    )
    typer.echo(f"[prep] Writing feature cache to {out_path} …")
    write_store(ds, out_path, processing_history=[step])
    typer.echo(f"[prep] Done. {N} samples written to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    labels_zip: Annotated[
        Path,
        typer.Option("--labels-zip", help="Path to training-labels.zip", exists=True),
    ],
    data_zip: Annotated[
        Path,
        typer.Option("--data-zip", help="Path to training-data-batch.zip", exists=True),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Output feature-cache Zarr path"),
    ],
    extract_dir: Annotated[
        Path | None,
        typer.Option(
            "--extract-dir", help="Directory for zip extraction (default: <out>/../cache)"
        ),
    ] = None,
    recipe: Annotated[
        Path | None,
        typer.Option("--recipe", help="Recipe YAML for split params"),
    ] = None,
    batch_ids: Annotated[
        str | None,
        typer.Option(
            "--batch-ids",
            help="Comma-separated batch IDs (0–7). Default: all available.",
        ),
    ] = None,
    skip_unsure: Annotated[
        bool,
        typer.Option("--skip-unsure/--no-skip-unsure", help="Skip 'unsure' labels"),
    ] = True,
    log_level: Annotated[
        str,
        typer.Option("--log-level"),
    ] = "INFO",
) -> None:
    """Build a feature-cache Zarr from pre-packaged cloud-detection training data.

    Extracts training-labels.zip and training-data-batch.zip, applies the original
    square-root scaling, assembles X (N, 2, 32, 32), label, split, and batch_id arrays,
    and writes a Zarr store compatible with pa-train-cloud.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    _extract_dir = extract_dir or (out.parent / "cache")

    parsed_batch_ids: list[int]
    if batch_ids:
        try:
            parsed_batch_ids = [int(b.strip()) for b in batch_ids.split(",") if b.strip()]
        except ValueError as e:
            typer.echo(f"[ERROR] --batch-ids parse error: {e}", err=True)
            raise typer.Exit(1) from e
    else:
        parsed_batch_ids = list(range(8))  # batches 0–7 are in the zip

    build_legacy_feature_cache(
        labels_zip=labels_zip,
        data_zip=data_zip,
        extract_dir=_extract_dir,
        out_path=out,
        batch_ids=parsed_batch_ids,
        skip_unsure=skip_unsure,
        recipe_path=recipe,
    )
