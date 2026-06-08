"""pa-prep-cloud-labels -- convert legacy batch-builder labels to interval CSV (Layer B).

Reads the pre-packaged training ZIPs and writes a ``(module, t_start_ns, t_end_ns, label)``
CSV that ``pa-features-cloud --label-csv`` can consume directly.

The join chain is:
  type_labeled.csv  feature_uid -> label
       |
  type_feature.csv  feature_uid -> pano_uid
       |
  type_pano.csv     pano_uid -> (module_id, frame_unix_t)

Each labeled point is expanded into a time interval of width ``2 * half_window_s`` centred on
``frame_unix_t``.  The default half-window of 5 s matches the original 10-second sampling
cadence so that each feature window maps to exactly one pipeline feature window.

All reading is done in-memory (BytesIO) -- no on-disk extraction, no BeeGFS small-file overhead.
"""

from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pa-prep-cloud-labels",
    no_args_is_help=True,
    help="Convert legacy batch-builder labels to interval CSV for pa-features-cloud.",
)

_LABEL_MAP = {"clear_night_sky": 0, "not_clear_cloudy": 1}


def _read_batch_tables(
    data_zip: Path, batch_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (feat_df, pano_df) for one batch, read entirely in-memory."""
    tgz_pattern = f"batch-id_{batch_id}.tar.gz"
    with zipfile.ZipFile(data_zip) as outer:
        matches = [n for n in outer.namelist() if tgz_pattern in n]
        if not matches:
            raise FileNotFoundError(
                f"No entry matching '{tgz_pattern}' in {data_zip.name}. "
                f"Available tar.gz: {[n for n in outer.namelist() if '.tar.gz' in n]}"
            )
        tgz_bytes = outer.read(matches[0])

    with tarfile.open(fileobj=io.BytesIO(tgz_bytes)) as tf:
        names = tf.getnames()

        feat_name = next((n for n in names if "type_feature" in n and n.endswith(".csv")), None)
        pano_name = next((n for n in names if "type_pano" in n and n.endswith(".csv")), None)
        if feat_name is None or pano_name is None:
            raise FileNotFoundError(
                f"Batch {batch_id}: missing type_feature or type_pano CSV in tar.gz. "
                f"Found: {[n for n in names if n.endswith('.csv')]}"
            )

        feat_df = pd.read_csv(io.BytesIO(tf.extractfile(feat_name).read()), index_col=0)  # type: ignore[union-attr]
        pano_df = pd.read_csv(io.BytesIO(tf.extractfile(pano_name).read()), index_col=0)  # type: ignore[union-attr]

    return feat_df[["feature_uid", "pano_uid"]], pano_df[["pano_uid", "module_id", "frame_unix_t"]]


def _read_batch_labels(labels_zip: Path, batch_id: int, skip_unsure: bool) -> pd.DataFrame:
    """Return labeled DataFrame with columns (feature_uid, label_int) for one batch."""
    label_pattern = f"batch-id_{batch_id}.type_labeled"
    with zipfile.ZipFile(labels_zip) as outer:
        # Outer zip contains inner zips (one per batch)
        inner_zips = [n for n in outer.namelist() if n.endswith(".zip")]
        label_df: pd.DataFrame | None = None
        for inner_name in inner_zips:
            inner_bytes = outer.read(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                csv_candidates = [n for n in inner.namelist() if label_pattern in n and n.endswith(".csv")]
                if csv_candidates:
                    label_df = pd.read_csv(io.BytesIO(inner.read(csv_candidates[0])), index_col=0)
                    break

    if label_df is None:
        raise FileNotFoundError(f"No labeled CSV for batch {batch_id} in {labels_zip.name}")

    if skip_unsure:
        label_df = label_df[label_df["label"] != "unsure"].copy()

    unknown = set(label_df["label"].unique()) - set(_LABEL_MAP)
    if unknown:
        raise ValueError(f"Batch {batch_id}: unknown label values {unknown}")

    label_df["label_int"] = label_df["label"].map(_LABEL_MAP).astype(np.int8)
    return label_df[["feature_uid", "label_int"]].reset_index(drop=True)


def build_interval_label_csv(
    labels_zip: Path,
    data_zip: Path,
    out_path: Path,
    batch_ids: list[int],
    half_window_s: float = 5.0,
    skip_unsure: bool = True,
) -> pd.DataFrame:
    """Build a ``(module, t_start_ns, t_end_ns, label)`` CSV from legacy batch ZIPs.

    Args:
        labels_zip: Path to ``training-labels.zip``.
        data_zip: Path to ``training-data-batch.zip``.
        out_path: Destination CSV path.
        batch_ids: Which batch IDs to include (0-7).
        half_window_s: Half the label interval width in seconds (default 5 s -> 10 s windows).
        skip_unsure: If True, drop samples labeled ``unsure``.

    Returns:
        The assembled DataFrame (also written to ``out_path``).
    """
    half_window_ns = int(half_window_s * 1e9)
    rows: list[dict[str, object]] = []

    for bid in sorted(batch_ids):
        logger.info("Processing batch %d …", bid)
        try:
            label_df = _read_batch_labels(labels_zip, bid, skip_unsure)
            feat_df, pano_df = _read_batch_tables(data_zip, bid)
        except FileNotFoundError as exc:
            logger.warning("Skipping batch %d: %s", bid, exc)
            continue

        # feature_uid → pano_uid → (module_id, frame_unix_t)
        merged = (
            label_df
            .merge(feat_df, on="feature_uid", how="left")
            .merge(pano_df, on="pano_uid", how="left")
        )
        n_missing = merged["frame_unix_t"].isna().sum()
        if n_missing:
            logger.warning("Batch %d: %d labels have no timestamp, skipping.", bid, n_missing)
            merged = merged.dropna(subset=["frame_unix_t"])

        for _, row in merged.iterrows():
            t_center_ns = int(float(row["frame_unix_t"]) * 1e9)
            rows.append({
                "module": str(int(row["module_id"])),
                "t_start_ns": t_center_ns - half_window_ns,
                "t_end_ns": t_center_ns + half_window_ns,
                "label": int(row["label_int"]),
            })

        logger.info("  batch %d: %d labeled samples", bid, len(merged))

    if not rows:
        raise RuntimeError("No labeled samples found — check batch_ids and zip paths.")

    df_out = pd.DataFrame(rows).sort_values(["module", "t_start_ns"]).reset_index(drop=True)

    n_cloudy = (df_out["label"] == 1).sum()
    n_clear = (df_out["label"] == 0).sum()
    logger.info(
        "Total: %d intervals — cloudy=%d clear=%d modules=%s",
        len(df_out), n_cloudy, n_clear, sorted(df_out["module"].unique()),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info("Written to %s", out_path)
    return df_out


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
        typer.Option("--out", help="Output CSV path (module,t_start_ns,t_end_ns,label)"),
    ],
    half_window_s: Annotated[
        float,
        typer.Option(
            "--half-window-s",
            help=(
                "Half the label interval width in seconds. "
                "Default 5.0 matches the 10-second batch sampling cadence."
            ),
        ),
    ] = 5.0,
    batch_ids: Annotated[
        str | None,
        typer.Option("--batch-ids", help="Comma-separated batch IDs (default: 0-7)."),
    ] = None,
    skip_unsure: Annotated[
        bool,
        typer.Option("--skip-unsure/--no-skip-unsure", help="Drop 'unsure' labels."),
    ] = True,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Convert legacy batch-builder labels to an interval CSV for pa-features-cloud.

    Reads training-labels.zip and training-data-batch.zip entirely in-memory and
    produces a (module, t_start_ns, t_end_ns, label) CSV compatible with
    ``pa-features-cloud --label-csv``.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    parsed_ids: list[int]
    if batch_ids:
        try:
            parsed_ids = [int(b.strip()) for b in batch_ids.split(",") if b.strip()]
        except ValueError as exc:
            typer.echo(f"[ERROR] --batch-ids parse error: {exc}", err=True)
            raise typer.Exit(1) from exc
    else:
        parsed_ids = list(range(8))

    df = build_interval_label_csv(
        labels_zip=labels_zip,
        data_zip=data_zip,
        out_path=out,
        batch_ids=parsed_ids,
        half_window_s=half_window_s,
        skip_unsure=skip_unsure,
    )

    typer.echo(
        f"[pa-prep-cloud-labels] {len(df)} intervals written to {out}\n"
        f"  cloudy={int((df['label']==1).sum())}  clear={int((df['label']==0).sum())}"
        f"  modules={sorted(df['module'].unique())}"
    )
