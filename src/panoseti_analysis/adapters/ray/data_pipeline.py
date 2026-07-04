"""
Ray Data scale-out for parallel PFF→L1 conversion (opt-in, experimental).

Layer B adapter — calls Layer A kernels (calibrate_img/ph) via the in-memory
path. Requires Ray with Ray Data (``ray[data]``).

Design
------
PFF→L1 is embarrassingly parallel over ``(file_idx, byte_range)`` chunks.
Each chunk:
  1. Opens a process-local ``PFFSequence`` (pickle-safe; handles re-opened lazily).
  2. Reads frames via ``iter_byte_range`` (frame-boundary-aligned).
  3. Builds an in-memory L0-layout Dataset (``sequence_to_dataset``).
  4. Optionally calibrates to L1 (``calibrate_img`` / ``calibrate_ph``).

Ray Data parallelises over chunks and distributes work across all cluster nodes,
including the second GPU node via the 400G RDMA fabric for result aggregation.

Usage
-----
    from panoseti_analysis.adapters.ray.data_pipeline import run_convert_ray_data

    result_refs = run_convert_ray_data(
        obs_dir="/mnt/beegfs/data/obs.pffd",
        dp="dp_img16.bpp_2.module_1",
        recipe=Path("recipes/img_calib_default.yml"),
        calibrate=True,
        chunk_frames=4096,
        num_cpus_per_worker=2,
    )
    # Each ref points to an xr.Dataset (L1) in the Ray object store.
    datasets = ray.get(result_refs)

Status: SKELETON — fan-out, per-chunk in-memory L0 build, and optional calibration
are implemented. Distributed write to Zarr and feature extraction are TODO stubs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr


# ── Chunk specification ──────────────────────────────────────────────────────


def _make_chunk_specs(
    file_paths: list[str],
    frame_size: int,
    chunk_frames: int,
) -> list[dict[str, Any]]:
    """Split each file into byte-range chunks aligned to ``chunk_frames`` frames."""
    specs: list[dict[str, Any]] = []
    global_frame = 0
    for file_idx, path in enumerate(file_paths):
        file_bytes = os.path.getsize(path)
        n_frames = file_bytes // frame_size
        for chunk_start in range(0, n_frames, chunk_frames):
            chunk_end = min(chunk_start + chunk_frames, n_frames)
            specs.append(
                {
                    "file_paths": file_paths,
                    "file_idx": file_idx,
                    "byte_start": chunk_start * frame_size,
                    "byte_end": chunk_end * frame_size,
                    "frame_start": global_frame + chunk_start,
                    "n_frames": chunk_end - chunk_start,
                }
            )
        global_frame += n_frames
    return specs


# ── Per-chunk worker ─────────────────────────────────────────────────────────


def _process_chunk(
    spec: dict[str, Any],
    *,
    calibrate: bool,
    recipe_path: str | None,
    run_configs: dict[str, Any] | None,
) -> xr.Dataset:
    """Read one byte-range chunk → in-memory L0 → optional L1.

    This function runs inside a Ray worker process. The ``PFFSequence`` is
    re-created locally (pickle-safe: handles dropped on ``__getstate__`` and
    reopened lazily on first access).
    """
    from pypff.io2 import PFFSequence

    from panoseti_analysis.io.zarr_compat import sequence_to_dataset

    seq = PFFSequence(spec["file_paths"])

    # Snap to frame-aligned boundaries then build the in-memory L0 Dataset.
    # sequence_to_dataset accepts frame indices, not byte offsets — convert.
    frame_size = seq.frame_config.frame_size if seq.frame_config else 1
    start_frame = spec["byte_start"] // frame_size
    stop_frame = spec["byte_end"] // frame_size

    ds_l0 = sequence_to_dataset(
        seq,
        start=start_frame,
        stop=stop_frame,
        run_configs=run_configs,
    )

    if not calibrate:
        return ds_l0

    # Calibrate to L1 using the shared _calibrate_ds helper.
    # recipe_path drives all calib params; None falls back to built-in defaults.
    from panoseti_analysis.adapters.calibrate import _calibrate_ds

    recipe_file = Path(recipe_path) if recipe_path is not None else None
    result = _calibrate_ds(ds_l0, recipe=recipe_file, fail_on_suspect=False)
    ds_l1 = result[
        0
    ]  # _CalibrateResult is (out_ds, ts_qc, l0_history, qc_report, kind, dp, cadence)
    return ds_l1


# ── Public entry point ───────────────────────────────────────────────────────


def run_convert_ray_data(
    obs_dir: str | Path,
    dp: str,
    *,
    calibrate: bool = False,
    recipe: Path | None = None,
    chunk_frames: int = 4096,
    num_cpus_per_worker: float = 1.0,
    run_configs: dict[str, Any] | None = None,
) -> list[Any]:  # list[ray.ObjectRef[xr.Dataset]]
    """Fan out PFF→L1 conversion across Ray workers using Ray Data.

    Parameters
    ----------
    obs_dir:
        Path to the ``.pffd`` observation directory.
    dp:
        Data product key (e.g. ``"dp_img16.bpp_2.module_1"``).
    calibrate:
        If True, calibrate each chunk to L1 in-memory after reading.
    recipe:
        Path to a calibration recipe YAML (e.g. ``recipes/img_calib_default.yml``).
        Used only when ``calibrate=True``.
    chunk_frames:
        Target number of frames per Ray Data chunk (default 4096 ≈ 400 ms at 100 µs cadence).
    num_cpus_per_worker:
        CPU fraction per map task. Fractional values allow over-subscription on
        CPU-only nodes (DFS nodes).
    run_configs:
        Optional ``run_configs`` dict embedded in each L0 Dataset's root attrs.
        Pass the output of ``PanosetiRun.run_configs`` for full provenance.

    Returns
    -------
    list[ray.ObjectRef]
        One Ray object reference per chunk. Call ``ray.get(refs)`` to materialise
        all L0/L1 Datasets into the driver process (only feasible for small subsets).
        For large observations, use ``.take_batch()`` or write via
        ``_write_chunk_to_zarr`` (TODO stub below).

    Notes
    -----
    This is a **skeleton** — the fan-out and in-memory pipeline are implemented.
    Merging chunks into a single Zarr store and distributed feature extraction
    are not yet implemented (see TODO stubs).
    """
    import ray
    import ray.data
    from pypff.io2 import PanosetiRun, PFFSequence

    obs_dir = Path(obs_dir)
    run = PanosetiRun(obs_dir)
    seq: PFFSequence = run.get_product(dp)

    if seq.frame_config is None:
        raise ValueError(f"No frame_config for {dp} — is the sequence non-empty?")

    frame_size = seq.frame_config.frame_size
    file_paths = [str(p) for p in seq.file_paths]

    chunk_specs = _make_chunk_specs(file_paths, frame_size, chunk_frames)
    if not chunk_specs:
        return []

    recipe_str = str(recipe) if recipe is not None else None

    # Build the Ray Data dataset from chunk specs (one row per chunk).
    ray_ds = ray.data.from_items(chunk_specs)

    # Map each chunk spec through the in-memory pipeline.
    # The result dataset stores one processed xr.Dataset per row via ray.put.
    result_refs: list[Any] = []

    def _map_fn(spec: dict[str, Any]) -> dict[str, Any]:
        ds = _process_chunk(
            spec,
            calibrate=calibrate,
            recipe_path=recipe_str,
            run_configs=run_configs,
        )
        ref = ray.put(ds)  # store in Ray object store; avoid copying back to driver
        return {"chunk_ref": ref, "frame_start": spec["frame_start"], "n_frames": spec["n_frames"]}

    processed = ray_ds.map(
        _map_fn,
        concurrency=len(chunk_specs),
        num_cpus=num_cpus_per_worker,
    )

    rows = processed.take_all()
    result_refs = [row["chunk_ref"] for row in rows]

    # TODO: merge chunks into a single L1 Zarr store (time-sorted concat + write_store).
    # Each ref is an independent L1 Dataset covering `chunk_frames` frames.
    # For production: open the target store, concat along "time", write via io/stores.py.
    #
    # TODO: Ray Data feature extraction pass:
    # features_ds = processed.map(_extract_features_fn, ...)  # → feature dicts
    # features_ds.write_parquet(out_dir / f"{dp}.features.parquet")

    return result_refs
