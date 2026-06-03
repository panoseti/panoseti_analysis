"""Pure-Python dev-loop pipeline driver.

Mirrors the Nextflow laptop DAG (subworkflows/local/ingest.nf + ml.nf) as a flat
in-memory sequence for notebook R&D and smoke-testing.  Nextflow remains the
production orchestrator (resume / SLURM / container).

**NOT a DAG engine.**  No scheduling, no resume, no parallelism.  One run, in memory.
Steps run in the same order as the Nextflow subworkflows; cross-step data flows via
the ``RunOutputs`` return value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import typer

from panoseti_analysis.adapters.calibrate import run_calibrate
from panoseti_analysis.adapters.convert import run_convert
from panoseti_analysis.adapters.hk import run_hk
from panoseti_analysis.adapters.manifest import run_manifest
from panoseti_analysis.adapters.nextflow.classify_cloud import run_classify
from panoseti_analysis.config.levels import infer_kind
from panoseti_analysis.config.models import Manifest, StoreLineage
from panoseti_analysis.io.stores import open_store


@dataclass
class RunOutputs:
    """Collected outputs of one pipeline run (mirrors the INGEST + ML emit channels)."""

    l0_stores: list[StoreLineage] = field(default_factory=list)
    hk_stores: list[StoreLineage] = field(default_factory=list)
    l1_stores: list[StoreLineage] = field(default_factory=list)
    l2_stores: list[StoreLineage] = field(default_factory=list)
    l0_manifest: Manifest | None = None
    l1_manifest: Manifest | None = None
    l2_manifest: Manifest | None = None


def run_pipeline(
    obs_dir: Path,
    out_dir: Path,
    *,
    model_path: Path | None = None,
    calib_recipe: Path | None = None,
    codec: str = "zstd",
    level: int = 5,
    shard_factor: int = 0,
) -> RunOutputs:
    """Run the full ingest + ML pipeline for one observation directory.

    Step order mirrors subworkflows/local/ingest.nf + ml.nf:

    1.  PFF → L0 Zarr stores  (run_convert)
    2.  HK Zarr stores         (run_hk, parallel branch in NF — sequential here)
    3.  L0 → L1 per (dp,module) via run_calibrate (ph or img dispatch)
    4.  L0 manifest, L1 manifest
    5.  img L1 → L2 cloud classification  (run_classify, img-only gate)
    6.  L2 manifest

    Args:
        obs_dir:       Path to an ``.pffd`` observation directory.
        out_dir:       Root output dir; ``L0/``, ``L1/``, ``L2/`` subdirs are created.
        model_path:    Path to a ``.pt`` model bundle for cloud classification.
                       Required when any img L1 stores are produced.
        calib_recipe:  Optional YAML recipe for calibration params (see
                       ``FileCalibrationResolver``).  Defaults reproduce current values.
        codec:         Zarr compression codec (default ``"zstd"``).
        level:         Compression level (default ``5``).
        shard_factor:  Inner chunks per shard (``0`` = no sharding).
    """
    obs_dir = Path(obs_dir)
    out_dir = Path(out_dir)

    l0_dir = out_dir / "L0"
    l1_dir = out_dir / "L1"
    l2_dir = out_dir / "L2"
    l0_dir.mkdir(parents=True, exist_ok=True)
    l1_dir.mkdir(parents=True, exist_ok=True)

    outputs = RunOutputs()

    # ── 1. Convert: PFF → L0 ─────────────────────────────────────────────────
    outputs.l0_stores = run_convert(
        obs_dir, l0_dir, codec=codec, level=level, shard_factor=shard_factor
    )

    # ── 2. HK (parallel branch in Nextflow; sequential here) ─────────────────
    outputs.hk_stores = run_hk(obs_dir, l0_dir, codec=codec, level=level)

    # ── 3. L0 → L1: calibrate each store ─────────────────────────────────────
    run_id: str = ""
    l1_lineage_files: list[Path] = []
    for rec in outputs.l0_stores:
        l0_path = l0_dir / rec.store
        kind = infer_kind(rec.dp)
        l1_name = rec.store.replace(".zarr", ".L1.zarr")
        l1_path = l1_dir / l1_name
        lineage_file = l1_dir / l1_name.replace(".zarr", ".lineage.json")
        l1_rec = run_calibrate(
            l0_path,
            l1_path,
            kind=kind,
            codec=codec,
            level=level,
            shard_factor=shard_factor,
            recipe=calib_recipe,
            lineage_out=lineage_file,
        )
        outputs.l1_stores.append(l1_rec)
        l1_lineage_files.append(lineage_file)
        if not run_id:
            ds = open_store(l0_path)
            run_id = str(ds.attrs.get("run_id", obs_dir.name))

    # ── 4. Manifests: L0 + L1 ────────────────────────────────────────────────
    # L0 lineage: write a combined file from run_convert's records
    l0_lineage_file = l0_dir / "l0_stores.lineage.json"
    l0_lineage_file.write_text(
        json.dumps([r.model_dump(mode="json") for r in outputs.l0_stores], indent=2)
    )

    outputs.l0_manifest = run_manifest(
        [l0_lineage_file],
        level="L0",
        run_id=run_id,
        out_path=l0_dir / "manifest.json",
        stores_dir=l0_dir,
    )
    outputs.l1_manifest = run_manifest(
        l1_lineage_files,
        level="L1",
        run_id=run_id,
        out_path=l1_dir / "manifest.json",
        stores_dir=l1_dir,
    )

    # ── 5. ML: classify img L1 stores ────────────────────────────────────────
    img_l1_recs = [r for r in outputs.l1_stores if r.kind == "img"]
    if img_l1_recs:
        if model_path is None:
            return outputs  # no model → skip classification; caller can run it later
        l2_dir.mkdir(parents=True, exist_ok=True)
        l2_lineage_files: list[Path] = []
        for rec in img_l1_recs:
            l1_path = l1_dir / rec.store
            l2_name = rec.store.replace(".L1.zarr", ".L2.zarr")
            l2_path = l2_dir / l2_name
            lineage_file = l2_dir / l2_name.replace(".zarr", ".lineage.json")
            l2_rec = run_classify(
                l1_store=l1_path,
                l2_store=l2_path,
                model_path=model_path,
                lineage_out=lineage_file,
                codec=codec,
                level=level,
            )
            outputs.l2_stores.append(l2_rec)
            l2_lineage_files.append(lineage_file)

        # ── 6. L2 manifest ────────────────────────────────────────────────────
        outputs.l2_manifest = run_manifest(
            l2_lineage_files,
            level="L2",
            run_id=run_id,
            out_path=l2_dir / "manifest.json",
            stores_dir=l2_dir,
        )

    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(
    add_completion=False,
    help=(
        "Pure-Python dev-loop pipeline driver (ingest + ML for one obs dir).\n\n"
        "For production use (resume / SLURM / containers) run Nextflow instead:\n"
        "  nextflow run . -profile laptop --steps ingest,ml --input_obs_dir <OBS> --outdir <OUT>"
    ),
)


@app.command()
def main(
    obs_dir: Path = typer.Argument(..., help="Path to a .pffd observation directory."),
    out_dir: Path = typer.Argument(..., help="Root output dir (L0/ L1/ L2/ created inside)."),
    model_path: Path | None = typer.Option(None, "--model-path", help="Cloud model .pt file."),
    calib_recipe: Path | None = typer.Option(
        None, "--calib-recipe", help="Calibration YAML recipe."
    ),
    codec: str = typer.Option("zstd", help="Zarr compression codec."),
    level: int = typer.Option(5, help="Compression level."),
    shard_factor: int = typer.Option(0, "--shard-factor", help="Inner chunks per shard (0=none)."),
) -> None:
    outputs = run_pipeline(
        obs_dir,
        out_dir,
        model_path=model_path,
        calib_recipe=calib_recipe,
        codec=codec,
        level=level,
        shard_factor=shard_factor,
    )
    typer.echo(f"L0 stores : {len(outputs.l0_stores)}")
    typer.echo(f"HK stores : {len(outputs.hk_stores)}")
    typer.echo(f"L1 stores : {len(outputs.l1_stores)}")
    typer.echo(f"L2 stores : {len(outputs.l2_stores)}")
    if outputs.l0_manifest:
        typer.echo(f"L0 manifest: {out_dir}/L0/manifest.json")
    if outputs.l1_manifest:
        typer.echo(f"L1 manifest: {out_dir}/L1/manifest.json")
    if outputs.l2_manifest:
        typer.echo(f"L2 manifest: {out_dir}/L2/manifest.json")


if __name__ == "__main__":
    app()
