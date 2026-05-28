# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **Nextflow 26.04 strict DSL2** pipeline that converts PanoSETI `.pffd` observation runs into calibrated Zarr v3 stores. It wraps the `pypff[zarr]` library (sibling directory `../pypff`) and adds per-data-product calibration steps.

## Commands

```bash
# Install Python dependencies
uv sync

# Run unit tests (no Nextflow required)
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_calibrate_ph.py::test_calibrate_ph_dtype

# Smoke-test the full pipeline (requires Nextflow ≥ 26.04 on PATH)
nextflow run . -profile laptop

# Custom input
nextflow run . -profile laptop --input_obs_dir /path/to/obs.pffd --outdir /tmp/results
```

## Architecture

### Nextflow structure

All Nextflow files use **strict DSL2** (`nextflow.enable.strict = true` in `nextflow.config`):
- `main.nf` — entry workflow; creates obs channel, calls `PFF_TO_ZARR`, fans out via `manifest.tsv`, calls `CALIBRATE`, publishes via `output {}` block (no `publishDir`).
- `modules/pff_to_zarr.nf` — wraps `bin/pff2zarr`; emits `l0_dir` + `manifest`.
- `modules/calibrate_ph.nf` — wraps `bin/calibrate_ph`; one task per ph store.
- `modules/calibrate_img.nf` — wraps `bin/calibrate_img`; one task per img store.
- `subworkflows/calibrate.nf` — filters channel by `kind` ('ph' vs 'img'), routes to appropriate module, re-attaches `kind` on output.
- `conf/laptop.config` — local executor, no container.
- `conf/hpc_slurm.config` — SLURM, Singularity; parameterized via `slurm_queue`/`slurm_account`.

### Python package (`src/panoseti_zarr_pipeline/`)

- `_common.py`: `open_l0(store)`, `write_l1(arrays, path, attrs, ...)`, `infer_kind(ds)`, `Stats` + `summarize(da)`, `make_preview_png`, `write_summary_json`. Shared by both calibration paths.
- `calibrate_ph.py`: pulse-height calibration for `ph256`/`ph1024` (`int16` ADC). Steps: optional ph1024 quabo position fix-up → add baseline offset → strided pedestal/sigma → subtract → n-σ mask → hot/dead mask.
- `calibrate_img.py`: image-mode calibration for `img8`/`img16` (`uint8`/`uint16` counts above threshold). Steps: cast to `int32` → 8×8 block median on strided subset → upsample → subtract → temporal supermedian → subtract → ADC→PE scaling → hot/dead mask.

### `bin/` entry-points

Executable scripts auto-added to `$PATH` inside Nextflow processes. Each is a ~15-line typer wrapper around the corresponding `src/panoseti_zarr_pipeline/...` function.

- `bin/pff2zarr` — calls `pypff.zarr.convert_run`, then `PanosetiZarrRun.list_products` to enumerate stores and emit `manifest.tsv` (columns: `product`, `store`, `kind`).
- `bin/calibrate_ph` — calls `calibrate_ph.calibrate_ph`.
- `bin/calibrate_img` — calls `calibrate_img.calibrate_img`.

### L1 store schema

Each L1 `.zarr` extends the L0 store (see `../pypff/docs/zarr_v3_spec.md`) with:
- New data variable: `pedestal_subtracted` (ph) or `median_subtracted` (img), dtype `float32`.
- New data variables: `hot_pixel_mask`, `dead_pixel_mask`, dtype `uint8`, shape `(H, W)`.
- Preserved from L0: `unix_t_ns`, `pkt_num`/`quabo_num` (or `quabo_<i>_pkt_num`).
- New root attrs: `panoseti_pff_zarr_l1_version = "0.1"`, `calibration` dict with all calibration params, L0's `run_configs` copied verbatim.
- Sidecar files inside the `.zarr` directory: `summary.json` (statistics), `preview.png` (quick-look).

## Key conventions

- `pypff` is installed in editable mode from `../pypff`. If you need to update pypff, do so there; changes take effect immediately.
- `infer_kind` reads `ds.attrs["data_product"]` (set by `pypff.zarr.convert_run`) — never sniff the kind from the filename.
- The `write_l1` function always drops the existing L1 store before writing (idempotent).
- Calibration functions write their own `summary.json` and `preview.png` — no separate summarize step needed in the Nextflow workflow.
- No Dask imports anywhere in `src/` or `bin/`. Computation uses xarray's default scheduler.
- `xr.open_zarr(..., consolidated=False)` everywhere — we never write consolidated metadata.

## Data product pixel semantics

| Product | dtype | Values | Calibration |
|---|---|---|---|
| `ph256` (16×16) | `int16` | ADC intensity counts; can be negative | pedestal subtraction + n-σ threshold |
| `ph1024` (32×32) | `int16` | Same; quabo 1↔2 position fix-up needed | same as ph256 |
| `img8` (32×32) | `uint8` | Counts above threshold in integration window; always ≥ 0 | block-median + temporal-median + ADC→PE |
| `img16` (32×32) | `uint16` | Same, wider range | same as img8 |

## HPC / Expanse

- `hpc/run_expanse.sh` — convenience launcher for SDSC Expanse.
- `hpc/legacy_dask/` — archived Dask cluster scripts from Andrea Zonca's original pipeline. No longer used; preserved for reference.
- Tower CLI notes in `AGENTS.md`.
