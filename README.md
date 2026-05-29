# panoseti-analysis

PANOSETI data-reduction monorepo: PFF → Zarr ingest/calibration pipelines for HPC
(and, later, real-time and ML). Built on a strict three-layer architecture with an
**nf-core**-templated Nextflow pipeline (branding off, PANOSETI namespace).

> Orientation for contributors and Claude Code: see [`CLAUDE.md`](CLAUDE.md).
> Storage conventions: see [`docs/storage_spec.md`](docs/storage_spec.md).

## Layout

```
src/panoseti_analysis/
  algorithms/   Layer A — pure kernels (xarray/numpy/pydantic; no I/O, no frameworks)
  adapters/     Layer B — thin Typer CLIs (pa-convert, pa-calibrate, pa-hk, pa-manifest, pa-pack)
  io/           filesystem boundary (open/write/checksum/pack; wraps pypff)
  config/       versions, data_level registry, Pydantic models
bin/            pa-* shims (on $PATH inside Nextflow processes)
main.nf, workflows/, subworkflows/local/, modules/local/, conf/   Layer C — Nextflow
pypff/          git submodule (PFF reader + PFF→Zarr v3 converter) — do not edit here
tests/          Layer A unit tests + io/adapter tests + boundary lint + e2e data
docs/           storage_spec.md + nf-core docs
```

## Quick start

```bash
uv sync                                              # workspace venv (incl. pypff submodule)
uv run pytest                                        # 57 tests: kernels, io, adapters
uv run ruff check src tests && uv run mypy src/panoseti_analysis

# Ingest pipeline on bundled test data (PFF -> L0 -> L1, level-major output):
nextflow run . -profile test,laptop --outdir results_smoke
ls results_smoke/L0 results_smoke/L1                 # per-(dp,module) .zarr + manifest.json
```

## Pipeline

```
.pffd ──pa-convert──▶ L0/ (per-(dp,module) Zarr + lineage)
                          │  fan-out by (dp,module)
              ┌───────────┴───────────┐
        pa-calibrate ph         pa-calibrate img      pa-hk ──▶ hk.<hashset>.zarr
     (pedestal + n-σ)        (block+temporal median)
              └───────────┬───────────┘
                          ▼
                       L1/ (calibrated Zarr) + manifest.json (lineage)
```

L0 mirrors native PFF order; **L1 guarantees monotonic `unix_t_ns`** (stable-sorted,
frames never dropped). Profiles: `laptop` (no container), `docker`, `hpc_slurm`
(SLURM + Apptainer, e.g. SDSC Expanse). Select steps with `--steps ingest`
(`reconstruct`/`ml` are stubs). Input is a `run_id,obs_dir` samplesheet (`--input`) or
the single-run convenience `--input_obs_dir`.

## License

MIT © 2026 PANOSETI.
