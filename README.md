# panoseti-analysis
PANOSETI analysis software including: real-time and HPC pipelines, using Nextflow and Ray to orchestrate actions, machine learning for automated analysis, and other algorithms.

Converts PanoSETI PFF observation data to calibrated Zarr v3 stores using
**Nextflow 26.04 strict DSL2**. Runs identically on a laptop (no container),
in Docker, and on SDSC Expanse (SLURM + Singularity).

## Pipeline stages

```
.pffd  ──►  PFF_TO_ZARR  ──►  L0/ (one .zarr per product × module)
                                    │
                       ┌────────────┴─────────────┐
                       ▼                          ▼
               CALIBRATE_PH                CALIBRATE_IMG
           (ph256, ph1024)              (img8, img16)
           pedestal subtraction         block-median subtraction
           n-σ thresholding             + temporal supermedian
                       │                          │
                       └────────────┬─────────────┘
                                    ▼
                             L1/ (calibrated .zarr + summary.json + preview.png)
```

**Key design choices (v0.3.0):**
- Per-product parallelism is Nextflow-native — no Dask cluster required.
- Output publishing uses the Nextflow 26 `output {}` block (no deprecated `publishDir`).
- Calibration runs synchronously in a single Python process per store; xarray + zstd
  writes ~GB/s on a laptop and scales to HPC via SLURM without code changes.
- `ph` and `img` products are calibrated differently:
  - `ph256`/`ph1024` (`int16` ADC intensities): pedestal subtracted, n-σ masked.
  - `img8`/`img16` (`uint8`/`uint16` counts above threshold): spatial block-median
    + temporal supermedian subtracted, then ADC→PE scaled.

---

## Relationship to `pypff`

This pipeline depends on `pypff[zarr]` (installed in editable mode from `../pypff`).
The L0 conversion step (`bin/pff2zarr`) calls `pypff.zarr.convert_run` directly — no
hand-written PFF parsing. The L0 stores follow the [pypff Zarr v3 spec](https://github.com/panoseti/pypff/blob/opt/docs/zarr_v3_spec.md).

---

## Profiles

| Profile | Executor | Container | When to use |
|---|---|---|---|
| `laptop` | local | none (uses `.venv`) | Development, quick smoke tests |
| `docker` | local | Docker | Reproducible local runs, CI |
| `hpc_slurm` | SLURM | Singularity | Expanse / any SLURM cluster |

---

## Quick start — laptop (no container)

```bash
# 1. Install Python dependencies (requires uv)
uv sync

# 2. Run against the bundled test data (~seconds)
nextflow run . -profile laptop

# Results in results/L0/ and results/L1/
ls results/L0/*.zarr results/L1/*.zarr
```

---

## Quick start — Docker

Build the image once (from the **parent** `panoseti/` directory, not this repo):

```bash
cd ..   # go to ~/panoseti/
docker build -t panoseti-zarr-pipeline:0.3.0 \
    -f panoseti_zarr_seqera/Dockerfile .
```

Then run:

```bash
cd panoseti_zarr_seqera
nextflow run . -profile docker
```

The `docker` profile uses the local executor with the container — no SLURM needed.

---

## Inspecting results

```python
import xarray as xr

# L0 store (raw, converted from PFF)
ds_l0 = xr.open_zarr("results/L0/obs_TEST.pffd.dp_ph256.bpp_2.module_1.debug_TRUNCATED.zarr",
                      consolidated=False)
print(ds_l0)               # images, unix_t_ns, pkt_num, quabo_num, …

# L1 store (calibrated)
ds = xr.open_zarr("results/L1/dp_ph256.bpp_2.module_1_L1.zarr", consolidated=False)
print(ds)                  # pedestal_subtracted, hot_pixel_mask, unix_t_ns, …
print(ds.attrs["calibration"])   # params used
```

Each L1 store also contains `summary.json` (statistics) and `preview.png` (quick-look plot).

---

## Repository layout

```
panoseti_zarr_seqera/
├── main.nf                      ← entry workflow (strict DSL2)
├── nextflow.config              ← params + profiles + outputDir + reports
├── Dockerfile                   ← image build (context = parent panoseti/ dir)
├── conf/
│   ├── laptop.config            ← local executor, no container
│   ├── docker.config            ← local executor + Docker
│   └── hpc_slurm.config         ← SLURM + Singularity
├── modules/
│   ├── pff_to_zarr.nf
│   ├── calibrate_ph.nf
│   └── calibrate_img.nf
├── subworkflows/
│   └── calibrate.nf             ← routes ph vs img products
├── bin/                         ← executable entry-points (auto on $PATH in processes)
│   ├── pff2zarr
│   ├── calibrate_ph
│   └── calibrate_img
├── src/panoseti_zarr_pipeline/  ← importable Python package
│   ├── _common.py               ← shared: open_l0, write_l1, Stats, preview PNG
│   ├── calibrate_ph.py
│   └── calibrate_img.py
├── tests/
│   ├── test_calibrate_ph.py
│   └── test_calibrate_img.py
├── hpc/
│   ├── run_expanse.sh           ← SDSC Expanse launcher
│   └── legacy_dask/             ← archived Dask cluster scripts (Andrea Zonca)
├── obs_TEST.pffd/               ← bundled test observation (img16 + ph256, truncated)
├── scripts/
│   └── bench_convert.py         ← PFF→Zarr codec/chunk benchmark harness
└── pyproject.toml
```

---

## Parameters

All defaults live in `nextflow.config` under `params { … }`. Override on the CLI or via `-params-file`.

| Parameter | Default | Description |
|---|---|---|
| `input_obs_dir` | `obs_TEST.pffd` | Input `.pffd` observation directory |
| `outdir` | `results/` | Base output directory (L0/, L1/ published here) |
| `codec` | `zstd` | Zarr compression codec |
| `level` | `5` | Compression level |
| `time_chunk` | `0` (auto) | Time chunk size; 0 = auto-sized by pypff |
| `ph_sigma` | `5.0` | n-σ threshold for pulse-height masking |
| `ph_offset` | `800` | ADC offset added before pedestal estimation |
| `ph_stride` | `200` | Frame stride for pedestal sampling |
| `img_stride` | `200` | Frame stride for block-median sampling |
| `img_block` | `8` | Spatial block size (pixels) |
| `img_adc_to_pe` | `1.5` | ADC counts per photoelectron |
| `slurm_queue` | `debug` | SLURM queue (`hpc_slurm` profile only) |
| `slurm_account` | `''` | SLURM account (`hpc_slurm` profile only) |

---

## Running on SDSC Expanse

Transfer data with Globus:
```bash
globus login
globus transfer <src_endpoint>:/path/to/obs.pffd \
    <expanse_endpoint>:/expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
    --recursive
```

Build the Singularity image from Docker (on a machine with Docker):
```bash
cd ~/panoseti
docker build -t panoseti-zarr-pipeline:0.3.0 -f panoseti_zarr_seqera/Dockerfile .
# Then on Expanse:
singularity pull panoseti-zarr-pipeline_0.3.0.sif docker://panoseti-zarr-pipeline:0.3.0
```

Submit via the convenience script:
```bash
bash hpc/run_expanse.sh \
    /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
    /expanse/lustre/scratch/$USER/panoseti/results
```

Or directly:
```bash
nextflow run . -profile hpc_slurm \
    --input_obs_dir /expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd \
    --outdir        /expanse/lustre/scratch/$USER/panoseti/results \
    --slurm_account sds166 \
    --slurm_queue   debug \
    -resume
```

Or via Seqera Tower:
```bash
source ~/.bashrc   # loads TOWER_ACCESS_TOKEN
tw launch panoseti_zarr \
    --workspace sdsc/panoseti \
    --profile hpc_slurm \
    --revision main \
    --params-file params.json
```

Example `params.json`:
```json
{
  "input_obs_dir": "/expanse/lustre/scratch/user/panoseti/inputs/obs.pffd",
  "outdir":        "/expanse/lustre/scratch/user/panoseti/results",
  "slurm_account": "sds166",
  "slurm_queue":   "debug"
}
```

---

## Tower CLI cheatsheet

```bash
# Inspect runs
tw runs list --workspace sdsc/panoseti --max 5
tw runs view --id <runId> --workspace sdsc/panoseti --status
tw runs view --id <runId> --workspace sdsc/panoseti download --type log | tail -n 80
tw runs view --id <runId> --workspace sdsc/panoseti tasks

# Relaunch
tw launch panoseti_zarr --workspace sdsc/panoseti --profile hpc_slurm --revision main
```

---

## Testing

Unit tests (no Nextflow or container required):
```bash
uv sync
uv run pytest tests/ -v
```

End-to-end smoke — laptop:
```bash
nextflow run . -profile laptop --input_obs_dir obs_TEST.pffd --outdir results_smoke
```

End-to-end smoke — Docker:
```bash
nextflow run . -profile docker --input_obs_dir obs_TEST.pffd --outdir results_smoke
```

---

## Troubleshooting

- **Docker image not found**: build it from the parent `panoseti/` directory with
  `docker build -t panoseti-zarr-pipeline:0.3.0 -f panoseti_zarr_seqera/Dockerfile .`
  The build context must be the parent directory because `pypff/` is a sibling of this repo.
- **`ZarrUserWarning` about `summary.json` / `preview.png`**: zarr-python warns about
  non-Zarr files inside a store directory. These files are intentional (per-store summaries)
  and the warning is harmless.
- **SLURM `debug` queue timeout**: the `debug` queue on Expanse has a 30-minute wall limit.
  Increase `process.time` in `conf/hpc_slurm.config` or switch to `--slurm_queue shared`.
- **Tower run stuck at `SUBMITTED`**: check that the Expanse compute environment is healthy
  and the queue has available slots. Use `tw runs view ... download --type log`.
