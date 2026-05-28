#!/bin/bash
# Run the PanoSETI Zarr pipeline on SDSC Expanse via Nextflow + SLURM.
#
# Usage:
#   bash hpc/run_expanse.sh
#   bash hpc/run_expanse.sh /path/to/obs.pffd /scratch/$USER/results
#
# Pre-requisites on Expanse:
#   module load singularity   (or apptainer)
#   module load nextflow      (or install via sdk: sdk install nextflow)
#   uv sync                   (only needed if NOT using container; for laptop-style runs)
#
# The hpc_slurm profile submits one SLURM job per calibration product.
# Parallelism is handled by Nextflow — no Dask cluster required.

set -euo pipefail

INPUT_OBS_DIR="${1:-/expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd}"
OUTDIR="${2:-/expanse/lustre/scratch/$USER/panoseti/results}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-sds166}"
SLURM_QUEUE="${SLURM_QUEUE:-debug}"

PIPELINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Pipeline dir : $PIPELINE_DIR"
echo "Input        : $INPUT_OBS_DIR"
echo "Output       : $OUTDIR"
echo "SLURM account: $SLURM_ACCOUNT  queue: $SLURM_QUEUE"
echo ""

nextflow run "$PIPELINE_DIR/main.nf" \
    -profile hpc_slurm \
    --input_obs_dir "$INPUT_OBS_DIR" \
    --outdir        "$OUTDIR" \
    --slurm_account "$SLURM_ACCOUNT" \
    --slurm_queue   "$SLURM_QUEUE" \
    -resume \
    "$@"

echo ""
echo "Done. Results in: $OUTDIR"
