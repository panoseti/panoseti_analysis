#!/usr/bin/env bash
# SDSC Expanse launcher for the ingest pipeline (SLURM + Apptainer).
#   bash scripts/launchers/expanse.sh [obs.pffd] [results_dir]
# Env overrides: SLURM_ACCOUNT, SLURM_QUEUE.
set -euo pipefail

INPUT_OBS_DIR="${1:-/expanse/lustre/scratch/$USER/panoseti/inputs/obs.pffd}"
OUTDIR="${2:-/expanse/lustre/scratch/$USER/panoseti/results}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-sds166}"
SLURM_QUEUE="${SLURM_QUEUE:-shared}"

PIPELINE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# Load site modules if available (names vary by cluster).
module load nextflow 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularitypro 2>/dev/null || true

# Shared Apptainer cache so concurrent tasks reuse one pulled image.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/expanse/lustre/scratch/$USER/.apptainer_cache}"

nextflow run "$PIPELINE_DIR" \
    -profile hpc_slurm \
    --input_obs_dir "$INPUT_OBS_DIR" \
    --outdir "$OUTDIR" \
    --slurm_account "$SLURM_ACCOUNT" \
    --slurm_queue "$SLURM_QUEUE" \
    -resume "${@:3}"
