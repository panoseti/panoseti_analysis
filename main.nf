#!/usr/bin/env nextflow
/*
 * PanoSETI Zarr Pipeline — Nextflow 26.04 strict DSL2
 *
 * Stages:
 *   1. PFF_TO_ZARR  : convert .pffd → per-(product, module) L0 Zarr v3 stores
 *   2. CALIBRATE    : pedestal subtraction (ph) or block-median subtraction (img)
 *
 * Profiles:  -profile laptop    (local executor, no container)
 *            -profile hpc_slurm (SLURM + singularity)
 *
 * Quick start:
 *   nextflow run . -profile laptop
 *
 * Override observation run:
 *   nextflow run . -profile laptop \
 *       --input_obs_dir /path/to/obs.pffd \
 *       --outdir /path/to/results
 */

include { PFF_TO_ZARR } from './modules/pff_to_zarr.nf'
include { CALIBRATE } from './subworkflows/calibrate.nf'

workflow {

    main:
    // ── Stage 1: PFF → L0 Zarr ───────────────────────────────────────────────
    obs = channel.fromPath(params.input_obs_dir, type: 'dir', checkIfExists: true)
    l0 = PFF_TO_ZARR(obs)

    // Fan out: parse manifest.tsv → one tuple per (product, l0_store, kind).
    // manifest.tsv columns: product <TAB> store <TAB> kind
    stores = l0.manifest
        .splitCsv(header: true, sep: '\t')
        .map { row -> tuple(row.product, file(row.store, checkIfExists: true), row.kind) }

    // ── Stage 2: Calibration (ph or img, in parallel per store) ──────────────
    l1 = CALIBRATE(stores)

    publish:
    l0_stores = l0.l0_dir
    l1_stores = l1.l1.map { product, store, kind -> store }
}

// ── Output block (replaces publishDir) ───────────────────────────────────────
// outputDir is set to params.outdir in nextflow.config.
output {
    l0_stores {
        path '.'
        mode 'copy'
        overwrite true
    }
    l1_stores {
        path 'L1'
        mode 'copy'
        overwrite true
    }
}
