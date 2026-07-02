#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Default parameters
params.make_plots = false

process INGEST_L0 {
    input:
    val run_id

    output:
    path "${run_id}_L0.zarr"

    script:
    """
    mkdir -p ${run_id}_L0.zarr
    echo "Raw counts for ${run_id}" > ${run_id}_L0.zarr/data.txt
    """
}

process CALIBRATE_L1 {
    input:
    path l0_zarr
    path demo_script

    output:
    path "L1.zarr"

    script:
    """
    python3 $demo_script $l0_zarr L1.zarr
    """
}

// STUDENTS: Add L2_QUICKLOOK process here

workflow {
    ch_runs = channel.of('run_1', 'run_2')

    l0_out = INGEST_L0(ch_runs)
    l1_out = CALIBRATE_L1(l0_out, file("${projectDir}/demo_l0_to_l1.py"))

    l1_out.view { it -> "Produced calibrated store: $it" }

    // STUDENTS: Add conditional execution for L2_QUICKLOOK here
}
