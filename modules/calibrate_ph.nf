/*
 * CALIBRATE_PH — Pulse-height calibration for ph256 and ph1024 data.
 *
 * Input:  tuple val(product), path(l0_store)
 * Output: tuple val(product), path("<product>_L1.zarr"), emit: l1
 */
process CALIBRATE_PH {
    label 'cpu_medium'

    input:
    tuple val(product), path(l0_store)

    output:
    tuple val(product), path("${product}_L1.zarr"), emit: l1

    script:
    """
    calibrate_ph \\
        "${l0_store}" \\
        "${product}_L1.zarr" \\
        --sigma ${params.ph_sigma} \\
        --offset ${params.ph_offset} \\
        --stride ${params.ph_stride} \\
        --codec ${params.codec} \\
        --level ${params.level}
    """
}
