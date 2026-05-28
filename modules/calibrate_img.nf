/*
 * CALIBRATE_IMG — Image-mode calibration for img8 and img16 data.
 *
 * Input:  tuple val(product), path(l0_store)
 * Output: tuple val(product), path("<product>_L1.zarr"), emit: l1
 */
process CALIBRATE_IMG {
    label 'cpu_medium'

    input:
    tuple val(product), path(l0_store)

    output:
    tuple val(product), path("${product}_L1.zarr"), emit: l1

    script:
    """
    calibrate_img \\
        "${l0_store}" \\
        "${product}_L1.zarr" \\
        --stride ${params.img_stride} \\
        --block ${params.img_block} \\
        --adc-to-pe ${params.img_adc_to_pe} \\
        --codec ${params.codec} \\
        --level ${params.level}
    """
}
