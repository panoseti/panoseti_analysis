process CALIBRATE_IMG {
    tag { "${meta.dp}.module_${meta.module}" }
    label 'cpu_fanout'

    input:
    tuple val(meta), path(l0_store)

    output:
    tuple val(meta), path("${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.zarr"), emit: store
    tuple val(meta), path("${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.lineage.json"), emit: lineage

    script:
    def out_store = "${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.zarr"
    def out_lineage = "${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.lineage.json"
    """
    pa-calibrate ${l0_store} ${out_store} \\
        --kind img \\
        --img-stride ${params.img_stride} \\
        --block ${params.img_block} \\
        --adc-to-pe ${params.img_adc_to_pe} \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --lineage-out ${out_lineage}
    """
}
