process CALIBRATE_PH {
    tag { "${meta.dp}.module_${meta.module}" }
    label 'cpu_fanout'

    input:
    tuple val(meta), path(l0_store)

    output:
    tuple val(meta), path("${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.zarr"), emit: store
    tuple val(meta), path("${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.lineage.json"), emit: lineage
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    out_store = "${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.zarr"
    def out_lineage = "${meta.run_id}.dp_${meta.dp}.module_${meta.module}.L1.lineage.json"
    """
    ${args} ${args2}
pa-calibrate ${l0_store} ${out_store} \\
        --kind ph \\
        --sigma ${params.ph_sigma} \\
        --offset ${params.ph_offset} \\
        --ph-stride ${params.ph_stride} \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --shard-factor ${params.shard_factor_l1} \\
        --lineage-out ${out_lineage}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
