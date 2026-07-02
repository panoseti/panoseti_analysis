process CLASSIFY_CLOUD_CPU {
    label 'cpu_fanout'

    input:
    tuple val(meta), path(l1_store)
    path  model_file
    path  model_json

    output:
    tuple val(meta), path("*.zarr"), emit: store
    tuple val(meta), path("*.json"), emit: lineage
    tuple val(meta), path("*.png") , emit: quicklook
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    l2_store       = "${meta.run_id}.cloud.module_${meta.module}.zarr"
    def lineage_file   = "${l2_store}.lineage.json"
    def quicklook_file = "${meta.run_id}.cloud.module_${meta.module}.quicklook.png"
    """
    ${args} ${args2}
pa-classify-cloud \\
        ${l1_store} \\
        ${l2_store} \\
        ${model_file} \\
        --lineage-out ${lineage_file} \\
        --quicklook-out ${quicklook_file}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
