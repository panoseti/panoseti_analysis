process L1_QUICKLOOK {
    cache false
    tag { "${meta.dp}.module_${meta.module}" }
    label 'cpu_light'

    input:
    tuple val(meta), path(l0_store), path(l1_store)

    output:
    tuple val(meta), path("*.png"), emit: quicklook
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    quicklook_file = "${meta.run_id}.dp_${meta.dp}.module_${meta.module}.quicklook.png"
    """
    ${args} ${args2}
pa-l1-quicklook \\
        ${l0_store} \\
        ${l1_store} \\
        ${quicklook_file}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
