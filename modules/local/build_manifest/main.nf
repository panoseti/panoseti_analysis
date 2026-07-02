process BUILD_MANIFEST {
    tag { "${run_id}:${level}" }
    label 'cpu_light'

    input:
    tuple val(run_id), path(lineage), val(level)

    output:
    tuple val(level), path("manifest.json"), emit: manifest
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    lins = (lineage instanceof List ? lineage : [lineage]).collect { it -> "--lineage ${it}" }.join(' ')
    """
    ${args} ${args2}
pa-manifest manifest.json --level ${level} --run-id ${run_id} ${lins}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
