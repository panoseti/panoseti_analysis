process BUILD_MANIFEST {
    tag { "${run_id}:${level}" }
    label 'cpu_light'

    input:
    tuple val(run_id), path(lineage), val(level)

    output:
    tuple val(level), path("manifest.json"), emit: manifest

    script:
    def lins = (lineage instanceof List ? lineage : [lineage]).collect { "--lineage ${it}" }.join(' ')
    """
    pa-manifest manifest.json --level ${level} --run-id ${run_id} ${lins}
    """
}
