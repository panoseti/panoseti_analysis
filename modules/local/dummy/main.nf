

process DUMMY {
    tag "$meta.id"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/YOUR-TOOL-HERE':
        'quay.io/biocontainers/YOUR-TOOL-HERE' }"

    input:tuple val(meta), path(input)

    output:
    tuple val(meta), path("*"), emit: output

    tuple val("${task.process}"), val('dummy'), eval("dummy --version"), topic: versions, emit: versions_dummy

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    """
    ${args} ${args2}


    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: ''

    """
    echo $args

    """
}
