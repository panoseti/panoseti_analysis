process BUILD_HK {
    tag { meta.run_id }
    label 'cpu_fanout'

    input:
    tuple val(meta), path(obs_dir)

    output:
    tuple val(meta), path("hk/hk.*.zarr"), optional: true, emit: stores
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    """
    ${args} ${args2}
pa-hk ${obs_dir} hk --codec ${params.codec} --level ${params.level}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
