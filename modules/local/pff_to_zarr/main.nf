process PFF_TO_ZARR {
    tag { meta.run_id }
    label 'io_heavy'

    input:
    tuple val(meta), path(obs_dir)

    output:
    tuple val(meta), path("*.zarr"), path("l0_lineage.json"), emit: l0
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    """
    ${args} ${args2}
pa-convert ${obs_dir} . \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --time-chunk ${params.time_chunk} \\
        --shard-factor ${params.shard_factor_l0} \\
        --lineage-out l0_lineage.json \\
        --checksum ${params.use_tensorstore && params.use_tensorstore.toString() != 'false' ? '--use-tensorstore' : ''} \\
        --max-workers ${task.cpus}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
