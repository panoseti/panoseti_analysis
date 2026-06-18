process PFF_TO_ZARR {
    tag { meta.run_id }
    label 'io_heavy'

    input:
    tuple val(meta), path(obs_dir)

    output:
    tuple val(meta), path("*.zarr"), path("l0_lineage.json"), emit: l0

    script:
    """
    pa-convert ${obs_dir} . \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --time-chunk ${params.time_chunk} \\
        --shard-factor ${params.shard_factor_l0} \\
        --lineage-out l0_lineage.json \\
        --checksum \\
        ${params.use_tensorstore && params.use_tensorstore.toString() != 'false' ? '--use-tensorstore \\' : ''}
        --max-workers ${task.cpus}
    """
}
