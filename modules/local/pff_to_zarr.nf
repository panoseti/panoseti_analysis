process PFF_TO_ZARR {
    tag { meta.run_id }
    label 'io_heavy'

    input:
    tuple val(meta), path(obs_dir)

    output:
    tuple val(meta), path("L0"), path("l0_lineage.json"), emit: l0

    script:
    """
    pa-convert ${obs_dir} . \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --time-chunk ${params.time_chunk} \\
        --lineage-out l0_lineage.json
    """
}
