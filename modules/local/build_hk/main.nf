process BUILD_HK {
    tag { meta.run_id }
    label 'cpu_fanout'

    input:
    tuple val(meta), path(obs_dir)

    output:
    tuple val(meta), path("hk/hk.*.zarr"), optional: true, emit: stores

    script:
    """
    pa-hk ${obs_dir} hk --codec ${params.codec} --level ${params.level}
    """
}
