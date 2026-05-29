process PROPAGATE_META {
    tag { "${meta.run_id}:${level}" }
    label 'cpu_light'

    input:
    tuple val(meta), path(l0_dir), val(level)

    output:
    tuple val(level), path("${level}/.panoseti-meta"), optional: true, emit: meta

    script:
    """
    mkdir -p ${level}
    if [ -d ${l0_dir}/.panoseti-meta ]; then
        cp -R ${l0_dir}/.panoseti-meta ${level}/.panoseti-meta
    fi
    """
}
