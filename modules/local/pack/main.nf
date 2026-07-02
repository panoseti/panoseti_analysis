process PACK {
    tag { store.name }
    label 'io_heavy'

    input:
    tuple val(meta), path(store), val(fmt)

    output:
    tuple val(meta), path("${store.name}.${fmt == 'tar' ? 'tar' : 'zarr.zip'}"), emit: packed
    path "versions.yml", emit: versions

    script:
    def args = task.ext.args ?: ''
    def args2 = task.ext.args2 ?: ''
    ext = fmt == 'tar' ? 'tar' : 'zarr.zip'
    """
    ${args} ${args2}
pa-pack ${store} ${store.name}.${ext} --format ${fmt}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        panoseti_analysis: \$(pa-run --version 2>&1 | sed 's/pa-run version //')
    END_VERSIONS
    """
}
