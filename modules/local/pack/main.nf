process PACK {
    tag { store.name }
    label 'io_heavy'

    input:
    tuple val(meta), path(store), val(fmt)

    output:
    tuple val(meta), path("${store.name}.${fmt == 'tar' ? 'tar' : 'zarr.zip'}"), emit: packed

    script:
    def ext = fmt == 'tar' ? 'tar' : 'zarr.zip'
    """
    pa-pack ${store} ${store.name}.${ext} --format ${fmt}
    """
}
