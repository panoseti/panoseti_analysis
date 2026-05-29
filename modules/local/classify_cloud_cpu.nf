process CLASSIFY_CLOUD_CPU {
    label 'cpu_fanout'
    
    input:
    tuple val(meta), path(l1_store)
    path(model_file)
    path(model_json)
    
    output:
    tuple val(meta), path("*.zarr")   , emit: store
    tuple val(meta), path("*.json")   , emit: lineage
    tuple val(meta), path("*.png")    , emit: quicklook
    
    script:
    def l2_store = "${meta.run_id}.cloud.module_${meta.module}.zarr"
    def lineage_file = "${l2_store}.lineage.json"
    def quicklook_file = "${meta.run_id}.cloud.module_${meta.module}.quicklook.png"
    """
    pa-classify-cloud \\
        ${l1_store} \\
        ${l2_store} \\
        ${model_file} \\
        --lineage-out ${lineage_file} \\
        --quicklook-out ${quicklook_file}
    """
}
