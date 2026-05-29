process CLASSIFY_CLOUD_CPU {
    label 'cpu_fanout'
    
    input:
    tuple val(meta), path(l1_store)
    path(model_file)
    path(model_json)
    
    output:
    tuple val(meta), path("*.zarr")   , emit: store
    tuple val(meta), path("*.json")   , emit: lineage
    
    script:
    def l2_store = "${meta.run_id}.cloud.module_${meta.module}.zarr"
    def lineage_file = "${l2_store}.lineage.json"
    """
    pa-classify-cloud \\
        ${l1_store} \\
        ${l2_store} \\
        ${model_file} \\
        --lineage-out ${lineage_file}
    """
}
