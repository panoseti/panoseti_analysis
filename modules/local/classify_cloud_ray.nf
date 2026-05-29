process CLASSIFY_CLOUD_RAY {
    label 'gpu_ray'
    
    input:
    tuple val(run_id), path(l1_stores)
    path(model_file)
    path(model_json)
    
    output:
    path("*.zarr")        , emit: stores
    tuple val(run_id), path("lineage.json")  , emit: lineage
    
    script:
    """
    export RAY_TMPDIR=\$TMPDIR
    
    # Write the list of stores to a file to avoid ARG_MAX issues
    for store in ${l1_stores.join(' ')}; do
        echo "\$store" >> stores.list
    done
    
    srun --nodes=\$SLURM_NNODES --ntasks-per-node=1 \\
         ray symmetric-run \\
         -- pa-ray-classify-cloud \\
            stores.list \\
            . \\
            ${model_file} \\
            --lineage-out lineage.json
    """
}
