process CLASSIFY_CLOUD_RAY {
    label 'gpu_ray'

    input:
    tuple val(run_id), path(l1_stores)
    path  model_file
    path  model_json

    output:
    path("*.zarr")                          , emit: stores
    tuple val(run_id), path("lineage.json") , emit: lineage
    path("*.png")                           , emit: quicklook

    script:
    def stores_arg = l1_stores instanceof List
        ? l1_stores.collect { it -> it.toString() }.join(' ')
        : l1_stores.toString()

    if (params.ray_launcher == "slurm") {
        // Expanse / SLURM: bring up a transient Ray cluster with ray symmetric-run,
        // then dispatch inference tasks across the allocation.
        """
        export RAY_TMPDIR=\${TMPDIR:-/tmp}
        for store in ${stores_arg}; do
            echo "\$store" >> stores.list
        done

        NUM_NODES=\${SLURM_JOB_NUM_NODES:-1}
        NUM_CPUS=\${SLURM_CPUS_PER_TASK:-4}
        NUM_GPUS=\${SLURM_GPUS_PER_NODE:-0}

        srun --nodes=\$NUM_NODES --ntasks-per-node=1 \\
             ray symmetric-run \\
             --min-nodes \$NUM_NODES \\
             --num-cpus \$NUM_CPUS \\
             --num-gpus \$NUM_GPUS \\
             -- pa-ray-classify-cloud \\
                stores.list \\
                . \\
                ${model_file} \\
                --launcher slurm \\
                --lineage-out lineage.json \\
                --quicklook-dir .
        """
    } else {
        // attach (RAL bare-metal) or standalone (CI/laptop): run the CLI directly.
        // init_ray() inside pa-ray-classify-cloud handles cluster attach vs. local spin-up.
        """
        export RAY_TMPDIR=\${TMPDIR:-/tmp}
        for store in ${stores_arg}; do
            echo "\$store" >> stores.list
        done

        pa-ray-classify-cloud \\
            stores.list \\
            . \\
            ${model_file} \\
            --launcher ${params.ray_launcher} \\
            --lineage-out lineage.json \\
            --quicklook-dir .
        """
    }
}
