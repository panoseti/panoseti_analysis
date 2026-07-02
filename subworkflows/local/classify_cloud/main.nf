//
// CLASSIFY_CLOUD: Routes L1 stores to either CPU fanout or Ray execution
//

include { CLASSIFY_CLOUD_CPU } from '../../../modules/local/classify_cloud_cpu/main'
include { CLASSIFY_CLOUD_RAY } from '../../../modules/local/classify_cloud_ray/main'

workflow CLASSIFY_CLOUD {

    take:
    ch_l1_stores  // channel: [ meta, l1_store ]
    model_file
    model_json
    use_ray       // boolean value from params

    main:
    ch_l2_stores    = channel.empty()
    ch_l2_lineage   = channel.empty()
    ch_l2_quicklook = channel.empty()

    if (use_ray) {
        // Group L1 stores by run_id so each Ray cluster invocation
        // handles all modules for a given run together
        ch_grouped = ch_l1_stores
            .map { meta, store -> tuple(meta.run_id, store) }
            .groupTuple()

        CLASSIFY_CLOUD_RAY(ch_grouped, model_file, model_json)

        // Flatten Ray outputs back to per-module streams.
        // Store names are <run_id>.cloud.module_<m>.zarr; parse only the well-defined suffix.
        // Quicklook names are <run_id>.cloud.module_<m>.quicklook.png.
        ch_l2_stores    = CLASSIFY_CLOUD_RAY.out.stores.flatMap().map { store ->
            def parts = store.name.tokenize('.')
            def run_id = parts[0..-4].join('.')
            tuple([run_id: run_id, level: 'L2', kind: 'cloud'], store)
        }
        ch_l2_lineage   = CLASSIFY_CLOUD_RAY.out.lineage
        ch_l2_quicklook = CLASSIFY_CLOUD_RAY.out.quicklook.flatMap().map { quicklook ->
            def parts = quicklook.name.tokenize('.')
            def run_id = parts[0..-5].join('.')
            tuple([run_id: run_id, level: 'L2', kind: 'cloud'], quicklook)
        }
    } else {
        CLASSIFY_CLOUD_CPU(ch_l1_stores, model_file, model_json)
        ch_l2_stores    = CLASSIFY_CLOUD_CPU.out.store
        ch_l2_lineage   = CLASSIFY_CLOUD_CPU.out.lineage
        ch_l2_quicklook = CLASSIFY_CLOUD_CPU.out.quicklook
    }

    emit:
    stores    = ch_l2_stores
    lineage   = ch_l2_lineage
    quicklook = ch_l2_quicklook
}
