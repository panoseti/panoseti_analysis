//
// CLASSIFY_CLOUD: Routes L1 stores to either CPU fanout or Ray execution
//

include { CLASSIFY_CLOUD_CPU } from '../../modules/local/classify_cloud_cpu'
include { CLASSIFY_CLOUD_RAY } from '../../modules/local/classify_cloud_ray'

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

        // Flatten the Ray outputs back into per-module meta streams if needed,
        // but for now we just emit them. The manifest builder will handle lineage arrays.
        ch_l2_stores    = CLASSIFY_CLOUD_RAY.out.stores.flatMap().map { store ->
            tuple([run_id: store.name.split('\\.')[0], level: 'L2', kind: 'cloud'], store)
        }
        ch_l2_lineage   = CLASSIFY_CLOUD_RAY.out.lineage
        ch_l2_quicklook = CLASSIFY_CLOUD_RAY.out.quicklook.flatMap().map { quicklook ->
            tuple([run_id: quicklook.name.split('\\.')[0], level: 'L2', kind: 'cloud'], quicklook)
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
