//
// ML — High-level ML subworkflow that composes specific models.
//

include { CLASSIFY_CLOUD } from './classify_cloud'
include { BUILD_MANIFEST as BUILD_MANIFEST_L2 } from '../../modules/local/build_manifest'

workflow ML {

    take:
    ch_l1_stores  // channel: [ meta, store ]

    main:
    // Only run cloud classification on img stores
    ch_img_stores = ch_l1_stores.filter { meta, _store -> meta.kind == 'img' }

    // Hardcode the model paths for Chunk 2, but these could be params
    model_file = file("${projectDir}/assets/models/cloud_detector_v1.pt")
    model_json = file("${projectDir}/assets/models/cloud_detector_v1.json")

    use_ray = params.use_ray ?: false

    CLASSIFY_CLOUD(ch_img_stores, model_file, model_json, use_ray)

    // L2 manifest aggregation
    BUILD_MANIFEST_L2(
        CLASSIFY_CLOUD.out.lineage
            .map { it ->
                // 'it' might be [meta, lineage_file] from CPU or [run_id, lineage_file] from Ray
                def run_id = it[0] instanceof Map ? it[0].run_id : it[0]
                def lineage_file = it[1]
                tuple(run_id, lineage_file)
            }
            .groupTuple()
            .map { run_id, lineages -> tuple(run_id, lineages, 'L2') }
    )

    emit:
    l2_stores     = CLASSIFY_CLOUD.out.stores
    l2_manifest   = BUILD_MANIFEST_L2.out.manifest.map { _level, m -> m }
    l2_quicklooks = CLASSIFY_CLOUD.out.quicklook
}
