//
// INGEST: PFF -> L0 Zarr -> L1 calibrated, per-(dp,module) fan-out + HK + manifests.
//

import groovy.json.JsonSlurper

include { PFF_TO_ZARR }                          from '../../modules/local/pff_to_zarr'
include { BUILD_HK }                             from '../../modules/local/build_hk'
include { CALIBRATE_PH }                         from '../../modules/local/calibrate_ph'
include { CALIBRATE_IMG }                        from '../../modules/local/calibrate_img'
include { BUILD_MANIFEST as BUILD_MANIFEST_L0 }  from '../../modules/local/build_manifest'
include { BUILD_MANIFEST as BUILD_MANIFEST_L1 }  from '../../modules/local/build_manifest'

workflow INGEST {

    take:
    ch_obs   // channel: [ meta(run), obs_dir ]

    main:
    PFF_TO_ZARR(ch_obs)
    BUILD_HK(ch_obs)

    // Fan out per (dp, module) by reading the L0 lineage JSON array.
    ch_l0_stores = PFF_TO_ZARR.out.l0.flatMap { meta, l0_dir, lineage ->
        def records = new JsonSlurper().parse(lineage.toFile())
        records.collect { rec ->
            def m = meta + [dp: rec.dp, module: rec.module, kind: rec.kind, level: 'L0', cadence_ns: rec.cadence_ns]
            tuple(m, file("${l0_dir}/${rec.store}"))
        }
    }

    ph_in  = ch_l0_stores.filter { _meta, _store -> _meta.kind == 'ph' }
    img_in = ch_l0_stores.filter { _meta, _store -> _meta.kind == 'img' }

    CALIBRATE_PH(ph_in)
    CALIBRATE_IMG(img_in)

    ch_l1_store   = CALIBRATE_PH.out.store.mix(CALIBRATE_IMG.out.store)
    ch_l1_lineage = CALIBRATE_PH.out.lineage.mix(CALIBRATE_IMG.out.lineage)

    // L0 manifest: one array lineage file per run.
    BUILD_MANIFEST_L0(
        PFF_TO_ZARR.out.l0.map { meta, _l0_dir, lineage -> tuple(meta.run_id, lineage, 'L0') }
    )

    // L1 manifest: group per-store lineage fragments by run.
    BUILD_MANIFEST_L1(
        ch_l1_lineage
            .map { meta, lineage -> tuple(meta.run_id, lineage) }
            .groupTuple()
            .map { run_id, lineages -> tuple(run_id, lineages, 'L1') }
    )

    emit:
    l0_stores   = ch_l0_stores
    l1_stores   = ch_l1_store
    hk_stores   = BUILD_HK.out.stores
                      .map { meta, stores -> tuple(meta, stores instanceof List ? stores : [stores]) }
                      .transpose()
    l0_manifest = BUILD_MANIFEST_L0.out.manifest.map { _level, m -> m }
    l1_manifest = BUILD_MANIFEST_L1.out.manifest.map { _level, m -> m }
}
