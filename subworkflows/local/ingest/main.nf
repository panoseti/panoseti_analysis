//
// INGEST: PFF -> L0 Zarr -> L1 calibrated, per-(dp,module) fan-out + HK + manifests.
//

include { PFF_TO_ZARR }                          from '../../../modules/local/pff_to_zarr/main'
include { BUILD_HK }                             from '../../../modules/local/build_hk/main'
include { CALIBRATE_PH }                         from '../../../modules/local/calibrate_ph/main'
include { CALIBRATE_IMG }                        from '../../../modules/local/calibrate_img/main'
include { BUILD_MANIFEST as BUILD_MANIFEST_L0 }  from '../../../modules/local/build_manifest/main'
include { BUILD_MANIFEST as BUILD_MANIFEST_L1 }  from '../../../modules/local/build_manifest/main'

workflow INGEST {

    take:
    ch_obs   // channel: [ meta(run), obs_dir ]

    main:
    PFF_TO_ZARR(ch_obs)
    BUILD_HK(ch_obs)

    // Fan out per (dp, module): pair each emitted store with its lineage record by name.
    ch_l0_stores = PFF_TO_ZARR.out.l0.flatMap { meta, stores, lineage ->
        def stores_list = stores instanceof List ? stores : [stores]
        def by_name = stores_list.collectEntries { it -> [(it.name): it] }
        def records = new groovy.json.JsonSlurper().parse(lineage.toFile())
        records.collect { rec ->
            def m = meta + [dp: rec.dp, module: rec.module, kind: rec.kind, level: 'L0', cadence_ns: rec.cadence_ns]
            tuple(m, by_name[rec.store])
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
        PFF_TO_ZARR.out.l0.map { meta, _stores, lineage -> tuple(meta.run_id, lineage, 'L0') }
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
