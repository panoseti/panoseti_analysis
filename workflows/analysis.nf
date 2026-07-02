/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    panoseti/analysis — master workflow (composes subworkflows by --steps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { INGEST }      from '../subworkflows/local/ingest/main'
include { RECONSTRUCT } from '../subworkflows/local/reconstruct/main'
include { ML }          from '../subworkflows/local/ml/main'

workflow ANALYSIS {

    take:
    ch_samplesheet // channel: [ meta(run), obs_dir ]
    _outdir

    main:
    def steps = (params.steps ?: 'ingest').tokenize(',')

    ch_l0_stores     = channel.empty()
    ch_l1_stores     = channel.empty()
    ch_hk_stores     = channel.empty()
    ch_l0_manifest   = channel.empty()
    ch_l1_manifest   = channel.empty()
    ch_l2_stores     = channel.empty()
    ch_l2_manifest   = channel.empty()
    ch_l2_quicklooks = channel.empty()

    if ('ingest' in steps) {
        INGEST(ch_samplesheet)
        ch_l0_stores   = INGEST.out.l0_stores
        ch_l1_stores   = INGEST.out.l1_stores
        ch_hk_stores   = INGEST.out.hk_stores
        ch_l0_manifest = INGEST.out.l0_manifest
        ch_l1_manifest = INGEST.out.l1_manifest
    }

    if ('reconstruct' in steps) {
        RECONSTRUCT(ch_l1_stores)   // STUB
    }
    if ('ml' in steps) {
        ML(ch_l1_stores)
        ch_l2_stores     = ML.out.l2_stores
        ch_l2_manifest   = ML.out.l2_manifest
        ch_l2_quicklooks = ML.out.l2_quicklooks
    }

    emit:
    l0_stores     = ch_l0_stores
    l1_stores     = ch_l1_stores
    hk_stores     = ch_hk_stores
    l0_manifest   = ch_l0_manifest
    l1_manifest   = ch_l1_manifest
    l2_stores     = ch_l2_stores
    l2_manifest   = ch_l2_manifest
    l2_quicklooks = ch_l2_quicklooks
}
