/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    panoseti/analysis — master workflow (composes subworkflows by --steps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { INGEST }      from '../subworkflows/local/ingest'
include { RECONSTRUCT } from '../subworkflows/local/reconstruct'
include { ML }          from '../subworkflows/local/ml'

workflow ANALYSIS {

    take:
    ch_samplesheet // channel: [ meta(run), obs_dir ]
    outdir

    main:
    def steps = (params.steps ?: 'ingest').tokenize(',')

    def l0_stores   = channel.empty()
    def l1_stores   = channel.empty()
    def hk_stores   = channel.empty()
    def l0_manifest = channel.empty()
    def l1_manifest = channel.empty()

    if ('ingest' in steps) {
        INGEST(ch_samplesheet)
        l0_stores   = INGEST.out.l0_stores
        l1_stores   = INGEST.out.l1_stores
        hk_stores   = INGEST.out.hk_stores
        l0_manifest = INGEST.out.l0_manifest
        l1_manifest = INGEST.out.l1_manifest
    }

    if ('reconstruct' in steps) {
        RECONSTRUCT(l1_stores)   // STUB
    }
    if ('ml' in steps) {
        ML(l1_stores)            // STUB
    }

    emit:
    l0_stores
    l1_stores
    hk_stores
    l0_manifest
    l1_manifest
}
