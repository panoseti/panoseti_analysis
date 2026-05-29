#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    panoseti/analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Github : https://github.com/panoseti/analysis
----------------------------------------------------------------------------------------
*/

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT FUNCTIONS / MODULES / SUBWORKFLOWS / WORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { ANALYSIS  } from './workflows/analysis'
include { PIPELINE_INITIALISATION } from './subworkflows/local/utils_nfcore_analysis_pipeline'
include { PIPELINE_COMPLETION     } from './subworkflows/local/utils_nfcore_analysis_pipeline'
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NAMED WORKFLOWS FOR PIPELINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// WORKFLOW: Run main analysis pipeline depending on type of input
//
workflow PANOSETI_ANALYSIS {

    take:
    samplesheet // channel: [ meta(run), obs_dir ]

    main:
    ANALYSIS (
        samplesheet,
        params.outdir,
    )

    emit:
    l0_stores   = ANALYSIS.out.l0_stores.map   { _meta, s -> s }
    l1_stores   = ANALYSIS.out.l1_stores.map   { _meta, s -> s }
    hk_stores   = ANALYSIS.out.hk_stores.map   { _meta, s -> s }
    l0_manifest = ANALYSIS.out.l0_manifest
    l1_manifest = ANALYSIS.out.l1_manifest
    l2_stores   = ANALYSIS.out.l2_stores.map   { _meta, s -> s }
    l2_manifest = ANALYSIS.out.l2_manifest
}
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow {

    main:
    //
    // SUBWORKFLOW: Run initialisation tasks
    //
    PIPELINE_INITIALISATION (
        params.version,
        params.validate_params,
        params.monochrome_logs,
        args,
        params.outdir,
        params.input,
        params.help,
        params.help_full,
        params.show_hidden,
        params.input_obs_dir
    )

    //
    // WORKFLOW: Run main workflow
    //
    PANOSETI_ANALYSIS (
        PIPELINE_INITIALISATION.out.samplesheet
    )
    //
    // SUBWORKFLOW: Run completion tasks
    //
    PIPELINE_COMPLETION (
        params.monochrome_logs,
    )

    publish:
    l0_stores   = PANOSETI_ANALYSIS.out.l0_stores
    l1_stores   = PANOSETI_ANALYSIS.out.l1_stores
    hk_stores   = PANOSETI_ANALYSIS.out.hk_stores
    l0_manifest = PANOSETI_ANALYSIS.out.l0_manifest
    l1_manifest = PANOSETI_ANALYSIS.out.l1_manifest
    l2_stores   = PANOSETI_ANALYSIS.out.l2_stores
    l2_manifest = PANOSETI_ANALYSIS.out.l2_manifest
}

// ── Level-major publishing (output {} block; no publishDir) ───────────────────
// The path closure returns the target DIRECTORY; Nextflow places each file inside it.
output {
    l0_stores   { path 'L0' }
    l1_stores   { path 'L1' }
    hk_stores   { path 'L0' }
    l0_manifest { path 'L0' }
    l1_manifest { path 'L1' }
    l2_stores   { path 'L2' }
    l2_manifest { path 'L2' }
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
