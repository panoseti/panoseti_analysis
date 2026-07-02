//
// RECONSTRUCT (STUB) — establishes the composition seam; not implemented in Chunk 1.
// A later chunk fills this in (event/track reconstruction, coincidence, stereo).
//

workflow RECONSTRUCT {


    take:
    ch_l1   // channel: [ meta, l1_store ]

    emit:
    // TODO(chunk>=2): reconstruction stages here (Layer B adapters over Layer A kernels).
    ch_l1.map { meta, _store -> meta }.filter { false }
}
