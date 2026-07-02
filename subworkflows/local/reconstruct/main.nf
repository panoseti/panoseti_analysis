//
// RECONSTRUCT (STUB) — establishes the composition seam; not implemented in Chunk 1.
// A later chunk fills this in (event/track reconstruction, coincidence, stereo).
//

include { DUMMY } from '../../../modules/local/dummy/main'

workflow RECONSTRUCT {

    take:
    ch_l1   // channel: [ meta, l1_store ]

    main:
    // TODO(chunk>=2): reconstruction stages here (Layer B adapters over Layer A kernels).
    DUMMY(ch_l1)

    emit:
    // TODO(chunk>=2): reconstruction stages here (Layer B adapters over Layer A kernels).
    ch_l1.map { meta, _store -> meta }.filter { false }
}
