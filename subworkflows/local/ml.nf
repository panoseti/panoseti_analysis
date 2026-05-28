//
// ML (STUB) — establishes the composition seam; not implemented in Chunk 1.
// A later chunk fills this in (training/inference; GPU/Ray via process labels + profiles).
//

workflow ML {

    take:
    ch_features   // channel: [ meta, store ]

    main:
    // TODO(chunk>=2): ML training/inference stages here.

    emit:
    products = channel.empty()
}
