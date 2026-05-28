/*
 * CALIBRATE — route each L0 store to the appropriate calibration process.
 *
 * Takes a channel of tuples: tuple(product, path(l0_store), kind)
 *   kind: 'ph' → CALIBRATE_PH, 'img' → CALIBRATE_IMG
 *
 * Emits a mixed channel of: tuple(product, path(l1_store), kind)
 */

include { CALIBRATE_PH } from '../modules/calibrate_ph.nf'
include { CALIBRATE_IMG } from '../modules/calibrate_img.nf'

workflow CALIBRATE {
    take:
    stores // tuple(product, path, kind)

    main:
    ph_in = stores
        .filter { product, store, kind -> kind == 'ph' }
        .map { product, store, kind -> tuple(product, store) }

    img_in = stores
        .filter { product, store, kind -> kind == 'img' }
        .map { product, store, kind -> tuple(product, store) }

    ph_out = CALIBRATE_PH(ph_in)
    img_out = CALIBRATE_IMG(img_in)

    // Re-attach kind for downstream (SUMMARIZE / output block)
    ph_tagged = ph_out.l1.map { product, store -> tuple(product, store, 'ph') }
    img_tagged = img_out.l1.map { product, store -> tuple(product, store, 'img') }

    emit:
    l1 = ph_tagged.mix(img_tagged)
}
