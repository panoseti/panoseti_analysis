/*
 * PFF_TO_ZARR — Convert a .pffd observation directory to L0 Zarr v3 stores.
 *
 * Emits:
 *   l0_dir   : path  — the L0/ directory containing all .zarr stores
 *   manifest : path  — manifest.tsv with columns: product<TAB>store<TAB>kind
 */
process PFF_TO_ZARR {
    label 'io_heavy'

    input:
    path obs_dir

    output:
    path 'L0', emit: l0_dir
    path 'manifest.tsv', emit: manifest

    script:
    """
    pff2zarr \\
        "${obs_dir}" \\
        . \\
        --codec ${params.codec} \\
        --level ${params.level} \\
        --time-chunk ${params.time_chunk}
    """
}
