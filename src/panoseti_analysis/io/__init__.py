"""Filesystem boundary: open/write Zarr stores, wrap pypff readers, checksum, pack.

This is the only layer (besides adapters) permitted to touch the filesystem or import
``zarr``/``pypff``. Kernels in ``algorithms/`` must never import from here.
"""
