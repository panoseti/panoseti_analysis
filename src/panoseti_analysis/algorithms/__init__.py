"""Layer A — pure science kernels.

Functions here take and return ``xarray.Dataset`` / ``numpy.ndarray`` / Pydantic
models and perform NO I/O. They must not import any orchestration or I/O framework
(ray, nextflow, grpc, slurm, typer, zarr, pypff) nor call ``.open_zarr``/``.to_zarr``;
a CI lint enforces this boundary.
"""
