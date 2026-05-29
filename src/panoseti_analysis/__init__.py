"""PANOSETI analysis package.

Three-layer architecture:
  - ``algorithms/`` — Layer A pure kernels (xarray/numpy/pydantic in and out; no I/O).
  - ``adapters/``   — Layer B thin CLIs invoked by Nextflow (the only place that does I/O).
  - ``io/`` + ``config/`` — filesystem boundary and shared registries/models.

Layer C (orchestration) lives in the repo's Nextflow ``*.nf`` files, which call only
the Layer B CLIs — never the kernels directly.
"""

__version__ = "0.1.0"
