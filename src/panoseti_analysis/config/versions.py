"""Storage version constants and canonical Zarr attribute key names.

Two independent version namespaces (see ``docs/storage_spec.md`` §0):

- ``panoseti_pff_zarr_version`` — owned by pypff; governs the L0 array layout.
- ``panoseti_analysis_storage_version`` — owned by this package; governs the level
  structure, manifest schema, HK store, ``timestamp_qc`` and PACK conventions.

No backward-compat constraints yet, so bump aggressively: an early incompatible
change jumps cleanly to ``"2.0"`` rather than contorting ``1.x``.
"""

from __future__ import annotations

#: Version of the panoseti_analysis storage conventions (level structure, manifest, HK, QC).
#: Bumped to 2.0: ProcessingStep/processing_history added to StoreLineage (breaking schema change).
PANOSETI_ANALYSIS_STORAGE_VERSION = "2.0"

#: Version of the per-run/per-level manifest.json schema.
#: Bumped to 2.0: StoreLineage gains processing_history (new field with default=[]).
MANIFEST_SCHEMA_VERSION = "2.0"

#: Version of the ProcessingStep / TrainingProvenance provenance record format.
PROVENANCE_SCHEMA_VERSION = "1.0"

# ── canonical root-attribute key names ────────────────────────────────────────
STORAGE_VERSION_KEY = "panoseti_analysis_storage_version"
PFF_ZARR_VERSION_KEY = "panoseti_pff_zarr_version"
DATA_LEVEL_KEY = "data_level"
CALIBRATION_KEY = "calibration"
TIMESTAMP_QC_KEY = "timestamp_qc"
