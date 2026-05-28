"""Per-run, per-level manifest assembly + validation — Layer A, pure.

The adapter (``pa-manifest``) computes the I/O-bound lineage fields (reads store attrs,
checksums) and passes ``StoreLineage`` records in; this kernel only merges/validates and
returns a typed ``Manifest``. It never touches the filesystem.
"""

from __future__ import annotations

from panoseti_analysis.config import levels
from panoseti_analysis.config.models import Manifest, StoreLineage
from panoseti_analysis.config.versions import MANIFEST_SCHEMA_VERSION


def build_level_manifest(
    run_id: str,
    level: str,
    entries: list[StoreLineage],
    *,
    storage_version: str,
    created_utc: str,
    seed: Manifest | None = None,
) -> Manifest:
    """Assemble and validate a manifest for one run/level.

    ``seed`` is the pypff L0 manifest (parsed by the adapter); its schema version is
    recorded for provenance. Raises ``ValueError`` on an unknown level or duplicate
    store names.
    """
    levels.validate_level(level)

    seen: set[str] = set()
    for entry in entries:
        if entry.store in seen:
            raise ValueError(f"duplicate store name in manifest: {entry.store!r}")
        seen.add(entry.store)

    return Manifest(
        panoseti_analysis_storage_version=storage_version,
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        level=level,
        created_utc=created_utc,
        seed_manifest_version=seed.manifest_schema_version if seed is not None else None,
        stores=list(entries),
    )
