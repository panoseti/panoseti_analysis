"""TDD for the manifest/lineage builder kernel (pure assembly + validation)."""

from __future__ import annotations

import pytest

from panoseti_analysis.algorithms.manifest import build_level_manifest
from panoseti_analysis.config.models import Manifest, StoreLineage
from panoseti_analysis.config.versions import (
    MANIFEST_SCHEMA_VERSION,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
)


def _entry(dp: str, store: str, **kw: object) -> StoreLineage:
    return StoreLineage(dp=dp, module="1", level="L1", kind="ph", store=store, n_frames=10, **kw)


def test_builds_manifest_from_entries() -> None:
    entries = [
        _entry("ph256", "run.dp_ph256.module_1.zarr"),
        _entry("ph1024", "run.dp_ph1024.module_1.zarr"),
    ]
    man = build_level_manifest(
        "run",
        "L1",
        entries,
        storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        created_utc="2026-05-28T00:00:00Z",
    )
    assert isinstance(man, Manifest)
    assert man.run_id == "run"
    assert man.level == "L1"
    assert man.manifest_schema_version == MANIFEST_SCHEMA_VERSION
    assert man.panoseti_analysis_storage_version == PANOSETI_ANALYSIS_STORAGE_VERSION
    assert [s.store for s in man.stores] == [e.store for e in entries]


def test_duplicate_store_names_rejected() -> None:
    dup = [_entry("ph256", "same.zarr"), _entry("ph1024", "same.zarr")]
    with pytest.raises(ValueError, match="duplicate"):
        build_level_manifest(
            "run",
            "L1",
            dup,
            storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            created_utc="t",
        )


def test_unknown_level_rejected() -> None:
    with pytest.raises(ValueError, match="unknown data_level"):
        build_level_manifest(
            "run",
            "L42",
            [],
            storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
            created_utc="t",
        )


def test_seed_version_is_recorded() -> None:
    seed = Manifest(
        panoseti_analysis_storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        manifest_schema_version="0.9",
        run_id="run",
        level="L0",
        created_utc="t",
    )
    man = build_level_manifest(
        "run",
        "L0",
        [_entry("ph256", "a.zarr")],
        storage_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        created_utc="t",
        seed=seed,
    )
    assert man.seed_manifest_version == "0.9"
