"""Dataset schema specifications for each (data_level, kind) pair.

Each ``SchemaSpec`` records which variables a Dataset at a given (level, kind) must
contain.  The registry is the single source of truth for "what must a Level-1 img store
have" — replacing scattered ``if var not in ds.data_vars: raise ValueError(...)`` guards
across kernels and adapters.

Extend by calling ``register_schema`` at import time; it is idempotent for identical
re-registrations and raises on a conflicting re-registration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SchemaSpec:
    """Contract for a Dataset at a specific (data_level, kind)."""

    level: str
    kind: str
    required_vars: tuple[str, ...]  # must be accessible via ds[key] (data_vars or coords)
    required_attrs: tuple[str, ...] = field(default=())


_SCHEMAS: dict[tuple[str, str], SchemaSpec] = {}


def register_schema(spec: SchemaSpec) -> None:
    """Register a schema spec.  Idempotent for identical re-registrations."""
    key = (spec.level, spec.kind)
    existing = _SCHEMAS.get(key)
    if existing is not None and existing != spec:
        raise ValueError(
            f"SchemaSpec for (level={spec.level!r}, kind={spec.kind!r}) already registered "
            f"with different metadata.\nExisting: {existing}\nNew: {spec}"
        )
    _SCHEMAS[key] = spec


def get_schema(level: str, kind: str) -> SchemaSpec:
    """Return the spec for (level, kind), raising ``KeyError`` if unregistered."""
    key = (level, kind)
    if key not in _SCHEMAS:
        raise KeyError(
            f"No schema registered for (level={level!r}, kind={kind!r}). Known: {sorted(_SCHEMAS)}"
        )
    return _SCHEMAS[key]


def known_schemas() -> list[tuple[str, str]]:
    return sorted(_SCHEMAS)


# ── built-in schemas ──────────────────────────────────────────────────────────
register_schema(SchemaSpec("L0", "img", required_vars=("images", "unix_t_ns")))
register_schema(SchemaSpec("L0", "ph", required_vars=("images", "unix_t_ns")))
register_schema(
    SchemaSpec(
        "L1",
        "img",
        required_vars=("median_subtracted", "hot_pixel_mask", "dead_pixel_mask", "unix_t_ns"),
    )
)
register_schema(
    SchemaSpec(
        "L1",
        "ph",
        required_vars=("pedestal_subtracted", "hot_pixel_mask", "dead_pixel_mask", "unix_t_ns"),
    )
)
register_schema(SchemaSpec("L2", "img", required_vars=("cloud_score", "cloud_label", "unix_t_ns")))
