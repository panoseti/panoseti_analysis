"""Extensible data-level registry and data-product kind inference.

``data_level`` is a *tagged string* validated against this in-code registry — not a
hardcoded enum — so new levels can be registered without a schema migration. Only L0
(raw, format-calibrated) and L1 (science-calibrated) are determined for Chunk 1; L2+
are reserved and intentionally undefined (forward design biases toward gammapy DL3
compatibility without committing to ctapipe level names).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Movie-mode ("img") vs pulse-height ("ph") product family.
Kind = Literal["ph", "img"]


@dataclass(frozen=True)
class LevelSpec:
    """Metadata for one registered data level."""

    name: str
    description: str
    determined: bool = True  # False for reserved/forward-looking levels (L2+).


_REGISTRY: dict[str, LevelSpec] = {}


def register_level(name: str, description: str, *, determined: bool = True) -> LevelSpec:
    """Register a data level. Idempotent for an identical re-registration."""
    existing = _REGISTRY.get(name)
    spec = LevelSpec(name=name, description=description, determined=determined)
    if existing is not None and existing != spec:
        raise ValueError(f"data_level {name!r} already registered with different metadata")
    _REGISTRY[name] = spec
    return spec


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def validate_level(name: str) -> str:
    """Return ``name`` if it is a registered level, else raise ``ValueError``."""
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown data_level {name!r}; registered: {sorted(_REGISTRY)}. "
            "Call register_level(...) to add a new one."
        )
    return name


def known_levels() -> list[str]:
    return sorted(_REGISTRY)


def get_level(name: str) -> LevelSpec:
    return _REGISTRY[validate_level(name)]


def infer_kind(data_product: str) -> Kind:
    """Map a pypff ``data_product`` attr (e.g. ``"ph256"``, ``"img16"``) to its kind.

    Never sniff the kind from a filename — read ``ds.attrs["data_product"]`` (set by
    ``pypff.zarr.convert_run``) and pass it here.
    """
    if data_product.startswith("ph"):
        return "ph"
    if data_product.startswith("img"):
        return "img"
    raise ValueError(f"cannot infer kind from data_product={data_product!r}")


# ── built-in levels (Chunk 1) ─────────────────────────────────────────────────
register_level("L0", "Raw, format-calibrated mirror of PFF in native frame order.")
register_level(
    "L1",
    "Science-calibrated (pedestal/median subtracted); unix_t_ns guaranteed monotonic.",
)
