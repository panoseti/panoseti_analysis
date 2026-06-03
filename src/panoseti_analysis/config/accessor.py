"""Registered xarray accessor ``pano`` — the single chokepoint for Dataset
validation and attr-stamping in the PANOSETI pipeline.

Registration happens as a side effect of importing this module (via
``config/__init__.py``), so ``ds.pano`` is available on any ``xr.Dataset``
whenever ``panoseti_analysis.config`` has been imported.  The accessor never
does I/O, never imports zarr/ray/io/adapters — it is Layer-A-safe.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from panoseti_analysis.config.levels import infer_kind, validate_level
from panoseti_analysis.config.schema import get_schema
from panoseti_analysis.config.versions import (
    CALIBRATION_KEY,
    DATA_LEVEL_KEY,
    PANOSETI_ANALYSIS_STORAGE_VERSION,
    STORAGE_VERSION_KEY,
    TIMESTAMP_QC_KEY,
)


@xr.register_dataset_accessor("pano")  # type: ignore[no-untyped-call]
class PanoAccessor:
    """Typed view of a PANOSETI ``xr.Dataset``: level/kind identity, validation, stamping."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    # ── identity ─────────────────────────────────────────────────────────────

    @property
    def level(self) -> str | None:
        """Registered data level from ``attrs['data_level']``, or ``None`` if unset."""
        return self._ds.attrs.get(DATA_LEVEL_KEY)

    @property
    def kind(self) -> str | None:
        """``'ph'`` or ``'img'`` inferred from ``attrs['data_product']``, or ``None``."""
        dp = self._ds.attrs.get("data_product")
        if dp is None:
            return None
        try:
            return infer_kind(str(dp))
        except ValueError:
            return None

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, *, level: str | None = None, kind: str | None = None) -> xr.Dataset:
        """Assert the Dataset satisfies the ``SchemaSpec`` for ``(level, kind)``.

        Uses ``level`` / ``kind`` from attrs if not passed explicitly.  Raises a
        ``ValueError`` that names exactly which variables or attrs are missing.
        Returns ``self._ds`` unchanged so the call can be chained.  No I/O.
        """
        lv = level or self.level
        kd = kind or self.kind
        if lv is None or kd is None:
            raise ValueError(
                f"Cannot validate: level={lv!r} and kind={kd!r} must both be determinable. "
                "Pass them explicitly or ensure attrs contain 'data_level' and 'data_product'."
            )
        spec = get_schema(lv, kd)
        missing_vars = [v for v in spec.required_vars if v not in self._ds]
        if missing_vars:
            raise ValueError(
                f"Dataset is missing required vars for (level={lv!r}, kind={kd!r}): "
                f"{missing_vars}.  Present data_vars: {list(self._ds.data_vars)}"
            )
        missing_attrs = [a for a in spec.required_attrs if a not in self._ds.attrs]
        if missing_attrs:
            raise ValueError(
                f"Dataset is missing required attrs for (level={lv!r}, kind={kd!r}): "
                f"{missing_attrs}.  Present attrs: {list(self._ds.attrs)}"
            )
        return self._ds

    # ── stamping ──────────────────────────────────────────────────────────────

    def stamp(
        self,
        *,
        data_level: str,
        calibration: dict[str, Any] | None = None,
        timestamp_qc: dict[str, Any] | None = None,
        carry_from: xr.Dataset | None = None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """Return a **new** Dataset with canonical attrs stamped.

        Builds the attrs dict in priority order (lowest → highest):
        1. ``carry_from.attrs`` — propagate source-level metadata (e.g. L0 → L1).
        2. ``self._ds.attrs`` — any attrs already on the output Dataset.
        3. Canonical stamp fields (``DATA_LEVEL_KEY``, ``STORAGE_VERSION_KEY``,
           ``CALIBRATION_KEY``, ``TIMESTAMP_QC_KEY``) — always win.
        4. ``extra_attrs`` — caller-supplied overrides (e.g. QC block).

        All values must be JSON-serializable (plain ``dict``/``str``/``int``/``float``,
        no pydantic objects or numpy scalars) so they survive zarr v3 round-trips.
        Use ``model.model_dump(mode="json")`` before passing pydantic models.
        """
        base: dict[str, Any] = {}
        if carry_from is not None:
            base.update(carry_from.attrs)
        base.update(self._ds.attrs)
        stamped: dict[str, Any] = {
            **base,
            DATA_LEVEL_KEY: validate_level(data_level),
            STORAGE_VERSION_KEY: PANOSETI_ANALYSIS_STORAGE_VERSION,
        }
        if calibration is not None:
            stamped[CALIBRATION_KEY] = calibration
        if timestamp_qc is not None:
            stamped[TIMESTAMP_QC_KEY] = timestamp_qc
        if extra_attrs:
            stamped.update(extra_attrs)
        return self._ds.assign_attrs(stamped)
