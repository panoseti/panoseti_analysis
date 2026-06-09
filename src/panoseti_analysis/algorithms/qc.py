"""QC-alongside-products — Layer A pure QC runner.

Usage pattern::

    # In a calibrate adapter, after producing ``out``:
    report = run_qc(out, level="L1", kind="img")
    # Then in io/qc.py (Layer B):
    out = out.pano.stamp(extra_attrs={"qc": report.model_dump(mode="json")})
    write_qc_sidecar(report, sidecar_path)

Checks are decorator-registered at import time.  New checks: add a function in
this module (or any module that gets imported) decorated with ``@qc_check``.

Design rule: this module is Layer A — no I/O, no zarr, no adapters imports.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from panoseti_analysis.algorithms.quicklook import frame_rate_hz

# ── result models ──────────────────────────────────────────────────────────────


class QCCheckResult(BaseModel):
    """Outcome of one QC check."""

    model_config = ConfigDict(frozen=True)

    key: str  # short machine-readable name (e.g. "hot_pixel_frac")
    value: float  # measured value
    threshold: float  # threshold used for pass/fail
    passed: bool  # True when the check is satisfied


class QCReport(BaseModel):
    """Aggregate QC report for one Dataset."""

    model_config = ConfigDict(frozen=True)

    level: str
    kind: str
    metrics: dict[str, float]  # key → measured value (all checks)
    checks: list[QCCheckResult]  # individual outcomes
    isgood: bool  # True iff all checks passed


# ── registry ───────────────────────────────────────────────────────────────────

_QC_REGISTRY: dict[tuple[str, str], list[Callable[[xr.Dataset], QCCheckResult]]] = defaultdict(list)


def qc_check(level: str, kind: str, key: str) -> Callable[..., Any]:
    """Decorator: register a QC check function for ``(level, kind)``."""

    def decorator(
        fn: Callable[[xr.Dataset], QCCheckResult],
    ) -> Callable[[xr.Dataset], QCCheckResult]:
        fn._qc_key = key  # type: ignore[attr-defined]
        fn._qc_level = level  # type: ignore[attr-defined]
        fn._qc_kind = kind  # type: ignore[attr-defined]
        _QC_REGISTRY[(level, kind)].append(fn)
        return fn

    return decorator


def run_qc(ds: xr.Dataset, *, level: str, kind: str) -> QCReport:
    """Run all registered checks for ``(level, kind)`` and return the report."""
    check_fns = _QC_REGISTRY.get((level, kind), [])
    results: list[QCCheckResult] = []
    metrics: dict[str, float] = {}
    for fn in check_fns:
        result = fn(ds)
        results.append(result)
        metrics[result.key] = result.value
    isgood = all(r.passed for r in results)
    return QCReport(level=level, kind=kind, metrics=metrics, checks=results, isgood=isgood)


def registered_checks(level: str, kind: str) -> list[str]:
    """Return the keys of all registered checks for ``(level, kind)``."""
    return [fn._qc_key for fn in _QC_REGISTRY.get((level, kind), [])]  # type: ignore[attr-defined]


# ── built-in checks ────────────────────────────────────────────────────────────

_HOT_PIXEL_THRESHOLD = 0.10  # >10% hot pixels → fail
_DEAD_PIXEL_THRESHOLD = 0.30  # >30% dead pixels → fail
_MIN_FRAME_RATE_HZ = 0.5  # frame rate must be positive and ≥ 0.5 Hz
_MAX_FINITE_LOSS = 0.05  # >5% non-finite values → fail


@qc_check("L1", "img", "hot_pixel_frac")
def _check_hot_pixel_frac(ds: xr.Dataset) -> QCCheckResult:
    mask = np.asarray(ds["hot_pixel_mask"].values)
    frac = float(np.sum(mask > 0)) / max(mask.size, 1)
    return QCCheckResult(
        key="hot_pixel_frac",
        value=frac,
        threshold=_HOT_PIXEL_THRESHOLD,
        passed=frac <= _HOT_PIXEL_THRESHOLD,
    )


@qc_check("L1", "img", "dead_pixel_frac")
def _check_dead_pixel_frac(ds: xr.Dataset) -> QCCheckResult:
    mask = np.asarray(ds["dead_pixel_mask"].values)
    frac = float(np.sum(mask > 0)) / max(mask.size, 1)
    return QCCheckResult(
        key="dead_pixel_frac",
        value=frac,
        threshold=_DEAD_PIXEL_THRESHOLD,
        passed=frac <= _DEAD_PIXEL_THRESHOLD,
    )


@qc_check("L1", "img", "frame_rate_hz")
def _check_frame_rate(ds: xr.Dataset) -> QCCheckResult:
    rate = frame_rate_hz(ds)
    value = float("nan") if np.isnan(rate) else rate
    passed = not np.isnan(rate) and rate >= _MIN_FRAME_RATE_HZ
    return QCCheckResult(
        key="frame_rate_hz",
        value=value,
        threshold=_MIN_FRAME_RATE_HZ,
        passed=passed,
    )


@qc_check("L1", "img", "finite_frac")
def _check_finite_frac(ds: xr.Dataset) -> QCCheckResult:
    arr = np.asarray(ds["median_subtracted"].values)
    n_total = arr.size
    n_nonfinite = int(np.sum(~np.isfinite(arr)))
    frac_loss = n_nonfinite / max(n_total, 1)
    return QCCheckResult(
        key="finite_frac",
        value=1.0 - frac_loss,
        threshold=1.0 - _MAX_FINITE_LOSS,
        passed=frac_loss <= _MAX_FINITE_LOSS,
    )


@qc_check("L1", "ph", "hot_pixel_frac")
def _check_ph_hot_pixel_frac(ds: xr.Dataset) -> QCCheckResult:
    mask = np.asarray(ds["hot_pixel_mask"].values)
    frac = float(np.sum(mask > 0)) / max(mask.size, 1)
    return QCCheckResult(
        key="hot_pixel_frac",
        value=frac,
        threshold=_HOT_PIXEL_THRESHOLD,
        passed=frac <= _HOT_PIXEL_THRESHOLD,
    )


@qc_check("L1", "ph", "dead_pixel_frac")
def _check_ph_dead_pixel_frac(ds: xr.Dataset) -> QCCheckResult:
    mask = np.asarray(ds["dead_pixel_mask"].values)
    frac = float(np.sum(mask > 0)) / max(mask.size, 1)
    return QCCheckResult(
        key="dead_pixel_frac",
        value=frac,
        threshold=_DEAD_PIXEL_THRESHOLD,
        passed=frac <= _DEAD_PIXEL_THRESHOLD,
    )


@qc_check("L1", "ph", "frame_rate_hz")
def _check_ph_frame_rate(ds: xr.Dataset) -> QCCheckResult:
    rate = frame_rate_hz(ds)
    value = float("nan") if np.isnan(rate) else rate
    passed = not np.isnan(rate) and rate >= _MIN_FRAME_RATE_HZ
    return QCCheckResult(
        key="frame_rate_hz",
        value=value,
        threshold=_MIN_FRAME_RATE_HZ,
        passed=passed,
    )
