"""QC I/O helpers — write sidecar JSON + stamp attrs (Layer B)."""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

import panoseti_analysis.config  # noqa: F401 — registers ds.pano accessor as side-effect
from panoseti_analysis.algorithms.qc import QCReport


def write_qc_sidecar(report: QCReport, path: str | Path) -> None:
    """Write the QC report as a JSON sidecar file."""
    Path(path).write_text(json.dumps(report.model_dump(mode="json"), indent=2))


def stamp_qc(ds: xr.Dataset, report: QCReport) -> xr.Dataset:
    """Return a new Dataset with the QC report stamped into ``ds.attrs["qc"]``."""
    return ds.pano.stamp(
        data_level=str(ds.attrs.get("data_level", report.level)),
        extra_attrs={"qc": report.model_dump(mode="json")},
    )
