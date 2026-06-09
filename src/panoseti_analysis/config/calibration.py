"""CalibrationResolver protocol + CalibrationResolution result model.

This module is pure (no I/O). ``io/calibration_source.py`` provides the concrete
file-backed implementation.

Convention table: ``_CAL_RECIPE_KEYS`` maps each ``cal_type`` string to the key that
should appear under the ``calibration:`` section of a recipe YAML.  Example::

    # recipes/calib_v1.yml
    name: calib_v1
    calibration:
      img:
        frame_stride: 200
        block_size: 8
        adc_to_pe: 1.5
      ph:
        sigma_threshold: 5.0
        baseline_offset: 800
        frame_stride: 200
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class CalibrationResolution(BaseModel):
    """JSON-clean record of how calibration parameters were resolved.

    Stamped into ``ds.attrs["calibration"]["resolution"]`` and into
    ``ProcessingStep.params`` so every L1 store is traceable to its calibration source.
    """

    model_config = ConfigDict(frozen=True)

    cal_type: str  # "img" | "ph"
    source: str  # recipe path or identifier (display/audit only)
    source_hash: str  # "sha256:..." of the source bytes
    relevance: str  # "file-recipe" today; future: "nearest-in-time", "default"
    params: dict[str, Any]  # the resolved params (JSON-clean; model-default keys always present)


@runtime_checkable
class CalibrationResolver(Protocol):
    """Protocol for objects that resolve calibration parameters.

    ``context`` carries run metadata (run_id, data_product, module) that future
    implementations can use for nearest-in-time or run-specific selection.  The
    current file-backed impl ignores it.
    """

    def resolve(self, cal_type: str, context: dict[str, Any]) -> CalibrationResolution: ...


# Canonical mapping: science type name → key under ``calibration:`` in a recipe YAML.
_CAL_RECIPE_KEYS: dict[str, str] = {
    "img": "img",
    "ph": "ph",
}
