"""FileCalibrationResolver — resolve calibration params from a recipe YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panoseti_analysis.config.calibration import (
    _CAL_RECIPE_KEYS,
    CalibrationResolution,
)
from panoseti_analysis.config.models import ImgCalibParams, PhCalibParams
from panoseti_analysis.config.recipes import load_recipe

# Defaults match the pydantic field defaults so L1 stores produced without a recipe
# remain bit-identical when a recipe that omits a key is later supplied.
_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "img": ImgCalibParams().model_dump(),
    "ph": PhCalibParams().model_dump(),
}


class FileCalibrationResolver:
    """Resolve calibration params from a YAML recipe file.

    Expected recipe structure::

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

    Missing keys fall back to the pydantic model defaults (bit-identical with prior
    output).  Unknown keys are silently ignored (the pydantic model constructor rejects
    them if passed, so the caller validates via ``ImgCalibParams(**params)``).
    """

    def __init__(self, recipe_path: str | Path) -> None:
        self._recipe_path = Path(recipe_path)
        self._raw, self._name, self._hash = load_recipe(recipe_path)

    @property
    def recipe_name(self) -> str:
        return self._name

    @property
    def recipe_hash(self) -> str:
        return self._hash

    def resolve(self, cal_type: str, context: dict[str, Any]) -> CalibrationResolution:
        """Return resolved params for ``cal_type``.

        Merges recipe values on top of model defaults; ``context`` is accepted for
        protocol compatibility but ignored by this implementation.
        """
        recipe_key = _CAL_RECIPE_KEYS.get(cal_type, cal_type)
        calib_section = self._raw.get("calibration", {})
        recipe_overrides: dict[str, Any] = calib_section.get(recipe_key, {})
        defaults = _DEFAULT_PARAMS.get(cal_type, {})
        merged = {**defaults, **recipe_overrides}
        return CalibrationResolution(
            cal_type=cal_type,
            source=str(self._recipe_path),
            source_hash=self._hash,
            relevance="file-recipe",
            params=merged,
        )
