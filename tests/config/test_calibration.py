"""Tests for CalibrationResolver protocol + FileCalibrationResolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoseti_analysis.config.calibration import CalibrationResolution, CalibrationResolver
from panoseti_analysis.config.models import ImgCalibParams, PhCalibParams
from panoseti_analysis.io.calibration_source import FileCalibrationResolver


@pytest.fixture
def calib_recipe(tmp_path: Path) -> Path:
    """Write a minimal calibration recipe YAML."""
    p = tmp_path / "calib_v1.yml"
    p.write_text(
        "name: calib_v1\ncalibration:\n  img:\n    frame_stride: 100\n    block_size: 4\n    adc_to_pe: 2.0\n  ph:\n    sigma_threshold: 3.0\n    baseline_offset: 600\n    frame_stride: 50\n"
    )
    return p


@pytest.fixture
def partial_calib_recipe(tmp_path: Path) -> Path:
    """Recipe with only the img section; ph should fall back to defaults."""
    p = tmp_path / "calib_partial.yml"
    p.write_text("name: calib_partial\ncalibration:\n  img:\n    adc_to_pe: 3.0\n")
    return p


def test_resolver_satisfies_protocol(calib_recipe: Path) -> None:
    resolver = FileCalibrationResolver(calib_recipe)
    assert isinstance(resolver, CalibrationResolver)


def test_img_params_loaded_from_recipe(calib_recipe: Path) -> None:
    resolver = FileCalibrationResolver(calib_recipe)
    res = resolver.resolve("img", {})
    assert isinstance(res, CalibrationResolution)
    assert res.cal_type == "img"
    assert res.relevance == "file-recipe"
    assert res.params["frame_stride"] == 100
    assert res.params["block_size"] == 4
    assert res.params["adc_to_pe"] == pytest.approx(2.0)


def test_ph_params_loaded_from_recipe(calib_recipe: Path) -> None:
    resolver = FileCalibrationResolver(calib_recipe)
    res = resolver.resolve("ph", {})
    assert res.params["sigma_threshold"] == pytest.approx(3.0)
    assert res.params["baseline_offset"] == 600
    assert res.params["frame_stride"] == 50


def test_stable_hash_across_calls(calib_recipe: Path) -> None:
    r1 = FileCalibrationResolver(calib_recipe)
    r2 = FileCalibrationResolver(calib_recipe)
    assert r1.recipe_hash == r2.recipe_hash
    assert r1.recipe_hash.startswith("sha256:")


def test_partial_recipe_falls_back_to_defaults(partial_calib_recipe: Path) -> None:
    resolver = FileCalibrationResolver(partial_calib_recipe)
    defaults_ph = PhCalibParams().model_dump()

    ph_res = resolver.resolve("ph", {})
    # ph section absent → full defaults
    assert ph_res.params == defaults_ph

    img_res = resolver.resolve("img", {})
    # Only adc_to_pe overridden; rest come from ImgCalibParams defaults
    assert img_res.params["adc_to_pe"] == pytest.approx(3.0)
    assert img_res.params["frame_stride"] == ImgCalibParams().frame_stride
    assert img_res.params["block_size"] == ImgCalibParams().block_size


def test_resolution_stamped_in_calibration_attrs(
    make_img,
    tmp_path: Path,
    calib_recipe: Path,  # type: ignore[no-untyped-def]
) -> None:
    """run_calibrate with recipe must stamp resolution into L1 attrs."""
    from panoseti_analysis.adapters.calibrate import run_calibrate
    from panoseti_analysis.io.stores import open_store, write_store

    l0 = tmp_path / "run.dp_img16.module_1.zarr"
    write_store(make_img(n=60, size=32), l0)
    l1 = tmp_path / "run.dp_img16.module_1.L1.zarr"

    run_calibrate(l0, l1, img_stride=5, recipe=calib_recipe)

    ds = open_store(l1)
    cal = ds.attrs["calibration"]
    assert "resolution" in cal, "resolution must be stamped into calibration attrs"
    res = cal["resolution"]
    assert res["cal_type"] == "img"
    assert res["relevance"] == "file-recipe"
    assert res["source_hash"].startswith("sha256:")
    assert res["params"]["adc_to_pe"] == pytest.approx(2.0)


def test_run_calibrate_without_recipe_unchanged(
    make_img,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """Without a recipe, calibration attrs must NOT contain 'resolution'."""
    from panoseti_analysis.adapters.calibrate import run_calibrate
    from panoseti_analysis.io.stores import open_store, write_store

    l0 = tmp_path / "run.dp_img16.module_1.zarr"
    write_store(make_img(n=60, size=32), l0)
    l1 = tmp_path / "run.dp_img16.module_1.L1.zarr"

    run_calibrate(l0, l1, img_stride=5)

    ds = open_store(l1)
    cal = ds.attrs["calibration"]
    assert "resolution" not in cal, "no recipe → no resolution key"
