"""Tests for config.recipes.load_recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from panoseti_analysis.config.recipes import load_recipe

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_load_recipe_returns_params_name_hash(tmp_path: Path) -> None:
    recipe_file = tmp_path / "test_r.yml"
    recipe_file.write_text("name: test_r\nvalue: 42\n")
    params, name, recipe_hash = load_recipe(recipe_file)
    assert params == {"name": "test_r", "value": 42}
    assert name == "test_r"
    assert recipe_hash.startswith("sha256:")


def test_recipe_hash_is_content_addressed(tmp_path: Path) -> None:
    content = "name: duplicate\nvalue: 1\n"
    file_a = tmp_path / "a.yml"
    file_b = tmp_path / "b.yml"
    file_a.write_text(content)
    file_b.write_text(content)
    _, _, hash_a = load_recipe(file_a)
    _, _, hash_b = load_recipe(file_b)
    assert hash_a == hash_b


def test_recipe_hash_changes_on_content_change(tmp_path: Path) -> None:
    file_a = tmp_path / "a.yml"
    file_b = tmp_path / "b.yml"
    file_a.write_text("name: recipe_a\nvalue: 1\n")
    file_b.write_text("name: recipe_b\nvalue: 2\n")
    _, _, hash_a = load_recipe(file_a)
    _, _, hash_b = load_recipe(file_b)
    assert hash_a != hash_b


def test_load_recipe_uses_stem_when_no_name(tmp_path: Path) -> None:
    recipe_file = tmp_path / "my_recipe.yml"
    recipe_file.write_text("value: 99\n")
    params, name, recipe_hash = load_recipe(recipe_file)
    assert name == "my_recipe"
    assert "name" not in params
    assert recipe_hash.startswith("sha256:")


def test_load_recipe_rejects_non_mapping(tmp_path: Path) -> None:
    recipe_file = tmp_path / "bad.yml"
    recipe_file.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_recipe(recipe_file)


def test_all_recipe_files_load() -> None:
    """Every file in recipes/ must load without error."""
    recipes_dir = Path(__file__).parent.parent.parent / "recipes"
    recipe_files = sorted(recipes_dir.glob("*.yml")) + sorted(recipes_dir.glob("*.yaml"))
    assert recipe_files, f"No recipe files found in {recipes_dir}"
    for recipe_path in recipe_files:
        params, name, recipe_hash = load_recipe(recipe_path)
        assert isinstance(params, dict), f"{recipe_path}: expected dict, got {type(params)}"
        assert isinstance(name, str) and name, f"{recipe_path}: name must be a non-empty string"
        assert recipe_hash.startswith("sha256:"), f"{recipe_path}: invalid hash prefix"
