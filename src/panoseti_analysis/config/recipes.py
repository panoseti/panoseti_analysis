"""Recipe loader — named science parameter bundles for the ML and calibration steps."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml  # PyYAML


def load_recipe(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """Load a YAML recipe file.

    Returns:
        (params, name, recipe_hash) where:
        - params: full YAML content as a dict
        - name: the ``name:`` field from the YAML, or the filename stem if absent
        - recipe_hash: ``"sha256:" + sha256hex(canonical_utf8_yaml_bytes)``
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    recipe_hash = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    params = yaml.safe_load(raw_bytes.decode("utf-8"))
    if not isinstance(params, dict):
        raise ValueError(f"Recipe {path} must be a YAML mapping, got {type(params).__name__}")
    name: str = params.get("name") or path.stem
    return params, name, recipe_hash
