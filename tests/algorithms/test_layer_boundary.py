"""Layer A purity guard: algorithms/ must not import I/O or orchestration frameworks.

This is the executable form of the architectural boundary (also wired into CI). It fails
if any ``algorithms/*.py`` imports a forbidden module or calls ``.open_zarr``/``.to_zarr``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import panoseti_analysis

_ALGO_DIR = Path(panoseti_analysis.__file__).parent / "algorithms"

# Frameworks / I/O that must never appear in a pure kernel.
_FORBIDDEN_ROOTS = {"ray", "nextflow", "grpc", "slurm", "typer", "zarr", "pypff"}
_FORBIDDEN_PREFIXES = ("panoseti_analysis.io", "panoseti_analysis.adapters")


def _algo_files() -> list[Path]:
    return sorted(p for p in _ALGO_DIR.glob("*.py"))


def test_algorithms_dir_is_non_empty() -> None:
    assert _algo_files(), "expected kernel modules under algorithms/"


def test_no_forbidden_imports() -> None:
    offenders: list[str] = []
    for path in _algo_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_ROOTS:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_ROOTS or node.module.startswith(_FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, "Layer A boundary violation:\n" + "\n".join(offenders)


def test_no_zarr_io_calls() -> None:
    offenders: list[str] = []
    for path in _algo_files():
        src = path.read_text()
        for needle in (".open_zarr(", ".to_zarr("):
            if needle in src:
                offenders.append(f"{path.name}: contains {needle}")
    assert not offenders, "Layer A must not do Zarr I/O:\n" + "\n".join(offenders)
