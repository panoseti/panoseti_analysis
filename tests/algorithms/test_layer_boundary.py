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
# Note: plain `ray` (core API) is allowed; only specific submodules are banned.
_FORBIDDEN_ROOTS = {
    "nextflow",
    "grpc",
    "slurm",
    "typer",
    "zarr",
    "pypff",
    # panoseti_grpc must not leak into Layer A (it's a transport adapter).
    "panoseti_grpc",
}
_FORBIDDEN_PREFIXES = ("panoseti_analysis.io", "panoseti_analysis.adapters")
# Banned ray submodules: training, tuning, and serving must not appear in kernels.
# plain `ray` (core API) is allowed.
_FORBIDDEN_RAY_SUBMODULES = {"ray.train", "ray.tune", "ray.serve"}

_CUDA_PATTERNS = (
    ".cuda(",
    "torch.cuda",
    '.to("cuda"',
    ".to('cuda'",
    'device="cuda"',
    "device='cuda'",
)


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
                    root = alias.name.split(".")[0]
                    if root in _FORBIDDEN_ROOTS:
                        offenders.append(f"{path.name}: import {alias.name}")
                    # ban ray.train / ray.tune submodules
                    if alias.name in _FORBIDDEN_RAY_SUBMODULES or any(
                        alias.name.startswith(s + ".") for s in _FORBIDDEN_RAY_SUBMODULES
                    ):
                        offenders.append(
                            f"{path.name}: import {alias.name} (ray.train/tune banned in Layer A)"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                full_mod = node.module
                if root in _FORBIDDEN_ROOTS or node.module.startswith(_FORBIDDEN_PREFIXES):
                    offenders.append(f"{path.name}: from {node.module} import ...")
                # ban from ray.train ... and from ray.tune ...
                if full_mod in _FORBIDDEN_RAY_SUBMODULES or any(
                    full_mod.startswith(s + ".") for s in _FORBIDDEN_RAY_SUBMODULES
                ):
                    offenders.append(
                        f"{path.name}: from {node.module} import ... (ray.train/tune banned in Layer A)"
                    )
    assert not offenders, "Layer A boundary violation:\n" + "\n".join(offenders)


def test_no_zarr_io_calls() -> None:
    offenders: list[str] = []
    for path in _algo_files():
        src = path.read_text()
        for needle in (".open_zarr(", ".to_zarr("):
            if needle in src:
                offenders.append(f"{path.name}: contains {needle}")
    assert not offenders, "Layer A must not do Zarr I/O:\n" + "\n".join(offenders)


def test_no_hardcoded_cuda_device() -> None:
    """algorithms/ must not hard-code CUDA device placement.

    The follow-the-model pattern ``device = next(model.parameters()).device``
    followed by ``.to(device)`` IS allowed — this check only bans explicit
    hard-coded ``cuda`` references.
    """
    offenders: list[str] = []
    for path in _algo_files():
        src = path.read_text()
        for pattern in _CUDA_PATTERNS:
            if pattern in src:
                offenders.append(f"{path.name}: contains {pattern!r}")
    assert not offenders, "Layer A must not hard-code CUDA device placement:\n" + "\n".join(offenders)


def test_follow_the_model_pattern_not_flagged(tmp_path: Path) -> None:
    """The follow-the-model pattern must not be flagged as a cuda violation."""
    code = (
        "import torch\n"
        "device = next(model.parameters()).device\n"
        "tensor = x.to(device)\n"
    )
    src = code
    for pattern in _CUDA_PATTERNS:
        assert pattern not in src, f"Pattern {pattern!r} incorrectly present in legitimate code"
