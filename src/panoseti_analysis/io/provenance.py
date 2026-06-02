"""Provenance I/O helpers — read/write processing_history from Zarr root attrs.

Layer B (io boundary): may import subprocess, importlib, etc.
Kernels in algorithms/ must never import from here.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from datetime import UTC, datetime
from typing import Any

from panoseti_analysis.config.models import ProcessingStep
from panoseti_analysis.paths import REPO_ROOT as _REPO_ROOT

logger = logging.getLogger(__name__)


def read_history(attrs: dict[str, Any]) -> list[ProcessingStep]:
    """Read processing_history from Zarr root attrs as a list of ProcessingStep.

    Returns [] if the key is absent, the value is not a list, or the value is None.
    Malformed individual entries are logged as warnings and skipped; the function
    never raises — an unreadable history is better than a crashed pipeline.
    """
    raw = attrs.get("processing_history")
    if not isinstance(raw, list):
        return []

    steps: list[ProcessingStep] = []
    for i, item in enumerate(raw):
        try:
            steps.append(ProcessingStep.model_validate(item))
        except Exception as exc:
            logger.warning("processing_history[%d] is malformed and will be skipped: %s", i, exc)
    return steps


def append_step(history: list[ProcessingStep], step: ProcessingStep) -> list[ProcessingStep]:
    """Return a new list with *step* appended; does not mutate the input list."""
    return [*list(history), step]


def capture_software(*, container: str | None = None) -> dict[str, str]:
    """Capture current software environment for ProcessingStep.software.

    Returns a dict with:
    - ``git_sha``: short SHA from ``git rev-parse --short HEAD`` (or ``"unknown"``
      if the working directory is not a git repo or git is unavailable).
    - ``package_version``: the installed ``panoseti-analysis`` package version.
    - ``container``: the Apptainer/Docker image path (only present when *container*
      is a non-None string, e.g. passed from the Nextflow task environment).
    """
    # git SHA — tolerate any failure gracefully
    git_sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except Exception:
        pass

    # Package version from installed metadata
    try:
        package_version = importlib.metadata.version("panoseti-analysis")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"

    info: dict[str, str] = {
        "git_sha": git_sha,
        "package_version": package_version,
    }
    if container is not None:
        info["container"] = container

    return info


def now_utc() -> str:
    """Return current UTC time as an ISO 8601 string (e.g. ``'2026-05-29T14:23:00Z'``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
