"""Guard: no large file may slip into the Ray working-dir package / cluster rsync.

Ray packages a ``working_dir`` honoring ``.gitignore`` **and** ``.rayignore``; the cluster
launcher (``cluster/ral_up.sh``) rsyncs source to every node. A forgotten experiment dump
that is excluded by neither would silently fan out to all nodes (or balloon a Serve package).

This test fails loudly, listing the offending paths + sizes, so the exclude lists stay
honest. Small miscellaneous files are fine — only files above ``THRESHOLD_MB`` are checked.
The 50 MB default matches the launcher's payload guard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THRESHOLD_MB = 50
THRESHOLD = THRESHOLD_MB * 1024 * 1024


def _git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


def _package_candidates() -> list[str]:
    """Repo-relative paths Ray would package before .rayignore: tracked + untracked-not-gitignored.

    Both sets already exclude everything matched by .gitignore (Ray honors .gitignore too),
    so anything here that is also large and not in .rayignore is a real leak.
    """
    paths: set[str] = set()
    for args in (["ls-files", "-z"], ["ls-files", "-z", "--others", "--exclude-standard"]):
        out = _git(*args).stdout
        paths.update(p for p in out.split("\0") if p)
    return sorted(paths)


def _rayignored(paths: list[str]) -> set[str]:
    """Subset of ``paths`` excluded by .rayignore (evaluated rules-only, ignoring the index)."""
    if not paths:
        return set()
    res = _git(
        "-c",
        f"core.excludesFile={REPO / '.rayignore'}",
        "check-ignore",
        "--no-index",
        "-z",
        "--stdin",
        stdin="\0".join(paths),
    )
    return {p for p in res.stdout.split("\0") if p}


def test_no_large_unignored_files() -> None:
    large = [
        p
        for p in _package_candidates()
        if (REPO / p).is_file() and (REPO / p).stat().st_size > THRESHOLD
    ]
    leaks = sorted(set(large) - _rayignored(large))
    if leaks:
        detail = "\n".join(f"  {(REPO / p).stat().st_size / 1048576:.0f} MB\t{p}" for p in leaks)
        raise AssertionError(
            f"Files > {THRESHOLD_MB} MB are neither .gitignore'd nor .rayignore'd and would "
            f"ship to every Ray node / Serve package:\n{detail}\n"
            "Add them to .gitignore (or .rayignore) — or delete the dump."
        )
