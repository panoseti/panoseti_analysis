"""Content checksums for Zarr directory stores (lineage provenance)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def checksum_store(store: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return ``"sha256:<hex>"`` over a store's files in sorted relative-path order.

    Hashes each file's relative path then its bytes, so the digest is stable across
    machines and reflects both layout and content. Reads the whole store (I/O-bound) —
    call only from adapters, never kernels.
    """
    root = Path(store)
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as fh:
            while chunk := fh.read(chunk_bytes):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compute_sha256(path: str | Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a single file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()
