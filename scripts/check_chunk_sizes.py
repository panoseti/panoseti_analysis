#!/usr/bin/env python
"""Lustre small-file guard: check emitted Zarr `images` chunk sizes.

Scans `*.zarr` stores under a directory and inspects the chunk blobs of the `images`
array (decision 6: gate the `images` array only — header arrays and HK are KB-scale by
nature and would false-positive). The single smallest (final, partial) chunk per array is
exempt. Fails if any remaining chunk is below --fail-below; warns below --warn-below.

Usage:
    python scripts/check_chunk_sizes.py <dir> [--array images] [--fail-below 1MiB] [--warn-below 4MiB]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MiB = 1024 * 1024


def _chunk_sizes(array_dir: Path) -> list[int]:
    # Zarr v3 stores chunks under "<array>/c/...". Exclude metadata (zarr.json).
    return [
        p.stat().st_size
        for p in array_dir.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--array", default="images")
    ap.add_argument("--fail-below", type=int, default=1 * _MiB)
    ap.add_argument("--warn-below", type=int, default=4 * _MiB)
    args = ap.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    checked = 0
    for store in sorted(args.root.rglob("*.zarr")):
        array_dir = store / args.array
        if not array_dir.is_dir():
            continue
        sizes = sorted(_chunk_sizes(array_dir))
        if not sizes:
            continue
        checked += 1
        # exempt the single smallest (final partial) chunk
        for size in sizes[1:]:
            if size < args.fail_below:
                failures.append(f"{store.name}/{args.array}: chunk {size}B < {args.fail_below}B")
            elif size < args.warn_below:
                warnings.append(f"{store.name}/{args.array}: chunk {size}B < {args.warn_below}B")

    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    print(f"checked {checked} store(s) for array '{args.array}'")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
