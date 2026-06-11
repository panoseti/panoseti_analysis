"""Lightweight benchmarking utilities for pipeline stages (Layer B).

Provides :class:`BenchResult` and :func:`stage_timer` for timing individual
pipeline stages with RSS memory tracking.

Related tool: ``pypff.profiling.Profiler`` — read-side latency measurements
used inside the PFF→Zarr converter.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class BenchResult:
    """Timing and throughput result for a single pipeline stage."""

    label: str
    seconds: float
    bytes_in: int  # 0 if not measured
    bytes_out: int  # 0 if not measured
    mb_per_sec: float  # bytes_out / seconds / 1e6, or bytes_in if bytes_out==0
    file_count: int  # 0 if not applicable
    peak_rss_mb: float  # peak RSS delta in MB during this stage


@contextmanager
def stage_timer(
    label: str,
    *,
    bytes_in: int = 0,
    bytes_out: int = 0,
    file_count: int = 0,
) -> Iterator[BenchResult]:
    """Context manager that records elapsed time and RSS delta.

    Yields a :class:`BenchResult` pre-populated with the provided metadata;
    ``seconds`` and ``peak_rss_mb`` are filled in upon exit so the caller has
    final values immediately after the ``with`` block.

    Example::

        with stage_timer("calibrate", bytes_in=store_bytes) as r:
            run_calibrate(...)
        print(r.seconds, r.mb_per_sec)
    """
    result = BenchResult(
        label=label,
        seconds=0.0,
        bytes_in=bytes_in,
        bytes_out=bytes_out,
        mb_per_sec=0.0,
        file_count=file_count,
        peak_rss_mb=0.0,
    )
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t_start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - t_start
        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss on Linux is in KB; on macOS it's in bytes — we target Linux.
        delta_mb = max(rss_after_kb - rss_before_kb, 0) / 1024.0

        result.seconds = elapsed
        result.peak_rss_mb = delta_mb

        if bytes_out > 0:
            result.mb_per_sec = bytes_out / elapsed / 1e6 if elapsed > 0 else 0.0
        elif bytes_in > 0:
            result.mb_per_sec = bytes_in / elapsed / 1e6 if elapsed > 0 else 0.0
        else:
            result.mb_per_sec = 0.0


def summarize(results: list[BenchResult]) -> str:
    """Format a list of :class:`BenchResult` as a human-readable plain-text table.

    Example output::

        Stage          | Time (s) | MB/s   | Files | Peak RSS (MB)
        calibrate      |    12.34 |  250.1 |     0 |         512.3
    """
    header = f"{'Stage':<20} | {'Time (s)':>8} | {'MB/s':>8} | {'Files':>5} | {'Peak RSS (MB)':>13}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.label:<20} | {r.seconds:>8.2f} | {r.mb_per_sec:>8.1f}"
            f" | {r.file_count:>5} | {r.peak_rss_mb:>13.1f}"
        )
    return "\n".join(rows)
