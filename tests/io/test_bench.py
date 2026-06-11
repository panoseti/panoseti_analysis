"""Tests for io/bench.py — stage_timer context manager and BenchResult."""

from __future__ import annotations

import time

from panoseti_analysis.io.bench import BenchResult, stage_timer, summarize


def test_stage_timer_fills_seconds_and_rss() -> None:
    """stage_timer fills seconds > 0 and peak_rss_mb >= 0 after the block."""
    with stage_timer("noop") as r:
        time.sleep(0.01)  # ensure measurable elapsed time

    assert r.seconds > 0
    assert r.peak_rss_mb >= 0.0


def test_mb_per_sec_uses_bytes_out() -> None:
    """mb_per_sec is computed from bytes_out when provided."""
    bytes_out = 100 * 1024 * 1024  # 100 MB
    with stage_timer("write", bytes_out=bytes_out) as r:
        time.sleep(0.1)

    expected = bytes_out / r.seconds / 1e6
    assert abs(r.mb_per_sec - expected) < 1.0  # within 1 MB/s tolerance


def test_mb_per_sec_falls_back_to_bytes_in() -> None:
    """mb_per_sec uses bytes_in when bytes_out is 0."""
    bytes_in = 50 * 1024 * 1024  # 50 MB
    with stage_timer("read", bytes_in=bytes_in) as r:
        time.sleep(0.05)

    expected = bytes_in / r.seconds / 1e6
    assert abs(r.mb_per_sec - expected) < 1.0


def test_mb_per_sec_zero_when_no_bytes() -> None:
    """mb_per_sec is 0.0 when neither bytes_in nor bytes_out are provided."""
    with stage_timer("cpu-only") as r:
        pass

    assert r.mb_per_sec == 0.0


def test_summarize_contains_label() -> None:
    """summarize returns a string that contains each result's label."""
    results = [
        BenchResult(
            label="calibrate",
            seconds=12.34,
            bytes_in=0,
            bytes_out=0,
            mb_per_sec=250.1,
            file_count=0,
            peak_rss_mb=512.3,
        ),
        BenchResult(
            label="ingest",
            seconds=3.0,
            bytes_in=1_000_000,
            bytes_out=0,
            mb_per_sec=1.0,
            file_count=10,
            peak_rss_mb=64.0,
        ),
    ]
    table = summarize(results)
    assert isinstance(table, str)
    assert "calibrate" in table
    assert "ingest" in table


def test_result_fields_preserved() -> None:
    """stage_timer preserves label, bytes_in, bytes_out, file_count."""
    with stage_timer("check", bytes_in=1024, bytes_out=2048, file_count=7) as r:
        pass

    assert r.label == "check"
    assert r.bytes_in == 1024
    assert r.bytes_out == 2048
    assert r.file_count == 7
