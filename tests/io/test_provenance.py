"""Tests for io.provenance — read_history, append_step, capture_software, now_utc.

TDD red-phase: written before the implementation.
"""

from __future__ import annotations

import re

from panoseti_analysis.config.models import ProcessingStep
from panoseti_analysis.io.provenance import (
    append_step,
    capture_software,
    now_utc,
    read_history,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_step_dict(**overrides: object) -> dict:
    base: dict = {
        "step_name": "convert",
        "step_version": "1.0.0",
        "params": {"chunk_frames": 512},
        "timestamp_utc": "2026-05-29T00:00:00Z",
    }
    base.update(overrides)
    return base


def _make_step(**overrides: object) -> ProcessingStep:
    return ProcessingStep(**_make_step_dict(**overrides))  # type: ignore[arg-type]


# ── read_history ──────────────────────────────────────────────────────────────


class TestReadHistory:
    def test_returns_empty_list_for_missing_key(self) -> None:
        attrs: dict = {"data_product": "ph256"}
        result = read_history(attrs)
        assert result == []

    def test_returns_empty_list_for_empty_dict(self) -> None:
        result = read_history({})
        assert result == []

    def test_returns_empty_list_for_empty_list_value(self) -> None:
        result = read_history({"processing_history": []})
        assert result == []

    def test_deserializes_single_step(self) -> None:
        attrs = {"processing_history": [_make_step_dict()]}
        result = read_history(attrs)
        assert len(result) == 1
        assert isinstance(result[0], ProcessingStep)
        assert result[0].step_name == "convert"
        assert result[0].step_version == "1.0.0"

    def test_deserializes_multiple_steps(self) -> None:
        attrs = {
            "processing_history": [
                _make_step_dict(step_name="convert"),
                _make_step_dict(step_name="calibrate_ph", params={"sigma_threshold": 5.0}),
            ]
        }
        result = read_history(attrs)
        assert len(result) == 2
        assert result[0].step_name == "convert"
        assert result[1].step_name == "calibrate_ph"
        assert result[1].params == {"sigma_threshold": 5.0}

    def test_skips_malformed_entry_and_keeps_valid(self) -> None:
        """A single bad entry must not crash — log warning and skip it."""
        attrs = {
            "processing_history": [
                _make_step_dict(step_name="convert"),
                {"not_a_valid": "step_dict"},  # missing required fields
                _make_step_dict(step_name="calibrate_ph"),
            ]
        }
        result = read_history(attrs)
        # two valid entries survive; malformed one is dropped
        assert len(result) == 2
        assert result[0].step_name == "convert"
        assert result[1].step_name == "calibrate_ph"

    def test_all_malformed_returns_empty(self) -> None:
        attrs = {
            "processing_history": [
                {"garbage": True},
                {"also_garbage": 123},
            ]
        }
        result = read_history(attrs)
        assert result == []

    def test_non_list_value_returns_empty(self) -> None:
        """A non-list value under 'processing_history' should not raise."""
        result = read_history({"processing_history": "not_a_list"})
        assert result == []

    def test_none_value_returns_empty(self) -> None:
        result = read_history({"processing_history": None})
        assert result == []

    def test_returns_list_of_processing_steps(self) -> None:
        attrs = {"processing_history": [_make_step_dict()]}
        result = read_history(attrs)
        assert all(isinstance(s, ProcessingStep) for s in result)


# ── append_step ───────────────────────────────────────────────────────────────


class TestAppendStep:
    def test_appends_step_to_empty_list(self) -> None:
        step = _make_step()
        result = append_step([], step)
        assert len(result) == 1
        assert result[0] is step

    def test_appends_step_to_existing_list(self) -> None:
        step1 = _make_step(step_name="convert")
        step2 = _make_step(step_name="calibrate_ph")
        result = append_step([step1], step2)
        assert len(result) == 2
        assert result[0].step_name == "convert"
        assert result[1].step_name == "calibrate_ph"

    def test_does_not_mutate_input_list(self) -> None:
        step1 = _make_step(step_name="convert")
        step2 = _make_step(step_name="calibrate_ph")
        original = [step1]
        original_len_before = len(original)
        append_step(original, step2)
        # input list must not be modified
        assert len(original) == original_len_before

    def test_returns_new_list_object(self) -> None:
        step1 = _make_step(step_name="convert")
        step2 = _make_step(step_name="calibrate_ph")
        original = [step1]
        result = append_step(original, step2)
        assert result is not original

    def test_preserves_existing_items_by_identity(self) -> None:
        step1 = _make_step(step_name="convert")
        step2 = _make_step(step_name="calibrate_ph")
        result = append_step([step1], step2)
        assert result[0] is step1


# ── capture_software ──────────────────────────────────────────────────────────


class TestCaptureSoftware:
    def test_returns_dict_with_git_sha_key(self) -> None:
        result = capture_software()
        assert "git_sha" in result

    def test_returns_dict_with_package_version_key(self) -> None:
        result = capture_software()
        assert "package_version" in result

    def test_git_sha_is_string(self) -> None:
        result = capture_software()
        assert isinstance(result["git_sha"], str)
        assert len(result["git_sha"]) > 0

    def test_package_version_is_string(self) -> None:
        result = capture_software()
        assert isinstance(result["package_version"], str)
        assert len(result["package_version"]) > 0

    def test_no_container_key_when_not_provided(self) -> None:
        result = capture_software()
        assert "container" not in result

    def test_container_key_present_when_provided(self) -> None:
        result = capture_software(container="ghcr.io/panoseti/ingest:1.2.3")
        assert "container" in result
        assert result["container"] == "ghcr.io/panoseti/ingest:1.2.3"

    def test_container_none_omits_key(self) -> None:
        result = capture_software(container=None)
        assert "container" not in result

    def test_all_values_are_strings(self) -> None:
        result = capture_software(container="img:latest")
        for k, v in result.items():
            assert isinstance(v, str), f"key {k!r} has non-string value {v!r}"

    def test_git_sha_is_unknown_or_short_hex(self) -> None:
        """git_sha must be either 'unknown' or a short hex string."""
        result = capture_software()
        sha = result["git_sha"]
        assert sha == "unknown" or re.match(r"^[0-9a-f]+$", sha), f"unexpected git_sha: {sha!r}"


# ── now_utc ───────────────────────────────────────────────────────────────────


class TestNowUtc:
    _ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_returns_string(self) -> None:
        result = now_utc()
        assert isinstance(result, str)

    def test_matches_iso8601_format(self) -> None:
        result = now_utc()
        assert self._ISO8601_RE.match(result), (
            f"now_utc() returned {result!r}, expected ISO 8601 like '2026-05-29T14:23:00Z'"
        )

    def test_ends_with_z(self) -> None:
        result = now_utc()
        assert result.endswith("Z")

    def test_two_calls_are_close(self) -> None:
        """Two consecutive calls should return the same or adjacent second."""
        import time

        t1 = now_utc()
        time.sleep(0.01)
        t2 = now_utc()
        # Both should start with the same year-month-day at minimum
        assert t1[:10] == t2[:10], f"dates differ: {t1!r} vs {t2!r}"
