"""Tests for the data-level registry and kind inference."""

from __future__ import annotations

import pytest

from panoseti_analysis.config import levels


def test_builtin_levels_registered() -> None:
    assert levels.is_registered("L0")
    assert levels.is_registered("L1")
    assert "L0" in levels.known_levels()
    assert levels.get_level("L1").determined is True


def test_validate_level_rejects_unknown() -> None:
    assert levels.validate_level("L0") == "L0"
    with pytest.raises(ValueError, match="unknown data_level"):
        levels.validate_level("L99")


def test_register_level_is_idempotent_but_rejects_conflict() -> None:
    levels.register_level("LTEST", "reserved", determined=False)
    levels.register_level("LTEST", "reserved", determined=False)  # identical re-register: OK
    assert levels.get_level("LTEST").determined is False
    with pytest.raises(ValueError, match="different metadata"):
        levels.register_level("LTEST", "something else", determined=True)


@pytest.mark.parametrize(
    ("data_product", "expected"),
    [("ph256", "ph"), ("ph1024", "ph"), ("img8", "img"), ("img16", "img")],
)
def test_infer_kind(data_product: str, expected: str) -> None:
    assert levels.infer_kind(data_product) == expected


def test_infer_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="cannot infer kind"):
        levels.infer_kind("weather")
