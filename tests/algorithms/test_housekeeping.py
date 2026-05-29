"""TDD for the housekeeping (HK) Dataset builder kernel."""

from __future__ import annotations

import numpy as np
import pytest

from panoseti_analysis.algorithms.housekeeping import build_hk_datasets


def test_builds_one_dataset_per_hashset() -> None:
    hk = {
        "quabo": {
            "unix_t_ns": np.array([1000, 2000, 3000], dtype="int64"),
            "DET_TEMP": np.array([10.0, 11.0, 12.0]),
        },
        "weather": {
            "timestamp": np.array([1000, 2000], dtype="int64"),
            "wind": np.array([3.0, 4.0]),
        },
    }
    out = build_hk_datasets(hk)
    assert set(out) == {"quabo", "weather"}

    q = out["quabo"]
    assert q.sizes["hk_time"] == 3
    # the time field is consumed into the hk_t_ns coordinate, not left as a data_var
    assert "hk_t_ns" in q.coords
    np.testing.assert_array_equal(q["hk_t_ns"].values, [1000, 2000, 3000])
    assert "DET_TEMP" in q.data_vars
    assert "unix_t_ns" not in q.data_vars
    assert q.attrs["hashset"] == "quabo"


def test_hashset_without_time_field_uses_index() -> None:
    hk = {"mount": {"az": np.array([1.0, 2.0, 3.0, 4.0])}}
    out = build_hk_datasets(hk)
    assert out["mount"].sizes["hk_time"] == 4
    np.testing.assert_array_equal(out["mount"]["hk_t_ns"].values, [0, 1, 2, 3])


def test_empty_hk_returns_empty_dict() -> None:
    assert build_hk_datasets({}) == {}


def test_ragged_fields_within_a_hashset_raise() -> None:
    hk = {"bad": {"a": np.array([1, 2, 3]), "b": np.array([1, 2])}}
    with pytest.raises(ValueError, match="length"):
        build_hk_datasets(hk)
