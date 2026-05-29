"""TDD for the cross-module coincidence-finder kernel."""

from __future__ import annotations

import numpy as np

from panoseti_analysis.algorithms.coincidence import (
    CoincidenceStream,
    find_coincidences,
)


def _stream(label: str, ts: list[int]) -> CoincidenceStream:
    return CoincidenceStream(label=label, unix_t_ns=np.array(sorted(ts), dtype="int64"))


def test_two_streams_form_groups_within_window() -> None:
    s0 = _stream("m0", [100, 200, 5000])
    s1 = _stream("m1", [105, 210, 9000])
    table = find_coincidences([s0, s1], window_ns=50, min_multiplicity=2)

    assert table.n_groups == 2
    assert table.group_id.size == 4  # two pairs

    # first group holds the 100/105 pair from both streams
    g0 = table.group_id == 0
    assert set(np.asarray(table.stream_label)[g0]) == {"m0", "m1"}
    assert set(table.unix_t_ns[g0].tolist()) == {100, 105}


def test_window_boundary_is_inclusive() -> None:
    s0 = _stream("a", [100])
    s1 = _stream("b", [150])
    table = find_coincidences([s0, s1], window_ns=50, min_multiplicity=2)
    assert table.n_groups == 1


def test_events_beyond_window_do_not_coincide() -> None:
    s0 = _stream("a", [100])
    s1 = _stream("b", [100_000])
    table = find_coincidences([s0, s1], window_ns=50, min_multiplicity=2)
    assert table.n_groups == 0
    assert table.group_id.size == 0


def test_min_multiplicity_three_needs_three_streams() -> None:
    s0 = _stream("a", [100])
    s1 = _stream("b", [110])
    table = find_coincidences([s0, s1], window_ns=50, min_multiplicity=3)
    assert table.n_groups == 0

    s2 = _stream("c", [120])
    table3 = find_coincidences([s0, s1, s2], window_ns=50, min_multiplicity=3)
    assert table3.n_groups == 1
    assert table3.group_id.size == 3
