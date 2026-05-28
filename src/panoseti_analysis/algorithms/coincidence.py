"""Cross-module coincidence finder (SETI multiplicity, gamma stereo) — Layer A, pure.

A temporal join over per-(dp,module) sorted int64 ``unix_t_ns`` indices — NOT a physical
consolidation of stores. Greedy anchored-window sweep over the merged, time-sorted events:
a group is emitted when >= ``min_multiplicity`` distinct streams fall within ``window_ns``
of the group's earliest event. Relies on the L1 monotonic contract (inputs sorted ascending).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoincidenceStream:
    """One input stream: a label and its ascending-sorted int64 timestamps."""

    label: str
    unix_t_ns: np.ndarray


@dataclass(frozen=True)
class CoincidenceTable:
    """Columnar membership table: one row per (group, event)."""

    group_id: np.ndarray  # (M,) int
    stream_label: list[str]  # (M,)
    local_index: np.ndarray  # (M,) int — index of the event within its source stream
    unix_t_ns: np.ndarray  # (M,) int64

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.group_id).size) if self.group_id.size else 0


def _empty_table() -> CoincidenceTable:
    return CoincidenceTable(
        group_id=np.empty(0, dtype="int64"),
        stream_label=[],
        local_index=np.empty(0, dtype="int64"),
        unix_t_ns=np.empty(0, dtype="int64"),
    )


def find_coincidences(
    streams: list[CoincidenceStream],
    *,
    window_ns: int,
    min_multiplicity: int = 2,
) -> CoincidenceTable:
    """Return coincidence groups across ``streams`` within ``window_ns``."""
    times: list[np.ndarray] = []
    stream_idx: list[np.ndarray] = []
    local_idx: list[np.ndarray] = []
    labels = [s.label for s in streams]
    for si, s in enumerate(streams):
        t = np.asarray(s.unix_t_ns, dtype="int64")
        times.append(t)
        stream_idx.append(np.full(t.size, si, dtype="int64"))
        local_idx.append(np.arange(t.size, dtype="int64"))

    if not times or sum(t.size for t in times) == 0:
        return _empty_table()

    all_t = np.concatenate(times)
    all_s = np.concatenate(stream_idx)
    all_l = np.concatenate(local_idx)
    order = np.argsort(all_t, kind="stable")
    all_t, all_s, all_l = all_t[order], all_s[order], all_l[order]
    n = all_t.size

    g_id: list[int] = []
    g_label: list[str] = []
    g_local: list[int] = []
    g_t: list[int] = []
    gid = 0
    i = 0
    while i < n:
        hi = int(np.searchsorted(all_t, all_t[i] + window_ns, side="right"))
        if np.unique(all_s[i:hi]).size >= min_multiplicity:
            for j in range(i, hi):
                g_id.append(gid)
                g_label.append(labels[int(all_s[j])])
                g_local.append(int(all_l[j]))
                g_t.append(int(all_t[j]))
            gid += 1
            i = hi
        else:
            i += 1

    if not g_id:
        return _empty_table()
    return CoincidenceTable(
        group_id=np.array(g_id, dtype="int64"),
        stream_label=g_label,
        local_index=np.array(g_local, dtype="int64"),
        unix_t_ns=np.array(g_t, dtype="int64"),
    )
