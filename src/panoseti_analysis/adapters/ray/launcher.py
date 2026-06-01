"""Three-mode Ray launcher for panoseti_analysis.

Modes:
  attach     — RAL bare-metal: attach to user's pre-running persistent cluster.
               ray.init(address=address or $RAY_ADDRESS or "auto")
  slurm      — Expanse HPC: attach to a symmetric-run transient cluster inside the allocation.
               ray.init(address="auto")
  standalone — CI / laptop: create a process-local cluster (NOT local_mode, real multi-process).
               ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
"""
from __future__ import annotations

import os
from typing import Any, Literal

import ray


def init_ray(
    launcher: Literal["attach", "slurm", "standalone"],
    *,
    address: str | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
) -> None:
    """Initialise Ray using the specified launcher mode.

    - ``attach``/``slurm``: connect to an existing cluster.
    - ``standalone``: spin up a process-local cluster (safe for CI; real multiprocess).
    """
    if ray.is_initialized():
        return

    if launcher == "attach":
        resolved = address or os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=resolved, ignore_reinit_error=True)

    elif launcher == "slurm":
        ray.init(address="auto", ignore_reinit_error=True)

    elif launcher == "standalone":
        kwargs: dict[str, Any] = {}
        if num_cpus is not None:
            kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            kwargs["num_gpus"] = num_gpus
        ray.init(**kwargs, ignore_reinit_error=True)

    else:
        raise ValueError(f"Unknown launcher mode {launcher!r}; expected attach|slurm|standalone")
