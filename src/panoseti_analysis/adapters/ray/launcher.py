"""Three-mode Ray launcher for panoseti_analysis.

Modes:
  attach     — RAL bare-metal: attach to user's pre-running persistent cluster.
               ray.init(address=address or $RAY_ADDRESS or "auto")
  slurm      — Expanse HPC: attach to a symmetric-run transient cluster inside the allocation.
               ray.init(address="auto")
  standalone — CI / laptop: create a process-local cluster (NOT local_mode, real multi-process).
               ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

runtime_env note (streaming pipeline)
--------------------------------------
When using Ray Serve in attach mode, source code must be shipped to remote workers
via ``runtime_env={"working_dir": ..., "env_vars": {"PYTHONPATH": "src:grpc/src"}}``
at ``ray.init()`` time — NOT per-actor.  Pass ``runtime_env`` to ``init_ray`` when
launching the streaming pipeline (``pa-stream-cloud``).  Training/batch adapters do
NOT need runtime_env (they run only on the head node or via Nextflow's container).
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
    runtime_env: dict[str, Any] | None = None,
) -> None:
    """Initialise Ray using the specified launcher mode.

    - ``attach``/``slurm``: connect to an existing cluster.
    - ``standalone``: spin up a process-local cluster (safe for CI; real multiprocess).

    Parameters
    ----------
    runtime_env : dict | None
        Optional Ray runtime_env dict passed to ``ray.init()``.  For the streaming
        pipeline, pass::

            runtime_env={
                "working_dir": "/path/to/panoseti_analysis",
                "env_vars": {"PYTHONPATH": "src:grpc/src"},
            }

        to ship source code to remote workers.  Leave None for training/batch adapters
        (they don't need code on remote workers beyond what the container provides).
    """
    if ray.is_initialized():
        return

    if launcher == "attach":
        resolved = address or os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=resolved, ignore_reinit_error=True, runtime_env=runtime_env)

    elif launcher == "slurm":
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)

    elif launcher == "standalone":
        kwargs: dict[str, Any] = {}
        if num_cpus is not None:
            kwargs["num_cpus"] = num_cpus
        if num_gpus is not None:
            kwargs["num_gpus"] = num_gpus
        if runtime_env is not None:
            kwargs["runtime_env"] = runtime_env
        ray.init(**kwargs, ignore_reinit_error=True)

    else:
        raise ValueError(f"Unknown launcher mode {launcher!r}; expected attach|slurm|standalone")
