"""Tests for the three-mode Ray launcher (adapters/ray/launcher.py)."""
from __future__ import annotations

import pytest
import ray

from panoseti_analysis.adapters.ray.launcher import init_ray


@pytest.fixture(autouse=True)
def shutdown_ray():
    """Ensure Ray is shut down before and after each test."""
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_init_ray_standalone_initializes() -> None:
    """standalone mode should start a real local cluster."""
    init_ray("standalone", num_cpus=2)
    assert ray.is_initialized()


def test_init_ray_is_idempotent() -> None:
    """Calling init_ray twice should not raise."""
    init_ray("standalone")
    init_ray("standalone")  # second call — already initialized, must be a no-op
    assert ray.is_initialized()


def test_init_ray_invalid_mode_raises() -> None:
    """Passing an unknown mode string should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown launcher mode"):
        init_ray("bogus")  # type: ignore[arg-type]


def test_init_ray_already_initialized_skips() -> None:
    """If Ray is already initialized, init_ray should return without error."""
    ray.init()
    assert ray.is_initialized()
    init_ray("attach")  # should be a no-op — early return branch
    assert ray.is_initialized()
