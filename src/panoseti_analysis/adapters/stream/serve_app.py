"""
CloudInferDeployment — Ray Serve deployment for cloud-detection inference.

This is a PERSISTENT Ray Serve deployment that:
  1. Loads the classifier bundle ONCE at startup (model + weights + provenance).
  2. Accepts xr.Dataset windows from FrameAccumulator actors via `infer()`.
  3. Runs predict_cloud_score (Layer A kernel) and returns the L2 Dataset.

GPU placement
-------------
The deployment is pinned to the gaming node (digilab-transmit, `accelerator_type:GAMING`)
by default via ray_actor_options, keeping the A6000s (`accelerator_type:A6000`) free for training.
Override with the node_ip or num_gpus arguments if needed.

Usage
-----
Called by the CLI (pa-stream-cloud) after ray.init(address="auto"):

    from panoseti_analysis.adapters.stream.serve_app import deploy_cloud_infer

    handle = deploy_cloud_infer(
        model_path="assets/models/cloud_detector_v1.pt",
        num_replicas=1,
    )
    # handle is a DeploymentHandle; accumulator actors call:
    #   await handle.infer.remote(ds_l1, params)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import xarray as xr

from panoseti_analysis.config.models import CloudInferParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


try:
    from ray import serve as _serve
except ModuleNotFoundError:  # pragma: no cover
    _serve = None  # type: ignore[assignment]


def _serve_deployment(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper for @serve.deployment that fails informatively without Ray."""
    if _serve is None:
        raise RuntimeError("Ray Serve is not installed; cannot create deployment")
    return _serve.deployment(*args, **kwargs)


@_serve_deployment(
    name="CloudInferDeployment",
    # GPU placement: pin to the gaming node (has accelerator_type:GAMING custom resource).
    # Serve controller runs on the head (receiver); replicas land on the gaming node.
    #
    # NOTE: panoseti_analysis + panoseti_grpc source is shipped to workers via
    # runtime_env at ray.init() level (see adapters/stream/cli.py and
    # adapters/ray/launcher.py).  DO NOT set runtime_env here — Ray 2.55+
    # requires working_dir at job level, not per-actor.
    ray_actor_options={
        "num_gpus": 1,
        "resources": {"accelerator_type:GAMING": 0.001},
    },
    max_ongoing_requests=8,
    autoscaling_config=None,  # fixed 1 replica by default
)
class CloudInferDeployment:
    """
    Ray Serve deployment for cloud-detection inference.

    Loads the classifier bundle once on startup; infers per window.
    The model is moved to the best available device (GPU if present).
    """

    def __init__(self, model_path: str) -> None:
        import os

        from panoseti_analysis.io.models import load_classifier

        p = Path(model_path)
        if not p.exists():
            # On remote Ray workers, CWD is set to the shipped working_dir root.
            # The model was shipped as part of working_dir; resolve relative to CWD.
            p = Path(os.getcwd()) / model_path
            if not p.exists():
                raise FileNotFoundError(
                    f"Model not found at '{model_path}' or relative to CWD '{os.getcwd()}'"
                )

        self._model, self._bundle = load_classifier(p)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info(
            "CloudInferDeployment ready: model=%s v=%s device=%s",
            self._bundle.model_name,
            self._bundle.model_version,
            self._device,
        )

    async def infer(
        self,
        ds_l1: xr.Dataset,
        params: CloudInferParams,
    ) -> xr.Dataset:
        """
        Run cloud-detection inference on one 60s window.

        Parameters
        ----------
        ds_l1 : xr.Dataset
            Must contain 'median_subtracted' (T, H, W) f32 and 'unix_t_ns' (T,) i64.
        params : CloudInferParams
            cadence_s and threshold.

        Returns
        -------
        xr.Dataset
            L2 dataset: cloud_score, cloud_label, feature_raw_fft, feature_deriv_fft,
            unix_t_ns over dim T_l2.
        """
        from panoseti_analysis.algorithms.cloud_detector import predict_cloud_score

        result = predict_cloud_score(ds_l1, self._model, params)
        return result

    def bundle_info(self) -> dict[str, Any]:
        """Return provenance info for the loaded model bundle."""
        return {
            "model_name": self._bundle.model_name,
            "model_version": self._bundle.model_version,
            "checksum": self._bundle.checksum,
            "input_spec": self._bundle.input_spec,
            "device": str(self._device),
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def deploy_cloud_infer(
    model_path: str | Path,
    num_replicas: int = 1,
    gpu_node_ip: str | None = None,
) -> Any:  # ray.serve.handle.DeploymentHandle
    """
    Deploy (or update) the CloudInferDeployment and return a handle.

    Parameters
    ----------
    model_path : str | Path
        Path to the .pt model file (the .json sidecar must sit alongside it).
    num_replicas : int
        Number of inference replicas.  1 is sufficient for 60s cadence.
    gpu_node_ip : str | None
        If set, pin replicas to this specific node IP instead of using the
        accelerator_type:GAMING resource constraint.  Useful for testing on the
        head node.

    Returns
    -------
    DeploymentHandle
        Awaitable handle for calling handle.infer.remote(ds, params).
    """
    ray_actor_opts: dict[str, Any] = {"num_gpus": 1}
    if gpu_node_ip is not None:
        ray_actor_opts["resources"] = {f"node:{gpu_node_ip}": 0.001}
    else:
        ray_actor_opts["resources"] = {"accelerator_type:GAMING": 0.001}

    deployment = CloudInferDeployment.options(  # type: ignore[attr-defined]
        num_replicas=num_replicas,
        ray_actor_options=ray_actor_opts,
    )
    if _serve is None:
        raise RuntimeError("Ray Serve is not installed; cannot deploy")
    app = deployment.bind(model_path=str(model_path))
    _serve.run(app, name="cloud_infer", route_prefix=None, blocking=False)
    handle = _serve.get_deployment_handle("CloudInferDeployment", app_name="cloud_infer")
    logger.info("CloudInferDeployment deployed: replicas=%d model=%s", num_replicas, model_path)
    return handle
