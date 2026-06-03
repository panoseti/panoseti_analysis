"""
pa-stream-cloud — real-time cloud detection from the live DaqData stream.

Attaches to the externally-owned Ray cluster (attach mode), deploys the Ray
Serve CloudInferDeployment, starts per-(module,dp) FrameAccumulator actors,
and runs the async DaqData StreamConsumer until interrupted.

Requires:
  - A running panoseti_grpc server with DaqData (and optionally MLInference) enabled.
  - The Ray cluster running and reachable via RAY_ADDRESS (or --ray-address).
  - The model bundle: <model_path>.pt + <model_path>.json.

Example
-------
    # Start the grpc server with ml_inference enabled:
    pseti-grpc server --config /path/to/server.toml  # ml_inference = true in server.toml

    # In a second shell, run the streaming pipeline:
    pa-stream-cloud \\
        --model-path assets/models/cloud_detector_v1.pt \\
        --recipe recipes/ml/stream_cloud_v1.yml \\
        --grpc-host localhost \\
        --archive-dir /mnt/beegfs/streams/
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from panoseti_analysis.config.recipes import load_recipe
from panoseti_analysis.paths import REPO_ROOT

app = typer.Typer(name="pa-stream-cloud", no_args_is_help=True)
logger = logging.getLogger(__name__)


@app.command()
def main(
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            help="Path to the .pt cloud detector model (the .json sidecar must exist alongside it).",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    recipe: Annotated[
        Path | None,
        typer.Option(
            "--recipe",
            help="Path to a stream_cloud_*.yml recipe (feature_cadence_s, threshold, grace_s).",
        ),
    ] = None,
    grpc_host: Annotated[
        str,
        typer.Option("--grpc-host", help="panoseti_grpc server hostname/IP."),
    ] = "localhost",
    grpc_port: Annotated[
        int,
        typer.Option("--grpc-port", help="panoseti_grpc server port."),
    ] = 50051,
    module_ids: Annotated[
        str | None,
        typer.Option(
            "--module-ids",
            help="Comma-separated module IDs to subscribe to (e.g. '1,2,3'). Empty = all.",
        ),
    ] = None,
    frame_limit: Annotated[
        int,
        typer.Option("--frame-limit", help="Stop after this many frames (-1 = unlimited)."),
    ] = -1,
    ray_address: Annotated[
        str | None,
        typer.Option(
            "--ray-address",
            help="Ray cluster address. Defaults to $RAY_ADDRESS or 'auto' (attach mode).",
        ),
    ] = None,
    gpu_node_ip: Annotated[
        str | None,
        typer.Option(
            "--gpu-node-ip",
            help=(
                "Pin the Serve inference replica to this specific node IP. "
                "Default: use accelerator_type:GAMING (digilab-transmit)."
            ),
        ),
    ] = None,
    num_replicas: Annotated[
        int,
        typer.Option("--num-replicas", help="Number of CloudInferDeployment replicas."),
    ] = 1,
    archive_dir: Annotated[
        Path | None,
        typer.Option(
            "--archive-dir",
            help="Archive flagged windows-of-interest to L2 Zarr under this directory.",
        ),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Python logging level (DEBUG/INFO/WARNING)."),
    ] = "INFO",
) -> None:
    """
    Run the real-time cloud-detection streaming pipeline.

    Attaches to the RAL Ray cluster, deploys CloudInferDeployment on the
    gaming GPU node, and consumes the DaqData.StreamImages feed until
    interrupted (Ctrl-C) or frame_limit is reached.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # --- Load recipe --------------------------------------------------------
    params, recipe_name, recipe_hash = _load_recipe_or_defaults(recipe)
    cadence_s: float = params.get("feature_cadence_s", 60.0)
    threshold: float = params.get("threshold", 0.5)
    grace_s: float = params.get("grace_s", 1.0)

    typer.echo(
        f"[pa-stream-cloud] recipe={recipe_name} hash={recipe_hash[:16]}... "
        f"cadence={cadence_s}s threshold={threshold}"
    )

    # --- Parse module IDs ---------------------------------------------------
    mids: list[int] = []
    if module_ids:
        try:
            mids = [int(x.strip()) for x in module_ids.split(",") if x.strip()]
        except ValueError as exc:
            typer.echo(f"[ERROR] --module-ids parse error: {exc}", err=True)
            raise typer.Exit(1) from exc

    # --- Init Ray (attach mode with runtime_env) ----------------------------
    # The source must be shipped to remote workers (e.g. gaming node) at init time.
    # Ray 2.55+ requires working_dir at job level (ray.init), NOT per-actor.
    from panoseti_analysis.adapters.ray.launcher import init_ray

    _launcher_mode = "attach"  # pa-stream-cloud always attaches to the externally-owned cluster
    _repo_root = str(REPO_ROOT)
    _streaming_runtime_env = {
        "working_dir": _repo_root,
        "env_vars": {"PYTHONPATH": "src:grpc/src"},
        "excludes": [".venv/", ".git/", "*.zarr/", "uv.lock", ".claude/"],
    }

    typer.echo(f"[pa-stream-cloud] Attaching to Ray cluster (working_dir={_repo_root})...")
    init_ray("attach", address=ray_address, runtime_env=_streaming_runtime_env)
    import ray

    res = ray.cluster_resources()  # type: ignore[no-untyped-call]  # Ray API is untyped
    typer.echo(
        f"[pa-stream-cloud] Cluster: CPU={res.get('CPU'):.0f} GPU={res.get('GPU'):.0f} "
        f"RAM={res.get('memory', 0) / 1e9:.1f}GB"
    )

    # --- Load model bundle for provenance tags -----------------------------
    from panoseti_analysis.io.models import load_classifier
    from panoseti_analysis.io.provenance import capture_software

    typer.echo(f"[pa-stream-cloud] Loading model bundle: {model_path}")
    _, bundle = load_classifier(model_path)
    software = capture_software()

    typer.echo(
        f"[pa-stream-cloud] Model: {bundle.model_name} v{bundle.model_version} "
        f"checksum={bundle.checksum[:24]}..."
    )

    # --- Deploy Ray Serve ---------------------------------------------------
    typer.echo(f"[pa-stream-cloud] Deploying CloudInferDeployment (replicas={num_replicas})...")
    from panoseti_analysis.adapters.stream.serve_app import deploy_cloud_infer

    # Pass model path relative to repo root so it resolves correctly on remote
    # Ray workers (CWD = shipped working_dir root) as well as locally.
    try:
        _serve_model_path: str = str(model_path.relative_to(_repo_root))
    except ValueError:
        _serve_model_path = str(model_path)

    handle = deploy_cloud_infer(
        model_path=_serve_model_path,
        num_replicas=num_replicas,
        gpu_node_ip=gpu_node_ip,
    )
    typer.echo("[pa-stream-cloud] CloudInferDeployment ready.")

    # --- Create FrameAccumulator actor factory ------------------------------
    from panoseti_analysis.adapters.stream.accumulator import FrameAccumulator

    def actor_factory(mid: int, dp: str) -> object:
        return FrameAccumulator.remote(  # type: ignore[attr-defined]  # .remote added by @ray.remote at runtime
            module_id=mid,
            data_product=dp,
            serve_handle=handle,
            ml_grpc_host=grpc_host,
            ml_grpc_port=grpc_port,
            cadence_s=cadence_s,
            grace_ns=int(grace_s * 1e9),
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            recipe_hash=recipe_hash,
            git_sha=software.get("git_sha", ""),
            threshold=threshold,
            archive_dir=str(archive_dir) if archive_dir else None,
            telemetry_host=grpc_host,
            telemetry_port=grpc_port,
        )

    # --- Run StreamConsumer -------------------------------------------------
    from panoseti_analysis.adapters.stream.consumer import StreamConsumer

    consumer = StreamConsumer(
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        include_data_products={"img16"},
        module_ids=mids,
        update_interval_seconds=1.0,
        actor_factory=actor_factory,
    )

    typer.echo(
        f"[pa-stream-cloud] Consuming DaqData.StreamImages at {grpc_host}:{grpc_port} "
        f"modules={mids if mids else 'all'} frame_limit={frame_limit}"
    )
    typer.echo("[pa-stream-cloud] Press Ctrl-C to stop.")

    try:
        asyncio.run(consumer.run(frame_limit=frame_limit))
    except KeyboardInterrupt:
        typer.echo("\n[pa-stream-cloud] Interrupted; shutting down.")
    except Exception as exc:
        typer.echo(f"[ERROR] StreamConsumer failed: {exc}", err=True)
        logger.exception("StreamConsumer error")
        raise typer.Exit(1) from exc
    finally:
        # Tear down Serve deployments (safe — does not stop the Ray cluster).
        try:
            import ray as _ray

            _ray.serve.shutdown()  # type: ignore[no-untyped-call]  # Ray Serve API is untyped
            typer.echo("[pa-stream-cloud] Serve shut down.")
        except Exception as exc:
            logger.warning("Serve shutdown error: %s", exc)
        # Only shut down the Ray runtime if we created the cluster (standalone mode).
        # In attach mode the cluster is externally owned and must outlive this process.
        if _launcher_mode == "standalone":
            ray.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_recipe_or_defaults(recipe_path: Path | None) -> tuple[dict[str, Any], str, str]:
    """Load recipe YAML or return built-in defaults."""
    if recipe_path is not None:
        params, name, recipe_hash = load_recipe(recipe_path)
        return params, name, recipe_hash
    # Built-in defaults (mirrors stream_cloud_v1.yml)
    defaults: dict[str, Any] = {
        "feature_cadence_s": 60.0,
        "threshold": 0.5,
        "grace_s": 1.0,
    }
    return defaults, "stream_cloud_default", "sha256:builtin"
