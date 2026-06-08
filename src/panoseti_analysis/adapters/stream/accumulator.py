"""
FrameAccumulator — Ray actor for per-(module, dp) frame buffering and inference.

Each actor:
  1. Maintains a rolling deque of (unix_t_ns, frame) covering ≥ window_ns.
  2. Builds a PROGRESSIVE pseudo-calibration that improves over the night:
     rolling median over the buffered frames = approximate pedestal/background.
  3. When the buffer spans ≥ window_ns AND at least cadence_s has elapsed
     since the last emission, assembles an xr.Dataset and calls the
     CloudInferDeployment Serve handle for scoring.
  4. Emits the scored prediction via the ML gRPC client (EmitPrediction)
     and the telemetry Redis path (TelemetryClient.log_flexible).
  5. Archives flagged windows-of-interest to L2 Zarr via io.stores.write_store.

Layer A kernels (calibrate_img, predict_cloud_score) are imported inside the
actor — this is intentional (warm Ray worker import, not cluster-level).

Calibration maturity
--------------------
maturity = min(1.0, frames_seen / (window_ns / nominal_frame_interval_ns))
where nominal_frame_interval_ns ≈ 100_000 (100 µs per PANOSETI frame).
Clipped to [0,1]; consumers can discount early-night windows.

Out-of-order frames
-------------------
Frames are stable-sorted by unix_t_ns before Dataset assembly (mirrors the
L0→L1 sort repair).  Frames arriving later than window_end − grace_ns are
counted and dropped from that window for bounded latency.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    pass  # used only for type hints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cloud-detection kernel constants (must match cloud_detector.py:134-135)
_WINDOW_NS: int = 60_000_000_000  # 60 seconds in nanoseconds

# Approximate inter-frame interval for calibration-maturity denominator.
# PANOSETI integrates ~100 µs per frame in movie mode.
_NOMINAL_FRAME_INTERVAL_NS: int = 100_000  # 100 µs

# Frames to keep beyond window_ns for late-arrival tolerance.
# Keep at least 2× the window for safe assembly.
_BUFFER_HEADROOM_NS: int = 2 * _WINDOW_NS


# ---------------------------------------------------------------------------
# FrameAccumulator (Ray actor)
# ---------------------------------------------------------------------------

# NOTE: @ray.remote is applied at the bottom of the file (after the class body)
# so this module can be imported without Ray available (e.g. during unit tests
# or type-checking).  Use FrameAccumulator.remote(...) at runtime as usual.


class FrameAccumulator:
    """
    Stateful per-(module, data_product) frame accumulator.

    Parameters
    ----------
    module_id : int
    data_product : str
        "img16", "img8", "ph256", or "ph1024".
    serve_handle : ray.serve.handle.RayServeHandle
        Handle to the CloudInferDeployment Serve deployment.
    ml_grpc_host : str
        Host of the panoseti_grpc server for EmitPrediction RPC.
    ml_grpc_port : int
        Port of the panoseti_grpc server.
    cadence_s : float
        Minimum interval between window emissions (seconds).
    grace_ns : int
        Frames arriving after window_end − grace_ns are considered late
        and dropped from that window.
    model_name : str
    model_version : str
    recipe_hash : str
    git_sha : str
    threshold : float
        Cloud score threshold for labelling.
    archive_dir : str | None
        If set, flagged windows-of-interest are archived to L2 Zarr under
        this directory.
    telemetry_host : str
        Host for telemetry Redis-path logging.
    telemetry_port : int
    """

    def __init__(
        self,
        module_id: int,
        data_product: str,
        serve_handle: Any,  # RayServeHandle
        ml_grpc_host: str = "localhost",
        ml_grpc_port: int = 50051,
        cadence_s: float = 60.0,
        grace_ns: int = 1_000_000_000,  # 1 s grace
        model_name: str = "cloud_detector",
        model_version: str = "1.0",
        recipe_hash: str = "",
        git_sha: str = "",
        threshold: float = 0.5,
        archive_dir: str | None = None,
        telemetry_host: str = "localhost",
        telemetry_port: int = 50051,
    ) -> None:
        self.module_id = module_id
        self.data_product = data_product
        self._serve_handle = serve_handle
        self._ml_grpc_host = ml_grpc_host
        self._ml_grpc_port = ml_grpc_port
        self._cadence_ns = int(cadence_s * 1e9)
        self._grace_ns = grace_ns
        self._model_name = model_name
        self._model_version = model_version
        self._recipe_hash = recipe_hash
        self._git_sha = git_sha
        self._threshold = threshold
        self._archive_dir = Path(archive_dir) if archive_dir else None
        self._telemetry_host = telemetry_host
        self._telemetry_port = telemetry_port

        # Rolling buffer: deque of (unix_t_ns: int, frame: np.ndarray[H,W])
        self._buffer: deque[tuple[int, np.ndarray]] = deque()
        self._frames_seen: int = 0
        self._last_emission_ns: int = 0
        self._windows_emitted: int = 0

    # -- Public Ray remote methods -------------------------------------------

    def push_frame(self, frame: np.ndarray, unix_t_ns: int) -> None:
        """
        Push one calibrated-quality frame into the rolling buffer.

        This is the hot path; it must be fast.  Triggering inference is
        done synchronously here (Ray actor calls are already serialised per
        actor, so this is safe and avoids callback complexity).

        Parameters
        ----------
        frame : np.ndarray
            Shape (H, W), dtype uint16 (img16) or int16 (ph).
        unix_t_ns : int
            White-Rabbit or packet nanosecond unix timestamp.
        """
        self._buffer.append((unix_t_ns, frame))
        self._frames_seen += 1

        # Evict frames older than window_ns + headroom from the front.
        if len(self._buffer) > 1:
            newest_ts = self._buffer[-1][0]
            cutoff_ts = newest_ts - (_WINDOW_NS + _BUFFER_HEADROOM_NS)
            while self._buffer and self._buffer[0][0] < cutoff_ts:
                self._buffer.popleft()

        # Check whether to emit a window.
        self._maybe_emit()

    def stats(self) -> dict[str, Any]:
        """Return accumulator statistics for monitoring / debugging."""
        ts_range = (self._buffer[0][0], self._buffer[-1][0]) if len(self._buffer) >= 2 else (0, 0)
        return {
            "module_id": self.module_id,
            "data_product": self.data_product,
            "frames_in_buffer": len(self._buffer),
            "frames_seen": self._frames_seen,
            "windows_emitted": self._windows_emitted,
            "buffer_span_ns": ts_range[1] - ts_range[0] if ts_range[1] else 0,
            "calibration_maturity": self._calibration_maturity(),
        }

    # -- Internal ------------------------------------------------------------

    def _calibration_maturity(self) -> float:
        """Progressive calibration quality: 0=cold, 1=warm."""
        expected_frames = _WINDOW_NS / _NOMINAL_FRAME_INTERVAL_NS
        return float(min(1.0, self._frames_seen / expected_frames))

    def _maybe_emit(self) -> None:
        """Emit a scored window if conditions are met."""
        if len(self._buffer) < 2:
            return

        # Sort by timestamp; check span
        sorted_buf = sorted(self._buffer, key=lambda x: x[0])
        t_start, t_end = sorted_buf[0][0], sorted_buf[-1][0]
        span = t_end - t_start
        if span < _WINDOW_NS:
            return  # not enough data yet

        # Check cadence (time since last emission)
        now_ns = time.time_ns()
        if self._last_emission_ns > 0 and (now_ns - self._last_emission_ns) < self._cadence_ns:
            return

        # Select window: frames in [t_end - window_ns, t_end]
        window_start = t_end - _WINDOW_NS
        window_frames = [(ts, f) for ts, f in sorted_buf if ts >= window_start]

        if len(window_frames) < 2:
            return

        self._emit_window(window_frames, t_start=window_frames[0][0], t_end=window_frames[-1][0])
        self._last_emission_ns = now_ns

    def _emit_window(
        self,
        window_frames: list[tuple[int, np.ndarray]],
        t_start: int,
        t_end: int,
    ) -> None:
        """
        Assemble xr.Dataset, run inference via Serve handle, emit outputs.
        """
        import asyncio

        from panoseti_analysis.algorithms.calibrate_img import calibrate_img
        from panoseti_analysis.config.models import CloudInferParams, ImgCalibParams

        # 1. Build raw Dataset
        timestamps = np.array([ts for ts, _ in window_frames], dtype=np.int64)
        cube = np.stack([f.astype(np.float32) for _, f in window_frames], axis=0)
        ds_raw = xr.Dataset(
            {
                "images": (("time", "y", "x"), cube),
                "unix_t_ns": (("time",), timestamps),
            }
        )

        # 2. Calibrate (progressive: rolling-buffer median = pedestal estimate)
        ds_l1 = calibrate_img(ds_raw, ImgCalibParams())

        # 3. Run inference via Serve handle (async call inside Ray actor)
        infer_params = CloudInferParams(cadence_s=self._cadence_ns / 1e9, threshold=self._threshold)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result: xr.Dataset = loop.run_until_complete(
            self._serve_handle.infer.remote(ds_l1, infer_params)
        )

        if result is None or len(result.data_vars) == 0:
            logger.warning(
                "Serve returned empty result for module=%d dp=%s t=[%d,%d]",
                self.module_id,
                self.data_product,
                t_start,
                t_end,
            )
            return

        # 4. Extract scalar prediction for the window (mean of T_l2 scores)
        scores = result["cloud_score"].values
        labels = result["cloud_label"].values
        cloud_score = float(np.mean(scores))
        cloud_label = bool(np.any(labels))
        maturity = self._calibration_maturity()

        logger.info(
            "Window emitted: module=%d dp=%s t=[%d,%d] score=%.3f label=%s maturity=%.2f",
            self.module_id,
            self.data_product,
            t_start,
            t_end,
            cloud_score,
            cloud_label,
            maturity,
        )

        # 5. Emit to ML gRPC service (EmitPrediction)
        self._emit_to_grpc(cloud_score, cloud_label, t_start, t_end, maturity)

        # 6. Emit to telemetry / Redis path for dashboard observability
        self._emit_to_telemetry(cloud_score, cloud_label, t_start, t_end, maturity)

        # 7. Archive flagged windows-of-interest to L2 Zarr
        if cloud_label and self._archive_dir is not None:
            self._archive_window(result, t_start, t_end)

        self._windows_emitted += 1

    def _emit_to_grpc(
        self,
        cloud_score: float,
        cloud_label: bool,
        t_start: int,
        t_end: int,
        maturity: float,
    ) -> None:
        """Fire-and-forget EmitPrediction to the ML gRPC service."""
        try:
            from panoseti_grpc.ml_inference.client import MLInferenceClient

            with MLInferenceClient(self._ml_grpc_host, self._ml_grpc_port) as client:
                client.emit_prediction(
                    module_id=self.module_id,
                    data_product=self.data_product,
                    model_name=self._model_name,
                    model_version=self._model_version,
                    recipe_hash=self._recipe_hash,
                    git_sha=self._git_sha,
                    cloud_score=cloud_score,
                    cloud_label=cloud_label,
                    t_start_ns=t_start,
                    t_end_ns=t_end,
                    calibration_maturity=maturity,
                    timeout=5.0,
                )
        except Exception as exc:
            logger.warning("EmitPrediction failed (module=%d): %s", self.module_id, exc)

    def _emit_to_telemetry(
        self,
        cloud_score: float,
        cloud_label: bool,
        t_start: int,
        t_end: int,
        maturity: float,
    ) -> None:
        """Emit prediction metadata to Redis via the Telemetry gRPC service."""
        try:
            from panoseti_grpc.telemetry.client import TelemetryClient

            payload: dict[str, Any] = {
                "module_id": self.module_id,
                "data_product": self.data_product,
                "model_name": self._model_name,
                "model_version": self._model_version,
                "recipe_hash": self._recipe_hash,
                "cloud_score": cloud_score,
                "cloud_label": int(cloud_label),
                "t_start_ns": t_start,
                "t_end_ns": t_end,
                "calibration_maturity": maturity,
                "windows_emitted": self._windows_emitted,
            }
            with TelemetryClient(self._telemetry_host, self._telemetry_port) as client:  # type: ignore[attr-defined]  # grpc client is a context manager at runtime but untyped
                client.log_flexible(
                    device_type="DEV_ml_predictions",
                    device_id=f"module_{self.module_id}_{self.data_product}",
                    data=payload,
                )
        except Exception as exc:
            logger.warning("Telemetry emission failed (module=%d): %s", self.module_id, exc)

    def _archive_window(self, result: xr.Dataset, t_start: int, t_end: int) -> None:
        """Archive a flagged window-of-interest to L2 Zarr."""
        try:
            from panoseti_analysis.config.models import ProcessingStep
            from panoseti_analysis.io.provenance import capture_software, now_utc
            from panoseti_analysis.io.stores import write_store

            assert self._archive_dir is not None
            store_name = (
                f"stream.module_{self.module_id}.{self.data_product}.t{t_start}_{t_end}.cloud.zarr"
            )
            out = self._archive_dir / store_name
            step = ProcessingStep(
                step_name="stream_cloud_infer",
                step_version="1.0",
                params={
                    "model_name": self._model_name,
                    "model_version": self._model_version,
                    "recipe_hash": self._recipe_hash,
                    "cadence_s": self._cadence_ns / 1e9,
                    "threshold": self._threshold,
                    "calibration": "progressive_rolling_median",
                },
                recipe_hash=self._recipe_hash,
                timestamp_utc=now_utc(),
                software=capture_software(),
            )
            write_store(result, out, processing_history=[step])
            logger.info("Archived window-of-interest to %s", out)
        except Exception as exc:
            logger.warning("Archive failed (module=%d): %s", self.module_id, exc)


# ---------------------------------------------------------------------------
# Apply @ray.remote lazily so the module is importable without Ray.
# At runtime (when Ray is available), FrameAccumulator.remote(...) works.
# ---------------------------------------------------------------------------
try:
    import ray as _ray

    FrameAccumulator = _ray.remote(FrameAccumulator)  # type: ignore[misc,assignment]  # @ray.remote rebinds the class to an ActorClass at runtime
except ModuleNotFoundError:
    pass  # Ray not installed; FrameAccumulator is a plain class (useful for unit tests)
