"""
StreamConsumer — async wrapper around DaqData.StreamImages.

Consumes the panoseti_grpc DaqData gRPC stream, extracts per-frame
nanosecond timestamps, and routes calibrated-quality frames to the
appropriate FrameAccumulator Ray actor.

Design notes
------------
* Per-frame ns timestamp lives in the PanoImage header Struct (not in a
  dedicated proto field). For [32,32] modules it's nested under quabo_0.
  We reuse panoseti_grpc.daq_data.resources.parse_pano_timestamps for this.

* StreamImagesResponse.timestamp is never populated by the server — do not rely on it.

* Only dp_img16 (32x32 MOVIE) frames are forwarded to cloud-detection
  accumulators.  Other data products can be added via include_data_products.

* The consumer is NOT a Ray actor — it runs in the asyncio event loop of
  the main driver process and submits frames via actor.push_frame.remote().
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for the actor-handle routing table.
# Key: (module_id, dp_name) — e.g. (1, "img16")
# Value: a FrameAccumulator Ray actor handle
# ---------------------------------------------------------------------------
_ActorHandle = Any  # ray.actor.ActorHandle; avoid importing ray at module level


class StreamConsumer:
    """
    Async consumer of DaqData.StreamImages.

    Wraps AioDaqDataClient.stream_images(), extracts per-frame timestamps,
    and dispatches each frame to the matching FrameAccumulator actor.

    Parameters
    ----------
    grpc_host : str
        Hostname/IP of the panoseti_grpc server (default: localhost).
    grpc_port : int
        gRPC port (default: 50051).
    include_data_products : set[str]
        Data products to forward.  Default: {"img16"} (cloud-detection path).
        Other products can be added but must have accumulators registered.
    module_ids : list[int] | None
        Module IDs to subscribe to.  None/empty = all modules.
    update_interval_seconds : float
        StreamImages update interval to request from the server.
    actor_factory : callable | None
        Called with (module_id, dp_name) -> actor_handle when a new
        (module, dp) pair is first seen.  If None, frames for unregistered
        (module, dp) pairs are dropped (useful for replay tests that pre-create
        actors and pass them via register_actor).
    """

    def __init__(
        self,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        include_data_products: set[str] | None = None,
        module_ids: list[int] | None = None,
        update_interval_seconds: float = 1.0,
        actor_factory: Callable[[int, str], _ActorHandle] | None = None,
    ) -> None:
        self._host = grpc_host
        self._port = grpc_port
        self._include_dps = include_data_products or {"img16"}
        self._module_ids = module_ids or []
        self._update_interval = update_interval_seconds
        self._actor_factory = actor_factory
        # (module_id, dp_name) -> actor handle
        self._actors: dict[tuple[int, str], _ActorHandle] = {}
        self._total_frames: int = 0
        self._dropped_frames: int = 0

    # -- Public API ----------------------------------------------------------

    def register_actor(self, module_id: int, dp_name: str, handle: _ActorHandle) -> None:
        """Pre-register an actor handle for a (module, dp) pair."""
        self._actors[(module_id, dp_name)] = handle
        logger.info("Registered actor for module=%d dp=%s", module_id, dp_name)

    async def run(
        self,
        *,
        frame_limit: int = -1,
        timeout: float = 36000.0,
    ) -> None:
        """
        Consume the StreamImages stream until frame_limit is reached or
        the server closes the stream.

        Parameters
        ----------
        frame_limit : int
            Stop after this many frames (across all modules/dps). -1 = unlimited.
        timeout : float
            gRPC call timeout in seconds.
        """
        from panoseti_grpc.daq_data.client import AioDaqDataClient

        logger.info(
            "StreamConsumer starting: host=%s port=%d dps=%s modules=%s",
            self._host,
            self._port,
            self._include_dps,
            self._module_ids or "all",
        )

        async with AioDaqDataClient(self._host, self._port) as client:
            async for parsed in client.stream_images(
                stream_movie_data=("img16" in self._include_dps or "img8" in self._include_dps),
                stream_pulse_height_data=("ph256" in self._include_dps or "ph1024" in self._include_dps),
                update_interval_seconds=self._update_interval,
                module_ids=self._module_ids,
                parse_pano_images=True,
                timeout=timeout,
            ):
                await self._dispatch(parsed)
                self._total_frames += 1
                if frame_limit > 0 and self._total_frames >= frame_limit:
                    logger.info("Frame limit %d reached; stopping consumer", frame_limit)
                    break

        logger.info(
            "StreamConsumer finished: total=%d dropped=%d",
            self._total_frames,
            self._dropped_frames,
        )

    # -- Internal ------------------------------------------------------------

    async def _dispatch(self, parsed: dict[str, Any]) -> None:
        """
        Extract frame metadata and submit to the matching accumulator actor.

        Parameters
        ----------
        parsed : dict
            Output of parse_pano_image(): includes 'module_id', 'type',
            'image_array' (ndarray), 'bytes_per_pixel', 'header' (with
            'pandas_unix_timestamp' and 'pkt_unix_timestamp' merged in).
        """
        module_id: int = parsed.get("module_id", 0)
        pano_type: str = parsed.get("type", "UNDEFINED")  # "MOVIE" or "PULSE_HEIGHT"
        image_array: np.ndarray = parsed["image_array"]
        bpp: int = parsed.get("bytes_per_pixel", 2)
        shape = image_array.shape

        # Derive dp_name from type+shape+bpp
        dp_name = _pano_type_to_dp_name(pano_type, shape, bpp)
        if dp_name not in self._include_dps:
            return  # not a tracked data product

        # Extract the per-frame nanosecond timestamp.
        # parse_pano_image merges parse_pano_timestamps into header,
        # so 'pandas_unix_timestamp' is already present as a pd.Timestamp.
        header = parsed.get("header", {})
        ts_ns = _extract_unix_t_ns(header)
        if ts_ns is None:
            logger.warning("module=%d dp=%s: no timestamp in header; dropping frame", module_id, dp_name)
            self._dropped_frames += 1
            return

        key = (module_id, dp_name)
        actor = self._actors.get(key)
        if actor is None:
            if self._actor_factory is not None:
                actor = self._actor_factory(module_id, dp_name)
                self._actors[key] = actor
                logger.info("Created new accumulator for module=%d dp=%s", module_id, dp_name)
            else:
                logger.debug("No actor for module=%d dp=%s; dropping frame", module_id, dp_name)
                self._dropped_frames += 1
                return

        # Submit to actor non-blocking (fire-and-forget; Ray queues the call)
        actor.push_frame.remote(image_array, ts_ns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pano_type_to_dp_name(pano_type: str, shape: tuple[int, ...], bpp: int) -> str:
    """Map (pano_type, shape, bpp) to a dp_name string matching PFF conventions."""
    if pano_type == "MOVIE":
        if shape == (32, 32) and bpp == 2:
            return "img16"
        if shape == (32, 32) and bpp == 1:
            return "img8"
    elif pano_type == "PULSE_HEIGHT":
        if shape == (16, 16):
            return "ph256"
        if shape == (32, 32):
            return "ph1024"
    return f"unknown_{pano_type}_{shape}_{bpp}"


def _extract_unix_t_ns(header: dict[str, Any]) -> int | None:
    """
    Extract an int64 unix timestamp in nanoseconds from a parsed PanoImage header.

    Prefers pkt_unix_timestamp (tv_sec / tv_usec, microsecond precision)
    over White-Rabbit wr_unix_timestamp (nanosecond precision from FPGA timing).
    For our 60-second windows, microsecond precision is sufficient.

    Returns None if no timestamp can be derived.
    """
    import decimal

    # After parse_pano_image, 'pandas_unix_timestamp' is a pd.Timestamp.
    pts = header.get("pandas_unix_timestamp")
    if pts is not None:
        try:
            # pd.Timestamp.value is int64 nanoseconds since epoch
            return int(pts.value)
        except (AttributeError, TypeError):
            pass

    # Fallback: derive directly from pkt_unix_timestamp (Decimal)
    pkt_ts = header.get("pkt_unix_timestamp")
    if pkt_ts is not None:
        try:
            return int(decimal.Decimal(str(pkt_ts)) * decimal.Decimal("1e9"))
        except (ValueError, TypeError):
            pass

    return None
