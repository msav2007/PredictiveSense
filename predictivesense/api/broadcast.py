"""Last-value-wins broadcast of :class:`StateSnapshot` to WebSocket clients.

The consumer thread calls :meth:`Broadcaster.publish`, which is an O(1) lock plus
assignment and never touches a client. Each client is served by
:func:`serve_state_client`, which polls the latest snapshot at a fixed rate and
sends it. A client whose send does not complete within a timeout is dropped; a
slow client can never slow the analysis loop and no per-client backlog is kept.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Protocol

from predictivesense.core.types import StateSnapshot
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.metrics import MetricRegistry

__all__ = ["Broadcaster", "serve_state_client", "WebSocketLike"]

_LOG = get_logger(__name__)


class WebSocketLike(Protocol):
    """The slice of a Starlette ``WebSocket`` this module uses."""

    async def send_text(self, data: str) -> Any: ...
    async def close(self, code: int = 1000) -> Any: ...


class Broadcaster:
    """Holds the single most recent snapshot and some client bookkeeping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: StateSnapshot | None = None
        self._clients = 0
        self._dropped_clients = 0

    def publish(self, snapshot: StateSnapshot) -> None:
        """Store the newest snapshot. Called from the analysis-loop thread."""

        with self._lock:
            self._latest = snapshot

    def latest(self) -> StateSnapshot | None:
        with self._lock:
            return self._latest

    @property
    def client_count(self) -> int:
        with self._lock:
            return self._clients

    @property
    def dropped_clients(self) -> int:
        with self._lock:
            return self._dropped_clients

    def _open(self) -> None:
        with self._lock:
            self._clients += 1

    def _close(self, *, dropped: bool) -> None:
        with self._lock:
            self._clients -= 1
            if dropped:
                self._dropped_clients += 1


async def serve_state_client(
    websocket: WebSocketLike,
    broadcaster: Broadcaster,
    *,
    rate_hz: float,
    send_timeout_s: float,
    metrics: MetricRegistry | None = None,
) -> bool:
    """Push snapshots to one client until it disconnects or is dropped.

    Returns ``True`` if the client was dropped for being too slow.

    When ``metrics`` is supplied, the emit->send delay for each snapshot this
    client receives is folded into ``stage_ws_out_ms`` - the Phase 8 broadcast
    hop that ``StateSnapshot.frame_age_ms`` never counted. With push-on-publish
    (Phase 8) this is the wake+send cost; on the legacy poll it also carries the
    up-to-``period`` scheduling wait.
    """

    period = 1.0 / rate_hz
    last_id: int | None = None
    dropped = False
    broadcaster._open()
    try:
        while True:
            await asyncio.sleep(period)
            snapshot = broadcaster.latest()
            if snapshot is None or snapshot.snapshot_id == last_id:
                continue
            if metrics is not None:
                delay_ms = (time.monotonic() - snapshot.emitted_ts) * 1000.0
                if delay_ms >= 0.0:
                    metrics.samples("stage_ws_out_ms", maxlen=4096).add(delay_ms)
            try:
                await asyncio.wait_for(
                    websocket.send_text(snapshot.to_wire_json()),
                    timeout=send_timeout_s,
                )
            except asyncio.TimeoutError:
                dropped = True
                _LOG.warning("dropping slow ws client after %.1fs send timeout", send_timeout_s)
                break
            except (asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001 - client gone / transport error
                _LOG.info("ws client ended: %r", exc)
                break
            last_id = snapshot.snapshot_id
    finally:
        broadcaster._close(dropped=dropped)
        if dropped:
            try:
                await websocket.close(code=1013)  # "try again later"
            except Exception:  # noqa: BLE001 - best effort
                pass
    return dropped
