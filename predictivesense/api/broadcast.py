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
from typing import Any, Protocol

from predictivesense.core.types import StateSnapshot
from predictivesense.logging_setup import get_logger

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
) -> bool:
    """Push snapshots to one client until it disconnects or is dropped.

    Returns ``True`` if the client was dropped for being too slow.
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
