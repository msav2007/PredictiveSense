"""Last-value-wins broadcast of :class:`StateSnapshot` to WebSocket clients.

The consumer thread calls :meth:`Broadcaster.publish`, which is an O(1) lock plus
assignment, records the newest snapshot, and **wakes every connected client**
via ``loop.call_soon_threadsafe`` - it still never touches a client socket. Each
client is served by :func:`serve_state_client`, which blocks on that wake
(``asyncio.Event``), sends the current snapshot, and enforces a maximum send
rate so a fast analysis loop cannot peg the event loop. A ``period`` fallback
timeout keeps a client refreshing even if a wake is ever missed. A client whose
send does not complete within a timeout is dropped; a slow client can never slow
the analysis loop and no per-client backlog is kept.

Phase 8: this replaced a fixed ``asyncio.sleep(1/rate_hz)`` poll that added up to
one poll period (100 ms at 10 Hz) between emission and send - latency the
``StateSnapshot.frame_age_ms`` never counted. Before/after in
``docs/phase-reports/phase8.md``.
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
        # (event-loop, wake-event) per connected client. publish() sets each
        # event via call_soon_threadsafe so the analysis thread never awaits.
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    def publish(self, snapshot: StateSnapshot) -> None:
        """Store the newest snapshot and wake every client. Analysis-loop thread."""

        with self._lock:
            self._latest = snapshot
            waiters = list(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # loop already closed / shutting down - the client task is gone
                pass

    def latest(self) -> StateSnapshot | None:
        with self._lock:
            return self._latest

    def _add_waiter(
        self, loop: asyncio.AbstractEventLoop, event: asyncio.Event
    ) -> None:
        with self._lock:
            self._waiters.append((loop, event))

    def _remove_waiter(self, event: asyncio.Event) -> None:
        with self._lock:
            self._waiters = [w for w in self._waiters if w[1] is not event]

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
    # Maximum send rate: never faster than twice the configured broadcast rate,
    # so a fast analysis loop cannot peg the event loop (section 7 "rate ceiling",
    # applied here at the fan-out rather than the sampler). No busy waiting - the
    # client blocks on an Event.
    min_send_interval = 0.5 / rate_hz
    last_id: int | None = None
    last_send = 0.0
    dropped = False
    loop = asyncio.get_running_loop()
    wake = asyncio.Event()
    broadcaster._open()
    broadcaster._add_waiter(loop, wake)
    try:
        while True:
            try:
                await asyncio.wait_for(wake.wait(), timeout=period)
            except asyncio.TimeoutError:
                pass  # fallback refresh - a wake was missed or the loop is idle
            wake.clear()
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
            now = time.monotonic()
            gap = now - last_send
            last_send = now
            if gap < min_send_interval:
                await asyncio.sleep(min_send_interval - gap)
    finally:
        broadcaster._remove_waiter(wake)
        broadcaster._close(dropped=dropped)
        if dropped:
            try:
                await websocket.close(code=1013)  # "try again later"
            except Exception:  # noqa: BLE001 - best effort
                pass
    return dropped
