"""``WS /ws/ingest`` - binary analysis frames from the browser Web Worker.

Handshake (JSON text):

    client -> {"type":"hello","client_ts_ms": <epoch ms>}
    server -> {"type":"hello_ack","server_mono": <monotonic s>,"rtt_probe": <id>}
    client -> {"type":"echo","rtt_probe": <id>}          (echoed immediately)

The server records ``offset = server_mono - client_ts_ms/1000`` and the echo
round-trip time. Then every binary message is decoded (4-byte length + JSON
header + JPEG) and handed to :class:`BrowserSource`. Malformed messages are
counted and dropped; a disconnect ends the handler while the analysis loop keeps
running (snapshots go ``stale=true``). Nothing here can raise into the loop.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from predictivesense.camera.browser import BrowserSource
from predictivesense.camera.framing import (
    ClockOffset,
    FramingError,
    clock_offset_seconds,
    decode_ingest_message,
    rtt_ms_from_monotonic,
)
from predictivesense.logging_setup import get_logger

__all__ = ["router"]

_LOG = get_logger(__name__)
_HANDSHAKE_TIMEOUT_S = 10.0

router = APIRouter()


@router.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket) -> None:
    app = websocket.app
    config = app.state.config

    if config.capture.owner != "browser":
        await websocket.close(code=4403)  # policy: backend owns the camera
        _LOG.info("ingest connection refused: capture.owner=%s", config.capture.owner)
        return

    source = getattr(app.state, "browser_source", None)
    if not isinstance(source, BrowserSource):
        await websocket.close(code=1011)
        _LOG.warning("ingest connection refused: no BrowserSource wired")
        return

    await websocket.accept()
    stats = app.state.ingest_stats

    if not await _handshake(websocket, source, stats):
        return

    max_bytes = config.capture.max_ingest_message_bytes
    frames = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None:
                continue  # stray text frame after the handshake
            if len(data) > max_bytes:
                source.note_malformed()
                stats["malformed"] += 1
                continue
            try:
                header, jpeg = decode_ingest_message(data, max_bytes=max_bytes)
            except FramingError as exc:
                source.note_malformed()
                stats["malformed"] += 1
                _LOG.debug("dropped malformed ingest message: %s", exc)
                continue
            await asyncio.to_thread(source.submit, header, jpeg)
            frames += 1
            stats["frames"] += 1
    except WebSocketDisconnect:
        pass
    except (RuntimeError, ConnectionError) as exc:
        _LOG.info("ingest socket ended: %r", exc)
    except Exception as exc:  # noqa: BLE001 - ingest must never kill the loop
        _LOG.error("ingest loop error (isolated): %r", exc)
    finally:
        _LOG.info("ingest client disconnected after %d frame(s)", frames)


async def _handshake(
    websocket: WebSocket, source: BrowserSource, stats: dict[str, float]
) -> bool:
    """Run hello/hello_ack/echo. Returns True on success (socket left open)."""

    try:
        hello_raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=_HANDSHAKE_TIMEOUT_S
        )
        hello = json.loads(hello_raw)
        client_ts_ms = float(hello["client_ts_ms"])

        server_mono = time.monotonic()
        probe_id = uuid.uuid4().hex[:8]
        await websocket.send_text(
            json.dumps(
                {"type": "hello_ack", "server_mono": server_mono, "rtt_probe": probe_id}
            )
        )
        await asyncio.wait_for(websocket.receive_text(), timeout=_HANDSHAKE_TIMEOUT_S)
        echoed_mono = time.monotonic()

        offset = ClockOffset(
            offset_s=clock_offset_seconds(client_ts_ms, server_mono),
            rtt_ms=rtt_ms_from_monotonic(server_mono, echoed_mono),
        )
        source.begin_client_session(offset)
        stats["clock_offset_s"] = offset.offset_s
        stats["clock_offset_rtt_ms"] = offset.rtt_ms
        stats["clients"] += 1
        return True
    except (asyncio.TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _LOG.warning("ingest handshake failed: %r", exc)
        try:
            await websocket.close(code=1002)
        except RuntimeError:
            pass
        return False
    except WebSocketDisconnect:
        return False
