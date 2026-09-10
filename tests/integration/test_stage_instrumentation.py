"""Phase 8 section 5 / test 10 - one analysed frame's journey is attributable
end to end, with no missing server stage.

Perception is off in ``browser_config`` (fast), so ``detector`` / ``pose`` /
``policy`` are legitimately absent; every *transport* stage
(worker_encode, ws_transit, decode, both single-slot buffer dwells,
mailbox_dwell, snapshot_build) must be present for an ingested frame, and the
loop must expose them on the snapshot as ``stage_*`` keys plus ``ws_out`` via
the broadcaster.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.camera.framing import encode_ingest_message
from predictivesense.core.types import IngestHeader

pytestmark = pytest.mark.integration

_W, _H = 64, 48
_TRANSPORT = (
    "worker_encode",
    "ws_transit",
    "decode",
    "src_buffer_dwell",
    "producer_handoff",
    "mailbox_dwell",
    "snapshot_build",
)


def _jpeg(tag: int) -> bytes:
    img = np.zeros((_H, _W, 3), dtype=np.uint8)
    img[:, tag % _W] = 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _handshake(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
    ack = json.loads(ws.receive_text())
    ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))


def test_stage_attribution_record_is_complete_for_an_analysed_frame(browser_config) -> None:
    app = create_app(browser_config)
    stage_snaps: list[dict] = []
    with TestClient(app) as client:
        reader_stop = time.monotonic() + 8.0
        with client.websocket_connect("/ws/state") as state, \
                client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)
            base = time.time() * 1000.0
            for i in range(60):
                cap = base + i * 33.0
                header = IngestHeader(
                    client_ts_ms=cap + 5.0, seq=i, w=_W, h=_H,
                    cap_ts_ms=cap, enc_ms=3.2,
                )
                ws.send_bytes(encode_ingest_message(header, _jpeg(i)))
                time.sleep(0.033)
            # drain /ws/state for the stage keys
            while time.monotonic() < reader_stop:
                try:
                    snap = json.loads(state.receive_text())
                except Exception:  # noqa: BLE001
                    break
                m = snap.get("metrics", {})
                if not snap.get("stale") and "stage_ws_transit_ms" in m:
                    stage_snaps.append(snap)
                    if len(stage_snaps) >= 5:
                        break

        loop = app.state.loop
        records = loop.frame_stages()

    assert loop.error is None
    assert stage_snaps, "no /ws/state snapshot ever carried stage_* keys"

    m = stage_snaps[-1]["metrics"]
    for stage in _TRANSPORT:
        key = f"stage_{stage}_ms"
        assert key in m, f"snapshot missing {key}"
        assert m[key] >= 0.0
    assert "stage_capture_to_snapshot_ms" in m and m["stage_capture_to_snapshot_ms"] > 0.0
    assert "capture_client_ts_ms" in m and m["capture_client_ts_ms"] > 0.0
    # broadcaster recorded the emit->send hop
    assert loop.metrics.samples("stage_ws_out_ms").count >= 1

    assert records, "loop kept no FrameStages record"
    last = records[-1]
    missing = last.missing_server_stages()
    # only the perception stages may be missing here (perception is off)
    assert set(missing) <= {"detector", "pose", "policy"}, missing
    assert last.capture_to_snapshot_ms is not None and last.capture_to_snapshot_ms > 0.0
    assert last.mailbox_dwell_ms is not None
