"""Phase 8 section 19 integration test 8 - a slow detector must not let the
end-to-end frame age grow without limit.

A fake perception engine sleeps ~220 ms per frame (slower than the ~66 ms
inter-frame interval), so without a staleness guard the mailbox frame keeps
ageing while the consumer is busy. With ``analysis.max_frame_age_ms`` set, an
over-age frame is dropped (`dropped_stale` rises) and the loop waits for a fresh
one, so the age the overlay sees stays bounded and does not climb across the run.
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
from predictivesense.config.settings import load_config
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import IngestHeader
from predictivesense.perception.types import PerceptionResult
from predictivesense.pipeline.loop import build_loop

pytestmark = pytest.mark.integration

_W, _H = 64, 48


class _SlowPerception:
    """Stand-in for a detector that is slower than the frame cadence."""

    def __init__(self, delay_s: float = 0.22) -> None:
        self._delay = delay_s

    def infer(self, frame, *, frame_index=None) -> PerceptionResult:
        time.sleep(self._delay)
        return PerceptionResult(detector_ms=self._delay * 1000.0, pose_ms=None)

    def info(self) -> dict:
        return {"detection_enabled": True, "pose_enabled": False, "provider": "fake"}


def _jpeg(i: int) -> bytes:
    img = np.zeros((_H, _W, 3), dtype=np.uint8)
    img[:, i % _W] = 255
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _handshake(ws) -> None:
    ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
    ack = json.loads(ws.receive_text())
    ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))


def _run(max_frame_age_ms: float, scheduler: str) -> dict:
    cfg = load_config("dev")
    cfg = cfg.model_copy(update={
        "source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER}),
        "analysis": cfg.analysis.model_copy(update={"max_frame_age_ms": max_frame_age_ms}),
        "consumer": cfg.consumer.model_copy(update={"scheduler": scheduler}),
        "perception": cfg.perception.model_copy(
            update={"detection_enabled": False, "pose_enabled": False}
        ),
    })
    loop = build_loop(cfg, perception=_SlowPerception())
    app = create_app(cfg, loop=loop)

    early: list[float] = []
    late: list[float] = []
    all_ages: list[float] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/state") as state, \
                client.websocket_connect("/ws/ingest") as ws:
            _handshake(ws)
            t0 = time.monotonic()
            base = time.time() * 1000.0
            i = 0
            while time.monotonic() - t0 < 7.0:
                cap = base + i * 66.0
                ws.send_bytes(encode_ingest_message(
                    IngestHeader(client_ts_ms=cap + 4.0, seq=i, w=_W, h=_H,
                                 cap_ts_ms=cap, enc_ms=3.0),
                    _jpeg(i),
                ))
                i += 1
                time.sleep(0.066)
                try:
                    state.settimeout(0.001) if hasattr(state, "settimeout") else None
                except Exception:  # noqa: BLE001
                    pass
            # drain the state socket
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    snap = json.loads(state.receive_text())
                except Exception:  # noqa: BLE001
                    break
                m = snap.get("metrics", {})
                age = snap.get("frame_age_ms")
                if age is not None and age >= 0 and not snap.get("stale"):
                    all_ages.append(age)
                    (early if len(all_ages) <= 15 else late).append(age)

        m = app.state.loop.latest.metrics
        info = app.state.browser_source.info()
        stats = app.state.loop.mailbox.stats()
    return {
        "ages": all_ages, "early": early, "late": late,
        "dropped_stale": m.get("dropped_stale", 0.0),
        "decoded": info["frames_submitted"],
        "buffer_dropped": info["buffer_dropped"],
        "consumed": stats.consumed, "mailbox_dropped": stats.dropped,
        "in_flight_ok": m.get("frames_in_flight", 0.0),
        "error": app.state.loop.error,
    }


def _p(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, int(q / 100 * (len(s) - 1)))
    return s[k]


def test_slow_detector_with_guard_keeps_frame_age_bounded() -> None:
    r = _run(max_frame_age_ms=180.0, scheduler="timer")
    assert r["error"] is None
    assert r["ages"], "no fresh snapshots collected"
    p95 = _p(r["ages"], 95)
    # bounded: not spiralling into multi-second ages
    assert p95 < 900.0, f"frame age p95 {p95} ms - guard did not bound it"
    # the guard actually fired (frames were too old to be worth analysing)
    assert r["dropped_stale"] >= 1.0
    # no unbounded growth: the tail of the run is not dramatically worse than
    # the head (allow 1.8x for warm-up + noise)
    if len(r["early"]) >= 5 and len(r["late"]) >= 5:
        assert _p(r["late"], 50) <= _p(r["early"], 50) * 1.8 + 120.0
    # every decoded frame is accounted for at drain
    analysed = r["consumed"] - r["dropped_stale"]
    in_flight = r["decoded"] - analysed - r["dropped_stale"] - r["buffer_dropped"] - r["mailbox_dropped"]
    assert 0 <= in_flight <= 3, (r["decoded"], analysed, in_flight)


def test_slow_detector_without_guard_still_terminates_and_reconciles() -> None:
    """Guard off: age is allowed to grow, but the run must not deadlock and the
    counters must still reconcile (drops are all mailbox-overwrite here)."""

    r = _run(max_frame_age_ms=0.0, scheduler="timer")
    assert r["error"] is None
    assert r["dropped_stale"] == 0.0
    analysed = r["consumed"]
    in_flight = r["decoded"] - analysed - r["buffer_dropped"] - r["mailbox_dropped"]
    assert 0 <= in_flight <= 3
