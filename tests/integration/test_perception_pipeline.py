"""Detector + pose over a fixture image, and the perception-off equivalence.

The ``models``-marked cases need ONNX weights on disk; they skip cleanly (never
fail, never silently pass) when ``models/`` is empty. The perception-off case
needs no weights and is a plain regression check that Phase 1.6 behaviour is
reproduced.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from predictivesense.api.app import create_app
from predictivesense.camera.framing import encode_ingest_message
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import Detection, Frame, IngestHeader, Pose
from predictivesense.perception.classes import KEYPOINT_NAMES
from predictivesense.pipeline.loop import build_loop

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "perception"
_FIXTURE_IMAGES = sorted(_FIXTURES.glob("*.jpg"))


def _frame_from(path: Path) -> Frame:
    img = cv2.imread(str(path))
    assert img is not None, f"could not read fixture {path}"
    img = np.ascontiguousarray(img)
    return Frame(
        frame_id=0, capture_ts=0.0, image=img,
        width=int(img.shape[1]), height=int(img.shape[0]),
        source_id="fixture", seq=0,
    )


@pytest.mark.models
@pytest.mark.parametrize("path", _FIXTURE_IMAGES, ids=lambda p: p.stem)
def test_detector_returns_valid_detections(require_models, perception_dev_config, path) -> None:
    from predictivesense.perception.detector import ObjectDetector

    det = ObjectDetector(perception_dev_config.perception.detector, provider="cpu")
    frame = _frame_from(path)
    detections = det.infer(frame)

    assert isinstance(detections, list)
    for d in detections:
        assert isinstance(d, Detection)
        x1, y1, x2, y2 = d.bbox
        assert 0.0 <= x1 < x2 <= frame.width + 1e-6
        assert 0.0 <= y1 < y2 <= frame.height + 1e-6
        assert 0.0 <= d.score <= 1.0
        assert d.frame_id == frame.frame_id
        assert d.class_name and isinstance(d.class_id, int)


@pytest.mark.models
@pytest.mark.parametrize("path", _FIXTURE_IMAGES, ids=lambda p: p.stem)
def test_pose_returns_valid_poses(require_models, perception_dev_config, path) -> None:
    from predictivesense.perception.pose import PoseEstimator

    pose = PoseEstimator(perception_dev_config.perception.pose, provider="cpu")
    frame = _frame_from(path)
    poses = pose.infer(frame)

    assert isinstance(poses, list)
    for p in poses:
        assert isinstance(p, Pose)
        assert len(p.keypoints) == len(KEYPOINT_NAMES) == 17
        for (kx, ky, kv) in p.keypoints:
            assert -1.0 <= kx <= frame.width + 1.0
            assert -1.0 <= ky <= frame.height + 1.0
            assert 0.0 <= kv <= 1.0
        x1, y1, x2, y2 = p.bbox
        assert x1 < x2 and y1 < y2
        assert 0.0 <= p.score <= 1.0


@pytest.mark.models
def test_person_is_detected_in_the_cabin_fixtures(require_models, perception_dev_config) -> None:
    """The developer's own footage frames contain a person - the pipeline must
    see at least one across the fixtures (a structural sanity check, not an
    accuracy claim)."""

    from predictivesense.perception.engine import build_perception

    engine = build_perception(perception_dev_config, strict=True)
    seen = set()
    for path in _FIXTURE_IMAGES:
        result = engine.infer(_frame_from(path))
        seen.update(d.class_name for d in result.detections)
    assert "person" in seen


@pytest.mark.models
@pytest.mark.slow
def test_preview_independence_holds_with_perception_running(require_models, dev_config) -> None:
    """BLOCK 3.7/31: the first phase where the analysis path is genuinely slow.
    A stalled consumer must still not block ingest, and with perception ON the
    mailbox stays single-slot. Records the drop rate under load."""

    cfg = dev_config.model_copy(
        update={
            "source": dev_config.source.model_copy(update={"kind": SourceKind.BROWSER}),
            "perception": dev_config.perception.model_copy(
                update={"detection_enabled": True, "pose_enabled": True}
            ),
        }
    )
    app = create_app(cfg)
    w, h = 96, 72
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, ::7] = 200
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()

    sent = 0
    with TestClient(app) as client:
        assert app.state.loop.perception is not None, "perception must be active for this test"
        with client.websocket_connect("/ws/ingest") as ws:
            ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
            ack = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))

            client.post("/api/debug/stall", params={"seconds": 3.0})
            t_end = time.monotonic() + 3.5
            base_ms = time.time() * 1000.0
            while time.monotonic() < t_end:
                header = IngestHeader(client_ts_ms=base_ms + sent * 20.0, seq=sent, w=w, h=h)
                started = time.monotonic()
                ws.send_bytes(encode_ingest_message(header, jpeg))
                assert time.monotonic() - started < 1.0  # send never blocks on the slow consumer
                sent += 1
                time.sleep(0.02)

            src = app.state.browser_source
            deadline = time.monotonic() + 3.0
            while src.info()["frames_submitted"] < sent and time.monotonic() < deadline:
                time.sleep(0.02)

        loop = app.state.loop
        info = app.state.browser_source.info()
        mailbox = loop.mailbox.stats()
        threads = loop.threads_alive()
        max_depth = loop.max_mailbox_depth
        loop_error = loop.error

    assert sent > 20
    assert info["frames_submitted"] == sent          # socket kept accepting throughout
    assert mailbox.depth <= 1 and max_depth <= 1
    assert threads["producer"] is True
    assert loop_error is None
    total = mailbox.consumed + mailbox.dropped
    drop_rate = mailbox.dropped / total if total else 0.0
    print(f"\n[preview-independence + perception] sent={sent} "
          f"consumed={mailbox.consumed} dropped={mailbox.dropped} drop_rate={drop_rate:.3f}")


def test_perception_off_reproduces_phase_1_6(dev_config) -> None:
    """dev_config has perception disabled (conftest). Snapshots carry empty
    detections/poses and the loop behaves exactly as Phase 1.6."""

    loop = build_loop(dev_config)
    assert loop.perception is None
    loop.run_for(1.5)
    assert loop.error is None
    snap = loop.latest
    assert snap is not None
    assert snap.detections == [] and snap.poses == []
    assert "detector_ms" not in snap.metrics
    assert "poses_per_frame" not in snap.metrics
