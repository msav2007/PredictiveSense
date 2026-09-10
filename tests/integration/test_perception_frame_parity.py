"""Phase 8 - PerceptionFrame is emitted with an identical structure by the
real-time loop and the recorded driver, and recorded mode stays deterministic
with a non-default pose cadence configured (section 19 tests 6 and 9).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.core.types import Frame, PerceptionFrame
from predictivesense.perception.engine import build_perception
from predictivesense.pipeline.loop import AnalysisLoop
from predictivesense.pipeline.recorded import RecordedDriver
from predictivesense.telemetry.metrics import MetricRegistry

pytestmark = [pytest.mark.integration, pytest.mark.models]


@pytest.fixture()
def cadence_config(perception_dev_config):
    return perception_dev_config.model_copy(update={
        "perception": perception_dev_config.perception.model_copy(
            update={"pose_cadence": "every_n:2"}
        ),
        # staleness guard off: this test hand-feeds frames and only checks the
        # PerceptionFrame shape, not frame age.
        "analysis": perception_dev_config.analysis.model_copy(
            update={"max_frame_age_ms": 0.0}
        ),
    })


def _frames_from(video: Path, limit: int = 20) -> list[Frame]:
    cap = cv2.VideoCapture(str(video))
    out: list[Frame] = []
    i = 0
    while len(out) < limit:
        ok, img = cap.read()
        if not ok:
            break
        img = np.ascontiguousarray(img)
        out.append(Frame(frame_id=i, capture_ts=i / 30.0, image=img,
                         width=img.shape[1], height=img.shape[0],
                         source_id="parity", seq=i))
        i += 1
    cap.release()
    return out


def test_realtime_and_recorded_emit_the_same_perception_frame_shape(
    require_models, cadence_config, fixture_video, tmp_path
) -> None:
    engine = build_perception(cadence_config, strict=True, warmup=False)

    # recorded path -> sidecar
    res = RecordedDriver(results_dir=tmp_path).run(
        fixture_video, replay_mode="asfast", run_id="pf", perception=engine
    )
    sidecar = Path(res["jsonl_path"]).with_name("recorded_pf.frames.jsonl")
    assert sidecar.is_file(), "recorded driver did not write the PerceptionFrame sidecar"
    rec_lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert rec_lines
    rec_frames = [PerceptionFrame.model_validate_json(x) for x in rec_lines]
    rec_keys = set(json.loads(rec_lines[0]).keys())

    # real-time path -> loop.perception_frames(); drive _iterate by hand, no threads
    engine2 = build_perception(cadence_config, strict=True, warmup=False)
    loop = AnalysisLoop(
        cadence_config,
        source=_DummySource(),
        mailbox=LatestFrameMailbox(),
        registry=MetricRegistry(),
        perception=engine2,
    )
    for fr in _frames_from(fixture_video, limit=len(rec_frames)):
        loop._iterate(fr)
    rt_frames = loop.perception_frames()
    assert rt_frames
    rt_keys = set(rt_frames[0].model_dump().keys())

    assert rt_keys == rec_keys, (rt_keys ^ rec_keys)
    # both are PerceptionFrame instances with the required identity fields
    for f in (*rec_frames[:1], *rt_frames[:1]):
        assert isinstance(f, PerceptionFrame)
        assert f.model_version and f.provider
        assert f.seq is not None and f.capture_ts is not None
        assert f.width > 0 and f.height > 0


def test_recorded_sidecar_is_byte_identical_across_runs_with_cadence(
    require_models, cadence_config, fixture_video, tmp_path
) -> None:
    engine = build_perception(cadence_config, strict=True, warmup=False)
    r1 = RecordedDriver(results_dir=tmp_path / "a").run(
        fixture_video, replay_mode="asfast", run_id="c", perception=engine
    )
    r2 = RecordedDriver(results_dir=tmp_path / "b").run(
        fixture_video, replay_mode="asfast", run_id="c", perception=engine
    )
    s1 = Path(r1["jsonl_path"]).with_name("recorded_c.frames.jsonl")
    s2 = Path(r2["jsonl_path"]).with_name("recorded_c.frames.jsonl")
    assert s1.read_bytes() == s2.read_bytes()
    # and the main JSONL is still byte-identical too
    assert Path(r1["jsonl_path"]).read_bytes() == Path(r2["jsonl_path"]).read_bytes()
    # every line parses back to a PerceptionFrame with the pose-cadence fields
    rows = [json.loads(x) for x in s1.read_text(encoding="utf-8").splitlines()]
    assert rows and all(
        {"pose_stale", "pose_frame_id", "pose_age_ms"} <= set(r) for r in rows
    )


class _DummySource:
    """Minimal FrameSource stand-in: the parity test drives _iterate directly."""

    def read(self):  # pragma: no cover - never called in this test
        return None

    def start(self):  # pragma: no cover
        pass

    def stop(self):  # pragma: no cover
        pass

    @property
    def is_running(self):  # pragma: no cover
        return False

    def info(self):
        return {}
