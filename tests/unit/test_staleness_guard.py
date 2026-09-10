"""Phase 8 section 8 - staleness guard + frame-counter reconciliation.

Unit-level: the guard and the counters are exercised by calling
``AnalysisLoop._iterate`` directly with hand-built frames of controlled age -
no threads, no real perception, no camera.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.camera.source import create_frame_source
from predictivesense.config.settings import load_config
from predictivesense.core.types import Frame
from predictivesense.pipeline.loop import AnalysisLoop
from predictivesense.telemetry.metrics import MetricRegistry

pytestmark = pytest.mark.unit


def _loop(max_frame_age_ms: float) -> AnalysisLoop:
    cfg = load_config("dev")
    cfg = cfg.model_copy(update={
        "perception": cfg.perception.model_copy(
            update={"detection_enabled": False, "pose_enabled": False}
        ),
        "analysis": cfg.analysis.model_copy(
            update={"max_frame_age_ms": max_frame_age_ms}
        ),
    })
    src = create_frame_source(cfg.source)
    return AnalysisLoop(cfg, src, LatestFrameMailbox(), registry=MetricRegistry())


def _frame(fid: int, age_ms: float) -> Frame:
    return Frame(
        frame_id=fid, capture_ts=time.monotonic() - age_ms / 1000.0,
        image=np.zeros((4, 4, 3), np.uint8), width=4, height=4,
        source_id="t", seq=fid,
    )


def test_frames_older_than_the_guard_are_dropped_and_counted_stale() -> None:
    loop = _loop(max_frame_age_ms=200.0)
    loop._iterate(_frame(0, age_ms=50.0))    # fresh -> analysed
    loop._iterate(_frame(1, age_ms=350.0))   # stale -> dropped
    loop._iterate(_frame(2, age_ms=10.0))    # fresh -> analysed
    snap = loop.latest
    assert snap is not None
    assert snap.metrics["dropped_stale"] == 1.0
    # the stale iteration still emitted a snapshot (frame_id None -> stale=True)
    assert loop._dropped_stale == 1


def test_guard_boundary_is_exact() -> None:
    loop = _loop(max_frame_age_ms=100.0)
    # age just under the bound passes; just over is dropped
    loop._iterate(_frame(0, age_ms=95.0))
    assert loop._dropped_stale == 0
    loop._iterate(_frame(1, age_ms=105.0))
    assert loop._dropped_stale == 1


def test_guard_disabled_at_zero_never_drops() -> None:
    loop = _loop(max_frame_age_ms=0.0)
    for i in range(5):
        loop._iterate(_frame(i, age_ms=5000.0))  # ancient, but guard is off
    assert loop._dropped_stale == 0


def test_counter_reconciliation_over_a_scripted_sequence() -> None:
    """captured == analysed + dropped_overwrite + dropped_stale + in_flight,
    exactly, at a quiescent point - driven through the real mailbox by hand."""

    loop = _loop(max_frame_age_ms=200.0)
    mbox = loop.mailbox

    captured = 0
    # 3 delivered and analysed
    for i in range(3):
        mbox.put(_frame(i, age_ms=10.0)); captured += 1
        loop._iterate(mbox.get())
    # 1 overwritten in the mailbox (put twice before a get)
    mbox.put(_frame(100, age_ms=10.0)); captured += 1
    mbox.put(_frame(101, age_ms=10.0)); captured += 1   # overwrites 100
    loop._iterate(mbox.get())                            # analyses 101
    # 1 stale-dropped
    mbox.put(_frame(200, age_ms=500.0)); captured += 1
    loop._iterate(mbox.get())                            # dropped_stale
    # 1 left sitting in the mailbox (in flight)
    mbox.put(_frame(300, age_ms=10.0)); captured += 1

    stats = mbox.stats()
    analysed = stats.consumed - loop._dropped_stale
    dropped_overwrite = stats.dropped
    dropped_stale = loop._dropped_stale
    in_flight = 1 if stats.depth else 0

    assert captured == analysed + dropped_overwrite + dropped_stale + in_flight
    assert analysed == 4          # frames 0,1,2,101
    assert dropped_overwrite == 1  # frame 100
    assert dropped_stale == 1      # frame 200
    assert in_flight == 1          # frame 300
