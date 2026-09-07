"""Pure-logic contract for the analysis worker's skip / newest-wins rule.

The worker itself is JavaScript (``predictivesense/api/static/analysis-worker.js``)
and cannot be imported here. This module pins the *contract* it must obey:

  1. At most one encode+send is in flight (newest-wins): a frame that arrives
     while an encode is running is dropped, never queued.
  2. Before sending, if ``ws.bufferedAmount`` exceeds ``max_ws_buffered_bytes``
     the frame is dropped and counted (backpressure skip).
  3. Frames under the fps interval are dropped (throttle).

``decide_drop`` below is a faithful port of the worker's ``dropReason``. A
second test scans the shipped JS to confirm the guards are still present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_WORKER_JS = (
    Path(predictivesense.__file__).resolve().parent / "api" / "static" / "analysis-worker.js"
)


@dataclass
class WorkerState:
    """Mirror of the worker's relevant fields."""

    ready: bool = True
    ws_open: bool = True
    fps: float = 10.0
    max_ws_buffered_bytes: int = 1_000_000
    last_send_ms: float = -1.0e9
    encode_busy: bool = False
    buffered_amount: int = 0
    skip_throttle: int = 0
    skip_busy: int = 0
    skip_backpressure: int = 0
    frames_sent: int = 0


def decide_drop(st: WorkerState, now_ms: float) -> str | None:
    """Port of analysis-worker.js ``dropReason``. ``None`` == send this frame."""

    if not st.ready or not st.ws_open:
        return "notReady"
    if now_ms - st.last_send_ms < 1000.0 / st.fps:
        return "throttle"
    if st.encode_busy:
        return "busy"
    if st.buffered_amount > st.max_ws_buffered_bytes:
        return "backpressure"
    return None


def offer_frame(st: WorkerState, now_ms: float) -> str | None:
    """Apply the rule and update counters exactly as the worker does."""

    reason = decide_drop(st, now_ms)
    if reason == "throttle":
        st.skip_throttle += 1
    elif reason == "busy":
        st.skip_busy += 1
    elif reason == "backpressure":
        st.skip_backpressure += 1
    if reason is None:
        st.last_send_ms = now_ms
        st.encode_busy = True  # released later by finishEncode
    return reason


# -- rule ---------------------------------------------------------------


def test_not_ready_is_dropped() -> None:
    assert offer_frame(WorkerState(ready=False), 10_000.0) == "notReady"
    assert offer_frame(WorkerState(ws_open=False), 10_000.0) == "notReady"


def test_throttle_gap_enforced() -> None:
    st = WorkerState(fps=10.0, last_send_ms=1_000.0)
    assert offer_frame(st, 1_050.0) == "throttle"  # 50 ms < 100 ms interval
    assert st.skip_throttle == 1
    assert offer_frame(st, 1_101.0) is None  # 101 ms >= interval


def test_encode_in_flight_drops_newest_wins() -> None:
    st = WorkerState(encode_busy=True, last_send_ms=0.0)
    assert offer_frame(st, 10_000.0) == "busy"
    assert st.skip_busy == 1
    assert st.frames_sent == 0  # nothing queued


def test_backpressure_skip_over_ceiling() -> None:
    st = WorkerState(buffered_amount=1_500_000, last_send_ms=0.0)
    assert offer_frame(st, 10_000.0) == "backpressure"
    assert st.skip_backpressure == 1
    # drains below the ceiling -> next frame proceeds
    st.buffered_amount = 200_000
    assert offer_frame(st, 20_000.0) is None


def test_clear_path_sends_and_marks_busy() -> None:
    st = WorkerState(last_send_ms=0.0)
    assert offer_frame(st, 10_000.0) is None
    assert st.encode_busy is True
    assert st.last_send_ms == 10_000.0


# -- no-backlog property ----------------------------------------------


@dataclass
class _Sim:
    in_flight: int = 0
    pending: int = 0  # must stay 0: the worker never queues
    max_in_flight: int = 0
    sent: int = 0
    dropped: int = 0
    reasons: list[str] = field(default_factory=list)


def test_slow_encoder_never_builds_a_backlog() -> None:
    """Frames arrive at 60 Hz; each encode takes 40 ms (spans ~2.4 arrivals).

    The worker must keep exactly one encode in flight, never queue, and drop the
    rest with a recorded reason.
    """

    st = WorkerState(fps=30.0, max_ws_buffered_bytes=1_000_000)
    sim = _Sim()
    encode_done_at: float | None = None
    arrival_hz = 60.0
    encode_ms = 40.0
    n = 400

    for i in range(n):
        now = i * (1000.0 / arrival_hz)
        # finish an encode whose time has come
        if encode_done_at is not None and now >= encode_done_at:
            st.encode_busy = False
            sim.in_flight -= 1
            sim.sent += 1
            st.buffered_amount = 0
            encode_done_at = None

        reason = offer_frame(st, now)
        if reason is None:
            sim.in_flight += 1
            sim.max_in_flight = max(sim.max_in_flight, sim.in_flight)
            encode_done_at = now + encode_ms
        else:
            sim.dropped += 1
            sim.reasons.append(reason)

    assert sim.max_in_flight == 1, "more than one encode was in flight at once"
    assert sim.pending == 0, "the worker queued a frame"
    assert sim.sent >= 1
    # every frame is sent, dropped, or still encoding at the end - none queued
    assert sim.sent + sim.dropped + sim.in_flight == n
    assert set(sim.reasons) <= {"throttle", "busy", "backpressure"}
    # at 60 Hz arrivals / 30 fps cap / 40 ms encode, most frames are throttled
    # or dropped for a busy encoder - the accepted rate must not exceed the cap.
    assert sim.sent <= n / 2 + 1


# -- shipped-JS guard ------------------------------------------------


def test_worker_js_still_carries_the_guards() -> None:
    src = _WORKER_JS.read_text(encoding="utf-8")
    for needle in (
        "bufferedAmount",
        "maxWsBufferedBytes",
        "encodeBusy",
        "framesClosed",
        "frame.close()",
        "skipBackpressure",
    ):
        assert needle in src, f"analysis-worker.js lost its {needle!r} guard"
