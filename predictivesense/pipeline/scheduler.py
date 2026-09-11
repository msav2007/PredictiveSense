"""Consumer scheduling strategies for the analysis loop (Phase 8, section 7).

Two strategies, chosen by ``consumer.scheduler``:

* ``timer`` (the pre-Phase-8 behaviour) - wake at ``consumer.sample_rate_hz`` and
  process whatever is in the mailbox. When inference is slower than the tick the
  loop runs late and reads a frame that was already old when it started.
* ``completion`` - block on the mailbox until the producer delivers a frame,
  process it, repeat. A ``max_analysis_rate_hz`` ceiling stops the loop pegging
  the CPU when frames and inference are both cheap. **No busy waiting**: the idle
  wait is ``threading.Condition.wait`` inside ``LatestFrameMailbox.get`` and the
  ceiling / stop / pause waits are ``Event.wait``.

The completion loop is a free function so it can be unit-tested with fakes for
the mailbox, the "run one iteration" callable and the control events (section 19
unit test 1) without constructing an :class:`~predictivesense.pipeline.loop.AnalysisLoop`.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.core.types import Frame

__all__ = ["run_completion_consumer", "IDLE_POLL_TIMEOUT_S"]

# When the mailbox is empty the consumer still wakes this often so a stop /
# pause / stall request is honoured promptly and a frameless (stale) snapshot is
# still emitted while no frames arrive. It is a wake-up bound, not a poll: a
# delivered frame releases the Condition immediately.
IDLE_POLL_TIMEOUT_S = 0.2


def run_completion_consumer(
    *,
    mailbox: LatestFrameMailbox,
    iterate_once: Callable[[Frame | None], None],
    should_stop: Callable[[], bool],
    stop_wait: Callable[[float], bool],
    is_paused: Callable[[], bool],
    stall_until: Callable[[], float],
    min_iter_period_s: float,
    now: Callable[[], float] = time.monotonic,
    idle_poll_timeout_s: float = IDLE_POLL_TIMEOUT_S,
) -> None:
    """Completion-driven consumer loop.

    ``iterate_once(frame)`` runs one analysis iteration (``frame`` may be
    ``None`` - it still emits a snapshot so ``/ws/state`` sees ``stale``).
    ``stop_wait(seconds)`` is the interruptible sleep (``Event.wait``);
    ``stall_until()`` returns the ``monotonic`` deadline of any active debug
    stall (0 = none).
    """

    last_iter_start = 0.0
    while not should_stop():
        if is_paused():
            stop_wait(0.05)
            last_iter_start = 0.0
            continue
        t = now()
        deadline = stall_until()
        if t < deadline:
            stop_wait(min(deadline - t, idle_poll_timeout_s))
            last_iter_start = 0.0
            continue
        # Rate ceiling: never start iterations faster than max_analysis_rate_hz.
        gap = t - last_iter_start
        if gap < min_iter_period_s:
            stop_wait(min_iter_period_s - gap)
            continue
        # Block until the producer delivers a frame (Condition.wait - not a
        # spin). A timeout returns None so stop / pause / stall stay responsive
        # and a stale snapshot is still emitted while the scene is quiet.
        frame = mailbox.get(block=True, timeout=idle_poll_timeout_s)
        if frame is not None:
            last_iter_start = now()
        iterate_once(frame)
