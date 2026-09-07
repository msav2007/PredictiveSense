"""The no-op analysis loop: a producer thread and a consumer thread.

The producer drives a :class:`FrameSource` into a :class:`LatestFrameMailbox`.
The consumer samples the mailbox at a configured rate, times each iteration, and
emits a :class:`StateSnapshot`. Both threads shut down cleanly and leave nothing
running. A thread that dies abnormally sets :attr:`AnalysisLoop.error`, which the
caller surfaces as a non-zero exit code.

This module is where ``mode`` awareness would live; the core never sees it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.camera.source import FrameSource, create_frame_source
from predictivesense.config.settings import AppConfig
from predictivesense.core.types import StateSnapshot
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.metrics import MetricRegistry

__all__ = ["AnalysisLoop", "build_loop"]

_LOG = get_logger(__name__)
_JOIN_TIMEOUT_S = 5.0
SnapshotListener = Callable[[StateSnapshot], None]


class AnalysisLoop:
    """Owns two threads and the snapshot cadence. Not reusable after :meth:`stop`."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        mailbox: LatestFrameMailbox,
        *,
        registry: MetricRegistry | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._mailbox = mailbox
        self.metrics = registry or MetricRegistry()

        self._sample_period = 1.0 / config.consumer.sample_rate_hz
        self._stale_after_ms = config.consumer.stale_after_ms

        self._stop_evt = threading.Event()
        self._producer: threading.Thread | None = None
        self._consumer: threading.Thread | None = None
        self._started = False
        self._stopped = False

        self._listeners: list[SnapshotListener] = []
        self._snapshot_id = 0
        self._latest: StateSnapshot | None = None
        self._last_frame_mono: float | None = None
        self._last_iter_ms = 0.0
        self._max_mailbox_depth = 0
        self.error: BaseException | None = None

    # -- wiring --------------------------------------------------------

    def add_snapshot_listener(self, listener: SnapshotListener) -> None:
        self._listeners.append(listener)

    @property
    def latest(self) -> StateSnapshot | None:
        return self._latest

    @property
    def max_mailbox_depth(self) -> int:
        return self._max_mailbox_depth

    # -- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        if self._stopped:
            raise RuntimeError("AnalysisLoop cannot be restarted")
        self._started = True

        _LOG.info(
            "analysis loop starting profile=%s mode=%s sample_rate=%.2f Hz",
            self._config.profile,
            self._config.mode.value,
            self._config.consumer.sample_rate_hz,
        )
        self._source.start()
        self._producer = threading.Thread(
            target=self._run_producer, name="ps-producer", daemon=False
        )
        self._consumer = threading.Thread(
            target=self._run_consumer, name="ps-consumer", daemon=False
        )
        self._producer.start()
        self._consumer.start()

    def stop(self) -> None:
        if not self._started or self._stopped:
            self._stopped = True
            return
        self._stopped = True
        self._stop_evt.set()

        for thread in (self._consumer, self._producer):
            if thread is not None:
                thread.join(timeout=_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    self._record_error(
                        RuntimeError(f"thread {thread.name} did not terminate"),
                        thread.name,
                    )

        try:
            self._source.stop()
        except Exception as exc:  # noqa: BLE001 - surface, do not swallow
            self._record_error(exc, "source.stop")

        stats = self._mailbox.stats()
        _LOG.info(
            "analysis loop stopped snapshots=%d consumed=%d dropped=%d "
            "max_mailbox_depth=%d loop_hz=%.2f producer_hz=%.2f error=%s",
            self._snapshot_id,
            stats.consumed,
            stats.dropped,
            self._max_mailbox_depth,
            self.metrics.rate("loop").hz(),
            self.metrics.rate("producer").hz(),
            self.error,
        )

    def run_for(self, seconds: float) -> None:
        """Start, run for ``seconds`` (or until a thread errors), then stop."""

        self.start()
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                if self.error is not None:
                    break
                self._stop_evt.wait(min(0.2, deadline - time.monotonic()))
        finally:
            self.stop()
        if self.error is not None:
            raise self.error

    def __enter__(self) -> "AnalysisLoop":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- threads --------------------------------------------------

    def _run_producer(self) -> None:
        producer_rate = self.metrics.rate("producer")
        try:
            while not self._stop_evt.is_set():
                frame = self._source.read()
                if frame is None:
                    if not self._source.is_running:
                        break
                    continue
                self._mailbox.put(frame)
                producer_rate.mark()
        except BaseException as exc:  # noqa: BLE001 - set flag, let thread exit
            self._record_error(exc, "producer")

    def _run_consumer(self) -> None:
        next_due = time.monotonic()
        try:
            while not self._stop_evt.is_set():
                now = time.monotonic()
                if now < next_due:
                    self._stop_evt.wait(next_due - now)
                    continue
                next_due += self._sample_period
                if next_due < time.monotonic():
                    next_due = time.monotonic() + self._sample_period
                self._iterate()
        except BaseException as exc:  # noqa: BLE001 - set flag, let thread exit
            self._record_error(exc, "consumer")

    # -- one iteration --------------------------------------------

    def _iterate(self) -> None:
        t0 = time.monotonic()

        frame = self._mailbox.get()
        stats = self._mailbox.stats()
        if stats.depth > self._max_mailbox_depth:
            self._max_mailbox_depth = stats.depth

        now = time.monotonic()
        if frame is not None:
            self._last_frame_mono = now
        self.metrics.rate("loop").mark(now=now)

        stale = self._last_frame_mono is None or (
            (now - self._last_frame_mono) * 1000.0 > self._stale_after_ms
        )

        capture_ts = frame.capture_ts if frame is not None else None
        emitted_ts = time.monotonic()
        frame_age_ms = (
            (emitted_ts - capture_ts) * 1000.0 if capture_ts is not None else None
        )

        iter_latency_ms = (time.monotonic() - t0) * 1000.0
        self._last_iter_ms = iter_latency_ms
        self.metrics.samples("iter_latency_ms").add(iter_latency_ms)

        self._snapshot_id += 1
        snapshot = StateSnapshot(
            snapshot_id=self._snapshot_id,
            mode=self._config.mode,
            frame_id=frame.frame_id if frame is not None else None,
            capture_ts=capture_ts,
            emitted_ts=emitted_ts,
            frame_age_ms=frame_age_ms,
            detections=[],
            poses=[],
            tracks=[],
            risk=None,
            metrics={
                "loop_rate_hz": self.metrics.rate("loop").hz(now=now),
                "producer_rate_hz": self.metrics.rate("producer").hz(now=now),
                "consumed": float(stats.consumed),
                "dropped": float(stats.dropped),
                "mailbox_depth": float(stats.depth),
                "iter_latency_ms": iter_latency_ms,
            },
            stale=stale,
        )
        self._latest = snapshot
        self._emit(snapshot)

    def _emit(self, snapshot: StateSnapshot) -> None:
        for listener in self._listeners:
            try:
                listener(snapshot)
            except Exception as exc:  # noqa: BLE001 - a bad listener must not kill the loop
                _LOG.error("snapshot listener %r failed: %s", listener, exc)

    # -- errors --------------------------------------------------

    def _record_error(self, exc: BaseException, where: str) -> None:
        if self.error is None:
            self.error = exc
        _LOG.error("analysis loop thread %s failed: %r", where, exc)

    # -- reporting ---------------------------------------------

    def summary(self) -> dict[str, Any]:
        stats = self._mailbox.stats()
        latency = self.metrics.samples("iter_latency_ms")
        return {
            "snapshots": self._snapshot_id,
            "consumed": stats.consumed,
            "dropped": stats.dropped,
            "max_mailbox_depth": self._max_mailbox_depth,
            "loop_rate_hz": self.metrics.rate("loop").hz(),
            "producer_rate_hz": self.metrics.rate("producer").hz(),
            "iter_latency_ms_p50": latency.p50,
            "iter_latency_ms_p95": latency.p95,
            "iter_latency_ms_max": latency.max,
            "iter_latency_samples": latency.count,
            "last_iter_latency_ms": self._last_iter_ms,
            "error": repr(self.error) if self.error is not None else None,
        }


def build_loop(
    config: AppConfig, *, registry: MetricRegistry | None = None
) -> AnalysisLoop:
    """Construct a loop with the frame source described by ``config``."""

    source = create_frame_source(config.source)
    mailbox = LatestFrameMailbox()
    return AnalysisLoop(config, source, mailbox, registry=registry)
