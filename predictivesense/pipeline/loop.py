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
from predictivesense.config.settings import AppConfig, ConfigError
from predictivesense.core.enums import SourceKind
from predictivesense.core.types import StateSnapshot
from predictivesense.logging_setup import get_logger
from predictivesense.telemetry.metrics import MetricRegistry

__all__ = ["AnalysisLoop", "build_loop", "build_camera_source"]

_LOG = get_logger(__name__)
_JOIN_TIMEOUT_S = 5.0
_MAX_STALL_S = 10.0  # debug consumer stall ceiling (preview-independence check)
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
        self._stall_until = 0.0  # time.monotonic() deadline; 0 = not stalled
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

    @property
    def source(self) -> FrameSource:
        return self._source

    @property
    def mailbox(self) -> LatestFrameMailbox:
        return self._mailbox

    def threads_alive(self) -> dict[str, bool]:
        return {
            "producer": bool(self._producer and self._producer.is_alive()),
            "consumer": bool(self._consumer and self._consumer.is_alive()),
        }

    def request_consumer_stall(self, seconds: float) -> float:
        """Artificially stall the consumer for up to 10 s (preview-independence).

        Documented hook for Block 13 check 3: the producer, the ingest socket
        and the browser preview keep running while the consumer pauses; mailbox
        drops climb and depth stays <= 1. Returns the clamped seconds applied.
        """

        applied = max(0.0, min(float(seconds), _MAX_STALL_S))
        self._stall_until = time.monotonic() + applied
        _LOG.warning("consumer stall requested: %.2fs (debug / preview-independence)", applied)
        return applied

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
                if now < self._stall_until:
                    # Debug stall: do not consume. The producer + ingest socket
                    # keep running; the mailbox fills and drops, depth stays <=1.
                    self._stop_evt.wait(min(self._stall_until - now, 0.2))
                    next_due = time.monotonic()
                    continue
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
        # perf_counter (monotonic, ~100 ns) times the iteration; monotonic's
        # 15.6 ms tick on this Windows build cannot resolve it. time.time is
        # never used for any duration.
        t0 = time.perf_counter()

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

        iter_latency_ms = (time.perf_counter() - t0) * 1000.0
        self._last_iter_ms = iter_latency_ms
        self.metrics.samples("iter_latency_ms").add(iter_latency_ms)

        loop_hz = self.metrics.rate("loop").hz(now=now)
        producer_hz = self.metrics.rate("producer").hz(now=now)
        total_seen = stats.consumed + stats.dropped
        metrics: dict[str, float] = {
            "loop_rate_hz": loop_hz,
            "producer_rate_hz": producer_hz,
            "consumed": float(stats.consumed),
            "dropped": float(stats.dropped),
            "mailbox_depth": float(stats.depth),
            "iter_latency_ms": iter_latency_ms,
            # Phase 1 keys (additive; no reshape).
            "capture_fps": producer_hz,
            "analysis_fps": loop_hz,
            "dropped_analysis_frames": float(stats.dropped),
            "drop_rate": (stats.dropped / total_seen) if total_seen else 0.0,
            "frame_age_ms": frame_age_ms if frame_age_ms is not None else -1.0,
        }
        self._merge_source_metrics(metrics)

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
            metrics=metrics,
            stale=stale,
        )
        self._latest = snapshot
        self._emit(snapshot)

    def _merge_source_metrics(self, metrics: dict[str, float]) -> None:
        """Fold source-specific counters into the snapshot metrics. Never raises.

        Only cheap last-value / counter reads - no percentile work on this hot
        path (Block 3.8 / Requirement 23).
        """

        try:
            info = self._source.info()
        except Exception as exc:  # noqa: BLE001 - a bad source must not kill the loop
            _LOG.debug("source.info() failed on hot path: %r", exc)
            return
        mapping = {
            "decode_ms": "decode_ms_last",
            "ingest_bytes_per_s": "ingest_bytes_per_s",
            "reconnects": "reconnects",
            "clock_offset_rtt_ms": "clock_offset_rtt_ms",
        }
        for out_key, in_key in mapping.items():
            value = info.get(in_key)
            if isinstance(value, (int, float)):
                metrics[out_key] = float(value)
        # Count this source's own newest-wins drops toward the analysis-drop total.
        buffer_dropped = info.get("buffer_dropped")
        if isinstance(buffer_dropped, (int, float)):
            metrics["dropped_analysis_frames"] += float(buffer_dropped)

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


def build_camera_source(config: AppConfig) -> FrameSource:
    """Construct the live frame source for ``config.source.kind``.

    ``SYNTHETIC`` -> the Phase 0 synthetic source (used by ``run_noop`` and the
    Phase 0 tests). ``BROWSER`` -> a :class:`BrowserSource` fed by the ingest
    socket (requires ``capture.owner == "browser"``). ``DEVICE`` -> a
    backend-owned :class:`DeviceSource` (requires ``capture.owner == "backend"``).
    ``FILE`` is rejected here - recorded video runs through ``RecordedDriver``,
    not the live loop.
    """

    kind = SourceKind(config.source.kind)
    if kind is SourceKind.SYNTHETIC:
        return create_frame_source(config.source)

    if kind is SourceKind.BROWSER:
        if config.capture.owner != "browser":
            raise ConfigError(
                "source.kind=browser requires capture.owner=browser "
                f"(got {config.capture.owner!r})"
            )
        from predictivesense.camera.browser import BrowserSource

        return BrowserSource(
            analysis_width=config.capture.analysis_width,
            analysis_height=config.capture.analysis_height,
        )

    if kind is SourceKind.DEVICE:
        if config.capture.owner != "backend":
            raise ConfigError(
                "source.kind=device requires capture.owner=backend "
                f"(got {config.capture.owner!r})"
            )
        from predictivesense.camera.device import DeviceSource

        return DeviceSource(
            config.capture.device_index,
            backend=config.capture.device_backend,
            request_width=config.capture.request_width,
            request_height=config.capture.request_height,
            request_fps=config.capture.request_fps,
            fourcc=config.capture.fourcc,
            buffer_size=config.capture.buffer_size,
            warmup_frames=config.capture.warmup_frames,
            open_timeout_s=config.capture.open_timeout_s,
            reconnect_initial_s=config.capture.reconnect_initial_s,
            reconnect_max_s=config.capture.reconnect_max_s,
        )

    if kind is SourceKind.FILE:
        raise ConfigError(
            "source.kind=file has no live loop; analyse recorded video with "
            "RecordedDriver (scripts/run_recorded.py or POST /api/analyze)."
        )

    # WEBRTC and any future kind.
    return create_frame_source(config.source)


def build_loop(
    config: AppConfig, *, registry: MetricRegistry | None = None
) -> AnalysisLoop:
    """Construct a loop with the frame source described by ``config``."""

    source = build_camera_source(config)
    mailbox = LatestFrameMailbox()
    return AnalysisLoop(config, source, mailbox, registry=registry)
