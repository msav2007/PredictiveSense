"""Small, bounded telemetry primitives.

Nothing here grows without a cap: :class:`Rate` and :class:`Samples` are backed
by ``collections.deque`` with a fixed ``maxlen``.
"""

from __future__ import annotations

import threading
import time
from collections import deque

__all__ = ["Counter", "Rate", "Samples", "Timer", "MetricRegistry"]

_DEFAULT_RATE_WINDOW_S = 5.0
_DEFAULT_RATE_MAXLEN = 4096
_DEFAULT_SAMPLES_MAXLEN = 100_000


class Counter:
    """A monotonically increasing (by default) float counter."""

    def __init__(self, name: str = "counter") -> None:
        self.name = name
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Rate:
    """Sliding-window event rate in hertz. Backed by a bounded deque."""

    def __init__(
        self,
        name: str = "rate",
        *,
        window_s: float = _DEFAULT_RATE_WINDOW_S,
        maxlen: int = _DEFAULT_RATE_MAXLEN,
    ) -> None:
        self.name = name
        self.window_s = float(window_s)
        self._events: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def mark(self, count: int = 1, *, now: float | None = None) -> None:
        ts = time.monotonic() if now is None else now
        with self._lock:
            for _ in range(max(count, 0)):
                self._events.append(ts)

    def hz(self, *, now: float | None = None) -> float:
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_s
        with self._lock:
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            n = len(self._events)
            if n < 2:
                return 0.0
            span = self._events[-1] - self._events[0]
        return (n - 1) / span if span > 0 else 0.0

    @property
    def value(self) -> float:
        return self.hz()


class Samples:
    """A bounded reservoir of float observations with percentile access."""

    def __init__(
        self, name: str = "samples", *, maxlen: int = _DEFAULT_SAMPLES_MAXLEN
    ) -> None:
        self.name = name
        self.maxlen = maxlen
        self._values: deque[float] = deque(maxlen=maxlen)
        self._count = 0
        self._max = float("-inf")
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._values.append(float(value))
            self._count += 1
            if value > self._max:
                self._max = float(value)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def max(self) -> float:
        with self._lock:
            return self._max if self._count else 0.0

    def percentile(self, pct: float) -> float:
        if not 0.0 <= pct <= 100.0:
            raise ValueError("pct must be within [0, 100]")
        with self._lock:
            data = sorted(self._values)
        if not data:
            return 0.0
        if len(data) == 1:
            return data[0]
        rank = (pct / 100.0) * (len(data) - 1)
        low = int(rank)
        frac = rank - low
        if low + 1 >= len(data):
            return data[-1]
        return data[low] + frac * (data[low + 1] - data[low])

    @property
    def p50(self) -> float:
        return self.percentile(50.0)

    @property
    def p95(self) -> float:
        return self.percentile(95.0)


class Timer:
    """Context manager measuring elapsed duration with ``time.monotonic``.

    Optionally records the elapsed milliseconds into a :class:`Samples` on exit.
    """

    def __init__(self, sink: Samples | None = None) -> None:
        self._sink = sink
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed_ms = (time.monotonic() - self._start) * 1000.0
        if self._sink is not None:
            self._sink.add(self.elapsed_ms)


class MetricRegistry:
    """A per-run bag of named metrics. Not a singleton; created and injected."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._rates: dict[str, Rate] = {}
        self._samples: dict[str, Samples] = {}
        self._lock = threading.Lock()

    def counter(self, name: str) -> Counter:
        with self._lock:
            return self._counters.setdefault(name, Counter(name))

    def rate(self, name: str, *, window_s: float = _DEFAULT_RATE_WINDOW_S) -> Rate:
        with self._lock:
            return self._rates.setdefault(name, Rate(name, window_s=window_s))

    def samples(
        self, name: str, *, maxlen: int = _DEFAULT_SAMPLES_MAXLEN
    ) -> Samples:
        with self._lock:
            return self._samples.setdefault(name, Samples(name, maxlen=maxlen))

    def as_dict(self) -> dict[str, float]:
        """Flat snapshot: counters by value, rates by hz, samples by p50/p95."""

        out: dict[str, float] = {}
        with self._lock:
            counters = list(self._counters.items())
            rates = list(self._rates.items())
            samples = list(self._samples.items())
        for name, c in counters:
            out[name] = c.value
        for name, r in rates:
            out[f"{name}_hz"] = r.hz()
        for name, s in samples:
            out[f"{name}_p50"] = s.p50
            out[f"{name}_p95"] = s.p95
        return out
