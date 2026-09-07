"""Telemetry primitives, CSV writer, and session manifest."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from predictivesense.telemetry.manifest import (
    MANIFEST_REQUIRED_KEYS,
    build_manifest,
    git_state,
)
from predictivesense.telemetry.metrics import (
    Counter,
    MetricRegistry,
    Rate,
    Samples,
    Timer,
)
from predictivesense.telemetry.writer import MetricsWriter

pytestmark = pytest.mark.unit


def test_counter_arithmetic() -> None:
    c = Counter("frames")
    c.inc()
    c.inc(4)
    assert c.value == 5.0
    c.set(10)
    assert c.value == 10.0


def test_rate_windowed_hz_from_injected_timestamps() -> None:
    r = Rate("loop", window_s=10.0)
    for i in range(11):  # 11 events spanning exactly 10 seconds -> 10 intervals
        r.mark(now=100.0 + i)
    assert r.hz(now=110.0) == pytest.approx(1.0, rel=0.01)


def test_rate_is_zero_with_fewer_than_two_events() -> None:
    assert Rate("x").hz(now=1.0) == 0.0


def test_timer_measures_and_feeds_sink() -> None:
    sink = Samples("iter")
    with Timer(sink) as t:
        time.sleep(0.02)
    assert t.elapsed_ms >= 10.0
    assert sink.count == 1
    assert sink.max >= 10.0


def test_samples_percentiles() -> None:
    s = Samples("lat")
    for v in range(1, 101):
        s.add(float(v))
    assert s.count == 100
    assert s.max == 100.0
    assert s.p50 == pytest.approx(50.5, abs=1.0)
    assert s.p95 == pytest.approx(95.05, abs=1.0)


def test_samples_is_bounded() -> None:
    s = Samples("lat", maxlen=100)
    for v in range(1000):
        s.add(float(v))
    assert len(s._values) == 100  # noqa: SLF001 - asserting the cap
    assert s.count == 1000


def test_registry_snapshot_keys() -> None:
    reg = MetricRegistry()
    reg.counter("frames").inc(3)
    reg.rate("loop").mark(now=1.0)
    reg.rate("loop").mark(now=2.0)
    reg.samples("iter").add(1.0)
    snap = reg.as_dict()
    assert snap["frames"] == 3.0
    assert "loop_hz" in snap
    assert "iter_p50" in snap and "iter_p95" in snap


def test_metrics_writer_emits_one_header_and_well_formed_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    fields = ["a", "b", "c"]
    with MetricsWriter(path, fields) as w:
        w.write_row({"a": 1, "b": 2, "c": 3})
        w.write_row({"a": 4, "b": 5, "c": 6, "ignored": 99})
    assert w.rows_written == 2

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == fields
    assert rows[1] == ["1", "2", "3"]
    assert rows[2] == ["4", "5", "6"]
    assert len(rows) == 3


def test_manifest_has_every_required_key_and_is_valid_json(tmp_path: Path, dev_config) -> None:
    manifest = build_manifest(dev_config, session_id="abc123", started_utc="2026-01-01T00:00:00.000000Z")
    assert manifest.ended_utc is None
    manifest.finalize()
    assert manifest.ended_utc is not None

    out = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in MANIFEST_REQUIRED_KEYS:
        assert key in payload, f"missing manifest key: {key}"
    assert isinstance(payload["config"], dict)
    assert payload["config"]["mode"] == "realtime"
    assert payload["seed"] == dev_config.source.seed
    assert isinstance(payload["cpu_count"], int) and payload["cpu_count"] >= 1
    assert isinstance(payload["total_ram_bytes"], int) and payload["total_ram_bytes"] > 0


def test_git_state_returns_typed_pair() -> None:
    commit, dirty = git_state()
    assert isinstance(commit, str) and commit
    assert isinstance(dirty, bool)
