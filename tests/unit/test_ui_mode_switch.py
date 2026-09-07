"""Mode switching tears capture down cleanly (Phase 1.6, Requirement 3.4 #12).

There is no headless browser here (no new dependency), so this pins the teardown
*chain* in the shipped JS by static analysis: leaving Real-time must stop every
track, drop srcObject, terminate the worker, and let the ingest socket close -
and app.js must invoke that path on the mode change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import predictivesense

pytestmark = pytest.mark.unit

_STATIC = Path(predictivesense.__file__).resolve().parent / "api" / "static"


def _read(rel: str) -> str:
    return (_STATIC / rel).read_text(encoding="utf-8")


def test_release_capture_stops_tracks_and_clears_srcobject() -> None:
    src = _read("features/camera-capture.js")
    assert "export function releaseCapture" in src
    # stopCurrentStream: stop every track, null srcObject, signal the worker down
    assert "getTracks().forEach((t) => t.stop())" in src
    assert "video.srcObject = null" in src
    assert 'emit("stream-stopped")' in src


def test_worker_is_terminated_on_stream_stopped() -> None:
    src = _read("features/analysis-client.js")
    assert 'on("stream-stopped", stopAnalysis)' in src
    assert "runtime.worker.terminate()" in src
    assert "runtime.worker = null" in src


def test_app_releases_capture_when_leaving_realtime() -> None:
    app = _read("app.js")
    assert 'state.mode === "recorded"' in app
    assert "releaseCapture()" in app
    assert "resumeCapture()" in app
    # the local-file object URL is revoked when leaving recorded mode
    assert "clearRecordedViewport()" in app
    vids = _read("features/videos.js")
    assert "URL.revokeObjectURL" in vids
