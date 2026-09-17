"""PHASE 11 PART A section 4 - track/pose freshness root-cause measurement.

    python scripts\\track_pose_freshness_audit.py --source data\\raw --live-seconds 40 --label before
    python scripts\\track_pose_freshness_audit.py --source data\\raw --live-seconds 40 --label after

Runs a REAL live real-time session (real ``/ws/ingest`` -> ``/ws/state``, real
mailbox/staleness/broadcast timing - a batch decode has no ingest gaps or
broadcast hop to measure) and publishes the two headline numbers section 4.1
asks for:

    displayed_pose_age_ms   = emitted_ts - the bound pose's own capture_ts
    displayed_track_age_ms  = emitted_ts - the track's last real detector hit

per non-stale snapshot, both computed by the analysis loop itself
(``predictivesense/pipeline/loop.py``) and carried on ``StateSnapshot.metrics``
- no new telemetry system, per the Phase 11 prompt section 22.

These are deliberately NOT ``frame_age_ms`` / ``stage_capture_to_snapshot_ms``
(Phase 8): those measure how old *this frame's own* detection is. A track can
be drawn from an OLDER detection than the current frame (coasting) and a pose
can be drawn from an even older binding (reused across several cycles) -
``displayed_*_age_ms`` measures what is actually on screen, not what was just
computed. Comparing the two locates the fault (section 4.1): if
``displayed_pose_age_ms`` is far larger than ``stage_capture_to_snapshot_ms``,
the delay is in what is kept and drawn, not in what is computed.

Also reports the buffer audit (section 4.4) as empirical bounds - every buffer
in the pipeline is single-slot or a bounded ring by construction (read directly
from the shipped code: ``BrowserSource._slot``, ``LatestFrameMailbox._slot``,
``Broadcaster`` push-only with no per-client queue, the analysis worker's
newest-wins ``encodeBusy``/backpressure drops, the browser ``store``'s single
``snapshot`` field) - this script additionally confirms empirically that
``mailbox_depth`` and ``frames_in_flight`` never exceed their declared bound
over the session.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.track_pose_freshness_audit")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def _load_frames(source: Path, limit: int) -> list[np.ndarray]:
    files = (
        [source]
        if source.is_file()
        else sorted(q for q in source.rglob("*") if q.suffix.lower() in _VIDEO_SUFFIXES)
    )
    imgs: list[np.ndarray] = []
    for f in files:
        cap = cv2.VideoCapture(str(f))
        try:
            while len(imgs) < limit:
                ok, img = cap.read()
                if not ok:
                    break
                imgs.append(np.ascontiguousarray(img))
        finally:
            cap.release()
        if len(imgs) >= limit:
            break
    return imgs


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return -1.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "p50": round(_percentile(values, 0.50), 1),
        "p95": round(_percentile(values, 0.95), 1),
        "max": round(max(values), 1) if values else -1.0,
        "mean": round(statistics.fmean(values), 1) if values else -1.0,
    }


def _run_live_session(cfg, imgs: list[np.ndarray], *, seconds: float) -> dict:
    import json as _json

    from fastapi.testclient import TestClient

    from predictivesense.api.app import create_app
    from predictivesense.camera.framing import encode_ingest_message
    from predictivesense.core.enums import SourceKind
    from predictivesense.core.types import IngestHeader, StateSnapshot

    e2e_cfg = cfg.model_copy(update={
        "source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER}),
    })
    jpegs: list[bytes] = []
    for img in imgs:
        small = cv2.resize(img, (cfg.capture.analysis_width, cfg.capture.analysis_height))
        ok, buf = cv2.imencode(
            ".jpg", small,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.capture.analysis_jpeg_quality * 100)],
        )
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return {"error": "no frames encoded"}

    snapshots_total = 0
    non_stale_total = 0
    displayed_pose_age_samples: list[float] = []
    displayed_track_age_samples: list[float] = []
    capture_to_snapshot_samples: list[float] = []
    tracker_update_ms_samples: list[float] = []
    mailbox_depth_max = 0
    frames_in_flight_max = 0.0
    dropped_stale_last = 0.0
    detector_cycle_ts: list[float] = []  # emitted_ts of every processed (non-stale) snapshot
    fresh_pose_ts: list[float] = []      # emitted_ts of snapshots with a brand-new pose bound

    app = create_app(e2e_cfg)
    period = 1.0 / cfg.capture.analysis_fps
    with TestClient(app) as client:
        stop = threading.Event()

        def _reader():
            nonlocal snapshots_total, non_stale_total, mailbox_depth_max
            nonlocal frames_in_flight_max, dropped_stale_last
            with client.websocket_connect("/ws/state") as sock:
                while not stop.is_set():
                    try:
                        snap = StateSnapshot.from_wire_json(sock.receive_text())
                    except Exception:
                        return
                    snapshots_total += 1
                    if snap.stale:
                        continue
                    non_stale_total += 1
                    m = snap.metrics or {}
                    detector_cycle_ts.append(snap.emitted_ts)
                    dpa = m.get("displayed_pose_age_ms", -1.0)
                    dta = m.get("displayed_track_age_ms", -1.0)
                    if dpa is not None and dpa >= 0.0:
                        displayed_pose_age_samples.append(dpa)
                        # A "fresh" pose binding this cycle is one whose age is
                        # ~0 - i.e. any track was just bound this exact cycle.
                        if any(t.pose_fresh for t in snap.tracks):
                            fresh_pose_ts.append(snap.emitted_ts)
                    if dta is not None and dta >= 0.0:
                        displayed_track_age_samples.append(dta)
                    c2s = m.get("stage_capture_to_snapshot_ms")
                    if isinstance(c2s, (int, float)) and c2s >= 0.0:
                        capture_to_snapshot_samples.append(float(c2s))
                    tum = m.get("tracker_update_ms")
                    if isinstance(tum, (int, float)) and tum >= 0.0:
                        tracker_update_ms_samples.append(float(tum))
                    depth = m.get("mailbox_depth")
                    if isinstance(depth, (int, float)):
                        mailbox_depth_max = max(mailbox_depth_max, int(depth))
                    fif = m.get("frames_in_flight")
                    if isinstance(fif, (int, float)):
                        frames_in_flight_max = max(frames_in_flight_max, float(fif))
                    ds = m.get("dropped_stale")
                    if isinstance(ds, (int, float)):
                        dropped_stale_last = float(ds)

        rt = threading.Thread(target=_reader, daemon=True)
        rt.start()
        with client.websocket_connect("/ws/ingest") as ws:
            ws.send_text(_json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
            ack = _json.loads(ws.receive_text())
            ws.send_text(_json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))
            t_end = time.monotonic() + seconds
            i = 0
            while time.monotonic() < t_end:
                jpeg = jpegs[i % len(jpegs)]
                cap_ms = time.time() * 1000.0
                header = IngestHeader(
                    client_ts_ms=time.time() * 1000.0, seq=i,
                    w=cfg.capture.analysis_width, h=cfg.capture.analysis_height,
                    cap_ts_ms=cap_ms, enc_ms=0.0,
                )
                ws.send_bytes(encode_ingest_message(header, jpeg))
                i += 1
                time.sleep(period)
        time.sleep(0.75)
        stop.set()
        rt.join(timeout=2.0)

    detector_interval_ms = [
        (b - a) * 1000.0 for a, b in zip(detector_cycle_ts, detector_cycle_ts[1:])
    ]
    pose_interval_ms = [
        (b - a) * 1000.0 for a, b in zip(fresh_pose_ts, fresh_pose_ts[1:])
    ]

    return {
        "seconds": seconds,
        "pose_cadence": cfg.perception.pose_cadence,
        "pose_max_reuse_ms": cfg.perception.pose_max_reuse_ms,
        "pose_max_age_ms": cfg.tracking.pose_max_age_ms,
        "snapshots_total": snapshots_total,
        "non_stale_snapshots": non_stale_total,
        "displayed_pose_age_ms": _stats(displayed_pose_age_samples),
        "displayed_track_age_ms": _stats(displayed_track_age_samples),
        "stage_capture_to_snapshot_ms": _stats(capture_to_snapshot_samples),
        "tracker_update_ms": _stats(tracker_update_ms_samples),
        "detector_tracker_observation_interval_ms": _stats(detector_interval_ms),
        "fresh_pose_observation_interval_ms": _stats(pose_interval_ms),
        "buffers": {
            "mailbox_depth_max_observed": mailbox_depth_max,
            "mailbox_depth_declared_bound": 1,
            "frames_in_flight_max_observed": frames_in_flight_max,
            "dropped_stale_total": dropped_stale_last,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 11 Part A track/pose display-freshness audit."
    )
    p.add_argument("--source", default="data/raw")
    p.add_argument("--frames", type=int, default=600, help="max frames loaded for the live loop")
    p.add_argument("--profile", default="dev")
    p.add_argument("--live-seconds", type=float, default=40.0)
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)
    configure_logging("INFO")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        config = load_config(args.profile)
    except (ConfigError, ValidationError) as exc:
        _LOG.error("configuration error: %s", exc)
        return 2

    source = Path(args.source)
    if not source.exists():
        _LOG.error("source not found: %s", source)
        return 2

    imgs = _load_frames(source, args.frames)
    if not imgs:
        _LOG.error("no frames decoded from %s", source)
        return 2
    _LOG.info("loaded %d frames from %s", len(imgs), source)

    label = args.label or re.sub(r"[^A-Za-z0-9_-]", "-", source.name or "clip")
    live = _run_live_session(config, imgs, seconds=args.live_seconds)
    if "error" in live:
        _LOG.error("live session failed: %s", live["error"])
        return 2
    _LOG.info(
        "live: %d snapshots (%d non-stale). displayed_pose_age_ms p50=%.1f p95=%.1f max=%.1f | "
        "displayed_track_age_ms p50=%.1f p95=%.1f max=%.1f | capture->snapshot p50=%.1f",
        live["snapshots_total"], live["non_stale_snapshots"],
        live["displayed_pose_age_ms"]["p50"], live["displayed_pose_age_ms"]["p95"],
        live["displayed_pose_age_ms"]["max"],
        live["displayed_track_age_ms"]["p50"], live["displayed_track_age_ms"]["p95"],
        live["displayed_track_age_ms"]["max"],
        live["stage_capture_to_snapshot_ms"]["p50"],
    )

    payload = {"label": label, "profile": args.profile, "source": str(source), "live_session": live}
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"track_pose_freshness_{label}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"track_pose_freshness_{label}.md").write_text(_markdown(payload), encoding="utf-8")
    _LOG.info("wrote %s", results_dir / f"track_pose_freshness_{label}.md")
    return 0


def _markdown(p: dict) -> str:
    live = p["live_session"]
    dpa = live["displayed_pose_age_ms"]
    dta = live["displayed_track_age_ms"]
    c2s = live["stage_capture_to_snapshot_ms"]
    tum = live["tracker_update_ms"]
    dti = live["detector_tracker_observation_interval_ms"]
    poi = live["fresh_pose_observation_interval_ms"]
    buf = live["buffers"]
    L = [
        f"# Track/pose display-freshness audit (Phase 11 Part A section 4) - `{p['label']}`",
        "",
        f"Live real-time session, {live['seconds']:.0f}s, source `{p['source']}`, profile "
        f"`{p['profile']}`. `pose_cadence`={live['pose_cadence']!r}, "
        f"`pose_max_reuse_ms`={live['pose_max_reuse_ms']}, "
        f"`tracking.pose_max_age_ms`={live['pose_max_age_ms']}.",
        "",
        "| measure | p50 | p95 | max | n |",
        "|---|---|---|---|---|",
        f"| **displayed_pose_age_ms** | {dpa['p50']} | {dpa['p95']} | **{dpa['max']}** | {dpa['n']} |",
        f"| **displayed_track_age_ms** | {dta['p50']} | {dta['p95']} | **{dta['max']}** | {dta['n']} |",
        f"| stage_capture_to_snapshot_ms (Phase 8, for comparison) | {c2s['p50']} | {c2s['p95']} | {c2s['max']} | {c2s['n']} |",
        f"| tracker_update_ms | {tum['p50']} | {tum['p95']} | {tum['max']} | {tum['n']} |",
        f"| detector/tracker observation interval (ms) | {dti['p50']} | {dti['p95']} | {dti['max']} | {dti['n']} |",
        f"| fresh pose observation interval (ms) | {poi['p50']} | {poi['p95']} | {poi['max']} | {poi['n']} |",
        "",
        "## Buffer audit (section 4.4)",
        "",
        f"- `mailbox_depth` observed max: **{buf['mailbox_depth_max_observed']}** "
        f"(declared bound: {buf['mailbox_depth_declared_bound']})",
        f"- `frames_in_flight` observed max: **{buf['frames_in_flight_max_observed']}** "
        "(transient 0..3 by construction - two single-slot buffers plus the producer's "
        "just-read frame; never unbounded, per `pipeline/loop.py::_add_frame_counters`)",
        f"- `dropped_stale` total this session: {buf['dropped_stale_total']}",
        "",
        "## Verdict",
        "",
        (
            f"`displayed_pose_age_ms` p95={dpa['p95']}ms vs "
            f"`stage_capture_to_snapshot_ms` p50={c2s['p50']}ms: "
            + (
                "the displayed pose is materially older than this frame's own capture->snapshot "
                "latency - the delay is in what is KEPT and DRAWN (the pose binding), not in what "
                "is computed."
                if dpa["p95"] > c2s["p50"] * 3
                else "the displayed pose age is within the same order of magnitude as "
                "capture->snapshot latency - no material kept-and-drawn staleness on this session."
            )
        ),
    ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
