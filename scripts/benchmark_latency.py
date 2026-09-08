"""Bounded-latency measurement: pose gating, thread count, policy cost, and
(Phase 5) end-to-end capture->overlay latency.

    python scripts\\benchmark_latency.py [--source data\\raw] [--threads 4,6,8]

Runs the real PerceptionEngine + RecognitionPolicy over decoded frames of the
developer's own footage and reports, before vs after each bounded change:

* detector / pose / combined per-frame latency p50 / p95 (BLOCK 11);
* pose-gating effect - pose off on no-person frames - on combined cost and the
  implied analysis FPS (1000 / combined p50);
* the recognition-policy cost per frame, asserted < 1 ms p95 (BLOCK 11);
* a small intra-op thread sweep on this 14-core machine.

Phase 5 adds an **end-to-end capture->snapshot** loopback: the real FastAPI app
with the browser-ingest source + perception + policy, frames pushed over
``/ws/ingest`` stamped with a real capture clock and the emitted
``StateSnapshot.frame_age_ms`` collected from ``/ws/state``. This is "what the
developer experiences as slow" minus only the browser's own capture-encode and
final paint (a live-camera Block 13 check). It also reports detector ms, pose
ms, analysis FPS, drop rate, frame age, CPU% and RSS.

Writes ``results/latency_2_5.{json,md}``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.engine import PerceptionEngine
from predictivesense.perception.policy import RecognitionPolicy

_LOG = get_logger("predictivesense.scripts.benchmark_latency")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _pct(xs, p):
    return round(float(np.percentile(xs, p)), 2) if xs else -1.0


def _frames(source: Path, limit: int):
    files = (
        [source] if source.is_file() else
        sorted(q for q in source.rglob("*") if q.suffix.lower() in (_VIDEO_SUFFIXES | _IMG_SUFFIXES))
    )
    n = 0
    for f in files:
        if f.suffix.lower() in _IMG_SUFFIXES:
            img = cv2.imread(str(f))
            if img is not None:
                n += 1
                yield np.ascontiguousarray(img)
            continue
        cap = cv2.VideoCapture(str(f))
        try:
            while n < limit:
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                yield np.ascontiguousarray(img)
        finally:
            cap.release()
        if n >= limit:
            return


def _run(engine: PerceptionEngine, policy: RecognitionPolicy, imgs, *, resize_to):
    det_ms, pose_ms, pol_ms, comb_ms = [], [], [], []
    pose_ran = 0
    for i, img in enumerate(imgs):
        if resize_to:
            img = cv2.resize(img, resize_to)
        fr = Frame(frame_id=i, capture_ts=float(i), image=np.ascontiguousarray(img),
                   width=img.shape[1], height=img.shape[0], source_id="bench", seq=i)
        r = engine.infer(fr)
        p0 = time.perf_counter()
        policy.apply(r.detections, frame_width=fr.width, frame_height=fr.height)
        pol = (time.perf_counter() - p0) * 1000.0
        if r.detector_ms is not None:
            det_ms.append(r.detector_ms)
        if r.pose_ms is not None:
            pose_ms.append(r.pose_ms)
            pose_ran += 1
        pol_ms.append(pol)
        comb_ms.append((r.detector_ms or 0.0) + (r.pose_ms or 0.0) + pol)
    return {
        "frames": len(comb_ms),
        "pose_ran": pose_ran,
        "detector_p50": _pct(det_ms, 50), "detector_p95": _pct(det_ms, 95),
        "pose_p50": _pct(pose_ms, 50), "pose_p95": _pct(pose_ms, 95),
        "policy_p50": _pct(pol_ms, 50), "policy_p95": _pct(pol_ms, 95),
        "combined_p50": _pct(comb_ms, 50), "combined_p95": _pct(comb_ms, 95),
        "implied_fps": round(1000.0 / _pct(comb_ms, 50), 2) if comb_ms and _pct(comb_ms, 50) > 0 else -1.0,
    }


def _engine(cfg, *, pose_requires_person, threads):
    pc = cfg.perception.model_copy(update={
        "pose_requires_person": pose_requires_person,
        "intra_op_threads": threads,
    })
    return PerceptionEngine(pc, warmup=True)


def _end_to_end(cfg, imgs, *, seconds: float):
    """Real app + browser-ingest source + perception + policy. Push frames over
    /ws/ingest at analysis_fps with a real capture clock; collect the emitted
    frame_age_ms and metrics from /ws/state. Returns the capture->snapshot
    latency distribution (the browser adds only its own encode + one paint)."""

    import json
    import threading

    import psutil
    from fastapi.testclient import TestClient

    from predictivesense.api.app import create_app
    from predictivesense.camera.framing import encode_ingest_message
    from predictivesense.core.enums import SourceKind
    from predictivesense.core.types import IngestHeader, StateSnapshot

    e2e_cfg = cfg.model_copy(update={
        "source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER}),
    })
    jpegs = []
    for img in imgs:
        small = cv2.resize(img, (cfg.capture.analysis_width, cfg.capture.analysis_height))
        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                int(cfg.capture.analysis_jpeg_quality * 100)])
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return None

    ages: list[float] = []
    det_ms: list[float] = []
    pose_ms: list[float] = []
    fps: list[float] = []
    drop: list[float] = []
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6
    proc.cpu_percent(None)

    app = create_app(e2e_cfg)
    period = 1.0 / cfg.capture.analysis_fps
    with TestClient(app) as client:
        stop = threading.Event()

        def _reader():
            with client.websocket_connect("/ws/state") as sock:
                while not stop.is_set():
                    try:
                        snap = StateSnapshot.from_wire_json(sock.receive_text())
                    except Exception:
                        return
                    if snap.frame_age_ms is not None and not snap.stale:
                        ages.append(float(snap.frame_age_ms))
                        m = snap.metrics
                        if m.get("detector_ms", -1) >= 0:
                            det_ms.append(m["detector_ms"])
                        if m.get("pose_ms", -1) >= 0:
                            pose_ms.append(m["pose_ms"])
                        if m.get("analysis_fps", 0) > 0:
                            fps.append(m["analysis_fps"])
                        drop.append(m.get("drop_rate", 0.0))

        rt = threading.Thread(target=_reader, daemon=True)
        rt.start()
        with client.websocket_connect("/ws/ingest") as ws:
            ws.send_text(json.dumps({"type": "hello", "client_ts_ms": time.time() * 1000.0}))
            ack = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "echo", "rtt_probe": ack["rtt_probe"]}))
            t_end = time.monotonic() + seconds
            i = 0
            while time.monotonic() < t_end:
                jpeg = jpegs[i % len(jpegs)]
                header = IngestHeader(
                    client_ts_ms=time.time() * 1000.0, seq=i,
                    w=cfg.capture.analysis_width, h=cfg.capture.analysis_height,
                )
                ws.send_bytes(encode_ingest_message(header, jpeg))
                i += 1
                time.sleep(period)
        time.sleep(0.5)
        stop.set()
        rt.join(timeout=2.0)

    cpu = proc.cpu_percent(None)
    rss1 = proc.memory_info().rss / 1e6
    return {
        "samples": len(ages),
        "frame_age_ms_p50": _pct(ages, 50),
        "frame_age_ms_p95": _pct(ages, 95),
        "frame_age_ms_max": round(max(ages), 2) if ages else -1.0,
        "detector_ms_p50": _pct(det_ms, 50),
        "pose_ms_p50": _pct(pose_ms, 50),
        "analysis_fps_p50": _pct(fps, 50),
        "drop_rate_mean": round(float(np.mean(drop)), 4) if drop else -1.0,
        "process_cpu_percent": round(cpu, 1),
        "rss_mb_before": round(rss0, 1),
        "rss_mb_after": round(rss1, 1),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Latency measurement.")
    p.add_argument("--source", default="data/raw")
    p.add_argument("--profile", default="dev")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--threads", default="4,6,8")
    p.add_argument("--end-to-end-seconds", type=float, default=12.0,
                   help="end-to-end capture->snapshot loopback duration; 0 to skip")
    p.add_argument("--only-end-to-end", action="store_true",
                   help="run ONLY the end-to-end loopback and print its JSON (used as a "
                        "clean subprocess so ORT-session contention does not skew it)")
    args = p.parse_args(argv)
    configure_logging("INFO")

    cfg = load_config(args.profile)
    src = Path(args.source)
    if not src.exists():
        _LOG.error("source not found: %s", src)
        return 2
    imgs = list(_frames(src, args.limit))
    if not imgs:
        _LOG.error("no frames decoded from %s", src)
        return 2
    _LOG.info("loaded %d frames from %s", len(imgs), src)

    if args.only_end_to_end:
        res = _end_to_end(cfg, imgs, seconds=args.end_to_end_seconds)
        print("E2E_JSON:" + json.dumps(res))
        return 0

    policy = RecognitionPolicy(cfg.policy)
    out: dict = {"profile": args.profile, "frames": len(imgs), "runs": {}}

    # The end-to-end loopback FIRST, in a clean subprocess: this benchmark builds
    # many ORT sessions in one process and by the thread sweep they all contend
    # (measured combined p50 balloons). The `before__pose_every_frame` run below
    # is the clean in-process reference for the per-model numbers.
    out["end_to_end"] = None
    if args.end_to_end_seconds > 0:
        _LOG.info("end-to-end capture->snapshot loopback (%.0fs, clean subprocess)...",
                  args.end_to_end_seconds)
        try:
            import subprocess

            cp = subprocess.run(
                [_sys.executable, __file__, "--only-end-to-end",
                 "--source", args.source, "--profile", args.profile,
                 "--limit", str(args.limit),
                 "--end-to-end-seconds", str(args.end_to_end_seconds)],
                capture_output=True, text=True, timeout=args.end_to_end_seconds + 180,
            )
            line = next((l for l in cp.stdout.splitlines() if l.startswith("E2E_JSON:")), None)
            out["end_to_end"] = json.loads(line[len("E2E_JSON:"):]) if line else {
                "error": (cp.stderr or cp.stdout or "no output")[-500:]
            }
        except Exception as exc:  # noqa: BLE001 - measurement, not production
            _LOG.warning("end-to-end loopback failed: %r", exc)
            out["end_to_end"] = {"error": repr(exc)}

    base = _engine(cfg, pose_requires_person=False, threads=cfg.perception.intra_op_threads)
    out["runs"]["before__pose_every_frame"] = _run(base, policy, imgs, resize_to=None)

    gated = _engine(cfg, pose_requires_person=True, threads=cfg.perception.intra_op_threads)
    out["runs"]["after__pose_gated_on_person"] = _run(gated, policy, imgs, resize_to=None)

    sweep = {}
    for t in [int(x) for x in args.threads.split(",") if x.strip()]:
        eng = _engine(cfg, pose_requires_person=False, threads=t)
        sweep[str(t)] = _run(eng, policy, imgs, resize_to=None)
    out["thread_sweep"] = sweep
    out["thread_sweep_note"] = (
        "cumulative in-process ORT-session contention - the `before__pose_every_frame` "
        "row is the clean per-model reference; the sweep rows are progressively skewed."
    )

    pol_p95 = out["runs"]["before__pose_every_frame"]["policy_p95"]
    out["policy_cost_under_1ms_p95"] = bool(0 <= pol_p95 < 1.0)

    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "latency_2_5.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (results_dir / "latency_2_5.md").write_text(_md(out), encoding="utf-8")
    _LOG.info("policy p95 %.3f ms (<1ms: %s)", pol_p95, out["policy_cost_under_1ms_p95"])
    _LOG.info("wrote %s", results_dir / "latency_2_5.md")
    return 0


def _md(o: dict) -> str:
    L = [f"# Phase 2.5 latency — {o['frames']} frames of the developer's footage", "",
         "All milliseconds, CPU provider. End-to-end capture→overlay latency needs a "
         "live browser and is a developer check (phase report Part 2).", "",
         "| run | frames | pose ran | detector p50/p95 | pose p50/p95 | policy p50/p95 | combined p50/p95 | implied FPS |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in o["runs"].items():
        L.append(f"| {name} | {r['frames']} | {r['pose_ran']} | {r['detector_p50']}/{r['detector_p95']} "
                 f"| {r['pose_p50']}/{r['pose_p95']} | {r['policy_p50']}/{r['policy_p95']} "
                 f"| {r['combined_p50']}/{r['combined_p95']} | {r['implied_fps']} |")
    L += ["", f"**Policy cost < 1 ms p95:** {o['policy_cost_under_1ms_p95']}", "",
          "## intra-op thread sweep (pose every frame)", "",
          "| threads | detector p50 | pose p50 | combined p50 | implied FPS |", "|---|---|---|---|---|"]
    for t, r in o["thread_sweep"].items():
        L.append(f"| {t} | {r['detector_p50']} | {r['pose_p50']} | {r['combined_p50']} | {r['implied_fps']} |")

    e = o.get("end_to_end")
    L += ["", "## End-to-end capture -> snapshot (real app loopback, perception ON)", ""]
    if not e:
        L.append("_(skipped)_")
    elif "error" in e:
        L.append(f"_(failed: {e['error']})_")
    else:
        L += [
            "Frames pushed over `/ws/ingest` at `analysis_fps` with a real capture "
            "clock; `StateSnapshot.frame_age_ms` collected from `/ws/state`. The live "
            "browser adds only its own capture-encode + one paint on top of this "
            "(Block 13 developer check).", "",
            "| measure | value |", "|---|---|",
            f"| samples | {e['samples']} |",
            f"| **frame age p50 / p95 / max (ms)** | **{e['frame_age_ms_p50']} / {e['frame_age_ms_p95']} / {e['frame_age_ms_max']}** |",
            f"| detector p50 (ms) | {e['detector_ms_p50']} |",
            f"| pose p50 (ms) | {e['pose_ms_p50']} |",
            f"| analysis FPS p50 | {e['analysis_fps_p50']} |",
            f"| drop rate (mean) | {e['drop_rate_mean']} |",
            f"| process CPU % | {e['process_cpu_percent']} |",
            f"| RSS MB (before -> after) | {e['rss_mb_before']} -> {e['rss_mb_after']} |",
        ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
