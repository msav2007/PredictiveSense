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
    enc_proxy_ms: list[float] = []
    for img in imgs:
        small = cv2.resize(img, (cfg.capture.analysis_width, cfg.capture.analysis_height))
        _t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", small,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                int(cfg.capture.analysis_jpeg_quality * 100)])
        _enc = (time.perf_counter() - _t0) * 1000.0
        if ok:
            jpegs.append(buf.tobytes())
            enc_proxy_ms.append(_enc)
    if not jpegs:
        return None

    ages: list[float] = []
    det_ms: list[float] = []
    pose_ms: list[float] = []
    fps: list[float] = []
    drop: list[float] = []
    # Phase 8: per-stage attribution collected from the snapshot `stage_*` keys.
    stage_names = (
        "worker_encode", "ws_transit", "decode", "src_buffer_dwell",
        "producer_handoff", "mailbox_dwell", "detector", "pose", "policy",
        "snapshot_build", "ws_out", "capture_to_snapshot",
    )
    stage_samples: dict[str, list[float]] = {n: [] for n in stage_names}
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
                        for _n in stage_names:
                            _v = m.get(f"stage_{_n}_ms")
                            if isinstance(_v, (int, float)) and _v >= 0:
                                stage_samples[_n].append(float(_v))

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
                cap_ms = time.time() * 1000.0  # "drawImage" instant
                # `jpeg` is pre-encoded; attribute the measured encode proxy so
                # the worker_encode stage is populated in the loopback table.
                header = IngestHeader(
                    client_ts_ms=time.time() * 1000.0, seq=i,
                    w=cfg.capture.analysis_width, h=cfg.capture.analysis_height,
                    cap_ts_ms=cap_ms,
                    enc_ms=round(enc_proxy_ms[i % len(enc_proxy_ms)], 2),
                )
                ws.send_bytes(encode_ingest_message(header, jpeg))
                i += 1
                time.sleep(period)
        time.sleep(0.5)
        stop.set()
        rt.join(timeout=2.0)

    cpu = proc.cpu_percent(None)
    rss1 = proc.memory_info().rss / 1e6
    stages = {
        n: {"p50": _pct(v, 50), "p95": _pct(v, 95), "n": len(v)}
        for n, v in stage_samples.items()
    }
    # ws_out is stamped by the broadcaster into the loop's registry, not onto the
    # snapshot - read it directly (same process).
    try:
        wso = app.state.loop.metrics.samples("stage_ws_out_ms")
        if wso.count:
            stages["ws_out"] = {"p50": round(wso.p50, 2), "p95": round(wso.p95, 2),
                                "n": wso.count}
    except Exception:  # noqa: BLE001 - measurement only
        pass
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
        "stages": stages,
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
    p.add_argument("--label", default="loopback",
                   help="label for the Phase 8 stage-attribution files "
                        "results/latency_stages_<label>.{json,md}")
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

    e = out.get("end_to_end")
    if isinstance(e, dict) and "error" not in e and e.get("stages"):
        _write_attribution(e, results_dir, args.label, cfg, args)
        _LOG.info("wrote %s", results_dir / f"latency_stages_{args.label}.md")
    return 0


def _machine_fingerprint(cfg) -> dict:
    """Minimal machine + runtime fingerprint for a stage-attribution file."""

    import platform

    import psutil

    fp: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "provider_config": cfg.perception.provider,
        "intra_op_threads": cfg.perception.intra_op_threads,
        "detector_input_size": cfg.perception.detector.input_size,
        "analysis": f"{cfg.capture.analysis_width}x{cfg.capture.analysis_height}"
                    f"@{cfg.capture.analysis_fps}fps q{cfg.capture.analysis_jpeg_quality}",
    }
    try:
        import onnxruntime as ort

        fp["onnxruntime"] = ort.__version__
        fp["available_providers"] = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        fp["onnxruntime"] = "unavailable"
    try:
        proc = _sys  # noqa: F841 - placeholder to keep structure obvious
        cpu = platform.processor()
        if cpu:
            fp["cpu"] = cpu
    except Exception:  # noqa: BLE001
        pass
    return fp


# Ordered stage list for the attribution table. `ws_out` and `overlay_paint` are
# attributed here only when a value is present (loopback has ws_out, no paint;
# the fake-camera headless run has both).
_ATTR_STAGES: tuple[tuple[str, str], ...] = (
    ("worker_encode", "browser JPEG encode"),
    ("ws_transit", "send -> server receive"),
    ("decode", "cv2.imdecode"),
    ("src_buffer_dwell", "BrowserSource slot dwell"),
    ("producer_handoff", "producer read -> mailbox put"),
    ("mailbox_dwell", "LatestFrameMailbox dwell"),
    ("detector", "detector inference"),
    ("pose", "pose inference"),
    ("policy", "recognition policy"),
    ("snapshot_build", "snapshot assembly"),
    ("ws_out", "emit -> /ws/state send"),
    ("overlay_paint", "capture -> overlay paint (browser)"),
)


def _write_attribution(e: dict, results_dir: Path, label: str, cfg, args) -> None:
    """Write results/latency_stages_<label>.{json,md} - each stage p50/p95 and
    its share of the end-to-end capture->snapshot p50."""

    stages = e.get("stages", {})
    total_p50 = e.get("frame_age_ms_p50") or 0.0
    cap_to_snap = stages.get("capture_to_snapshot", {}).get("p50") or total_p50

    rows = []
    attributed_p50 = 0.0
    for key, human in _ATTR_STAGES:
        s = stages.get(key)
        if not s or not s.get("n"):
            continue
        p50 = s["p50"]
        p95 = s["p95"]
        share = (p50 / cap_to_snap * 100.0) if cap_to_snap else 0.0
        if key not in ("ws_out", "overlay_paint"):
            attributed_p50 += p50
        rows.append({
            "stage": key, "what": human, "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2), "pct_of_end_to_end": round(share, 1),
            "n": s["n"],
        })
    unattributed = round(max(0.0, cap_to_snap - attributed_p50), 2)

    payload = {
        "label": label,
        "kind": "loopback (server pipeline; recorded frames; no live camera, no browser paint)",
        "source": args.source,
        "machine": _machine_fingerprint(cfg),
        "samples": e.get("samples"),
        "end_to_end": {
            "capture_to_snapshot_ms_p50": round(cap_to_snap, 2),
            "frame_age_ms_p50": e.get("frame_age_ms_p50"),
            "frame_age_ms_p95": e.get("frame_age_ms_p95"),
            "frame_age_ms_max": e.get("frame_age_ms_max"),
            "analysis_fps_p50": e.get("analysis_fps_p50"),
            "drop_rate_mean": e.get("drop_rate_mean"),
        },
        "stages": rows,
        "server_stages_attributed_p50_ms": round(attributed_p50, 2),
        "unattributed_p50_ms": unattributed,
        "notes": (
            "Loopback: frames are pushed over a real /ws/ingest with a real capture "
            "clock through the real server pipeline; the browser's getUserMedia, "
            "the Web Worker canvas draw and the final overlay paint are NOT in this "
            "path. worker_encode here is a cv2.imencode proxy. overlay_paint and "
            "true capture->paint come from the fake-camera headless run "
            "(results/browser_metrics_*.json) and real-camera values are the "
            "developer's physical verification."
        ),
    }
    (results_dir / f"latency_stages_{label}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    m = e.get("machine", {})  # noqa: F841
    fp = payload["machine"]
    L = [
        f"# Latency stage attribution - `{label}`",
        "",
        f"**Kind:** {payload['kind']}",
        "",
        f"**Machine:** {fp.get('cpu', 'CPU n/a')} · {fp['logical_cores']} logical cores "
        f"· {fp['ram_gb']} GB · {fp['platform']} · Python {fp['python']} · "
        f"onnxruntime {fp.get('onnxruntime')} · providers {fp.get('available_providers')}",
        "",
        f"**Config:** provider `{fp['provider_config']}` · intra_op {fp['intra_op_threads']} "
        f"· detector input {fp['detector_input_size']} · analysis {fp['analysis']}",
        "",
        f"**Samples:** {payload['samples']} analysed frames · "
        f"end-to-end capture->snapshot p50 **{payload['end_to_end']['capture_to_snapshot_ms_p50']} ms** "
        f"· frame_age p50/p95/max "
        f"{payload['end_to_end']['frame_age_ms_p50']}/{payload['end_to_end']['frame_age_ms_p95']}/"
        f"{payload['end_to_end']['frame_age_ms_max']} ms · analysis FPS "
        f"{payload['end_to_end']['analysis_fps_p50']} · drop rate "
        f"{payload['end_to_end']['drop_rate_mean']}",
        "",
        "| stage | what | p50 ms | p95 ms | % of end-to-end | n |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| `{r['stage']}` | {r['what']} | {r['p50_ms']} | {r['p95_ms']} "
            f"| {r['pct_of_end_to_end']}% | {r['n']} |"
        )
    L += [
        "",
        f"Server stages attributed (sum of p50): **{payload['server_stages_attributed_p50_ms']} ms** "
        f"of {payload['end_to_end']['capture_to_snapshot_ms_p50']} ms "
        f"(unattributed / overlap: {payload['unattributed_p50_ms']} ms).",
        "",
        payload["notes"],
        "",
    ]
    (results_dir / f"latency_stages_{label}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


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
