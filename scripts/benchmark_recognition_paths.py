"""Phase 6 BLOCK 8 - three bounded, measured recognition-responsiveness
experiments under ONE canonical latency protocol. No model swap, no
quantisation, no threading redesign, no architecture change.

    python scripts\\benchmark_recognition_paths.py --source data\\raw --frames 300

Experiments
-----------
1. **Pose gating** (`perception.pose_requires_person` false vs true) - combined
   per-frame cost, end-to-end frame age p50/p95, analysis FPS, and the change in
   pose availability on frames where a person *is* present.
2. **Detector input size** 480 vs 640 - detector latency AND the detection count
   of the classes actually in the primary tier (the small objects that motivated
   640 - watch / spectacles / charger - are outside the model vocabulary at any
   size).
3. **Canonical latency protocol** - two protocols defined and both re-reported so
   the Phase 2 (`detector p50 46.8 ms`) and Phase 5 (`65-72 ms`) figures are
   reconciled rather than left contradictory.

Writes ``results/recognition_paths.{json,md}``. Numbers are latency/frequency,
never accuracy - the footage is unlabelled.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.engine import PerceptionEngine
from predictivesense.perception.policy import RecognitionPolicy

_LOG = get_logger("predictivesense.scripts.benchmark_recognition_paths")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
_WARMUP_DISCARD = 5  # frames dropped from every timing series (Protocol A)


def _pct(xs, p):
    return round(float(np.percentile(xs, p)), 2) if len(xs) else -1.0


def _frames(source: Path, limit: int):
    files = (
        [source]
        if source.is_file()
        else sorted(q for q in source.rglob("*") if q.suffix.lower() in (_VIDEO_SUFFIXES | _IMG_SUFFIXES))
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


def _engine(cfg, *, pose_requires_person: bool, input_size: int) -> PerceptionEngine:
    pc = cfg.perception.model_copy(
        update={
            "pose_requires_person": pose_requires_person,
            "detector": cfg.perception.detector.model_copy(update={"input_size": input_size}),
        }
    )
    return PerceptionEngine(pc, warmup=True)


def _run_isolated(engine: PerceptionEngine, policy: RecognitionPolicy, imgs, primary: set[str]):
    """Protocol A: isolated, in-process, warm. One engine, one ORT session pair,
    frames at native resolution, first `_WARMUP_DISCARD` frames dropped. Reports
    the model's own measured inference time from PerceptionResult."""

    det_ms, pose_ms, comb_ms = [], [], []
    pose_ran = pose_person_frames = person_frames = 0
    primary_hits = secondary_hits = other_hits = 0
    primary_scores: list[float] = []  # Phase 8 §10: score distribution, primary tier
    for i, img in enumerate(imgs):
        fr = Frame(
            frame_id=i,
            capture_ts=float(i),
            image=np.ascontiguousarray(img),
            width=img.shape[1],
            height=img.shape[0],
            source_id="bench",
            seq=i,
        )
        r = engine.infer(fr, frame_index=i)
        has_person = any(d.class_name == "person" for d in r.detections)
        if i >= _WARMUP_DISCARD:
            if r.detector_ms is not None:
                det_ms.append(r.detector_ms)
            if r.pose_ms is not None:
                pose_ms.append(r.pose_ms)
                pose_ran += 1
                if has_person:
                    pose_person_frames += 1
            if has_person:
                person_frames += 1
            comb_ms.append((r.detector_ms or 0.0) + (r.pose_ms or 0.0))
            for d in r.detections:
                if d.class_name in primary:
                    primary_hits += 1
                    primary_scores.append(float(d.score))
                elif d.class_name:
                    secondary_hits += 1
                else:
                    other_hits += 1
    return {
        "frames_timed": len(comb_ms),
        "detector_p50": _pct(det_ms, 50),
        "detector_p95": _pct(det_ms, 95),
        "pose_p50": _pct(pose_ms, 50),
        "pose_p95": _pct(pose_ms, 95),
        "combined_p50": _pct(comb_ms, 50),
        "combined_p95": _pct(comb_ms, 95),
        "implied_fps": round(1000.0 / _pct(comb_ms, 50), 2) if _pct(comb_ms, 50) > 0 else -1.0,
        "pose_ran_frames": pose_ran,
        "person_frames": person_frames,
        "pose_available_on_person_frames": pose_person_frames,
        "primary_tier_detections": primary_hits,
        "non_primary_detections": secondary_hits,
        # Phase 8 §10: does a latency win quietly lose or weaken detections?
        "primary_score_p10": round(_pct(primary_scores, 10), 4),
        "primary_score_p50": round(_pct(primary_scores, 50), 4),
        "primary_score_p90": round(_pct(primary_scores, 90), 4),
        "primary_score_mean": round(sum(primary_scores) / len(primary_scores), 4)
        if primary_scores else -1.0,
    }


def _end_to_end(profile: str, source: str, *, pose_requires_person: bool, seconds: float) -> dict:
    """Protocol B: the real FastAPI app + browser-ingest source + perception +
    policy, run in a clean subprocess so ORT-session contention does not skew it.
    Returns capture->snapshot frame age + analysis FPS + drop rate."""

    import subprocess

    cp = subprocess.run(
        [
            _sys.executable,
            __file__,
            "--only-end-to-end",
            "--source",
            source,
            "--profile",
            profile,
            "--end-to-end-seconds",
            str(seconds),
            "--pose-requires-person",
            "1" if pose_requires_person else "0",
        ],
        capture_output=True,
        text=True,
        timeout=seconds + 240,
    )
    line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("E2E_JSON:")), None)
    if not line:
        return {"error": (cp.stderr or cp.stdout or "no output")[-800:]}
    return json.loads(line[len("E2E_JSON:") :])


def _end_to_end_impl(cfg, source: str, *, pose_requires_person: bool, seconds: float) -> dict:
    import threading

    import psutil
    from fastapi.testclient import TestClient

    from predictivesense.api.app import create_app
    from predictivesense.camera.framing import encode_ingest_message
    from predictivesense.core.enums import SourceKind
    from predictivesense.core.types import IngestHeader, StateSnapshot

    imgs = list(_frames(Path(source), 200))
    if not imgs:
        return {"error": f"no frames from {source}"}

    e2e_cfg = cfg.model_copy(
        update={
            "source": cfg.source.model_copy(update={"kind": SourceKind.BROWSER}),
            "perception": cfg.perception.model_copy(
                update={"pose_requires_person": pose_requires_person}
            ),
        }
    )
    jpegs = []
    for img in imgs:
        small = cv2.resize(img, (cfg.capture.analysis_width, cfg.capture.analysis_height))
        ok, buf = cv2.imencode(
            ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.capture.analysis_jpeg_quality * 100)]
        )
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return {"error": "no jpegs encoded"}

    ages: list[float] = []
    det_ms: list[float] = []
    pose_ms: list[float] = []
    fps: list[float] = []
    drop: list[float] = []
    proc = psutil.Process()
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
                header = IngestHeader(
                    client_ts_ms=time.time() * 1000.0,
                    seq=i,
                    w=cfg.capture.analysis_width,
                    h=cfg.capture.analysis_height,
                )
                ws.send_bytes(encode_ingest_message(header, jpegs[i % len(jpegs)]))
                i += 1
                time.sleep(period)
        time.sleep(0.5)
        stop.set()
        rt.join(timeout=2.0)

    return {
        "pose_requires_person": pose_requires_person,
        "samples": len(ages),
        "frame_age_ms_p50": _pct(ages, 50),
        "frame_age_ms_p95": _pct(ages, 95),
        "frame_age_ms_max": round(max(ages), 2) if ages else -1.0,
        "detector_ms_p50": _pct(det_ms, 50),
        "pose_ms_p50": _pct(pose_ms, 50),
        "pose_frames": len(pose_ms),
        "analysis_fps_p50": _pct(fps, 50),
        "drop_rate_mean": round(float(np.mean(drop)), 4) if drop else -1.0,
        "process_cpu_percent": round(proc.cpu_percent(None), 1),
    }


def _md(o: dict) -> str:
    p = o["protocol"]
    iso = o["isolated"]
    L = [
        f"# Phase 6 - recognition responsiveness ({o['frames']} frames, {o['source']})",
        "",
        "Latency and detection **frequencies**, CPU provider, this machine. Not accuracy - "
        "the footage is unlabelled.",
        "",
        "## Canonical latency protocol (BLOCK 8.22)",
        "",
        "Two protocols are defined; every latency figure in this project is now tagged with one.",
        "",
        "**Protocol A - isolated, in-process, warm.** One `PerceptionEngine` "
        "(`warmup=True`), a single ORT detector+pose session pair, frames decoded from "
        f"`{o['source']}` at native resolution, `intra_op_threads={p['intra_op_threads']}`, "
        f"`provider={p['provider']}`, pose every frame, the first {p['warmup_discard']} frames "
        "discarded. Reports the model's own measured inference time "
        "(`PerceptionResult.detector_ms` / `.pose_ms`). This is the Phase 5 "
        "\"per-frame CPU inference (uncontended in-process run)\" number.",
        "",
        "**Protocol B - end-to-end app loopback.** The real FastAPI app + browser-ingest "
        "source + perception + policy in a clean subprocess; frames pushed over `/ws/ingest` "
        f"at `analysis_fps={p['analysis_fps']}` with a real capture clock; "
        "`StateSnapshot.frame_age_ms` collected from `/ws/state`. The live browser adds only "
        "its own capture-encode + one paint on top (a developer check).",
        "",
        "### Reconciliation of prior figures",
        "",
        "| report | figure as published | how it was measured | Protocol A? | this run |",
        "|---|---|---|---|---|",
        f"| Phase 2 | detector p50 **46.8 ms** | `benchmark_providers.py` - detector and "
        f"pose timed in **separate passes**, no per-frame interleaving | **no** (lower bound) "
        f"| detector p50 **{iso['every_frame_640']['detector_p50']} ms** |",
        f"| Phase 5 | detector p50 **65-72 ms** | isolated in-process, both ORT sessions per "
        f"frame, developer footage | **yes** | detector p50 "
        f"**{iso['every_frame_640']['detector_p50']} ms** |",
        "",
        "Phase 2's figure is not a regression - it is a different measurement (one model at "
        "a time). Running both sessions per frame, as the analysis loop does (Protocol A), "
        "roughly doubles detector p50. The interleaved-vs-separate-pass distinction, not any "
        "single value, is what reconciles the three figures. The absolute Protocol A number "
        "is a **range** (~65-115 ms on this machine, load-dependent). The Phase 2 and Phase 5 "
        "reports carry a one-line pointer to this protocol.",
        "",
        "## Experiment 1 - pose gating (BLOCK 8.20)",
        "",
        "| run | detector p50/p95 | pose p50/p95 | combined p50/p95 | implied FPS | pose ran | "
        "person frames | pose available on person frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in (
        ("every_frame_640", "pose every frame (shipped)"),
        ("gated_640", "pose_requires_person = true"),
    ):
        r = iso[key]
        L.append(
            f"| {label} | {r['detector_p50']}/{r['detector_p95']} | {r['pose_p50']}/{r['pose_p95']} "
            f"| {r['combined_p50']}/{r['combined_p95']} | {r['implied_fps']} | {r['pose_ran_frames']} "
            f"| {r['person_frames']} | {r['pose_available_on_person_frames']} |"
        )
    e2e = o.get("end_to_end", {})
    L += ["", "End-to-end (Protocol B):", "",
          "| run | frame age p50/p95/max | analysis FPS p50 | drop rate | pose frames |",
          "|---|---|---|---|---|"]
    for key in ("every_frame", "gated"):
        r = e2e.get(key, {})
        if "error" in r:
            L.append(f"| {key} | _(failed: {r['error'][:80]})_ | | | |")
        elif r:
            L.append(
                f"| {key} | {r['frame_age_ms_p50']}/{r['frame_age_ms_p95']}/{r['frame_age_ms_max']} "
                f"| {r['analysis_fps_p50']} | {r['drop_rate_mean']} | {r['pose_frames']} |"
            )
    L += ["", f"**Verdict:** {o['verdicts']['pose_gating']}", ""]

    L += [
        "## Experiment 2 - detector input size 320 / 480 / 640 (Phase 8 §10)",
        "",
        "| input | detector p50/p95 | combined p50/p95 | implied FPS | primary-tier detections | "
        "primary score p10/p50/p90 | non-primary detections |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, size in (("every_frame_320", 320), ("every_frame_480", 480), ("every_frame_640", 640)):
        r = iso[key]
        L.append(
            f"| {size} | {r['detector_p50']}/{r['detector_p95']} | {r['combined_p50']}/{r['combined_p95']} "
            f"| {r['implied_fps']} | {r['primary_tier_detections']} "
            f"| {r.get('primary_score_p10')}/{r.get('primary_score_p50')}/{r.get('primary_score_p90')} "
            f"| {r['non_primary_detections']} |"
        )
    L += ["", f"**Verdict:** {o['verdicts']['input_size']}", ""]
    return "\n".join(L) + "\n"


def _verdicts(iso: dict, e2e: dict) -> dict:
    ef, gt = iso["every_frame_640"], iso["gated_640"]
    p480, p640 = iso["every_frame_480"], iso["every_frame_640"]

    # pose gating helps only if it cuts combined cost AND does not drop pose on
    # person frames.
    person_loss = ef["pose_available_on_person_frames"] - gt["pose_available_on_person_frames"]
    saved = ef["combined_p50"] - gt["combined_p50"]
    if gt["person_frames"] >= max(1, int(0.6 * gt["frames_timed"])):
        pose_gating = (
            f"NO CHANGE JUSTIFIED. A person is present in {gt['person_frames']}/"
            f"{gt['frames_timed']} frames on this footage, so gating skips pose almost never "
            f"(combined p50 {ef['combined_p50']} -> {gt['combined_p50']} ms, "
            f"pose lost on {person_loss} person frame(s)). Keep `pose_requires_person: false`."
        )
    elif saved > 5 and person_loss <= 0:
        pose_gating = (
            f"HELPS. Gating cut combined p50 by {saved:.1f} ms with no pose lost on person "
            f"frames. Consider `pose_requires_person: true`."
        )
    else:
        pose_gating = (
            f"NO CHANGE JUSTIFIED. Combined p50 moved {ef['combined_p50']} -> "
            f"{gt['combined_p50']} ms; pose availability on person frames changed by "
            f"{-person_loss}. Not worth the behavioural risk."
        )

    lat_gain = p640["detector_p50"] - p480["detector_p50"]
    prim_loss = p640["primary_tier_detections"] - p480["primary_tier_detections"]
    if lat_gain > 8 and prim_loss <= max(2, int(0.03 * p640["primary_tier_detections"])):
        input_size = (
            f"480 IS VIABLE. Detector p50 {p640['detector_p50']} -> {p480['detector_p50']} ms "
            f"(-{lat_gain:.1f}) and primary-tier detections {p640['primary_tier_detections']} -> "
            f"{p480['primary_tier_detections']} (loss {prim_loss}). The small objects 640 was "
            f"chosen for (watch/spectacles/charger) are outside the vocabulary at any size. "
            f"A default change to 480 is defensible; left to the developer to confirm on more "
            f"footage before flipping `detector.input_size`."
        )
    else:
        input_size = (
            f"NO CHANGE JUSTIFIED. 640 -> 480 saved only {lat_gain:.1f} ms detector p50 and "
            f"lost {prim_loss} primary-tier detection(s). Keep `detector.input_size: 640`."
        )
    return {"pose_gating": pose_gating, "input_size": input_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 recognition-responsiveness experiments.")
    ap.add_argument("--source", default="data/raw")
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--end-to-end-seconds", type=float, default=12.0, help="0 to skip Protocol B")
    ap.add_argument("--only-end-to-end", action="store_true")
    ap.add_argument("--pose-requires-person", default="0")
    args = ap.parse_args(argv)
    configure_logging("INFO")

    cfg = load_config(args.profile)
    src = Path(args.source)
    if not src.exists():
        _LOG.error("source not found: %s", src)
        return 2

    if args.only_end_to_end:
        res = _end_to_end_impl(
            cfg,
            args.source,
            pose_requires_person=args.pose_requires_person == "1",
            seconds=args.end_to_end_seconds,
        )
        print("E2E_JSON:" + json.dumps(res))
        return 0

    imgs = list(_frames(src, args.frames))
    if not imgs:
        _LOG.error("no frames decoded from %s", src)
        return 2
    _LOG.info("loaded %d frames from %s", len(imgs), src)

    primary = set(cfg.policy.vocabulary.primary)
    policy = RecognitionPolicy(cfg.policy)

    isolated: dict = {}
    for name, pose_gate, size in (
        ("every_frame_640", False, 640),
        ("gated_640", True, 640),
        ("every_frame_480", False, 480),
        ("every_frame_320", False, 320),  # Phase 8 §10: full 320/480/640 sweep
    ):
        _LOG.info("isolated run: %s", name)
        eng = _engine(cfg, pose_requires_person=pose_gate, input_size=size)
        isolated[name] = _run_isolated(eng, policy, imgs, primary)

    end_to_end: dict = {}
    if args.end_to_end_seconds > 0:
        for name, gate in (("every_frame", False), ("gated", True)):
            _LOG.info("end-to-end loopback (%s, clean subprocess)...", name)
            try:
                end_to_end[name] = _end_to_end(
                    args.profile, args.source,
                    pose_requires_person=gate, seconds=args.end_to_end_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - measurement, not production
                end_to_end[name] = {"error": repr(exc)}

    out = {
        "profile": args.profile,
        "source": args.source,
        "frames": len(imgs),
        "protocol": {
            "intra_op_threads": cfg.perception.intra_op_threads,
            "provider": cfg.perception.provider,
            "analysis_fps": cfg.capture.analysis_fps,
            "warmup_discard": _WARMUP_DISCARD,
        },
        "isolated": isolated,
        "end_to_end": end_to_end,
    }
    out["verdicts"] = _verdicts(isolated, end_to_end)

    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "recognition_paths.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (results_dir / "recognition_paths.md").write_text(_md(out), encoding="utf-8")
    _LOG.info("pose gating : %s", out["verdicts"]["pose_gating"])
    _LOG.info("input size  : %s", out["verdicts"]["input_size"])
    _LOG.info("wrote %s", results_dir / "recognition_paths.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
