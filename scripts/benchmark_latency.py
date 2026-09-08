"""Phase 2.5 bounded-latency measurement: pose gating, thread count, policy cost.

    python scripts\\benchmark_latency.py [--source data\\raw] [--threads 4,6,8]

Runs the real PerceptionEngine + RecognitionPolicy over decoded frames of the
developer's own footage and reports, before vs after each bounded change:

* detector / pose / combined per-frame latency p50 / p95 (BLOCK 11);
* pose-gating effect - pose off on no-person frames - on combined cost and the
  implied analysis FPS (1000 / combined p50);
* the recognition-policy cost per frame, asserted < 1 ms p95 (BLOCK 11);
* a small intra-op thread sweep on this 14-core machine.

End-to-end capture->overlay latency also needs a live browser and is a
developer-performed check (see the phase report, Part 2). Writes
``results/latency_2_5.{json,md}``.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 2.5 latency measurement.")
    p.add_argument("--source", default="data/raw")
    p.add_argument("--profile", default="dev")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--threads", default="4,6,8")
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

    policy = RecognitionPolicy(cfg.policy)
    out: dict = {"profile": args.profile, "frames": len(imgs), "runs": {}}

    base = _engine(cfg, pose_requires_person=False, threads=cfg.perception.intra_op_threads)
    out["runs"]["before__pose_every_frame"] = _run(base, policy, imgs, resize_to=None)

    gated = _engine(cfg, pose_requires_person=True, threads=cfg.perception.intra_op_threads)
    out["runs"]["after__pose_gated_on_person"] = _run(gated, policy, imgs, resize_to=None)

    sweep = {}
    for t in [int(x) for x in args.threads.split(",") if x.strip()]:
        eng = _engine(cfg, pose_requires_person=False, threads=t)
        sweep[str(t)] = _run(eng, policy, imgs, resize_to=None)
    out["thread_sweep"] = sweep

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
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
