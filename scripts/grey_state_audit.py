"""Phase 9 root-cause instrumentation (docs/phase-reports/phase9.md section 4).

    python scripts\\grey_state_audit.py --source data\\raw --frames 600 --live-seconds 20

Two independent measurements over the developer's own real footage
(``data/raw``), published BEFORE any rendering change, per the four
hypotheses in the Phase 9 prompt (secondary-tier de-emphasis, the
low-confidence band, unknown_low_confidence/unknown_margin, and snapshot
staleness):

1. **BATCH** - the real detector + :class:`RecognitionPolicy` run
   frame-by-frame over the full clip (same code path as
   ``scripts/policy_audit.py``), tabulating every detection's
   ``policy_state``, ``tier`` and ``low_confidence_band`` membership. This is
   content-driven and wants a large, complete sample, so it runs offline
   without the live timing/mailbox machinery subsampling frames.
2. **LIVE SESSION** - the real FastAPI app, real ``/ws/ingest`` + real
   ``/ws/state``, real analysis-loop timing (Phase 8's
   ``LatestFrameMailbox`` + staleness guard), pushing the same clip's frames
   at ``analysis_fps`` for ``--live-seconds`` seconds. ``StateSnapshot.stale``
   is a property of frame-arrival *timing*, not content, so it can only be
   observed by actually running the live loop - a batch pass over decoded
   frames has no ingest gaps to be stale about. Reuses the same
   ``create_app`` + ``TestClient`` + ``/ws/ingest`` push loop pattern as
   ``scripts/benchmark_latency.py``'s ``_end_to_end`` (Phase 8), but counts
   *every* incoming snapshot (stale included) instead of filtering staleness
   out.

**These are frequencies on the developer's own unlabelled footage, not
accuracy.** No rendering change may be made without this table (BLOCK: Phase
9 section 4).

Writes ``results/grey_state_frequency.{json,md}``.
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
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.policy import POLICY_STATES, RecognitionPolicy

from scripts._eval_common import model_spec

_LOG = get_logger("predictivesense.scripts.grey_state_audit")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_TIERS = ("primary", "secondary", "implausible", "unlisted")


def _frames(source: Path, limit: int):
    files = (
        [source]
        if source.is_file()
        else sorted(
            q for q in source.rglob("*")
            if q.suffix.lower() in (_VIDEO_SUFFIXES | _IMG_SUFFIXES)
        )
    )
    n = 0
    for f in files:
        if f.suffix.lower() in _IMG_SUFFIXES:
            img = cv2.imread(str(f))
            if img is not None:
                n += 1
                yield np.ascontiguousarray(img)
            if n >= limit:
                return
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


def _batch(detector: ObjectDetector, policy: RecognitionPolicy, band: tuple[float, float],
           imgs: list[np.ndarray]) -> dict:
    """Offline pass: detector + policy over every decoded frame. Tabulates
    policy_state, tier, and low_confidence_band membership (independent of
    which policy_state the detection landed in, so the table shows how often
    the band and the policy's own threshold *disagree*, per the root-cause
    trace's fifth finding)."""

    state_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    state_x_tier: Counter = Counter()
    band_member = 0
    band_member_by_state: Counter = Counter()
    band_lo, band_hi = band
    raw_total = 0
    frames_done = 0

    for i, img in enumerate(imgs):
        frames_done += 1
        fr = Frame(
            frame_id=i, capture_ts=float(i), image=np.ascontiguousarray(img),
            width=int(img.shape[1]), height=int(img.shape[0]),
            source_id="grey_state_audit", seq=i,
        )
        raw = detector.infer(fr)
        raw_total += len(raw)
        outcome = policy.apply(raw, frame_width=fr.width, frame_height=fr.height)
        for d in outcome.detections:
            state_counts[d.policy_state] += 1
            tier_counts[d.tier] += 1
            state_x_tier[f"{d.policy_state}|{d.tier}"] += 1
            in_band = band_lo <= d.score < band_hi
            if in_band:
                band_member += 1
                band_member_by_state[d.policy_state] += 1

    total = sum(state_counts.values())
    return {
        "frames": frames_done,
        "raw_detections": raw_total,
        "accepted_detections": total,
        "policy_state_counts": {s: state_counts.get(s, 0) for s in POLICY_STATES},
        "policy_state_share": {
            s: round(state_counts.get(s, 0) / total, 4) if total else 0.0
            for s in POLICY_STATES
        },
        "tier_counts": {t: tier_counts.get(t, 0) for t in _TIERS},
        "state_x_tier_counts": dict(state_x_tier),
        "low_confidence_band": list(band),
        "band_member_count": band_member,
        "band_member_share_of_total": round(band_member / total, 4) if total else 0.0,
        "band_member_by_state": dict(band_member_by_state),
        "band_member_note": (
            "Membership is computed for EVERY detection regardless of policy_state "
            "(score in [lo, hi)); the overlay (overlay.js) only ever *applies* the "
            "dashed/dimmed band styling to policy_state=='accepted' detections - the "
            "band_member_by_state breakdown shows how many band-member detections "
            "landed in a different state (e.g. unknown_low_confidence) where the "
            "band check never fires, because that state's own de-emphasis already "
            "dominates the render."
        ),
    }


def _live_session(cfg, imgs: list[np.ndarray], *, seconds: float) -> dict:
    """Real FastAPI app, real /ws/ingest -> /ws/state, real Phase 8 analysis
    loop timing. Pushes the clip's frames at analysis_fps for `seconds` and
    counts EVERY incoming StateSnapshot (stale included) - this is the only
    way to observe `stale` and `pose_stale`, which are frame-arrival-timing
    properties the offline batch pass above cannot produce."""

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
    stale_total = 0
    pose_stale_total = 0
    state_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    detections_seen = 0
    stale_run_lengths: list[int] = []
    _cur_stale_run = [0]

    app = create_app(e2e_cfg)
    period = 1.0 / cfg.capture.analysis_fps
    with TestClient(app) as client:
        stop = threading.Event()

        def _reader():
            nonlocal snapshots_total, stale_total, pose_stale_total, detections_seen
            with client.websocket_connect("/ws/state") as sock:
                while not stop.is_set():
                    try:
                        snap = StateSnapshot.from_wire_json(sock.receive_text())
                    except Exception:
                        return
                    snapshots_total += 1
                    if snap.stale:
                        stale_total += 1
                        _cur_stale_run[0] += 1
                    else:
                        if _cur_stale_run[0] > 0:
                            stale_run_lengths.append(_cur_stale_run[0])
                        _cur_stale_run[0] = 0
                    if snap.pose_stale:
                        pose_stale_total += 1
                    for d in snap.detections:
                        detections_seen += 1
                        state_counts[d.policy_state] += 1
                        tier_counts[d.tier] += 1

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

    if _cur_stale_run[0] > 0:
        stale_run_lengths.append(_cur_stale_run[0])

    return {
        "seconds": seconds,
        "stale_after_ms": cfg.consumer.stale_after_ms,
        "snapshots_total": snapshots_total,
        "stale_snapshots": stale_total,
        "stale_share": round(stale_total / snapshots_total, 4) if snapshots_total else 0.0,
        "pose_stale_snapshots": pose_stale_total,
        "pose_stale_share": round(pose_stale_total / snapshots_total, 4) if snapshots_total else 0.0,
        "detections_seen_across_all_snapshots": detections_seen,
        "policy_state_counts_live": dict(state_counts),
        "tier_counts_live": dict(tier_counts),
        "stale_run_count": len(stale_run_lengths),
        "stale_run_lengths_snapshots": stale_run_lengths,
        "note": (
            "A 'stale run' is a maximal sequence of consecutive stale snapshots "
            "between two non-stale ones - each run is one observed "
            "active-to-grey-to-active toggle. detections_seen_across_all_snapshots "
            "is necessarily small: stale=True implies an empty detections list "
            "(pipeline/loop.py), and analysis_fps subsamples relative to the "
            "broadcast rate."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 9 grey-state root-cause frequency audit.")
    p.add_argument("--source", default="data/raw", help="clip, or directory of clips/stills")
    p.add_argument("--frames", type=int, default=600, help="max real frames for the batch pass")
    p.add_argument("--model", default="yolo11n")
    p.add_argument("--profile", default="dev")
    p.add_argument("--live-seconds", type=float, default=20.0,
                   help="live real-time /ws/ingest->/ws/state session duration; 0 to skip")
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

    spec = model_spec(args.model, config, models_dir=_REPO / "models")
    try:
        detector = ObjectDetector(
            spec.detector_config, provider=config.perception.provider, warmup=True,
            intra_op_threads=config.perception.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start detector: %s", exc)
        return 2
    policy = RecognitionPolicy(config.policy)

    imgs = list(_frames(source, args.frames))
    if not imgs:
        _LOG.error("no frames decoded from %s", source)
        return 2
    _LOG.info("loaded %d frames from %s for the batch pass", len(imgs), source)

    band = tuple(config.perception.detector.low_confidence_band)
    batch = _batch(detector, policy, band, imgs)
    _LOG.info(
        "batch: %d frames, %d raw dets -> %s (band[%.2f,%.2f) member share %.3f)",
        batch["frames"], batch["raw_detections"], batch["policy_state_counts"],
        band[0], band[1], batch["band_member_share_of_total"],
    )

    live = None
    if args.live_seconds > 0:
        _LOG.info("live real-time session (%.0fs, same clip, real /ws/ingest + /ws/state)...",
                  args.live_seconds)
        try:
            live = _live_session(config, imgs, seconds=args.live_seconds)
            _LOG.info(
                "live: %d snapshots, stale %.1f%% (%d runs), pose_stale %.1f%%",
                live["snapshots_total"], live["stale_share"] * 100.0,
                live["stale_run_count"], live["pose_stale_share"] * 100.0,
            )
        except Exception as exc:  # noqa: BLE001 - measurement only, report and continue
            _LOG.warning("live session failed: %r", exc)
            live = {"error": repr(exc)}

    payload = {
        "header": (
            "GREY-STATE ROOT-CAUSE FREQUENCY TABLE (Phase 9 section 4). Batch = "
            "detector+policy over every decoded frame of the source (content-driven "
            "states: policy_state, tier, low_confidence_band). Live session = the "
            "real FastAPI app's real-time loop over the same clip (timing-driven "
            "state: snapshot staleness)."
        ),
        "model": spec.name,
        "profile": args.profile,
        "source": str(source),
        "policy_thresholds": policy.active_thresholds(),
        "policy_config": {
            "default_threshold": config.policy.default_threshold,
            "margin_min": config.policy.margin_min,
            "low_confidence_band": list(band),
        },
        "consumer_stale_after_ms": config.consumer.stale_after_ms,
        "batch": batch,
        "live_session": live,
    }
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "grey_state_frequency.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "grey_state_frequency.md").write_text(_markdown(payload), encoding="utf-8")
    _LOG.info("wrote %s", results_dir / "grey_state_frequency.md")
    return 0


def _markdown(p: dict) -> str:
    b = p["batch"]
    L = [
        "# Grey-state root-cause frequency table (Phase 9 section 4)",
        "",
        f"> {p['header']}",
        "",
        f"Model `{p['model']}` · profile `{p['profile']}` · source `{p['source']}` · "
        f"{b['frames']} real frames · {b['raw_detections']} raw detections · "
        f"{b['accepted_detections']} detections after policy.",
        "",
        "## 1. `policy_state` frequency (batch, real footage)",
        "",
        "| policy_state | count | share |",
        "|---|---|---|",
    ]
    for s in POLICY_STATES:
        L.append(f"| {s} | {b['policy_state_counts'][s]} | {b['policy_state_share'][s]:.3f} |")
    L += [
        "",
        "## 2. `tier` frequency (batch, real footage)",
        "",
        "| tier | count |",
        "|---|---|",
    ]
    for t in _TIERS:
        L.append(f"| {t} | {b['tier_counts'].get(t, 0)} |")
    L += [
        "",
        "## 3. `low_confidence_band` membership (batch, real footage)",
        "",
        f"Band: `{b['low_confidence_band']}` (config `perception.detector.low_confidence_band`, "
        f"independent of `policy.default_threshold` = {p['policy_config']['default_threshold']}).",
        "",
        f"**{b['band_member_count']}** of {b['accepted_detections']} detections "
        f"({b['band_member_share_of_total']:.3f}) fall in the band, broken down by "
        "the policy_state they actually landed in:",
        "",
        "| policy_state | band members | note |",
        "|---|---|---|",
    ]
    for s in POLICY_STATES:
        n = b["band_member_by_state"].get(s, 0)
        note = "overlay applies dashed/dimmed styling here" if s == "accepted" else "overlay never checks the band for this state"
        L.append(f"| {s} | {n} | {note} |")
    L += ["", b["band_member_note"], ""]

    live = p.get("live_session")
    L += ["", "## 4. Snapshot staleness (live real-time session, same clip)", ""]
    if not live:
        L.append("_(skipped: --live-seconds 0)_")
    elif "error" in live:
        L.append(f"_(failed: {live['error']})_")
    else:
        L += [
            f"{live['seconds']:.0f}s live session over `/ws/ingest` -> `/ws/state`, "
            f"`consumer.stale_after_ms` = {live['stale_after_ms']}.",
            "",
            "| measure | value |", "|---|---|",
            f"| snapshots received | {live['snapshots_total']} |",
            f"| **stale snapshots (share)** | **{live['stale_snapshots']} ({live['stale_share']:.3f})** |",
            f"| pose_stale snapshots (share) | {live['pose_stale_snapshots']} ({live['pose_stale_share']:.3f}) |",
            f"| stale runs (active→grey→active toggles observed) | {live['stale_run_count']} |",
            f"| stale run lengths (consecutive snapshots) | {live['stale_run_lengths_snapshots']} |",
            f"| detections seen across all live snapshots | {live['detections_seen_across_all_snapshots']} |",
            "",
            live["note"],
        ]
    L += ["", "## Active per-class thresholds", "",
          "```", json.dumps(p["policy_thresholds"], indent=2), "```", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
