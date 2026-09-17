"""Phase 9 section 7.2 - derive tracker thresholds from measurement, not invention.

    python scripts\\track_threshold_derivation.py --source data\\raw --seconds 40

`n_init`, `max_age` and the association thresholds must come from "the
measured detection interval and the measured miss rate on real footage"
(Phase 9 prompt, section 7.2) - not picked by feel. This script runs the real
FastAPI app's real-time analysis loop (same live harness as
`scripts/grey_state_audit.py` / `scripts/benchmark_latency.py`'s
`_end_to_end`) over the developer's own footage (`data/raw`) and measures, at
the ACTUAL real-time analysis cadence (not the video's native frame rate -
recorded/batch passes analyse every frame and are lossless per section 13.3,
which is the wrong cadence to derive a real-time coasting budget from):

* the inter-analysis-cycle interval (ms) between consecutive non-stale
  snapshots - this is the unit `max_age` must be expressed in;
* per raw class, the presence bitmap across the ordered sequence of
  non-stale snapshots, from which "miss run" (consecutive absences between
  two presences) and "presence run" (consecutive presences) lengths are
  computed.

`max_age_frames` is set at the 95th percentile of observed miss-run lengths
(plus one cycle of margin); `max_age_ms` is that many cycles at the measured
p50 interval. `n_init` is chosen from the presence-run-length distribution:
if length-1 runs (single-cycle blips) are common, a low n_init would promote
noise to a confirmed identity, so n_init is raised; the exact rule is applied
and shown in the output, not just the final number - see the JSON/MD for the
formula and the counts it was applied to.

Writes ``results/track_thresholds.{json,md}``. `docs/decisions.md` must cite
this file for the numbers actually adopted in `predictivesense/tracking/`.
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
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from predictivesense.config.settings import ConfigError, ValidationError, load_config
from predictivesense.logging_setup import configure_logging, get_logger

_LOG = get_logger("predictivesense.scripts.track_threshold_derivation")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# States a tracker would actually receive a box for - excludes rejected_size
# (policy's own judgement: too small / bad aspect) and suppressed_implausible
# (confidently an out-of-domain class) since neither is a candidate object.
_TRACK_ELIGIBLE_STATES = frozenset(
    {"accepted", "accepted_secondary", "unknown_low_confidence", "unknown_margin"}
)


def _pct(xs: list[float], p: float) -> float:
    return round(float(np.percentile(xs, p)), 2) if xs else -1.0


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


def _run_lengths(bitmap: list[bool]) -> tuple[list[int], list[int]]:
    """Return (presence_run_lengths, miss_run_lengths). A miss run only counts
    if it is bounded by a presence on both sides (an edge-trailing absence at
    the very start/end of the session is not a "the object was there and
    momentarily lost" event - it is "never seen" / "session ended")."""

    presence_runs: list[int] = []
    miss_runs: list[int] = []
    i = 0
    n = len(bitmap)
    # Trim leading/trailing False so only interior runs are counted.
    lo = 0
    while lo < n and not bitmap[lo]:
        lo += 1
    hi = n
    while hi > lo and not bitmap[hi - 1]:
        hi -= 1
    i = lo
    while i < hi:
        val = bitmap[i]
        j = i
        while j < hi and bitmap[j] == val:
            j += 1
        (presence_runs if val else miss_runs).append(j - i)
        i = j
    return presence_runs, miss_runs


def _live_presence_session(cfg, imgs: list[np.ndarray], *, seconds: float) -> dict:
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

    # One entry per non-stale snapshot, in arrival order: (wall_ms, {raw_class_name,...}).
    sequence: list[tuple[float, set[str]]] = []
    stale_count = [0]
    total_count = [0]

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
                    total_count[0] += 1
                    if snap.stale:
                        stale_count[0] += 1
                        continue
                    present = {
                        (d.raw_class_name or d.class_name)
                        for d in snap.detections
                        if d.policy_state in _TRACK_ELIGIBLE_STATES
                    }
                    sequence.append((time.monotonic() * 1000.0, present))

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
                header = IngestHeader(
                    client_ts_ms=time.time() * 1000.0, seq=i,
                    w=cfg.capture.analysis_width, h=cfg.capture.analysis_height,
                    cap_ts_ms=time.time() * 1000.0, enc_ms=0.0,
                )
                ws.send_bytes(encode_ingest_message(header, jpeg))
                i += 1
                time.sleep(period)
        time.sleep(0.75)
        stop.set()
        rt.join(timeout=2.0)

    if len(sequence) < 3:
        return {
            "error": f"only {len(sequence)} non-stale snapshots observed; need >=3",
            "total_snapshots": total_count[0], "stale_snapshots": stale_count[0],
        }

    intervals_ms = [
        sequence[i][0] - sequence[i - 1][0] for i in range(1, len(sequence))
    ]

    classes = sorted({c for _, present in sequence for c in present})
    presence_runs_all: list[int] = []
    miss_runs_all: list[int] = []
    per_class: dict[str, dict] = {}
    for cls in classes:
        bitmap = [cls in present for _, present in sequence]
        pres, miss = _run_lengths(bitmap)
        presence_runs_all += pres
        miss_runs_all += miss
        per_class[cls] = {
            "cycles_present": sum(bitmap),
            "presence_runs": pres,
            "miss_runs": miss,
        }

    n_singleton = sum(1 for r in presence_runs_all if r == 1)
    singleton_share = (
        n_singleton / len(presence_runs_all) if presence_runs_all else 0.0
    )
    # Rule (documented, not silent): if single-cycle presence "blips" make up
    # more than 20% of all presence runs, a track confirmed on a single hit
    # would too often be detector noise - require 3 consecutive hits instead
    # of 2. Applied mechanically below; the counts that drove it are in the
    # output so the rule can be re-checked against more footage later.
    n_init = 3 if singleton_share > 0.20 else 2

    interval_p50 = _pct(intervals_ms, 50)
    interval_p95 = _pct(intervals_ms, 95)
    miss_p95_cycles = int(np.ceil(np.percentile(miss_runs_all, 95))) if miss_runs_all else 0
    # +1 cycle margin: max_age is "survives while consecutive_misses < max_age",
    # so a track must outlive the worst observed miss run, not just match it.
    max_age_frames = max(2, miss_p95_cycles + 1)
    max_age_ms = round(max_age_frames * interval_p50, 1) if interval_p50 > 0 else -1.0

    return {
        "seconds": seconds,
        "total_snapshots": total_count[0],
        "stale_snapshots": stale_count[0],
        "non_stale_snapshots_used": len(sequence),
        "interval_ms_p50": interval_p50,
        "interval_ms_p95": interval_p95,
        "classes_observed": classes,
        "presence_run_lengths_all": presence_runs_all,
        "miss_run_lengths_all": miss_runs_all,
        "singleton_presence_runs": n_singleton,
        "singleton_presence_share": round(singleton_share, 4),
        "miss_run_p95_cycles": miss_p95_cycles,
        "derived": {
            "n_init": n_init,
            "n_init_rule": (
                "3 if singleton_presence_share > 0.20 else 2 "
                f"(measured singleton_presence_share={singleton_share:.4f} over "
                f"{len(presence_runs_all)} presence runs)"
            ),
            "max_age_frames": max_age_frames,
            "max_age_frames_rule": (
                "max(2, ceil(p95(miss_run_lengths_cycles)) + 1) "
                f"(measured miss_run_p95_cycles={miss_p95_cycles} over "
                f"{len(miss_runs_all)} interior miss runs)"
            ),
            "max_age_ms": max_age_ms,
            "max_age_ms_rule": "max_age_frames * interval_ms_p50",
        },
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Derive Phase 9 tracker thresholds from measurement.")
    p.add_argument("--source", default="data/raw")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--profile", default="dev")
    p.add_argument("--seconds", type=float, default=40.0)
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
    imgs = list(_frames(source, args.frames))
    if not imgs:
        _LOG.error("no frames decoded from %s", source)
        return 2
    _LOG.info("loaded %d frames from %s; running %.0fs live session", len(imgs), source, args.seconds)

    result = _live_presence_session(config, imgs, seconds=args.seconds)
    if "error" in result:
        _LOG.error("derivation failed: %s", result["error"])
        results_dir = Path(config.eval.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "track_thresholds.json").write_text(
            json.dumps({"error": result["error"], "raw": result}, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    d = result["derived"]
    _LOG.info(
        "derived: n_init=%d (%s) | max_age_frames=%d max_age_ms=%.1f (%s) | interval p50/p95 %.1f/%.1f ms",
        d["n_init"], d["n_init_rule"], d["max_age_frames"], d["max_age_ms"],
        d["max_age_frames_rule"], result["interval_ms_p50"], result["interval_ms_p95"],
    )

    payload = {
        "header": (
            "Phase 9 section 7.2 - n_init/max_age derived from a live real-time "
            "session (real /ws/ingest -> /ws/state, real analysis-loop timing) "
            "over data/raw, NOT invented constants."
        ),
        "profile": args.profile,
        "source": str(source),
        "result": result,
    }
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "track_thresholds.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "track_thresholds.md").write_text(_markdown(payload), encoding="utf-8")
    _LOG.info("wrote %s", results_dir / "track_thresholds.md")
    return 0


def _markdown(p: dict) -> str:
    r = p["result"]
    d = r["derived"]
    L = [
        "# Tracker threshold derivation (Phase 9 section 7.2)",
        "",
        f"> {p['header']}",
        "",
        f"Profile `{p['profile']}` · source `{p['source']}` · {r['seconds']:.0f}s live session · "
        f"{r['total_snapshots']} snapshots ({r['stale_snapshots']} stale, "
        f"{r['non_stale_snapshots_used']} non-stale used).",
        "",
        "## Measured analysis-cycle interval",
        "",
        "| measure | value (ms) |", "|---|---|",
        f"| interval p50 | {r['interval_ms_p50']} |",
        f"| interval p95 | {r['interval_ms_p95']} |",
        "",
        "## Presence / miss run-length distributions (per non-stale analysis cycle)",
        "",
        f"Classes observed: `{r['classes_observed']}`.",
        "",
        f"- {len(r['presence_run_lengths_all'])} interior presence runs, "
        f"{r['singleton_presence_runs']} of length 1 (share {r['singleton_presence_share']:.3f}).",
        f"- {len(r['miss_run_lengths_all'])} interior miss runs; p95 = {r['miss_run_p95_cycles']} cycles.",
        "",
        "## Derived thresholds", "",
        "| parameter | value | rule applied |", "|---|---|---|",
        f"| `n_init` | {d['n_init']} | {d['n_init_rule']} |",
        f"| `max_age` (frames) | {d['max_age_frames']} | {d['max_age_frames_rule']} |",
        f"| `max_age` (ms) | {d['max_age_ms']} | {d['max_age_ms_rule']} |",
        "",
        "## Per-class detail", "",
        "| class | cycles present | presence runs | miss runs |", "|---|---|---|---|",
    ]
    for cls, v in r["per_class"].items():
        L.append(f"| {cls} | {v['cycles_present']} | {v['presence_runs']} | {v['miss_runs']} |")
    L += ["", "Only one recorded clip is available (`data/raw`) at time of writing - these "
          "thresholds should be re-derived once more sessions are recorded (see "
          "docs/decisions.md and the phase report's Not verified / limitations section).", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
