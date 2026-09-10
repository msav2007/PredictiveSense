"""Execution-provider benchmark for the Phase 2 perception models.

    python scripts\\benchmark_providers.py --provider cpu
    #  ... then, in the isolated .venv-dml (see BLOCK 9):
    python scripts\\benchmark_providers.py --provider dml

Runs the detector and the pose model over a fixed fixture clip and reports, per
model: warm-up ms, inference p50 / p95 / max ms, throughput, peak process RSS,
and - for any provider other than ``cpu`` - a numerical agreement check against
the CPU baseline (mean absolute box difference and class-agreement rate) so a
faster provider that changes the results is not adopted silently.

Writes ``results/providers_<provider>.json`` (machine-readable, includes the
per-frame detections needed as the next run's baseline) and
``results/providers_<provider>.md`` (a table).

``onnxruntime``, ``onnxruntime-directml`` and ``onnxruntime-openvino`` share the
module name and cannot coexist - this script only ever imports whichever build is
installed in the active interpreter.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from predictivesense.config.settings import load_config
from predictivesense.core.types import Frame
from predictivesense.logging_setup import configure_logging, get_logger
from predictivesense.perception.detector import ObjectDetector
from predictivesense.perception.pose import PoseEstimator

_LOG = get_logger("predictivesense.scripts.benchmark_providers")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 >= len(ordered):
        return ordered[-1]
    return ordered[low] + frac * (ordered[low + 1] - ordered[low])


def _default_fixture() -> Path:
    raw = _REPO_ROOT / "data" / "raw"
    if raw.is_dir():
        clips = sorted(raw.rglob("*.webm")) + sorted(raw.rglob("*.mp4"))
        if clips:
            return clips[0]
    fx = _REPO_ROOT / "tests" / "fixtures" / "fixture_clip.avi"
    if fx.is_file():
        return fx
    raise FileNotFoundError(
        "no fixture clip found - record one under data/raw/ or regenerate "
        "tests/fixtures/fixture_clip.avi"
    )


def _load_frames(path: Path, limit: int) -> list[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open fixture clip {path}")
    frames: list[Frame] = []
    idx = 0
    try:
        while len(frames) < limit:
            ok, img = cap.read()
            if not ok:
                break
            img = np.ascontiguousarray(img)
            frames.append(
                Frame(
                    frame_id=idx,
                    capture_ts=float(idx),
                    image=img,
                    width=int(img.shape[1]),
                    height=int(img.shape[0]),
                    source_id="bench",
                    seq=idx,
                )
            )
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"fixture clip {path} decoded zero frames")
    return frames


def _dets_key(dets) -> list[dict]:
    rows = [
        {
            "class_name": d.class_name,
            "score": round(d.score, 4),
            "bbox": [round(v, 2) for v in d.bbox],
        }
        for d in dets
    ]
    rows.sort(key=lambda r: (r["class_name"], -r["score"]))
    return rows


def _agreement(baseline: list[list[dict]], current: list[list[dict]]) -> dict:
    """Compare two runs' per-frame detections."""

    box_diffs: list[float] = []
    class_hits = 0
    class_total = 0
    count_match = 0
    frames = min(len(baseline), len(current))
    for i in range(frames):
        b, c = baseline[i], current[i]
        if len(b) == len(c):
            count_match += 1
        for jb, jc in zip(b, c):
            class_total += 1
            if jb["class_name"] == jc["class_name"]:
                class_hits += 1
            box_diffs.append(
                float(np.mean(np.abs(np.array(jb["bbox"]) - np.array(jc["bbox"]))))
            )
    return {
        "frames_compared": frames,
        "count_match_rate": round(count_match / frames, 4) if frames else 0.0,
        "class_agreement_rate": round(class_hits / class_total, 4) if class_total else 1.0,
        "mean_abs_box_diff_px": round(statistics.fmean(box_diffs), 3) if box_diffs else 0.0,
        "max_abs_box_diff_px": round(max(box_diffs), 3) if box_diffs else 0.0,
    }


def _bench_model(name: str, infer, frames: list[Frame], proc: psutil.Process) -> dict:
    # warm
    for f in frames[:3]:
        infer(f)
    times: list[float] = []
    per_frame_dets: list[list[dict]] = []
    peak_rss = proc.memory_info().rss
    for f in frames:
        t0 = time.perf_counter()
        out = infer(f)
        times.append((time.perf_counter() - t0) * 1000.0)
        per_frame_dets.append(_dets_key(out) if name == "detector" else [])
        peak_rss = max(peak_rss, proc.memory_info().rss)
    p50 = _pct(times, 50)
    row = {
        "model": name,
        "frames": len(frames),
        "infer_ms_p50": round(p50, 2),
        "infer_ms_p95": round(_pct(times, 95), 2),
        "infer_ms_max": round(max(times), 2),
        "throughput_fps": round(1000.0 / p50, 2) if p50 > 0 else 0.0,
        "peak_rss_mb": round(peak_rss / 1e6, 1),
    }
    return row | {"_per_frame_dets": per_frame_dets}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark an ONNX execution provider.")
    p.add_argument(
        "--provider",
        choices=("auto", "cpu", "cuda", "directml", "dml", "openvino"),
        required=True,
    )
    p.add_argument("--fixture", default=None, help="clip path (default: data/raw/ or the test fixture)")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--profile", default="dev")
    p.add_argument("--results-dir", default=str(_REPO_ROOT / "results"))
    args = p.parse_args(argv)
    configure_logging("INFO")
    try:  # the Windows console is cp1252; the report table has non-latin-1 glyphs
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    fixture = Path(args.fixture) if args.fixture else _default_fixture()
    _LOG.info("provider=%s fixture=%s frames<=%d", args.provider, fixture, args.frames)
    frames = _load_frames(fixture, args.frames)
    _LOG.info("decoded %d frames (%dx%d)", len(frames), frames[0].width, frames[0].height)

    pcfg = load_config(args.profile).perception
    proc = psutil.Process()
    proc.cpu_percent(None)

    try:
        detector = ObjectDetector(
            pcfg.detector, provider=args.provider, warmup=True,
            intra_op_threads=pcfg.intra_op_threads,
        )
        pose = PoseEstimator(
            pcfg.pose, provider=args.provider, warmup=True,
            intra_op_threads=pcfg.intra_op_threads,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _LOG.error("cannot start %s session: %s", args.provider, exc)
        return 2

    det_row = _bench_model("detector", detector.infer, frames, proc)
    pose_row = _bench_model("pose", pose.infer, frames, proc)
    cpu_pct = proc.cpu_percent(None)

    det_dets = det_row.pop("_per_frame_dets")
    pose_row.pop("_per_frame_dets", None)
    det_row["warmup_ms"] = round(detector.warmup_ms, 1)
    pose_row["warmup_ms"] = round(pose.warmup_ms, 1)
    det_row["ep_in_use"] = detector.ep_name
    pose_row["ep_in_use"] = pose.ep_name

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Machine fingerprint - so results from two laptops are never conflated
    # (Phase 8 section 14). GPU utilisation / memory are "where practical": not
    # captured here (no vendor tool wired); the developer records them on the
    # NVIDIA machine from `nvidia-smi` alongside this run.
    import platform as _plat

    try:
        import onnxruntime as _ort

        _ort_ver = _ort.__version__
        _ort_eps = list(_ort.get_available_providers())
    except Exception:  # noqa: BLE001
        _ort_ver, _ort_eps = "unknown", []
    machine = {
        "cpu": _plat.processor() or _plat.machine(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": _plat.platform(),
        "python": _plat.python_version(),
        "onnxruntime": _ort_ver,
        "available_providers": _ort_eps,
        "intra_op_threads": pcfg.intra_op_threads,
        "detector_input_size": pcfg.detector.input_size,
        "requested_provider": args.provider,
        "active_ep_detector": detector.ep_name,
        "active_ep_pose": pose.ep_name,
        "model_detector": det_row.get("model") and Path(str(pcfg.detector.model_path)).name,
        "gpu_util_percent": None,   # developer records from nvidia-smi on the NVIDIA box
        "gpu_mem_mb": None,
    }

    agreement = None
    if args.provider not in ("cpu",):
        base_path = results_dir / "providers_cpu.json"
        if base_path.is_file():
            base = json.loads(base_path.read_text(encoding="utf-8"))
            agreement = _agreement(base.get("detector_per_frame_dets", []), det_dets)
            _LOG.info("agreement vs CPU baseline: %s", agreement)
        else:
            _LOG.warning("no CPU baseline at %s - run --provider cpu first", base_path)

    payload = {
        "provider": args.provider,
        "machine": machine,
        "fixture": str(fixture),
        "frames": len(frames),
        "frame_size": [frames[0].width, frames[0].height],
        "detector": det_row,
        "pose": pose_row,
        "combined_ms_p50": round(det_row["infer_ms_p50"] + pose_row["infer_ms_p50"], 2),
        "process_cpu_percent": round(cpu_pct, 1),
        "agreement_vs_cpu": agreement,
        "detector_per_frame_dets": det_dets,
    }
    json_path = results_dir / f"providers_{args.provider}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = _markdown(payload)
    md_path = results_dir / f"providers_{args.provider}.md"
    md_path.write_text(md, encoding="utf-8")
    _LOG.info("wrote %s and %s", json_path, md_path)
    print("\n" + md)
    return 0


def _markdown(p: dict) -> str:
    d, po = p["detector"], p["pose"]
    mm = p.get("machine", {})
    lines = [
        f"# Execution-provider benchmark - `{p['provider']}`",
        "",
        f"**Machine:** {mm.get('cpu', '?')} · {mm.get('logical_cores', '?')} cores "
        f"· {mm.get('ram_gb', '?')} GB · {mm.get('os', '?')} · Python "
        f"{mm.get('python', '?')} · onnxruntime {mm.get('onnxruntime', '?')} "
        f"· EPs {mm.get('available_providers', [])} · intra_op "
        f"{mm.get('intra_op_threads', '?')} · detector input "
        f"{mm.get('detector_input_size', '?')} · GPU util/mem "
        f"{mm.get('gpu_util_percent')}/{mm.get('gpu_mem_mb')} "
        f"(record from nvidia-smi on the NVIDIA box)",
        "",
        f"Fixture: `{p['fixture']}` · {p['frames']} frames · "
        f"{p['frame_size'][0]}x{p['frame_size'][1]} · EP actually in use: "
        f"detector `{d['ep_in_use']}` / pose `{po['ep_in_use']}`",
        "",
        "| model | warm-up ms | p50 ms | p95 ms | max ms | throughput fps | peak RSS MB |",
        "|---|---|---|---|---|---|---|",
        f"| detector | {d['warmup_ms']} | {d['infer_ms_p50']} | {d['infer_ms_p95']} "
        f"| {d['infer_ms_max']} | {d['throughput_fps']} | {d['peak_rss_mb']} |",
        f"| pose | {po['warmup_ms']} | {po['infer_ms_p50']} | {po['infer_ms_p95']} "
        f"| {po['infer_ms_max']} | {po['throughput_fps']} | {po['peak_rss_mb']} |",
        "",
        f"Combined detector+pose p50: **{p['combined_ms_p50']} ms** · "
        f"process CPU during run: {p['process_cpu_percent']} %",
        "",
    ]
    ag = p["agreement_vs_cpu"]
    if p["provider"] == "cpu":
        lines.append("_This run is the CPU baseline; other providers are checked against it._")
    elif ag is not None:
        lines += [
            "## Numerical agreement vs the CPU baseline",
            "",
            f"- frames compared: {ag['frames_compared']}",
            f"- detection-count match rate: {ag['count_match_rate']}",
            f"- class-agreement rate (matched pairs): {ag['class_agreement_rate']}",
            f"- mean |box diff|: {ag['mean_abs_box_diff_px']} px · max: {ag['max_abs_box_diff_px']} px",
            "",
            "A provider is only adopted if it is faster **and** agrees "
            "(class-agreement ~1.0, mean |box diff| within a couple of px).",
        ]
    else:
        lines.append("_No CPU baseline found - run `--provider cpu` first for the agreement check._")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
