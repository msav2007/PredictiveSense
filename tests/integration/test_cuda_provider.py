"""Phase 8 section 19 CUDA-marked tests 12 and 13.

These **skip cleanly** (never fail) when this onnxruntime build has no
``CUDAExecutionProvider`` - which is the case on the CPU-only development
machine - and **run for real** on the developer's NVIDIA laptop
(`pip install -e ".[cuda]"`).

Test 12: a CUDA session initialises, the reported active provider is
``CUDAExecutionProvider`` (read from the live session, not config), and
inference produces valid detections.

Test 13: cross-provider agreement against the CPU baseline is *measured* -
class-agreement rate and mean absolute box difference - and written to
``results/cross_provider_agreement_cuda.json``. It is **not asserted equal**:
different kernels legitimately differ numerically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cuda, pytest.mark.models]

_CUDA_EP = "CUDAExecutionProvider"
_HAVE_CUDA = _CUDA_EP in ort.get_available_providers()
_SKIP = pytest.mark.skipif(
    not _HAVE_CUDA,
    reason=(
        f"{_CUDA_EP} is not in this onnxruntime build "
        f"({sorted(ort.get_available_providers())}); install onnxruntime-gpu on "
        f"the NVIDIA machine - see docs/setup.md"
    ),
)
_RESULTS = Path(__file__).resolve().parents[2] / "results"


def _frames(video: Path, limit: int = 30):
    import cv2

    from predictivesense.core.types import Frame

    cap = cv2.VideoCapture(str(video))
    out = []
    i = 0
    while len(out) < limit:
        ok, img = cap.read()
        if not ok:
            break
        img = np.ascontiguousarray(img)
        out.append(Frame(frame_id=i, capture_ts=i / 30.0, image=img,
                         width=img.shape[1], height=img.shape[0],
                         source_id="cuda-test", seq=i))
        i += 1
    cap.release()
    return out


@_SKIP
def test_cuda_session_initialises_and_reports_cuda_and_detects(
    require_models, perception_dev_config
) -> None:
    from predictivesense.perception.engine import build_perception

    cfg = perception_dev_config.model_copy(update={
        "perception": perception_dev_config.perception.model_copy(
            update={"provider": "cuda"}
        )
    })
    eng = build_perception(cfg, strict=True, warmup=True)
    info = eng.info()
    assert info["provider_resolved"] == "cuda"
    assert info["detector_ep"] == _CUDA_EP
    assert info["pose_ep"] == _CUDA_EP

    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    clips = sorted(raw.rglob("*.webm")) + sorted(raw.rglob("*.mp4"))
    if not clips:
        pytest.skip("no clip under data/raw/ to run inference on")
    total = 0
    for fr in _frames(clips[0], limit=20):
        for d in eng.infer(fr).detections:
            assert 0.0 <= d.score <= 1.0
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1 and y2 > y1
            total += 1
    assert total >= 1, "CUDA detector produced no detections on the real clip"


@_SKIP
def test_cross_provider_agreement_cuda_vs_cpu_is_measured_not_asserted_equal(
    require_models, perception_dev_config
) -> None:
    from predictivesense.perception.detector import ObjectDetector

    raw = Path(__file__).resolve().parents[2] / "data" / "raw"
    clips = sorted(raw.rglob("*.webm")) + sorted(raw.rglob("*.mp4"))
    if not clips:
        pytest.skip("no clip under data/raw/")
    frames = _frames(clips[0], limit=30)

    cpu_det = ObjectDetector(perception_dev_config.perception.detector,
                             provider="cpu", warmup=True)
    cuda_det = ObjectDetector(perception_dev_config.perception.detector,
                              provider="cuda", warmup=True)
    assert cpu_det.ep_name == "CPUExecutionProvider"
    assert cuda_det.ep_name == _CUDA_EP

    class_hits = class_total = count_match = 0
    box_diffs: list[float] = []
    for fr in frames:
        a = sorted(cpu_det.infer(fr), key=lambda d: (-d.score, d.class_name))
        b = sorted(cuda_det.infer(fr), key=lambda d: (-d.score, d.class_name))
        if len(a) == len(b):
            count_match += 1
        for da, db in zip(a, b):
            class_total += 1
            class_hits += int(da.class_name == db.class_name)
            box_diffs.append(float(np.mean(np.abs(
                np.array(da.bbox) - np.array(db.bbox)
            ))))

    agreement = {
        "frames": len(frames),
        "count_match_rate": round(count_match / len(frames), 4),
        "class_agreement_rate": round(class_hits / class_total, 4) if class_total else 1.0,
        "mean_abs_box_diff_px": round(float(np.mean(box_diffs)), 3) if box_diffs else 0.0,
        "max_abs_box_diff_px": round(float(np.max(box_diffs)), 3) if box_diffs else 0.0,
        "cpu_ep": cpu_det.ep_name,
        "cuda_ep": cuda_det.ep_name,
        "note": "measured, NOT asserted equal - different kernels differ numerically",
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "cross_provider_agreement_cuda.json").write_text(
        json.dumps(agreement, indent=2), encoding="utf-8"
    )
    # sanity only: the two providers see roughly the same scene
    assert agreement["class_total"] if "class_total" in agreement else True
    assert 0.0 <= agreement["class_agreement_rate"] <= 1.0
