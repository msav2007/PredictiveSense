# Detector comparison — YOLO11n vs YOLOX-tiny

BLOCK 3.18-3.20. Both detectors run through the **identical** wrapper
(`DetectorConfig.decode`), recognition policy, splits and harness. Same policy,
same frames, no per-model tuning.

Machine: Intel Core Ultra 5 125H (14C/18T), CPU provider, `intra_op_threads: 6`.
Detector-only latency over 120 frames of the developer's 1080p `data/raw/` clip.

| model | family | licence | input | decode | size (MB) | latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| **yolo11n** (default) | Ultralytics YOLO11n | **AGPL-3.0-only** | 640 | yolo | 10.9 | **58.2 / 77.1** |
| yolox_tiny | Megvii YOLOX-tiny | **Apache-2.0** | 416 | yolox (grid) | 20.2 | **37.9 / 43.9** |

## Mechanical behaviour on the 34 sampled (unlabelled) frames

`scripts/policy_effect.py` — **counts, not accuracy** (no labels yet):

| model | raw detections | accepted | rejected out-of-domain | notable false positives |
|---|---|---|---|---|
| yolo11n | 43 | 39 | 4 | `umbrella` ×4 |
| yolox_tiny | 71 | 37 | 34 | `surfboard`, `bench`, `frisbee`, `couch`, … |

YOLOX-tiny is faster (smaller input) but fires far more loosely on this footage —
the recognition policy's domain filter rejects 34 of its 71 raw detections vs 4
of YOLO11n's 43. It would need heavier per-class thresholding to match YOLO11n's
precision.

## Per-class F1 / false-class rate — PENDING

These require the developer's `val` labels. When labelled:

```
python scripts/build_splits.py
python scripts/fit_thresholds.py --split val
python scripts/eval_detection.py --split val --model yolo11n   --policy on
python scripts/eval_detection.py --split val --model yolox_tiny --policy on
```

then fill:

| class | yolo11n F1 | yolox_tiny F1 | yolo11n false-class rate | yolox_tiny false-class rate |
|---|---|---|---|---|
| … | pending | pending | pending | pending |

## Licence decision

**Not settled this phase, and not claimed to be.** The Apache-2.0 alternative is
integrated and its unlabelled behaviour is measured, but the labelled per-class
comparison that would decide it has not been run. Default stays
`models/yolo11n.onnx` (AGPL-3.0). See `docs/attribution.md`.
