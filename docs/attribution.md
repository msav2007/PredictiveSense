# Attribution

Every runtime and dev dependency, and why it is here. No cloud SDK or outbound
network client is present. Phase 2 adds an ONNX inference runtime and two
pretrained model files (fetched by a manual setup script, never at runtime).

## ⚠️ OPEN LICENCE DECISION - Phase 2 models are AGPL-3.0

The Phase 2 detector and pose estimator are **Ultralytics YOLO11** models
(`yolo11n`, `yolo11n-pose`), pre-exported to ONNX and published by Ultralytics.
**Ultralytics YOLO is licensed AGPL-3.0-only.** The model *files* are not
committed (only `models/manifest.json` with their hashes and licence metadata),
but shipping or running this repository with these weights makes the AGPL-3.0
terms relevant to how PredictiveSense may itself be released.

This is recorded as an **open, unresolved decision** - it is *not* resolved here:

- If the project is released under a non-AGPL licence, these weights must be
  swapped for a permissively-licensed detector, or a commercial Ultralytics
  licence obtained, or the whole thing kept AGPL.
- The perception code is **model-agnostic** (see `docs/decisions.md`): switching
  to another ONNX detector/pose model is a config change (`perception.detector.
  model_path` / `perception.pose.model_path` + class list from model metadata),
  not a code change. So the decision can be deferred without lock-in.
- `models/manifest.json` carries `license: "AGPL-3.0-only"` and a `license_note`
  per model; `scripts/fetch_models.py` also writes `models/LICENSE-AGPL-3.0.txt`
  beside the weights (git-ignored).

### Phase 2.5 - evidence gathered, decision still deferred (2026-09-08)

Phase 2.5 wired **Megvii YOLOX-tiny** (`yolox_tiny.onnx`, **Apache-2.0**) through
the identical detector wrapper (new `decode: "yolox"` variant), recognition
policy, splits and harness - the permissively-licensed alternative BLOCK 3.18
asks for. It is fetched + SHA-256-verified by `scripts/fetch_models.py`
(`models/manifest.json` `detector_alt`; `.onnx` still git-ignored).

**What is measured** (`results/model_comparison.md`, `results/latency_2_5.md`,
`results/policy_effect_unlabelled_*.md`): both models over the identical 34
sampled frames of the developer's footage, and isolated latency. On *unlabelled*
frames YOLOX-tiny fires far more loosely (71 raw detections vs YOLO11n's 43; 34
out-of-domain rejections by the policy vs 4) - it would need heavier per-class
thresholding. Detection *accuracy* (per-class F1, false-class rate) needs the
developer's `val` labels and is **pending**.

**Decision: still deferred, and this phase does NOT claim it is settled.** The
alternative is integrated and its behaviour is partially measured, but the
labelled `val` comparison that would actually decide it has not been run (no
labels yet). Until then the default stays `models/yolo11n.onnx` (AGPL-3.0). When
`val` is labelled, `scripts/eval_detection.py --model yolox_tiny --policy on`
completes `results/model_comparison.md` and the decision is made on evidence.

## Runtime dependencies

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `fastapi` | 0.115.6 | MIT | HTTP + WebSocket framework for `GET /health`, `GET /api/config`, `WS /ws/state`, `GET /`, and the Phase 1 `/api/cameras`, `/ws/ingest`, `/api/videos`, `/api/analyze`, `/api/record/upload`, `/api/clips`, `/api/debug/stall`, `/static/*`. |
| `uvicorn[standard]` | 0.32.1 | BSD-3-Clause | ASGI server that runs the app (`scripts/run_app.py`). The `standard` extra provides the `websockets` implementation used by `WS /ws/state` and `WS /ws/ingest`. `uvloop` is excluded on Windows by its own environment marker. |
| `pydantic` | 2.10.4 | MIT | Immutable, fully validated contract models in `core/types.py` (incl. `SourceInfo`, `ClipManifest`, `IngestHeader`); the `StateSnapshot` JSON wire codec. |
| `pydantic-settings` | 2.6.1 | MIT | Typed settings model with strict (`extra="forbid"`) validation and environment overrides in `config/settings.py`, including the Phase 1 `capture` / `recorder` / `video` sections. |
| `PyYAML` | 6.0.2 | MIT | Parse the `dev` / `eval` YAML profiles under `config/profiles/`. |
| `numpy` | 2.2.1 | BSD-3-Clause | `Frame.image` array type (`HxWx3`, BGR, `uint8`); the seeded RNG for `SyntheticSource`; `np.frombuffer` for the ingest JPEG buffer. |
| `psutil` | 6.1.1 | BSD-3-Clause | Process RSS, total RAM, and logical CPU count for the session manifest and the no-op run's memory assertion. The standard library exposes none of these portably on Windows. |
| `opencv-python` | 4.10.0.84 | Apache-2.0 (OpenCV); MIT (the PyPI packaging) | `cv2.VideoCapture` for `DeviceSource` (MSMF/DSHOW) and `FileSource` (recorded video); `cv2.imdecode` for the browser ingest path; `cv2.VideoWriter` for the deterministic MJPG test fixture; clip-duration probe. Imported **only** under `predictivesense/camera/`. Prebuilt wheels only - no runtime download, no CUDA. |
| `python-multipart` | 0.0.20 | Apache-2.0 | `multipart/form-data` parsing for `POST /api/record/upload` (Starlette requires it for `UploadFile` / `Form`). |
| `onnxruntime` | 1.24.4 | MIT | Phase 2 perception: runs the ONNX object detector and pose estimator (`predictivesense/perception/`). CPU build only in the main `.venv`. `onnxruntime-directml` and `onnxruntime-openvino` share the module name and **must not** be co-installed - the DirectML benchmark uses a separate `.venv-dml`. Prebuilt wheels only - no runtime download, **no CUDA**. Imported only under `predictivesense/perception/`. Transitive: `protobuf`, `flatbuffers`, `sympy`+`mpmath`, `coloredlogs`+`humanfriendly`, `pyreadline3`. |

**Phase 2.5 adds no runtime dependency.** `predictivesense/dataset/` and
`predictivesense/eval/` are stdlib + `numpy` only. The evaluation *scripts* that
decode frame images use `cv2.imread` and live in `scripts/` (outside the scanned
package), so the `cv2`/`onnxruntime` import scoping is unchanged. The labelling
tool is plain ES modules from `static/label/`.

## Model files (not committed - `models/manifest.json` only)

| Model | Family | Task | Licence | Source |
|---|---|---|---|---|
| `yolo11n.onnx` | Ultralytics YOLO11n | detection (COCO-80) | **AGPL-3.0-only** | `github.com/ultralytics/assets` release `v8.3.0`, pre-exported ONNX (opset 22, dynamic input) |
| `yolo11n-pose.onnx` | Ultralytics YOLO11n-pose | pose (17 keypoints) | **AGPL-3.0-only** | same release, input locked to 640x640 |
| `yolox_tiny.onnx` | Megvii YOLOX-tiny | detection (COCO-80) | **Apache-2.0** | `github.com/Megvii-BaseDetection/YOLOX` release `0.1.1rc0`, input locked to 416x416, raw head (grid decode). Phase 2.5 permissive-licence comparison detector - see the open-decision box above. |

Fetched and hash-verified by `python scripts/fetch_models.py` against
`models/manifest.json`. See the open-decision box above.

## Training dependencies (`train` extra, Phase 11 Part B - not previously recorded here)

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `torch` | 2.14.0 (CPU wheel) | BSD-3-Clause | Trains the crop classifier (`predictivesense/training/`) - the ONLY place it is imported (`tests/unit/test_no_forbidden_imports.py` enforces this; the serving app never needs it, see `perception/classifier.py`). |
| `torchvision` | 0.29.0 (CPU wheel) | BSD-3-Clause | Transform utilities used by `predictivesense/training/crop_data.py`. |
| `onnxscript` | 0.7.2 | Apache-2.0/MIT | Required by `torch.onnx.export`'s dynamo-capable path; training-only. |

## Datasets

| Dataset | Version / subset | Licence situation | Why it is needed |
|---|---|---|---|
| Object Learning Studio samples (`data/objects/`) | ongoing, developer-collected | Our own - the originality claim for the paper | The Studio's own captured positives/negatives; never leaves `data/objects/`, never redistributed as raw images with the repo. |
| Phase 2.5 evaluation set (`data/eval/`) | ongoing, developer-collected | Our own | Held-out test data for detection eval; strictly separate from training data (`tests/unit/test_dataset_separation.py`). |
| **Open Images V7** (Phase 12 external import) | `open-images-v7-train-5000`, FiftyOne export, downloaded 2026-09-16, `train` split, 5000 images | **Annotations**: CC-BY-4.0 (Google, Open Images V7). **Images**: individually licensed by their original Flickr authors - overwhelmingly CC-BY-2.0 in the classes imported so far (`watch`), but **not uniformly one licence**; each imported sample's per-image `license`/`author`/`source_url` is preserved verbatim in `data/external/open-images-v7/<class>/manifest.json` (section 6.1). **Redistribution constraint: images are NOT copied into this repository at all** (`data/external/` is git-ignored; only manifests, SHA-256 hashes and `results/external_class_mapping.json` are committed) - so no image redistribution question arises for the repository itself, but any paper artefact that embeds one of these images directly must carry that specific image's own licence/attribution from the manifest, not a blanket claim. | Combined with (or, for `watch` specifically, used in place of - see `docs/decisions.md` Phase 12) Studio samples to train the Phase 11 crop classifier on real, human-annotated boxes for classes too laborious to self-collect from scratch. |

**Import-time dependency, never a runtime one (`external` extra):** `pandas` (BSD-3-Clause) is used only by `scripts/audit_external_dataset.py` and `scripts/import_external_dataset.py` to parse the FiftyOne export's CSVs directly from disk. Neither script, nor anything under `predictivesense/`, imports the `fiftyone` package itself - the one-time download was a manual, separate step; see the environment-contamination note in `docs/decisions.md` Phase 12 for why `fiftyone` must never be installed into this project's own `.venv` again.

## Algorithm attribution - Phase 9 tracker (no new dependency)

`predictivesense/tracking/` implements its association and lifecycle logic
in-repository - **no tracking library or package was added** (stdlib +
`numpy` only, same rule as `predictivesense/dataset/`/`predictivesense/eval/`
above). The algorithm *family* is cited here per the Phase 9 prompt's
requirement to attribute the approach, not copy an implementation:

- **SORT** (Simple Online and Realtime Tracking) - Bewley, A., Ge, Z.,
  Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple online and realtime
  tracking.* IEEE International Conference on Image Processing (ICIP).
  `arxiv.org/abs/1602.00763`. Source of the IoU-based greedy association +
  constant-velocity motion-prediction pattern this tracker's stage-1
  association and `TrackState.predicted_bbox` follow (this implementation
  uses a linear constant-velocity extrapolation of the box centre, not
  SORT's Kalman filter - a deliberate simplification given the detector's
  own per-frame localisation noise on this hardware was not measured to
  justify a full Kalman state).
- **ByteTrack** - Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z.,
  Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-object tracking by
  associating every detection box.* European Conference on Computer Vision
  (ECCV). `arxiv.org/abs/2110.06864`. Source of the two-stage association
  pattern this tracker's `association.py`/`tracker.py` follow: a first pass
  on high-scoring detections, a second IoU-only pass recovering
  already-tracked objects from lower-scoring detections that never creates a
  new track identity.

Both are cited for their algorithmic ideas; no code from either project's
reference implementation was copied. Every threshold (`n_init`, `max_age`)
is independently measured on this project's own footage
(`docs/decisions.md`), not carried over from either paper's tuning.

## Optional dependencies (`camera` extra)

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `pygrabber` | 0.2 | MIT | Human-readable DirectShow camera names for backend enumeration (`GET /api/cameras`). Windows-only. Never imported at import time; a missing or throwing `pygrabber` degrades to `"Camera <index>"` and logs once at INFO. |
| `comtypes` | 1.4.16 | MIT | `pygrabber`'s only dependency (COM interop). Pulled in transitively by the `camera` extra. |

## Optional dependencies (`external` extra, Phase 12)

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `pandas` | 3.0.5 | BSD-3-Clause | CSV parsing for `scripts/audit_external_dataset.py` / `scripts/import_external_dataset.py` only - never imported under `predictivesense/` (enforced by `tests/unit/test_no_forbidden_imports.py`, which also forbids `fiftyone` anywhere in the package). |

## Dev dependencies

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `pytest` | 8.3.4 | MIT | Test runner; markers `unit`, `integration`, `slow`, `hardware` (the last gated behind `--run-hardware`). |
| `httpx` | 0.28.1 | BSD-3-Clause | Required by `fastapi.testclient.TestClient` for the API and WebSocket integration tests. Used only in `tests/`, never imported by `predictivesense/`. |

## Transitive dependencies

Pinned exactly in `requirements.lock.txt`, produced by
`pip freeze --exclude-editable` from the clean install (UTF-8, no BOM). Notable
transitives: `starlette` (FastAPI's ASGI toolkit), `websockets` / `httptools` /
`watchfiles` / `python-dotenv` (uvicorn `standard` extra), `anyio` + `sniffio`
(Starlette / httpx async core), `click` + `h11` (uvicorn / httpx), `comtypes`
(pygrabber).

## Standard library

`asyncio`, `threading`, `csv`, `json`, `struct` (ingest framing), `subprocess`
(for `git` state only), `platform`, `pathlib`, `dataclasses`/`enum`/`typing`,
`logging`, `argparse`, `uuid`, `datetime`, `collections.deque`, `statistics`
(benchmark report only), `time` (`monotonic` / `perf_counter` for all
durations).
