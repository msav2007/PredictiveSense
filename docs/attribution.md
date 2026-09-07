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

## Model files (not committed - `models/manifest.json` only)

| Model | Family | Task | Licence | Source |
|---|---|---|---|---|
| `yolo11n.onnx` | Ultralytics YOLO11n | detection (COCO-80) | **AGPL-3.0-only** | `github.com/ultralytics/assets` release `v8.3.0`, pre-exported ONNX (opset 22, dynamic input) |
| `yolo11n-pose.onnx` | Ultralytics YOLO11n-pose | pose (17 keypoints) | **AGPL-3.0-only** | same release, input locked to 640x640 |

Fetched and hash-verified by `python scripts/fetch_models.py` against
`models/manifest.json`. See the open-decision box above.

## Optional dependencies (`camera` extra)

| Package | Version | Licence | Why it is needed |
|---|---|---|---|
| `pygrabber` | 0.2 | MIT | Human-readable DirectShow camera names for backend enumeration (`GET /api/cameras`). Windows-only. Never imported at import time; a missing or throwing `pygrabber` degrades to `"Camera <index>"` and logs once at INFO. |
| `comtypes` | 1.4.16 | MIT | `pygrabber`'s only dependency (COM interop). Pulled in transitively by the `camera` extra. |

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
