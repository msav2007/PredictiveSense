# Attribution

Every runtime and dev dependency, and why it is here. Phase 0 adds nothing that
is not needed to satisfy the Phase 0 prompt. No model, cloud SDK, camera, or
network client is present.

## Runtime dependencies

| Package | Version | Why it is needed |
|---|---|---|
| `fastapi` | 0.115.6 | HTTP + WebSocket framework for `GET /health`, `GET /api/config`, `WS /ws/state`, `GET /`. |
| `uvicorn[standard]` | 0.32.1 | ASGI server that runs the app (`scripts/run_app.py`). The `standard` extra provides the `websockets` implementation used by `WS /ws/state`. `uvloop` is excluded on Windows by its own environment marker. |
| `pydantic` | 2.10.4 | Immutable, fully validated contract models in `core/types.py`; also the `StateSnapshot` JSON wire codec. |
| `pydantic-settings` | 2.6.1 | Typed settings model with strict (`extra="forbid"`) validation and environment overrides in `config/settings.py`. |
| `PyYAML` | 6.0.2 | Parse the `dev` / `eval` YAML profiles under `config/profiles/`. |
| `numpy` | 2.2.1 | `Frame.image` array type (`HxWx3`, BGR, `uint8`) and the seeded RNG that makes `SyntheticSource` deterministic. |
| `psutil` | 6.1.1 | Process RSS, total RAM, and logical CPU count for the session manifest and the no-op run's memory assertion. The standard library exposes none of these portably on Windows. |

## Dev dependencies

| Package | Version | Why it is needed |
|---|---|---|
| `pytest` | 8.3.4 | Test runner; markers `unit`, `integration`, `slow`, `hardware`. |
| `httpx` | 0.28.1 | Required by `fastapi.testclient.TestClient` for the API and WebSocket integration tests. Used only in `tests/`, never imported by `predictivesense/`. |

## Transitive dependencies

Pinned exactly in `requirements.lock.txt`, produced by `pip freeze` from the
clean install. Notable transitives: `starlette` (FastAPI's ASGI toolkit),
`websockets` / `httptools` / `watchfiles` / `python-dotenv` (uvicorn `standard`
extra), `anyio` + `sniffio` (Starlette / httpx async core), `click` +
`h11` (uvicorn / httpx).

## Standard library

`asyncio`, `threading`, `csv`, `json`, `subprocess` (for `git` state only),
`platform`, `pathlib`, `dataclasses`/`enum`/`typing`, `logging`, `argparse`,
`uuid`, `datetime`, `collections.deque`, `time` (`monotonic` / `perf_counter`
for all durations).
