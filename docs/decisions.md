# Decision log

One line per decision. Newest at the bottom. A contract rename or reshape in a
later phase must be recorded here.

## Phase 0

- **2026-09-07** Contracts are frozen **Pydantic v2** models (not dataclasses): gives the `StateSnapshot` JSON round-trip, field validation, and mutation-raises for free, and Pydantic is already a dependency via `pydantic-settings`.
- **2026-09-07** `Frame` sets `arbitrary_types_allowed=True` for `image: np.ndarray`; `Frame` is never compared by `==` or hashed in code or tests (numpy would make that ambiguous).
- **2026-09-07** Configuration uses `pydantic-settings` + one YAML file per profile under `config/profiles/`; `extra="forbid"` on every section so an unknown field is an error, not a silent ignore.
- **2026-09-07** `mode` accepts exactly `realtime` / `recorded` via the `Mode` enum; any other value is a `ValidationError`. No driver is implemented in Phase 0.
- **2026-09-07** Logging uses the **standard library `logging`** module (no new dependency): a single stdout `StreamHandler`, configurable level, format `"%(asctime)s %(levelname)-7s %(name)s | %(message)s"`. No `print()` in `predictivesense/`.
- **2026-09-07** `SyntheticSource.capture_ts` is a real `time.monotonic()` reading, nudged forward by 1e-9 s when the clock does not advance between reads, to guarantee a *strictly* increasing sequence.
- **2026-09-07** Per-frame synthetic image = the seeded base image rolled by `frame_id % width` columns: deterministic, cheap, visibly changing.
- **2026-09-07** `MailboxStats` lives in `core/types.py` (a frozen contract), so `camera/mailbox.py` imports it from `core` rather than defining its own.
- **2026-09-07** Non-synthetic `SourceKind` values raise `NotImplementedError` naming phase **P1**, from `camera/source.create_frame_source`.
- **2026-09-07** WebSocket broadcast is **poll-based, last-value-wins**: the broadcaster stores only the newest snapshot; each client's server task wakes at `broadcast.rate_hz` and sends the current snapshot, skipping any it missed. A send that exceeds `broadcast.client_send_timeout_s` drops that client. No per-client queue.
- **2026-09-07** `telemetry.Rate` and `telemetry.Samples` are `deque(maxlen=...)` (4096 / 100 000). At Phase 0 rates a 60 s run stays well under the latency cap; if it were ever exceeded, percentiles would be over the most recent samples.
- **2026-09-07** Metrics CSV schema: `elapsed_s, wall_utc, snapshot_id, frame_id, stale, loop_rate_hz, producer_rate_hz, consumed, dropped, mailbox_depth, iter_latency_ms, rss_mb`.
- **2026-09-07** `scripts/run_noop.py` always configures logging at `INFO` regardless of the profile's level, so the measurement summary is always emitted.
- **2026-09-07** Session manifest is a Pydantic model written as pretty JSON; `git_commit`/`git_dirty` come from `git rev-parse HEAD` and `git status --porcelain`, falling back to `("unknown", False)` with a logged warning if git is unavailable.
- **2026-09-07** pytest runs with `--import-mode=importlib`; `tests/conftest.py` puts the repo root on `sys.path` so the un-packaged `scripts/` directory is importable from `test_noop_run.py`.
- **2026-09-07** Python **3.11.9** is used, matching the Phase 0 prompt. It was installed on request because the machine previously had only 3.13 and 3.14.
