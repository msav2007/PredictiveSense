# Architecture (Phase 0)

## Shape

```
SyntheticSource --frames--> LatestFrameMailbox --newest--> AnalysisLoop consumer
                                                              |
                                                              v
                                                        StateSnapshot
                                                         /          \
                                              MetricsWriter (CSV)   Broadcaster --> WS clients
```

Everything is injectable. There is no global mutable state, no singleton, and no
import-time side effect. Threads exist only in `predictivesense/pipeline/`.

## Threading model

- **Producer thread** (`ps-producer`): calls `SyntheticSource.read()` in a loop
  and `put()`s each frame into the mailbox. `read()` paces itself to the
  configured `source.target_fps`.
- **Consumer thread** (`ps-consumer`): wakes at `consumer.sample_rate_hz`, does
  one `mailbox.get()`, times the iteration, builds a `StateSnapshot`, and hands
  it to every registered listener.
- **Shutdown**: `AnalysisLoop.stop()` sets an `Event`, joins both threads with a
  5 s timeout, then stops the source. A thread still alive after the join, or a
  thread that raised, sets `AnalysisLoop.error`; `run_for()` re-raises it and the
  scripts exit non-zero.

## Bounded everywhere

- `LatestFrameMailbox` holds exactly one frame. `put()` on a full slot drops the
  waiting frame and increments `dropped`. Depth is always 0 or 1.
- The broadcaster keeps only the single most recent snapshot. Each WS client is
  served by a poll loop at `broadcast.rate_hz` that sends the current snapshot;
  intermediate snapshots for a client that cannot keep up are simply skipped
  (last-value-wins). A client whose send does not finish within
  `broadcast.client_send_timeout_s` is dropped. `publish()` is an O(1) lock plus
  assignment and never touches a client, so a slow client cannot slow the loop.
- `telemetry.Rate` and `telemetry.Samples` are `deque`s with a fixed `maxlen`
  (rate window buffer 4096; latency reservoir 100 000).

## Mode awareness

`mode` (`realtime` | `recorded`) is validated in config and carried on every
`StateSnapshot`. Only `predictivesense/pipeline/` may branch on it. Phase 0
implements neither driver; the field exists so later phases do not reshape the
config or the wire format.

## `StateSnapshot` wire format

Broadcast over `WS /ws/state` and returned (shape-compatible) by the loop. JSON
is produced by `StateSnapshot.to_wire_json()` (Pydantic v2 `model_dump_json`)
and parsed by `StateSnapshot.from_wire_json()`. Enums serialise to their string
values; `None` is JSON `null`.

| Field | JSON type | Phase 0 value |
|---|---|---|
| `snapshot_id` | number | monotonic from 1 |
| `mode` | string | `"realtime"` or `"recorded"` |
| `frame_id` | number \| null | id of the frame this snapshot describes, else null |
| `capture_ts` | number \| null | `time.monotonic()` at capture, else null |
| `emitted_ts` | number | `time.monotonic()` at emission |
| `frame_age_ms` | number \| null | `(emitted_ts - capture_ts) * 1000`, else null |
| `detections` | array | always `[]` |
| `poses` | array | always `[]` |
| `tracks` | array | always `[]` |
| `risk` | object \| null | always `null` |
| `metrics` | object (string -> number) | `loop_rate_hz`, `producer_rate_hz`, `consumed`, `dropped`, `mailbox_depth`, `iter_latency_ms` |
| `stale` | boolean | `true` when no frame arrived within `consumer.stale_after_ms` |

Later phases may add fields. Renaming or removing one requires a line in
`docs/decisions.md`.

## Metrics CSV schema

Written by `scripts/run_noop.py` via `telemetry.MetricsWriter`. One header row,
then one row per `noop.csv_sample_period_s` seconds:

```
elapsed_s, wall_utc, snapshot_id, frame_id, stale, loop_rate_hz,
producer_rate_hz, consumed, dropped, mailbox_depth, iter_latency_ms, rss_mb
```

## Session manifest

Written by `scripts/run_noop.py` via `telemetry.SessionManifest` as pretty JSON.
Keys: `session_id`, `started_utc`, `ended_utc`, `git_commit`, `git_dirty`,
`config` (resolved), `python_version`, `platform`, `cpu_count`,
`total_ram_bytes`, `seed`, plus an `extra.run` block with the no-op run stats.

## What does not exist yet

No perception, tracking, scene, temporal, risk, policy, audio, or research code.
No camera, microphone, speaker, WebRTC, model weights, or recorded-video reader.
No Docker, CI, database, or message broker. See Block 7 of the Phase 0 prompt.
