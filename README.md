# PredictiveSense

A research prototype that observes an indoor cabin scene, tracks objects and
people over time, and predicts developing safety risks early enough to warn the
user with a spoken alert and a fixed preventive recommendation.

Two input modes share one perception -> tracking -> temporal -> risk core:

- **Mode A - realtime:** live camera.
- **Mode B - recorded:** a recorded video analysed retrospectively.

## Phase 0 scope

Phase 0 is the foundation only: typed contracts, configuration, a deterministic
synthetic frame source, a bounded single-slot frame mailbox, a no-op analysis
loop, a FastAPI app with a state WebSocket, telemetry, a session manifest, and
the test suite. **There is no detector, pose model, tracker, risk model, camera
access, WebRTC, TTS, or overlay in this phase.**

Target machine: Windows, Intel Core Ultra 5 125H, integrated Intel Arc GPU,
16 GB RAM. No NVIDIA GPU, no CUDA. All processing is local; nothing is
downloaded or uploaded at runtime.

## Commands (PowerShell)

Working directory: `C:\Users\mummi\Documents\Projects\PredictiveSense`.

```powershell
# 0. Go to the project
cd C:\Users\mummi\Documents\Projects\PredictiveSense

# 1. Clean environment (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 2. Full test suite
pytest -q

# 3. Fast subset (skip slow tests)
pytest -q -m "not slow"

# 4. Start the application (dev profile) -> http://127.0.0.1:8000/
python scripts\run_app.py --profile dev

# 5. 60-second instrumented no-op run (writes results\)
python scripts\run_noop.py --profile dev --seconds 60

# 6. Regenerate the lockfile after a dependency change
pip freeze > requirements.lock.txt
```

## Configuration

Two profiles under `config/profiles/`: `dev` and `eval`. An unknown profile, an
unknown field, or a value that fails validation raises a clear error and exits
non-zero. Environment overrides use the `PS_` prefix with `__` as the nested
delimiter (for example `PS_API__PORT=9001`).

