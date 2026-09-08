"""``GET /api/videos`` + ``POST /api/analyze`` - Mode B recorded-video analysis."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from predictivesense.logging_setup import get_logger
from predictivesense.pipeline.recorded import RecordedDriver

__all__ = ["router"]

_LOG = get_logger(__name__)
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}

router = APIRouter()


class AnalyzeRequest(BaseModel):
    path: str
    replay_mode: str | None = None


def _input_dir(request: Request) -> Path:
    return Path(request.app.state.config.video.input_dir)


def _resolve_within(base: Path, candidate: str) -> Path:
    """Resolve ``candidate`` and confirm it stays inside ``base``. 400 otherwise."""

    base_res = base.resolve()
    raw = Path(candidate)
    target = (raw if raw.is_absolute() else base_res / raw).resolve()
    if target != base_res and base_res not in target.parents:
        raise HTTPException(status_code=400, detail="path is outside data/videos/")
    return target


@router.get("/api/videos")
async def list_videos(request: Request) -> list[dict[str, object]]:
    base = _input_dir(request)
    if not base.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES:
            rows.append(
                {
                    "path": path.relative_to(base).as_posix(),
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                }
            )
    return rows


@router.post("/api/analyze")
async def analyze(request: Request, body: AnalyzeRequest) -> dict[str, object]:
    base = _input_dir(request)
    target = _resolve_within(base, body.path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"video not found: {body.path}")

    config = request.app.state.config
    replay_mode = body.replay_mode or config.video.replay_mode
    if replay_mode not in ("realtime", "asfast"):
        raise HTTPException(status_code=400, detail="replay_mode must be realtime|asfast")

    driver = RecordedDriver(results_dir=config.results_dir)
    perception = getattr(request.app.state, "perception", None)
    policy = getattr(request.app.state, "policy", None)
    try:
        result = driver.run(
            target,
            replay_mode=replay_mode,
            config_profile=config.profile,
            perception=perception,
            policy=policy,
        )
    except (RuntimeError, OSError) as exc:
        _LOG.error("recorded analysis failed for %s: %r", target, exc)
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}") from exc

    return {
        "run_id": result["run_id"],
        "jsonl_path": result["jsonl_path"],
        "frames": result["frames"],
        "replay_mode": result["replay_mode"],
        "perception": bool(result.get("perception_enabled")),
        "detections": result.get("total_detections", 0),
        "poses": result.get("total_poses", 0),
    }
