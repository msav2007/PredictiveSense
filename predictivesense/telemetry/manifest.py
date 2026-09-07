"""The session manifest: everything needed to reproduce and attribute a run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from pydantic import BaseModel, Field

from predictivesense.config.settings import AppConfig
from predictivesense.logging_setup import get_logger

__all__ = [
    "SessionManifest",
    "build_manifest",
    "git_state",
    "MANIFEST_REQUIRED_KEYS",
    "utc_now_iso",
]

_LOG = get_logger(__name__)

MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "session_id",
    "started_utc",
    "ended_utc",
    "git_commit",
    "git_dirty",
    "config",
    "python_version",
    "platform",
    "cpu_count",
    "total_ram_bytes",
    "seed",
)


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with a ``Z`` suffix."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_state(repo_root: Path | None = None) -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)``. ``("unknown", False)`` if git is unavailable."""

    root = repo_root or _repo_root()
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return sha, bool(porcelain.strip())
    except (subprocess.SubprocessError, OSError) as exc:
        _LOG.warning("git state unavailable: %s", exc)
        return "unknown", False


class SessionManifest(BaseModel):
    """JSON artifact written to ``results/`` at the end of every run."""

    session_id: str
    started_utc: str
    ended_utc: str | None = None
    git_commit: str
    git_dirty: bool
    config: dict[str, Any]
    python_version: str
    platform: str
    cpu_count: int
    total_ram_bytes: int
    seed: int
    extra: dict[str, Any] = Field(default_factory=dict)

    def finalize(self, ended_utc: str | None = None) -> "SessionManifest":
        """Stamp the end time. Returns ``self`` for chaining."""

        self.ended_utc = ended_utc or utc_now_iso()
        return self

    def write(self, path: str | Path) -> Path:
        """Write pretty JSON to ``path`` and return it."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        _LOG.info("session manifest written: %s", out)
        return out


def build_manifest(
    config: AppConfig,
    *,
    session_id: str | None = None,
    started_utc: str | None = None,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> SessionManifest:
    """Assemble a manifest from the resolved config and the host environment."""

    commit, dirty = git_state(repo_root)
    vm = psutil.virtual_memory()
    return SessionManifest(
        session_id=session_id or uuid.uuid4().hex,
        started_utc=started_utc or utc_now_iso(),
        ended_utc=None,
        git_commit=commit,
        git_dirty=dirty,
        config=config.as_json_dict(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
        total_ram_bytes=int(vm.total),
        seed=config.source.seed,
        extra=extra or {},
    )
