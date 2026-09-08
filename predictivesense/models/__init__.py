"""Model version registry (Phase 4) - read / validate / resolve only.

No training, no fine-tuning, no new model, no activation logic beyond reading the
``active`` flag. See :mod:`predictivesense.models.registry`.
"""

from __future__ import annotations

from predictivesense.models.registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelVersion,
    default_registry_path,
)

__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ModelVersion",
    "default_registry_path",
]
