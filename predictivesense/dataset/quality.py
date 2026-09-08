"""Per-frame provenance and labelling-progress accounting.

The provenance record is written by ``scripts/build_eval_frames.py`` and carried
on every image in the COCO store (``ps_provenance``). The unseeded-subset helpers
back the recall-bias check the harness reports separately (BLOCK 3.1.4).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from predictivesense.dataset.coco_store import CocoStore

__all__ = ["FrameProvenance", "ProgressSummary", "unseeded_image_ids", "seeded_image_ids"]


@dataclass(frozen=True)
class FrameProvenance:
    """Where one sampled evaluation frame came from."""

    source_clip: str
    timestamp_s: float
    session_id: str
    camera_device: str
    condition_tags: list[str] = field(default_factory=list)
    frame_index: int = -1
    clip_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgressSummary:
    """Snapshot of labelling progress for the Research group / the report."""

    images: int
    labelled: int
    seeded: int
    unseeded: int
    unseeded_fraction: float
    annotations: int
    sessions: int
    per_class: dict[str, int]
    per_session: dict[str, dict[str, int]]
    min_unseeded_fraction: float
    meets_unseeded_target: bool

    @classmethod
    def from_store(cls, store: CocoStore, *, min_unseeded_fraction: float) -> "ProgressSummary":
        c = store.counts()
        return cls(
            images=c["images"],
            labelled=c["labelled"],
            seeded=c["seeded"],
            unseeded=c["unseeded"],
            unseeded_fraction=round(c["unseeded_fraction"], 4),
            annotations=c["annotations"],
            sessions=c["sessions"],
            per_class=c["per_class"],
            per_session=c["per_session"],
            min_unseeded_fraction=min_unseeded_fraction,
            meets_unseeded_target=(
                c["labelled"] > 0 and c["unseeded_fraction"] >= min_unseeded_fraction
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unseeded_image_ids(store: CocoStore, image_ids: list[int] | None = None) -> list[int]:
    """Labelled image ids whose frame was labelled with seeding OFF."""

    ids = image_ids if image_ids is not None else store.image_ids()
    return [
        iid
        for iid in ids
        if store.image(iid).get("labelled") and not store.image(iid).get("seeded")
    ]


def seeded_image_ids(store: CocoStore, image_ids: list[int] | None = None) -> list[int]:
    ids = image_ids if image_ids is not None else store.image_ids()
    return [
        iid
        for iid in ids
        if store.image(iid).get("labelled") and store.image(iid).get("seeded")
    ]
