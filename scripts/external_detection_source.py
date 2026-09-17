"""Stage 5 section 2.1: exhaustive multi-class requery of a raw FiftyOne
Open Images V7 export.

Mirrors ``scripts/audit_external_dataset.py`` / ``scripts/import_external_dataset.py``
exactly for how the raw export is read (``detections.csv``/``classes.csv``/
``image_ids.csv`` under ``<root>/<split>/...``, never through the ``fiftyone``
package). The one thing this module does differently from
``import_external_dataset.py`` is the shape of the query itself:

``import_external_dataset.py`` (Phase 12, feeds the crop classifier) queries
ONE ps_class's mids at a time and writes one manifest per class - correct for
a single-label crop dataset, but a landmine for a multi-class DETECTION
dataset: an image pulled in for class A that also contains an unboxed
instance of class B silently teaches a detector that B is background
(stage5 section 2.1). ``query_active_classes`` below queries every mid in the
ACTIVE class set in a single pass over the whole image population, so an
image entering the dataset for any reason is annotated with every active
class it actually contains - never just the class that found it.

Requires the ``[external]`` extra (pandas) - never imported by
``predictivesense/`` (``tests/unit/test_no_forbidden_imports.py`` enforces
this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "RawBox",
    "load_class_names",
    "load_image_ids",
    "load_licenses",
    "query_active_classes",
]

_DETECTION_COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
_LICENSE_COLS = ["ImageID", "License", "Author", "OriginalURL", "Title"]
_CHUNK_ROWS = 500_000


@dataclass(frozen=True)
class RawBox:
    """One Open-Images-V7 box, fractional coordinates, exactly as read."""

    mid: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def load_class_names(metadata_dir: Path) -> dict[str, str]:
    classes_csv = metadata_dir / "classes.csv"
    if not classes_csv.is_file():
        raise FileNotFoundError(f"classes.csv not found: {classes_csv}")
    df = pd.read_csv(classes_csv, header=None, names=["mid", "name"])
    return dict(zip(df["mid"], df["name"]))


def load_image_ids(data_dir: Path) -> set[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"image data dir not found: {data_dir}")
    return {p.stem for p in data_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}


def load_licenses(metadata_dir: Path, wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    """Per-image licence/attribution, filtered to ``wanted_ids`` -
    ``image_ids.csv`` is also the full, unfiltered OI master file."""

    path = metadata_dir / "image_ids.csv"
    out: dict[str, dict[str, str]] = {}
    for chunk in pd.read_csv(path, usecols=_LICENSE_COLS, chunksize=_CHUNK_ROWS, dtype=str):
        chunk = chunk[chunk["ImageID"].isin(wanted_ids)]
        for _, row in chunk.iterrows():
            out[row["ImageID"]] = {
                "license": row.get("License") or "",
                "author": row.get("Author") or "",
                "source_url": row.get("OriginalURL") or "",
                "title": row.get("Title") or "",
            }
    return out


def query_active_classes(
    labels_csv: Path,
    image_ids: set[str],
    active_mids: set[str],
) -> dict[str, list[RawBox]]:
    """ONE pass over ``detections.csv``. For every image in ``image_ids``
    that has at least one box whose ``LabelName`` is in ``active_mids``,
    return ALL such boxes (every active class present, not just whichever
    class the image was originally sought for).

    There is no per-class pre-filtering of ``image_ids`` before this scan -
    that absence is what makes the requery exhaustive (stage5 section 2.1):
    every image in ``image_ids`` is checked against every active mid in the
    same pass, so a class B box on an image found "for" class A is never
    missed.
    """

    if not active_mids:
        raise ValueError("active_mids must be non-empty - an empty active set would silently drop every box")

    hits: dict[str, list[RawBox]] = {}
    for chunk in pd.read_csv(
        labels_csv, usecols=_DETECTION_COLS, chunksize=_CHUNK_ROWS, dtype={"ImageID": str, "LabelName": str},
    ):
        chunk = chunk[chunk["ImageID"].isin(image_ids) & chunk["LabelName"].isin(active_mids)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hits.setdefault(row.ImageID, []).append(
                RawBox(mid=row.LabelName, xmin=float(row.XMin), xmax=float(row.XMax), ymin=float(row.YMin), ymax=float(row.YMax))
            )
    return hits
