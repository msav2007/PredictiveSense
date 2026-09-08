"""Phase 4 Object Learning Studio - environment-specific object *data collection*.

This package stores images and metadata; it trains nothing. The three modules:

* :mod:`predictivesense.objects.vocab` - the fixed condition vocabularies, the
  sample ``role`` set, the object ``kind`` / ``status`` sets, and the seed
  ``confusable_with`` map derived from the developer's observed failure modes.
* :mod:`predictivesense.objects.registry` - :class:`ObjectProfile` CRUD over
  ``data/objects/objects.json`` with stable slug ids, atomic writes and soft
  delete (a deleted profile's folder moves to ``data/objects/_deleted/``).
* :mod:`predictivesense.objects.samples` - :class:`ObjectSample` records in
  ``data/objects/<id>/manifest.json``, one bounding box per sample, condition
  tags, quality fields and full provenance; atomic manifest writes; sample
  deletion is also soft.
* :mod:`predictivesense.objects.quality` - Laplacian variance, dHash and
  coverage accounting, pure numpy, no new dependency.

The object dataset is *training data* and is kept strictly separate from the
Phase 2.5 evaluation set under ``data/eval`` (``docs/decisions.md``).
"""

from __future__ import annotations

from predictivesense.objects.registry import (
    ObjectProfile,
    ObjectRegistry,
    ObjectStoreError,
)
from predictivesense.objects.samples import ObjectSample, SampleStore

__all__ = [
    "ObjectProfile",
    "ObjectRegistry",
    "ObjectStoreError",
    "ObjectSample",
    "SampleStore",
]
