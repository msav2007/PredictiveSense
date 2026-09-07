"""Frame sources and the bounded frame mailbox.

Phase 0 ships one source (:class:`~predictivesense.camera.synthetic.SyntheticSource`)
and no hardware or file access whatsoever.
"""

from predictivesense.camera.mailbox import LatestFrameMailbox
from predictivesense.camera.source import FrameSource, create_frame_source
from predictivesense.camera.synthetic import SyntheticSource

__all__ = [
    "FrameSource",
    "create_frame_source",
    "SyntheticSource",
    "LatestFrameMailbox",
]
