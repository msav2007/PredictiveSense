"""Frame sources and the bounded frame mailbox.

Phase 1 adds three real sources - :class:`~predictivesense.camera.device.DeviceSource`
(backend-owned OpenCV camera), :class:`~predictivesense.camera.file_source.FileSource`
(recorded video), and :class:`~predictivesense.camera.browser.BrowserSource`
(fed by the browser Web Worker over ``WS /ws/ingest``) - plus device
enumeration and the binary ingest framing / clock-offset helpers.

``cv2`` is imported only within this package.
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
