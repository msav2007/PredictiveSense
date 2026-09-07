"""Binary ingest framing: round-trip, and every malformed input is rejected."""

from __future__ import annotations

import json
import struct

import cv2
import numpy as np
import pytest

from predictivesense.camera.framing import (
    FramingError,
    decode_ingest_message,
    encode_ingest_message,
    looks_like_jpeg,
)
from predictivesense.core.types import IngestHeader

pytestmark = pytest.mark.unit

_MAX = 2_000_000


def _jpeg(w: int = 64, h: int = 48) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:20, 10:20] = 200
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_header_payload_round_trip() -> None:
    header = IngestHeader(client_ts_ms=1234.5, seq=7, w=64, h=48)
    jpeg = _jpeg()
    message = encode_ingest_message(header, jpeg)

    got_header, got_jpeg = decode_ingest_message(message, max_bytes=_MAX)
    assert got_header == header
    assert got_jpeg == jpeg
    assert looks_like_jpeg(got_jpeg)


def test_truncated_message_is_rejected() -> None:
    message = encode_ingest_message(IngestHeader(client_ts_ms=1.0, seq=0, w=8, h=8), _jpeg())
    with pytest.raises(FramingError):
        decode_ingest_message(message[:3], max_bytes=_MAX)  # shorter than the length prefix
    with pytest.raises(FramingError):
        decode_ingest_message(message[: 4 + 5], max_bytes=_MAX)  # truncated inside header


def test_oversize_message_is_rejected() -> None:
    message = encode_ingest_message(IngestHeader(client_ts_ms=1.0, seq=0, w=8, h=8), _jpeg())
    with pytest.raises(FramingError):
        decode_ingest_message(message, max_bytes=len(message) - 1)


def test_non_jpeg_payload_is_rejected() -> None:
    header = json.dumps({"client_ts_ms": 1.0, "seq": 0, "w": 8, "h": 8}).encode()
    message = struct.pack(">I", len(header)) + header + b"this is not a jpeg"
    with pytest.raises(FramingError):
        decode_ingest_message(message, max_bytes=_MAX)


def test_bad_json_header_is_rejected() -> None:
    body = b"{not json"
    message = struct.pack(">I", len(body)) + body + _jpeg()
    with pytest.raises(FramingError):
        decode_ingest_message(message, max_bytes=_MAX)


def test_implausible_header_length_is_rejected() -> None:
    message = struct.pack(">I", 10_000_000) + b"{}" + _jpeg()
    with pytest.raises(FramingError):
        decode_ingest_message(message, max_bytes=_MAX)


def test_decode_error_is_a_valueerror_subclass() -> None:
    # The ingest handler catches FramingError; confirm it is catchable narrowly
    # and is not a bare Exception surprise.
    assert issubclass(FramingError, ValueError)
