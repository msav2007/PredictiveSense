"""Binary ``WS /ws/ingest`` message framing and clock-offset arithmetic.

Wire format of one analysis-frame message (all big-endian):

    | 4 bytes: uint32 header length H | H bytes: UTF-8 JSON IngestHeader | JPEG |

The JSON header is ``{"client_ts_ms": float, "seq": int, "w": int, "h": int}``.

Decoding never raises out of the caller's message loop: every malformed input
(truncation, oversize, bad JSON, non-JPEG payload) is reported as
:class:`FramingError`, which the ingest handler counts and drops.

Clock offset: the client's epoch-millisecond clock is mapped onto the server's
``time.monotonic()`` timeline so that ``frame_age_ms`` is meaningful. The offset
is ``server_mono - client_ts_ms / 1000`` measured at the hello/ack exchange; the
round-trip time of the echo is recorded separately as the offset's error bar and
is *not* folded into the offset.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from predictivesense.core.types import IngestHeader

__all__ = [
    "FramingError",
    "ClockOffset",
    "HEADER_LEN_BYTES",
    "encode_ingest_message",
    "decode_ingest_message",
    "looks_like_jpeg",
    "clock_offset_seconds",
    "capture_ts_seconds",
    "rtt_ms_from_monotonic",
]

HEADER_LEN_BYTES = 4
_JPEG_SOI = b"\xff\xd8\xff"
_JPEG_EOI = b"\xff\xd9"
# A hard ceiling independent of config, guarding struct.unpack against absurd
# header-length claims before any allocation.
_MAX_HEADER_JSON_BYTES = 4096


class FramingError(ValueError):
    """A binary ingest message could not be decoded. Counted and dropped."""


@dataclass(frozen=True)
class ClockOffset:
    """Result of the ingest hello/ack/echo exchange.

    ``offset_s`` maps client epoch seconds to server monotonic seconds:
    ``capture_ts = client_ts_ms / 1000 + offset_s``. ``rtt_ms`` is the measured
    echo round-trip and is the reported uncertainty on every frame age.
    """

    offset_s: float
    rtt_ms: float


def looks_like_jpeg(payload: bytes) -> bool:
    """True when ``payload`` starts with the JPEG SOI marker and ends with EOI."""

    return (
        len(payload) >= len(_JPEG_SOI) + len(_JPEG_EOI)
        and payload.startswith(_JPEG_SOI)
        and payload.rstrip(b"\x00").endswith(_JPEG_EOI)
    )


def encode_ingest_message(header: IngestHeader, jpeg: bytes) -> bytes:
    """Frame ``header`` + ``jpeg`` into one binary message."""

    body = json.dumps(
        {
            "client_ts_ms": header.client_ts_ms,
            "seq": header.seq,
            "w": header.w,
            "h": header.h,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return struct.pack(">I", len(body)) + body + bytes(jpeg)


def decode_ingest_message(data: bytes, *, max_bytes: int) -> tuple[IngestHeader, bytes]:
    """Parse one binary message. Raises :class:`FramingError` on any defect."""

    if len(data) > max_bytes:
        raise FramingError(f"message of {len(data)} bytes exceeds cap {max_bytes}")
    if len(data) < HEADER_LEN_BYTES:
        raise FramingError("message shorter than the 4-byte header length")

    (header_len,) = struct.unpack(">I", data[:HEADER_LEN_BYTES])
    if header_len == 0 or header_len > _MAX_HEADER_JSON_BYTES:
        raise FramingError(f"implausible header length {header_len}")
    header_end = HEADER_LEN_BYTES + header_len
    if len(data) < header_end:
        raise FramingError("message truncated inside the JSON header")

    try:
        raw = json.loads(data[HEADER_LEN_BYTES:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise FramingError("header JSON is not an object")

    try:
        header = IngestHeader(
            client_ts_ms=float(raw["client_ts_ms"]),
            seq=int(raw["seq"]),
            w=int(raw["w"]),
            h=int(raw["h"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FramingError(f"header fields invalid: {exc}") from exc

    payload = data[header_end:]
    if not payload:
        raise FramingError("message carries an empty payload")
    if not looks_like_jpeg(payload):
        raise FramingError("payload is not a JPEG (missing SOI/EOI markers)")
    return header, payload


def clock_offset_seconds(client_ts_ms: float, server_mono: float) -> float:
    """``server_mono - client_ts_ms / 1000`` - the hello/ack offset."""

    return server_mono - client_ts_ms / 1000.0


def capture_ts_seconds(client_ts_ms: float, offset_s: float) -> float:
    """Map a client epoch-ms timestamp onto the server monotonic timeline."""

    return client_ts_ms / 1000.0 + offset_s


def rtt_ms_from_monotonic(sent_mono: float, echoed_mono: float) -> float:
    """Round-trip time of the echo probe, in milliseconds."""

    return max(0.0, (echoed_mono - sent_mono) * 1000.0)
