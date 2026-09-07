"""Clock-offset arithmetic for the ingest hello/ack/echo exchange."""

from __future__ import annotations

import pytest

from predictivesense.camera.framing import (
    ClockOffset,
    capture_ts_seconds,
    clock_offset_seconds,
    rtt_ms_from_monotonic,
)

pytestmark = pytest.mark.unit


def test_offset_maps_client_epoch_onto_server_monotonic() -> None:
    # Client's epoch-ms clock reads 1_700_000_000_000 when the server's
    # monotonic clock reads 12_345.678.
    client_hello_ms = 1_700_000_000_000.0
    server_mono = 12_345.678
    offset = clock_offset_seconds(client_hello_ms, server_mono)

    # A frame the client stamps at the same instant maps back to ~server_mono.
    assert capture_ts_seconds(client_hello_ms, offset) == pytest.approx(server_mono)


def test_capture_ts_is_monotonic_across_a_simulated_sequence() -> None:
    client_hello_ms = 1_700_000_000_000.0
    server_mono = 500.0
    offset = clock_offset_seconds(client_hello_ms, server_mono)

    caps = [
        capture_ts_seconds(client_hello_ms + i * 100.0, offset) for i in range(50)
    ]
    assert caps == sorted(caps)
    assert len(set(caps)) == len(caps)
    # 100 ms client spacing survives the mapping.
    assert caps[1] - caps[0] == pytest.approx(0.1)


def test_rtt_recorded_and_non_negative() -> None:
    assert rtt_ms_from_monotonic(10.0, 10.021) == pytest.approx(21.0)
    assert rtt_ms_from_monotonic(10.0, 9.999) == 0.0  # clamped, never negative

    offset = ClockOffset(
        offset_s=clock_offset_seconds(1_000.0, 5.0),
        rtt_ms=rtt_ms_from_monotonic(5.0, 5.03),
    )
    assert offset.rtt_ms == pytest.approx(30.0)
    assert offset.offset_s == pytest.approx(4.0)
