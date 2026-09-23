"""The second real engine that proves the swap is a config change, not a
claim (phase_specification Task 3). Replays a RECORDED AssemblyAI response
payload through AssemblyAI's own parsing path -- NOT Soniox's -- because
routing replay through `SonioxEngine`'s pure functions would exercise one
vendor's schema twice and prove only that the registry maps three names, not
that the seam survives a structurally different response shape. AssemblyAI's
schema already differs materially from Soniox's (SDK event/turn objects vs.
token frames) and is already implemented and contract-tested in
`benchmark/engines/assemblyai.py`, so this costs nothing extra and makes the
proof real.

The recorded payload below is reused verbatim from the AssemblyAI contract
test (`benchmark/tests/test_engines_contract.py`,
`test_assemblyai_turn_event_to_segment_offline_via_sdk_parser`) rather than
invented -- Phase 1 already captured and contract-tested this exact shape.
"""

from __future__ import annotations

import time
from typing import Optional

from benchmark.engines.assemblyai import turn_event_to_segment
from benchmark.engines.base import Segment

# Reused verbatim from benchmark/tests/test_engines_contract.py's recorded
# AssemblyAI v3 Turn event payload.
RECORDED_TURN_PAYLOAD = {
    "type": "Turn",
    "turn_order": 1,
    "turn_is_formatted": True,
    "end_of_turn": True,
    "transcript": "hello world",
    "end_of_turn_confidence": 0.95,
    "words": [
        {"start": 0, "end": 500, "confidence": 0.9, "text": "hello", "word_is_final": True},
        {"start": 500, "end": 1000, "confidence": 0.9, "text": "world", "word_is_final": True},
    ],
    "language_code": "en",
}

DEFAULT_CADENCE_BYTES = 32000  # 1.0s of 16 kHz mono PCM16, matching the fake engine's default


class ReplayStreamingSession:
    """Accumulates audio and, once `cadence_bytes` worth has been sent, emits
    ONE segment produced by replaying `RECORDED_TURN_PAYLOAD` through
    AssemblyAI's own SDK parser (`StreamingClient._parse_message`) and the
    existing pure response handler (`turn_event_to_segment`) -- the same
    path `benchmark/tests/test_engines_contract.py` already exercises
    offline, with no network and no credentials."""

    def __init__(self, *, cadence_bytes: int, sample_rate: int) -> None:
        del sample_rate  # the recorded payload carries its own token timings
        self._cadence_bytes = cadence_bytes
        self._bytes_received = 0
        self._emitted = False
        self._session_start_wall = time.monotonic()
        self._pending: list[Segment] = []
        self._closed = False

    def send_audio(self, pcm16: bytes) -> None:
        if self._closed or self._emitted or not pcm16:
            return
        self._bytes_received += len(pcm16)
        if self._bytes_received >= self._cadence_bytes:
            self._emitted = True
            segment = self._replay_recorded_turn()
            if segment is not None:
                self._pending.append(segment)

    def poll(self, timeout_s: float = 0.0) -> tuple[Segment, ...]:
        del timeout_s  # nothing to wait on -- segments are produced synchronously by send_audio
        segments = tuple(self._pending)
        self._pending.clear()
        return segments

    def close(self) -> tuple[Segment, ...]:
        self._closed = True
        segments = tuple(self._pending)
        self._pending.clear()
        return segments

    def _replay_recorded_turn(self) -> Optional[Segment]:
        from assemblyai.streaming.v3.client import StreamingClient

        event = StreamingClient._parse_message(RECORDED_TURN_PAYLOAD)
        now_wall = time.monotonic()
        return turn_event_to_segment(event, self._session_start_wall, now_wall)


class ReplayStreamingEngine:
    """Satisfies `StreamingEngine`. Construction never touches the network --
    there is no network to touch -- matching every other adapter's
    construction contract."""

    name = "replay"

    def __init__(self, *, cadence_bytes: int = DEFAULT_CADENCE_BYTES) -> None:
        self._cadence_bytes = cadence_bytes

    def open_session(self, *, sample_rate: int, label: str = "") -> ReplayStreamingSession:
        del label  # the replay engine's behaviour never depends on which stream it serves
        return ReplayStreamingSession(cadence_bytes=self._cadence_bytes, sample_rate=sample_rate)
