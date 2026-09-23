"""The Engine protocol every candidate implements.

Adding or swapping an engine must never require touching score.py — score.py and
report.py depend only on the shapes defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass(frozen=True)
class Segment:
    """One emitted transcription segment.

    `audio_end_s` is the segment's position on the audio clock (when its audio
    ends). `ready_wall_s` is when the segment became available on the engine's own
    clock (virtual real-time clock for the local pair, wall clock for the cloud
    adapters). `ready_wall_s - audio_end_s` is the latency this phase measures —
    never a batch-decode number (see phase_specification `key_links`).
    """

    text: str
    audio_end_s: float
    ready_wall_s: float
    detected_language: Optional[str] = None


@dataclass(frozen=True)
class EngineResult:
    hypothesis_text: str
    segments: tuple[Segment, ...]
    metadata: dict = field(default_factory=dict)


class Engine(Protocol):
    """Every candidate implements this. `transcribe()` is the only method that may
    touch the network or the GPU; construction must never require either."""

    name: str

    def transcribe(
        self, audio_path: str, slice_: Optional[tuple[float, float]] = None
    ) -> EngineResult: ...


# --- Phase 2 (app/): the live-source extension, in the same module -----------------
#
# `Engine.transcribe()` is file-shaped (a whole clip in, a whole result out).
# Phase 2 needs a live source instead, so the protocol is extended here, in
# place, rather than defined a second time somewhere under app/ (phase_specification
# S2 / STT-03). `Segment` above is reused unchanged. One concrete class may
# satisfy both protocols -- see benchmark/engines/soniox.py's SonioxEngine,
# which gains `open_session()` alongside its existing `transcribe()`.


class StreamingSession(Protocol):
    """One live stream's worth of engine state. Owned by exactly one thread
    (phase_specification S5) -- no cross-thread socket sharing, and therefore
    no dependence on any library's threading guarantees."""

    def send_audio(self, pcm16: bytes) -> None: ...

    def poll(self, timeout_s: float = 0.0) -> tuple[Segment, ...]:
        """Must return promptly and must never raise on 'nothing yet' -- a read
        timeout is normal and returns an empty tuple."""
        ...

    def close(self) -> tuple[Segment, ...]:
        """Flushes the engine's tail and returns any final segments."""
        ...


class StreamingEngine(Protocol):
    """Construction must never touch the network -- only `open_session()` does,
    matching `Engine`'s "construction never touches the network or the GPU"
    contract above."""

    name: str

    def open_session(self, *, sample_rate: int, label: str = "") -> StreamingSession: ...
