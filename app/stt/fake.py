"""A deterministic, offline `StreamingEngine` (phase_specification S9). Needs
no credentials and costs nothing, so this plan's automated gate -- and a full
rehearsal -- run with zero egress. Real code, not a test double hidden in a
test file: plan 02-02 builds a deliberately slow sibling from this same shape
to test lag/drop handling (via `poll_delay_s`, which already models a
consume rate slower than realtime -- a delay per poll call, with the capture
pacer producing frames on an independent wall clock, is what makes the queue
back up deterministically).

Plan 03-01 extends this with an optional `texts` sequence that the session
cycles through, one string per emitted segment -- feeding `app/tests/rtl_probe.py`
distinct, verbatim code-switched strings (D-04) without touching the existing
single-`text` behaviour or its instance-id suffix, which every prior test
still relies on unchanged."""

from __future__ import annotations

import itertools
import time
from typing import Optional, Sequence

from benchmark.engines.base import Segment

DEFAULT_TEXT = "This is a fake transcript segment."
DEFAULT_CADENCE_BYTES = 32000  # 1.0s of 16 kHz mono PCM16 at the default sample rate
DEFAULT_LANGUAGE = "en"

_instance_counter = itertools.count(1)


class FakeStreamingSession:
    """Accumulates audio and emits one segment every `cadence_bytes` of PCM16
    received, with a configurable per-poll delay so callers (including plan
    02-02's lag tests) can simulate a slow engine without a real one."""

    def __init__(
        self,
        *,
        cadence_bytes: int,
        text: str,
        detected_language: Optional[str],
        poll_delay_s: float,
        sample_rate: int,
        texts: Optional[Sequence[str]] = None,
    ) -> None:
        self._cadence_bytes = cadence_bytes
        self._text = text
        # When `texts` is supplied, each emitted segment cycles to the next
        # string instead of repeating `text` -- existing single-text
        # behaviour (and its instance-id suffix) is completely untouched
        # when `texts` is left `None` (the default for every caller before
        # this plan).
        self._text_cycle = itertools.cycle(texts) if texts else None
        self._detected_language = detected_language
        self._poll_delay_s = poll_delay_s
        self._sample_rate = sample_rate
        self._bytes_received = 0
        self._bytes_at_last_emit = 0
        self._session_start = time.monotonic()
        self._pending: list[Segment] = []
        self._closed = False

    def send_audio(self, pcm16: bytes) -> None:
        if self._closed or not pcm16:
            return
        self._bytes_received += len(pcm16)
        while self._bytes_received - self._bytes_at_last_emit >= self._cadence_bytes:
            self._bytes_at_last_emit += self._cadence_bytes
            audio_end_s = self._bytes_at_last_emit / 2 / self._sample_rate  # 16-bit mono
            text = next(self._text_cycle) if self._text_cycle is not None else self._text
            self._pending.append(
                Segment(
                    text=text,
                    audio_end_s=audio_end_s,
                    ready_wall_s=time.monotonic() - self._session_start,
                    detected_language=self._detected_language,
                )
            )

    def poll(self, timeout_s: float = 0.0) -> tuple[Segment, ...]:
        del timeout_s  # nothing to wait on -- segments are produced synchronously by send_audio
        if self._poll_delay_s > 0:
            time.sleep(self._poll_delay_s)
        segments = tuple(self._pending)
        self._pending.clear()
        return segments

    def close(self) -> tuple[Segment, ...]:
        self._closed = True
        segments = tuple(self._pending)
        self._pending.clear()
        return segments


class FakeStreamingEngine:
    """Satisfies `StreamingEngine`. Construction never touches the network --
    there is no network to touch -- matching every real adapter's
    construction contract.

    Carries a per-instance identity (`instance_id`), auto-assigned unless the
    caller passes one explicitly. Plan 02-02's dual-stream tests use this to
    prove two sides run through genuinely independent engine instances (not
    one shared session wearing two labels): when `text` is left at its
    default, the emitted segment text embeds `instance_id`, so two engines
    constructed from the SAME zero-arg registry factory still produce
    distinguishably different segment text. An explicit `text` override (as
    plan 02-01's marker test uses) is used verbatim, with no suffix."""

    name = "fake"

    def __init__(
        self,
        *,
        text: Optional[str] = None,
        texts: Optional[Sequence[str]] = None,
        cadence_bytes: int = DEFAULT_CADENCE_BYTES,
        detected_language: Optional[str] = DEFAULT_LANGUAGE,
        poll_delay_s: float = 0.0,
        instance_id: Optional[int] = None,
    ) -> None:
        self.instance_id = instance_id if instance_id is not None else next(_instance_counter)
        self._text = text if text is not None else f"{DEFAULT_TEXT} (engine #{self.instance_id})"
        # `texts`, if given, takes over emitted-segment text entirely (cycled,
        # verbatim, no instance-id suffix) -- used by `app/tests/rtl_probe.py`
        # to feed the seven D-04 code-switched strings through the real
        # pipeline. Leaves every existing caller (which never passes `texts`)
        # completely unaffected.
        self._texts = tuple(texts) if texts else None
        self._cadence_bytes = cadence_bytes
        self._detected_language = detected_language
        self._poll_delay_s = poll_delay_s

    def open_session(self, *, sample_rate: int, label: str = "") -> FakeStreamingSession:
        del label  # the fake engine's behaviour never depends on which stream it serves
        return FakeStreamingSession(
            cadence_bytes=self._cadence_bytes,
            text=self._text,
            texts=self._texts,
            detected_language=self._detected_language,
            poll_delay_s=self._poll_delay_s,
            sample_rate=sample_rate,
        )
