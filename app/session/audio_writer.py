"""Opt-in, per-stream WAV retention (phase_specification U3). Uses the
standard-library `wave` writer only -- no new dependency.

**Retention defaults to OFF** and is enabled only by the caller passing
`retain=True` (wired to `--retain-audio` in `app/main.py`). This is a
privacy decision, not a preference -- see phase_specification U3 for the
full rationale: the common case for `start` is a REAL CUSTOMER CALL, and the
Phase 1 benchmark this audio would otherwise feed uploads to a second-tier
cloud transcription vendor that has never been approved for real customer
audio at all (named in U3, deliberately not repeated here -- see the
module-scope note below on why no vendor identifier belongs in this file).

Fed the SAME 16 kHz mono PCM16 frames the STT worker consumes, including the
capture pacer's zero-filled idle frames (phase_specification S3) -- the
file's timeline must match the transcript's, gap for gap, or it is useless
as the benchmark input it exists to produce.

Can NEVER block or fail the capture path (phase_specification T-02-22): a
write error records one event on the session and permanently disables
retention for the remainder of the session; capture and transcription
continue regardless. No STT vendor is ever named in this module's
executable code (STT-03).
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

from app.session.store import SessionStore

logger = logging.getLogger(__name__)

SAMPLE_WIDTH_BYTES = 2  # paInt16
CHANNELS = 1  # each stream is captured and resampled to mono at the capture boundary (S4)


class AudioWriter:
    """One instance per speaker per session. `write()` is a no-op once
    disabled -- by construction, not by a caller remembering to check a
    flag first."""

    def __init__(self, session_dir: Path, *, speaker: str, sample_rate: int, store: SessionStore) -> None:
        self._speaker = speaker
        self._store = store
        self._path = session_dir / f"{speaker}.wav"
        self._enabled = True
        self._wav: Optional["wave.Wave_write"] = None
        try:
            self._wav = wave.open(str(self._path), "wb")
            self._wav.setnchannels(CHANNELS)
            self._wav.setsampwidth(SAMPLE_WIDTH_BYTES)
            self._wav.setframerate(sample_rate)
        except OSError as exc:
            self._disable(exc, when="open")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    def write(self, pcm16: bytes) -> None:
        if not self._enabled or not pcm16:
            return
        try:
            self._wav.writeframes(pcm16)  # type: ignore[union-attr]
        except OSError as exc:
            self._disable(exc, when="write")

    def close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except OSError as exc:  # noqa: BLE001 -- best-effort close; never raise from close()
                logger.warning(
                    "audio writer close failed speaker=%s error_type=%s",
                    self._speaker, type(exc).__name__,
                )
            self._wav = None

    def _disable(self, exc: OSError, *, when: str) -> None:
        self._enabled = False
        self._wav = None
        logger.warning(
            "audio retention disabled speaker=%s when=%s error_type=%s",
            self._speaker, when, type(exc).__name__,
        )
        self._store.append_event(
            event="audio-retention-error",
            speaker=self._speaker,
            detail={"when": when, "error_type": type(exc).__name__, "message": str(exc)},
        )
