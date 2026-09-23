"""Audio level measurement: what a human means by "is this mic set right?".

Pure arithmetic over PCM16 -- no device, no stream, no I/O -- so every band
boundary and every degenerate input is provable without hardware.

Two decisions here, both paid for in a real interview:

1. Peaks, not the window mean. Speech is intermittent; a mean over the whole
   capture window is mostly measuring the pauses, so a LONGER window scores
   LOWER for the same voice. The real machine's microphone read 127 over an
   8-second window and 34 over a 30-second one while the user talked
   throughout, and the conclusion drawn was that the mic was broken. It was
   not. `peak_dbfs` scores the loudest half-second, which is the part that
   answers the actual question.

2. dBFS, not a raw 0-32768 integer. The bands below are the ones an audio
   engineer would use, and a number a human can compare against a rule of
   thumb is the difference between a level check that gets believed and one
   that gets ignored. Being ignored is the real cost: a genuinely silent
   device (an already-lost interview, when a headset dropped its stereo
   output the moment its hands-free microphone endpoint opened) reads 0 here,
   and a check that cries wolf on a working mic is exactly what makes a real
   zero easy to wave away.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

FULL_SCALE = 32768.0  # int16 magnitude that maps to 0 dBFS

# The scoring window. Long enough that a single sample transient cannot pass
# a dead mic, short enough that ordinary pauses between sentences do not drag
# the score down.
PEAK_WINDOW_S = 0.5

# Reported in place of -inf for a digitally silent window. `log10(0)` is a
# crash and `-inf` is not a number a human should ever read off a pre-call
# report, so silence is floored to a value that still sorts and prints.
SILENCE_FLOOR_DBFS = -120.0

BAND_SILENCE = "silence"
BAND_TOO_QUIET = "too quiet"
BAND_LOW = "low"
BAND_GOOD = "good"
BAND_HOT = "hot"
BAND_CLIPPING = "clipping"

SILENCE_MAX_DBFS = -50.0
TOO_QUIET_MAX_DBFS = -36.0
GOOD_MIN_DBFS = -18.0
GOOD_MAX_DBFS = -6.0
CLIPPING_MIN_DBFS = -1.0


@dataclass(frozen=True)
class LevelMeasurement:
    """`peak_dbfs` is the loudest `PEAK_WINDOW_S` of the capture, `band` names
    where that lands, and `window_s` is what was actually scored -- shorter
    than `PEAK_WINDOW_S` when the capture itself was, which a caller may want
    to say out loud rather than trust silently."""

    peak_dbfs: float
    band: str
    window_s: float

    @property
    def is_usable(self) -> bool:
        """Loud enough to be worth sending to an engine and not clipped. The
        two mid bands are deliberately included: `low` is quieter than ideal
        but transcribes fine, and a pre-call check that fails on merely
        imperfect is a check that gets skipped."""
        return self.band in (BAND_LOW, BAND_GOOD, BAND_HOT)

    def describe(self) -> str:
        return f"{self.peak_dbfs:.1f} dBFS ({self.band})"


def classify_dbfs(dbfs: float) -> str:
    if dbfs >= CLIPPING_MIN_DBFS:
        return BAND_CLIPPING
    if dbfs >= GOOD_MAX_DBFS:
        return BAND_HOT
    if dbfs >= GOOD_MIN_DBFS:
        return BAND_GOOD
    if dbfs >= TOO_QUIET_MAX_DBFS:
        return BAND_LOW
    if dbfs >= SILENCE_MAX_DBFS:
        return BAND_TOO_QUIET
    return BAND_SILENCE


def _as_samples(pcm16: Union[bytes, bytearray, memoryview, np.ndarray]) -> np.ndarray:
    if isinstance(pcm16, np.ndarray):
        return pcm16.astype(np.float64, copy=False)
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float64)


def peak_window_rms(samples: np.ndarray, *, window_samples: int) -> float:
    """The largest RMS over any `window_samples`-long span, computed exactly
    (every offset, not a hop) from a prefix sum of squares -- one pass, so a
    30-second window costs nothing worth optimising. A capture shorter than
    the window is scored whole rather than rejected: a short capture is still
    evidence, it is just weaker evidence, and the caller is told how long it
    actually was."""
    if samples.size == 0:
        return 0.0
    width = min(window_samples, samples.size)
    if width <= 0:
        return 0.0
    squares = np.cumsum(np.concatenate(([0.0], samples ** 2)))
    window_sums = squares[width:] - squares[:-width]
    return float(np.sqrt(np.max(window_sums) / width))


def measure_level(
    pcm16: Union[bytes, bytearray, memoryview, np.ndarray], *, sample_rate: int
) -> LevelMeasurement:
    """Scores a capture by its loudest `PEAK_WINDOW_S`. An empty capture and an
    all-zero (digitally silent) one both land on the floor rather than raising
    or producing `-inf`; they are indistinguishable to a human anyway, and both
    mean the same thing -- nothing arrived."""
    samples = _as_samples(pcm16)
    window_samples = max(1, int(round(sample_rate * PEAK_WINDOW_S)))
    scored_samples = min(window_samples, samples.size)
    window_s = scored_samples / sample_rate if sample_rate > 0 else 0.0

    rms = peak_window_rms(samples, window_samples=window_samples)
    if rms <= 0.0:
        dbfs = SILENCE_FLOOR_DBFS
    else:
        dbfs = max(SILENCE_FLOOR_DBFS, 20.0 * float(np.log10(rms / FULL_SCALE)))
    return LevelMeasurement(peak_dbfs=dbfs, band=classify_dbfs(dbfs), window_s=window_s)


# How far back `RecentLevel` looks. Long enough that one sentence keeps a
# stream "loud" across the pause that follows it, short enough that sound from
# a minute ago cannot vouch for a feed that has since gone dead.
DEFAULT_RECENT_WINDOW_S = 15.0


class RecentLevel:
    """The loudest thing a LIVE stream has carried recently.

    `measure_level` scores a finished capture held whole in memory, which is
    what `precheck` has and what a running interview never does. This is the
    live counterpart: frames are pushed as they are produced and the peak is
    read on demand, so nothing accumulates and no audio is retained beyond the
    window -- one float per frame, never the samples themselves. That matters
    for more than memory: a session that did not ask for `--retain-audio` must
    not grow a buffer of the interview anywhere.

    Peak rather than mean, for the reason `PEAK_WINDOW_S` documents and the
    audio research of 2026-08-15 measured: a mean over a window that includes
    pauses scores a talking person lower the longer you listen, which is how a
    perfectly good microphone reads as "too quiet"."""

    def __init__(self, *, window_s: float = DEFAULT_RECENT_WINDOW_S) -> None:
        self._window_s = window_s
        self._frames: "deque[tuple[float, float]]" = deque()

    def push(self, pcm16: Union[bytes, bytearray, memoryview, np.ndarray], *, at_s: float) -> None:
        """One frame's RMS, kept with its timestamp. O(1) amortised -- this
        runs on the capture thread, between a device read and a queue put."""
        samples = _as_samples(pcm16)
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        self._frames.append((at_s, rms))
        self._trim(at_s)

    def _trim(self, now_s: float) -> None:
        cutoff = now_s - self._window_s
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def peak_dbfs(self, *, at_s: float) -> "Optional[float]":
        """The loudest frame still inside the window, or `None` when nothing
        is -- which means "not measured", never "quiet". Callers are required
        to tell those apart; `app/session/feed_watch.py` fails closed on
        `None` precisely so a stalled pusher cannot read as silence."""
        self._trim(at_s)
        if not self._frames:
            return None
        rms = max(rms for _, rms in self._frames)
        if rms <= 0.0:
            return SILENCE_FLOOR_DBFS
        return max(SILENCE_FLOOR_DBFS, 20.0 * float(np.log10(rms / FULL_SCALE)))
