"""Correcting the interviewer latency a microphone toggle inflates.

**The bug, and why it survives being obvious.** `latency_s` is
`ready_wall_s - audio_end_s` (`store.py:298`). `ready_wall_s` is wall time since
the session opened. `audio_end_s` is the ENGINE's position in the audio it
received, taken from the engine's own token timings rather than from the capture
path's push counter. Those two clocks agree only while audio is flowing
continuously.

(The engine is not named here on purpose: gate 1 forbids any STT vendor
identifier in `app/session/` and `app/capture/` executable code, and a module
docstring is executable code as far as the AST is concerned. The gate caught
exactly that in this file's first draft.) Release the microphone and the interviewer's audio
timeline stops while the wall clock does not, so when the device comes back
every later interviewer segment reports a latency inflated by the entire
off-duration, permanently, for the rest of the session. A ninety-second pause
makes a healthy 0.9s pipeline report 90.9s.

Nothing about that is visible in the number. It is a plausible-looking float in
a field that has been trustworthy all session, and `session.json`'s
`latency_summary` -- written once at close -- carries it into every later report
as a measurement.

**Corrected at READ time, deliberately, rather than at capture.** The
information needed is already on disk: `microphone` events carry `at_s` on the
same wall clock `ready_wall_s` uses, recorded precisely because a gap on the
interviewer track is otherwise indistinguishable from a silent operator. So the
correction is arithmetic over records that already exist, which means it fixes
every session ALREADY RECORDED -- including any run made before this file
existed -- and it needs no change to the capture path, which cannot be tested
without hardware and is the last place to take an untested risk.

**Only the interviewer is corrected, and that asymmetry is the point.** The
microphone is the interviewer's device; the interviewee arrives over WASAPI
loopback and is untouched by the toggle. Correcting both would silently move a
number that was never wrong.

Pure: no I/O, no clock, deterministic. Returns counts and seconds, never text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

EVENT_MICROPHONE = "microphone"
INTERVIEWER = "interviewer"


@dataclass(frozen=True)
class OffWindow:
    """One stretch with the interviewer's device released.

    `ended_at_s` is `None` when the session closed with the microphone still
    off. That window is real and is left open rather than discarded or given an
    invented end -- an invented end would silently under- or over-correct every
    segment after it, and there are usually none, because a released microphone
    produces no interviewer segments."""

    began_at_s: float
    ended_at_s: "Optional[float]" = None

    def paused_before(self, at_s: float) -> float:
        """Seconds of THIS window that had elapsed by `at_s`."""
        if at_s <= self.began_at_s:
            return 0.0
        end = self.ended_at_s if self.ended_at_s is not None else at_s
        return max(0.0, min(at_s, end) - self.began_at_s)


def off_windows(records: "Iterable[Mapping[str, Any]]") -> "tuple[OffWindow, ...]":
    """Every stretch the microphone was off, from the `microphone` events.

    A repeated `enabled: false` does not open a second window and a repeated
    `enabled: true` does not close one that is not open. The control channel and
    the overlay can both issue a toggle, and a duplicate is a UI artefact rather
    than a second release of the device -- treating it as one would double-count
    a pause that happened once."""
    windows: "list[OffWindow]" = []
    open_at: "Optional[float]" = None
    for record in records:
        if record.get("kind") != "event" or record.get("event") != EVENT_MICROPHONE:
            continue
        enabled = (record.get("detail") or {}).get("enabled")
        at_s = record.get("at_s")
        if not isinstance(at_s, (int, float)):
            continue
        if enabled is False and open_at is None:
            open_at = float(at_s)
        elif enabled is True and open_at is not None:
            windows.append(OffWindow(began_at_s=open_at, ended_at_s=float(at_s)))
            open_at = None
    if open_at is not None:
        windows.append(OffWindow(began_at_s=open_at))
    return tuple(windows)


def paused_before(windows: "Sequence[OffWindow]", at_s: float) -> float:
    return sum(window.paused_before(at_s) for window in windows)


@dataclass(frozen=True)
class Correction:
    """What the toggle cost this session's interviewer latency figures."""

    corrected: "tuple[float, ...]"
    original: "tuple[float, ...]"
    windows: int
    total_off_s: float

    @property
    def applied(self) -> bool:
        """True when any figure actually moved. A session that never toggled
        the microphone is not 'corrected by zero' -- it needs no correction at
        all, and a report that says otherwise invites the reader to distrust
        numbers that were always right."""
        return any(
            abs(new - old) > 1e-9 for new, old in zip(self.corrected, self.original)
        )


def correct_interviewer_latencies(
    records: "Sequence[Mapping[str, Any]]",
) -> Correction:
    """Interviewer `latency_s` values with the microphone-off time removed.

    A segment whose `ready_wall_s` falls INSIDE an off window is corrected by
    the part of the window elapsed so far, not by the whole of it. That case is
    rare -- it means the engine finalized an utterance just after the device was
    released -- but crediting the whole window would over-correct into a
    negative latency, and a negative would be the one value obviously wrong
    enough to be noticed and 'fixed' by clamping, which would hide it again.

    Clamped at zero for the same reason `store.py` clamps: a negative latency is
    not a measurement, and the events and the engine's own timings come from
    different sources that can disagree by milliseconds."""
    windows = off_windows(records)
    corrected: "list[float]" = []
    original: "list[float]" = []
    for record in records:
        if record.get("kind") != "segment" or record.get("speaker") != INTERVIEWER:
            continue
        latency = record.get("latency_s")
        ready = record.get("ready_wall_s")
        if not isinstance(latency, (int, float)):
            continue
        original.append(float(latency))
        if not isinstance(ready, (int, float)):
            # No wall position means no way to say how much pause preceded it.
            # Carried through UNCHANGED rather than dropped: dropping would
            # quietly shrink the sample the median is taken over.
            corrected.append(float(latency))
            continue
        corrected.append(max(0.0, float(latency) - paused_before(windows, float(ready))))

    return Correction(
        corrected=tuple(corrected),
        original=tuple(original),
        windows=len(windows),
        total_off_s=sum(
            (w.ended_at_s - w.began_at_s) for w in windows if w.ended_at_s is not None
        ),
    )
