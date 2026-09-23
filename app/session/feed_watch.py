"""Is one side of the interview producing no transcript while the other is?

`app/capture/supervisor.py` already answers a neighbouring question and it is
NOT this one. It watches frame ARRIVAL: `idle` means the device stopped
handing over buffers. On 2026-08-17 the device handed over buffers for two
minutes with every sample zero -- a Bluetooth headset had moved playback to
its hands-free endpoint while the capture stayed bound to the stereo one --
and every frame-level check was satisfied throughout. Frames arrived. Nothing
was transcribed. Nothing said so, and the session was discovered afterwards.

So this watches the OUTPUT: segments. The judgement is COMPARATIVE, which is
the only thing that makes it safe to act on.

  "This side has produced nothing"
      -- not evidence. For the first minute that is the interviewer asking a
         question into a room that has not answered yet.

  "This side has produced nothing WHILE THE OTHER SIDE HAS BEEN TALKING"
      -- evidence. One live feed proves the interview is under way, the engine
         is reachable and the pipeline transcribes; against that, a second
         feed at zero is a fault rather than a pause.

That asymmetry is the whole mechanism, and every threshold here exists to
protect the FIRST alert. An alarm that fires during ordinary silence is
ignored within one interview, and an ignored alarm is worse than none: it is
the same green banner wearing a warning's clothes.

Pure state and arithmetic -- no clock, no store, no I/O. `elapsed_s` is passed
in, so every window and every edge is provable without waiting for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Mapping, Optional, Sequence

# How long a side may produce nothing, while the other is producing, before it
# is called out. Longer than any natural gap in a conversation: a considered
# answer to a hard question, a long interviewer preamble, and the pause while
# somebody reads a screen all fit inside it.
DEFAULT_SILENCE_S = 45.0

# The microphone's own window, and the reason the two sides no longer share
# one. Measured 2026-08-18: a real 10-minute interview produced 186
# interviewee segments against 105 interviewer ones and tripped this alert
# twice on the MICROPHONE -- at 6:04 and 7:41 -- each time after 45s of the
# interviewer simply listening while the interviewee answered.
#
# Both alerts satisfied the comparative rule exactly: the other side was
# producing. They were false anyway, because 45s of interviewer silence is not
# a symptom -- it is the goal. The Mom Test's entire posture is that the
# interviewer should be the side producing LESS, so the shared window put a
# detector on the target behaviour, and an alarm that fires when things go
# RIGHT is the ignored alarm this module's docstring warns about.
#
# Three minutes is chosen to sit past a long answer plus the follow-up silence
# that invites another one, while still being far inside an interview where
# the microphone has genuinely died. The microphone is not exempt -- a dead
# one costs the coach its whole coverage signal -- it is just held to a window
# only a genuinely absent interviewer can reach.
SPEAKER_INTERVIEWER = "interviewer"
DEFAULT_INTERVIEWER_SILENCE_S = 180.0

# Re-said on this cadence while the fault persists. Said once at minute one and
# never again, for something that lasts the whole interview, is how an operator
# ends up believing it resolved itself.
DEFAULT_REPEAT_S = 120.0

# How much the OTHER side must have produced before its liveness counts as
# proof. Two segments is an opening question; it is not a working interview.
DEFAULT_MIN_OTHER_SEGMENTS = 3

# Below this, a capture is carrying nothing a person could have said. Kept
# numerically equal to `app.capture.levels.SILENCE_MAX_DBFS` and duplicated
# rather than imported, so this module stays pure state and arithmetic with no
# dependency on the capture stack -- the same posture that lets every window
# here be tested without a device.
QUIET_DBFS_MAX = -50.0

# Sides where a quiet capture means "nobody is talking" rather than "the feed
# is dead", and so must NOT be alerted on. The microphone only: an interviewer
# who is listening produces silence by design.
#
# The interviewee is deliberately absent, and inverting it there would be the
# worst change anyone could make to this file. On 2026-08-17 the loopback
# carried two minutes of zeros while the call was plainly audible in the room
# -- digital silence WAS the fault, because the capture had been left bound to
# an endpoint that stopped receiving. Suppressing on quiet there blinds this
# module to the case it exists for.
DEFAULT_IDLE_WHEN_QUIET_FOR = frozenset({SPEAKER_INTERVIEWER})

KIND_NOT_TRANSCRIBING = "not-transcribing"
KIND_RECOVERED = "recovered"


@dataclass(frozen=True)
class FeedWatchConfig:
    silence_s: float = DEFAULT_SILENCE_S
    repeat_s: float = DEFAULT_REPEAT_S
    min_other_segments: int = DEFAULT_MIN_OTHER_SEGMENTS
    # Per-side overrides of `silence_s`. A mapping rather than an
    # `interviewer_silence_s` field so this class keeps saying nothing about
    # WHICH sides exist -- `FeedWatch` is handed its speakers and the two are
    # free to disagree, which is what the replay and single-side runs need.
    silence_s_by_speaker: "Mapping[str, float]" = field(
        default_factory=lambda: {SPEAKER_INTERVIEWER: DEFAULT_INTERVIEWER_SILENCE_S}
    )

    idle_when_quiet_for: "AbstractSet[str]" = DEFAULT_IDLE_WHEN_QUIET_FOR
    quiet_dbfs_max: float = QUIET_DBFS_MAX

    def silence_for(self, speaker: str) -> float:
        """This side's window. One lookup, so no caller can read the shared
        default where an override exists -- the bug this replaced was two
        seams disagreeing about one threshold."""
        return self.silence_s_by_speaker.get(speaker, self.silence_s)


@dataclass(frozen=True)
class FeedAlert:
    """One thing to say, once. `since_s` is carried so the message can be
    checked against what the operator just heard in the room -- "nothing for
    46s" is falsifiable by a person, "a feed looks wrong" is not."""

    speaker: str
    kind: str
    since_s: float


class FeedWatch:
    """Call `note_segment` from the segment hook, `poll` from the status
    poller. Both sides are watched, on DIFFERENT windows: watching only the
    interviewee would encode the assumption that a dead microphone is
    survivable, and a dead microphone costs the coach its entire coverage
    signal -- but the two sides are not symmetrical in what their silence
    means, so one window could not stay true for both. See
    `DEFAULT_INTERVIEWER_SILENCE_S`."""

    def __init__(self, *, speakers: Sequence[str], config: FeedWatchConfig) -> None:
        self._speakers = tuple(speakers)
        self._config = config
        self._last_at: "dict[str, Optional[float]]" = {s: None for s in self._speakers}
        self._counts: "dict[str, int]" = {s: 0 for s in self._speakers}
        # The count on the OTHER sides at the moment this side last spoke --
        # subtracted later to get "how much have they said since". Storing the
        # delta rather than a window of timestamps keeps this O(1) per segment
        # on the STT thread, where the segment hook runs.
        self._others_at_last: "dict[str, int]" = {s: 0 for s in self._speakers}
        self._alerted_at: "dict[str, Optional[float]]" = {s: None for s in self._speakers}

    def note_segment(self, speaker: str, *, at_s: float) -> None:
        """A finalized segment arrived. Runs on the STT stream thread, so it
        does arithmetic and nothing else."""
        if speaker not in self._counts:
            return
        self._counts[speaker] += 1
        self._last_at[speaker] = at_s
        self._others_at_last[speaker] = self._other_count(speaker)

    def _other_count(self, speaker: str) -> int:
        return sum(count for other, count in self._counts.items() if other != speaker)

    def _since_s(self, speaker: str, elapsed_s: float) -> float:
        last = self._last_at[speaker]
        return elapsed_s if last is None else elapsed_s - last

    def is_silent(self, speaker: str, elapsed_s: "Optional[float]" = None) -> bool:
        """The same judgement `poll` acts on, exposed for the status banner so
        there is one opinion about this fact rather than two.

        With no `elapsed_s`, reports what the last `poll` concluded -- which is
        what the poller wants, having just called it."""
        if elapsed_s is None:
            return self._alerted_at[speaker] is not None
        return self._is_silent_now(speaker, elapsed_s)

    def _is_silent_now(
        self,
        speaker: str,
        elapsed_s: float,
        levels: "Optional[Mapping[str, Optional[float]]]" = None,
    ) -> bool:
        if self._since_s(speaker, elapsed_s) < self._config.silence_for(speaker):
            return False
        others_since = self._other_count(speaker) - self._others_at_last[speaker]
        if others_since < self._config.min_other_segments:
            return False
        return not self._quiet_enough_to_excuse(speaker, levels)

    def _quiet_enough_to_excuse(
        self, speaker: str, levels: "Optional[Mapping[str, Optional[float]]]"
    ) -> bool:
        """Whether a measured level explains this side's silence innocently.

        Fails CLOSED on purpose: no `levels` mapping, no entry for this side,
        or a `None` reading all mean "not measured", and an unmeasured side is
        reported exactly as it was before the probe existed. A probe that
        failed open would disable the detector the moment level plumbing
        broke, which is the hardest failure of all to notice -- the banner
        stays green and nothing anywhere says why."""
        if speaker not in self._config.idle_when_quiet_for:
            return False
        if not levels:
            return False
        dbfs = levels.get(speaker)
        if dbfs is None:
            return False
        return dbfs < self._config.quiet_dbfs_max

    def poll(
        self,
        *,
        elapsed_s: float,
        levels: "Optional[Mapping[str, Optional[float]]]" = None,
    ) -> "tuple[FeedAlert, ...]":
        """Everything newly worth saying. Edge-triggered, so a caller may poll
        as often as it likes; a fault that has already been reported returns
        nothing until `repeat_s` has passed."""
        alerts: "list[FeedAlert]" = []
        for speaker in self._speakers:
            silent = self._is_silent_now(speaker, elapsed_s, levels)
            alerted_at = self._alerted_at[speaker]
            if silent:
                due = alerted_at is None or (elapsed_s - alerted_at) >= self._config.repeat_s
                if due:
                    self._alerted_at[speaker] = elapsed_s
                    alerts.append(FeedAlert(
                        speaker=speaker,
                        kind=KIND_NOT_TRANSCRIBING,
                        since_s=self._since_s(speaker, elapsed_s),
                    ))
            elif alerted_at is not None:
                self._alerted_at[speaker] = None
                alerts.append(FeedAlert(
                    speaker=speaker,
                    kind=KIND_RECOVERED,
                    since_s=self._since_s(speaker, elapsed_s),
                ))
        return tuple(alerts)
