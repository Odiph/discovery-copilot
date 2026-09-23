"""The next-question panel's DISPLAY policy: hold the current question on
screen until the interviewee actually answers it.

**The defect this exists for (2026-08-15, the first real coached interview).**
`CoverageDetector` moves a survey item to `asked` the moment the INTERVIEWER's
own utterance matches it, and `app/main.py`'s `_survey_broadcast_callback`
broadcast that change immediately. So the panel advanced while the interviewer
was still speaking -- `coverage: MATCHED survey item 's1'` fired at 00:11, mid
sentence -- and the question moved out from under them as they read it. The
interviewer's own words: *"The questions jumped when I finished reading them,
wait for the answer before shifting."*

**What this module does NOT do, and must never do: delay coverage state.**
`CoverageState` is the durable record of what has been asked, and the coach's
own `coverage_state_provider` reads it to decide what to suggest next. Delaying
it would make the coach propose a question the interviewer just asked. Only the
DISPLAY lags. This module sits between the two and holds nothing else.

**Three ways the hold releases**, because a hold with only the happy path
sticks forever on a question that is never answered:

1. an interviewee segment finalizes (the wanted behaviour -- they answered);
2. coverage changes AGAIN while held -- the interviewer asked something else
   without waiting, so the panel must not keep showing a question they have
   visibly moved past;
3. `hold_timeout_s` elapses.

**The timeout is evaluated LAZILY, on the next event, never by a timer.** There
is no ticker on the server and this module deliberately does not add a thread
for a display concern. The consequence is precise and worth stating: if the
hold expires and then NOTHING further happens -- no segment from either
speaker, no new connection -- the panel keeps showing the held question until
something does. That is the correct trade. A stalled panel matters only while
the interview is moving, and an interview that is moving produces segments
every few seconds.

Thread-safety is not optional here: `note_coverage_change` is called from
`CoverageDetector`'s own worker thread and `note_segment` from the STT stream
thread, so every read-modify-write below is under one lock.

This module holds no vendor client, makes no call, and must NEVER log segment
or question TEXT -- only ids and counts, matching `app/coach/coverage.py`'s own
S7/T-04-11 discipline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SPEAKER_INTERVIEWEE = "interviewee"

# How long a question may sit unanswered before the panel gives up waiting and
# advances anyway. Long enough that a thoughtful pause never trips it, short
# enough that a question nobody answers does not freeze the panel for the rest
# of the interview.
DEFAULT_HOLD_TIMEOUT_S = 45.0


class SurveyDisplayHold:
    """Decides WHEN a coverage change reaches the next-question panel.

    Every method returns either the `asked` set that should now be broadcast,
    or `None` meaning "broadcast nothing, the panel is already correct". A
    caller that treats `None` as an empty set would blank the panel on every
    held turn -- the exact failure `hub.survey_payload`'s own docstring warns
    about -- so the two are deliberately different types of answer.
    """

    def __init__(
        self,
        *,
        initial_asked: "frozenset[str]" = frozenset(),
        hold_timeout_s: float = DEFAULT_HOLD_TIMEOUT_S,
        clock: "Callable[[], float]" = time.monotonic,
    ) -> None:
        """`initial_asked` is REQUIRED to be the coverage detector's own
        replayed state, not an assumed empty set: coverage state is rebuilt
        from the durable record at construction, so a session resumed after a
        crash starts with questions already asked. Defaulting to empty would
        show question one to an interviewer who is half way down the survey.

        `clock` is injected so the timeout is testable without sleeping, and
        is `time.monotonic` rather than wall-clock so a clock adjustment
        mid-interview cannot make a hold expire instantly or never.
        """
        self._lock = threading.Lock()
        self._displayed: "frozenset[str]" = frozenset(initial_asked)
        self._pending: "Optional[frozenset[str]]" = None
        self._held_since: Optional[float] = None
        self._hold_timeout_s = hold_timeout_s
        self._clock = clock

    # -- reads ----------------------------------------------------------------

    def asked_for_display(self) -> "frozenset[str]":
        """The set the panel should currently be rendering.

        Called by `app/main.py`'s `survey_provider` on every fresh connect.
        It must go through the hold rather than reading coverage state live:
        a phone reconnecting mid-interview would otherwise be handed the
        newest state and skip the held question -- the same defect this
        module exists to fix, arriving through a different door.

        Expires a stale hold as a side effect (see the module docstring on
        lazy expiry), which is why this is a method and not a property.
        """
        with self._lock:
            self._expire_if_stale_locked()
            return self._displayed

    @property
    def is_holding(self) -> bool:
        """Diagnostic only -- never a branch in broadcast logic."""
        with self._lock:
            return self._pending is not None

    # -- events ---------------------------------------------------------------

    def note_coverage_change(self, asked: "frozenset[str]") -> "Optional[frozenset[str]]":
        """A survey item just moved to `asked`. Returns the set to broadcast,
        or `None` to hold.

        Takes the raw `asked` set rather than a `CoverageState` so this module
        never imports the coach package for a display concern.
        """
        with self._lock:
            asked = frozenset(asked)
            if asked == self._displayed:
                # Nothing actually changed on screen; a re-broadcast would be
                # noise. Any hold in flight is left exactly as it was.
                return None

            if self._pending is not None:
                # Release reason 2: the interviewer asked something else
                # without an answer landing. Keeping the old question on
                # screen now would be showing them a question they have
                # visibly moved past -- worse than advancing early.
                logger.info("survey hold released reason=coverage-changed-again")
                return self._release_locked(asked)

            self._pending = asked
            self._held_since = self._clock()
            return None

    def note_segment(self, *, speaker: str) -> "Optional[frozenset[str]]":
        """A transcript segment finalized. Returns the set to broadcast, or
        `None`.

        Takes the speaker alone, never the `SegmentRecord`: nothing here needs
        the text, and a signature that cannot receive it cannot leak it.
        """
        with self._lock:
            if self._pending is None:
                return None
            if speaker == SPEAKER_INTERVIEWEE:
                # Release reason 1: they answered. The whole point.
                logger.info("survey hold released reason=answered")
                return self._release_locked(self._pending)
            return self._expire_if_stale_locked()

    # -- internals ------------------------------------------------------------

    def _release_locked(self, asked: "frozenset[str]") -> "frozenset[str]":
        self._displayed = asked
        self._pending = None
        self._held_since = None
        return asked

    def _expire_if_stale_locked(self) -> "Optional[frozenset[str]]":
        """Release reason 3. Returns the set to broadcast if the hold just
        expired, else `None`."""
        if self._pending is None or self._held_since is None:
            return None
        if self._clock() - self._held_since < self._hold_timeout_s:
            return None
        logger.info("survey hold released reason=timeout")
        return self._release_locked(self._pending)
