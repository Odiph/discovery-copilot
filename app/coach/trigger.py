"""When to spend a coach cycle: once per interviewee turn, `delay_s` into it.

**Why this exists at all.** The coach used to fire on every interviewee
segment, and segments are ~5-word engine finalizations, so a 60-word answer
bought ~12 cycles. `benchmark/eval/trigger_sim.py` replayed the real 26-minute
transcript against the measured 7.9s cycle and scored each candidate rule on
the only thing that matters -- the share of interviewer turns that BEGAN with a
fresh suggestion already finished:

    every-segment (then)  184 cycles, 183 dropped, 50% of turns covered
    turn-end               71 cycles,   0 dropped,  0% -- an 8s cycle lands
                                                       inside the next question
    turn+3s (this)         71 cycles,   0 dropped, 81%

So firing on every fragment was never carelessness; it was a LATENCY
workaround -- fire early and often so something is on screen by the time it is
needed -- and the queue was dropping 183 of 184 triggers to do it. `turn+3s`
wins on both axes at once, which is why it is worth the gate below rather than
just a rate limit.

**This gate must see BOTH speakers.** A turn boundary is a speaker change, so
deciding "3s into the interviewee's turn" is impossible from interviewee
segments alone -- which is all `SuggestionPipeline` was ever wired to receive.
That is the whole reason this is a separate object rather than a condition
inside `submit`.

**A turn starts at the END of the previous segment, never at `audio_start_s`.**
`app/stt/live.py` fabricates `audio_start_s` as the end of the SAME speaker's
previous segment, so on the first segment of a turn it points back past the
whole of the other speaker's turn. `trigger_sim.py` was written against that
field first and reported 0% coverage for every rule at every latency; a broken
denominator does not raise, it just politely answers no. Only `audio_end_s` is
trustworthy, and only end-to-end deltas are used here.

**The live rule is a strictly later moment than the simulated one.** The
simulation fired at `turn.start_s + delay_s`, a clock instant. Nothing wakes
this gate but a finalized segment, so it fires on the first interviewee segment
that ENDS at least `delay_s` into its turn -- up to one segment (~1-2s) after
the simulation would have. `trigger_sim.py` cannot see that gap, and no
simulation can; it is what the live rehearsal is for.

Runs on the STT threads. Never blocks, never raises, never touches the network.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from app.session.store import SegmentRecord

logger = logging.getLogger(__name__)

SPEAKER_INTERVIEWEE = "interviewee"

# Chosen by measurement, not taste -- see the table above. 5s scored 62% and
# turn-end 0%, so this is a peak rather than a slope, and moving it is a
# decision to re-run `trigger_sim.py` about.
DEFAULT_TURN_DELAY_S = 3.0

# The second gate, added 2026-08-15 after the first live coached interview.
#
# `turn+3s` bounds how OFTEN a cycle fires. It says nothing about whether the
# turn has yet said anything, and the live run showed that gap costing real
# suggestions: the coach fired on turns whose text so far was `'Um, I guess'`
# and `'So, uh, I'`, and the critic then flagged the resulting question
# `is_generic` -- correctly, because a question CANNOT engage with an utterance
# that carries no content. 4 of the 7 suppressions that interview triggered on
# a one-or-two-word utterance, and the critic's own rationales named it: "the
# interviewee said 'Um, I', which offers no specific content to build upon".
#
# So the coach was being asked to be specific about nothing, and then judged
# for failing. This floor removes the ask instead of punishing the answer.
#
# Counted over the WHOLE TURN so far, never the triggering segment alone. A
# segment is a ~5-word engine finalization, so a per-segment floor would stall
# a genuinely long answer that happened to be split into short pieces -- the
# opposite of the intent.
#
# 12 was measured, not picked: see `benchmark/eval/trigger_floor.py`, which
# replays the real transcript and reports, per floor, how many cycles survive
# and how many of the observed fragment-triggered cycles are removed.
DEFAULT_MIN_TURN_WORDS = 12


class TurnTrigger:
    """Forwards at most one segment per interviewee turn, `delay_s` in.

    Wire it on BOTH speakers' `segment_hook`s: interviewer segments are what
    close a turn, and dropping them silently reverts the rule to
    every-segment-with-extra-steps.
    """

    def __init__(
        self,
        *,
        downstream: "Callable[[SegmentRecord], None]",
        delay_s: float = DEFAULT_TURN_DELAY_S,
        min_turn_words: int = DEFAULT_MIN_TURN_WORDS,
    ) -> None:
        self._downstream = downstream
        self._delay_s = float(delay_s)
        self._min_turn_words = int(min_turn_words)
        # The two speakers finalize on separate threads, so every field below
        # is shared mutable state and the decision has to be atomic with the
        # update -- otherwise two segments of one turn can both read
        # `_fired is False` and both fire.
        self._lock = threading.Lock()
        # Monotonic: the latest segment END seen from EITHER speaker. Guarded
        # against going backwards because a late arrival from the other thread
        # would otherwise drag the next turn's start earlier and make a
        # too-short turn look long enough to fire.
        self._clock = 0.0
        self._turn_speaker: Optional[str] = None
        self._turn_start = 0.0
        self._fired = False
        # Words seen SO FAR in the current turn, reset on a speaker change.
        # Only the count is kept, never the text: this object runs on the STT
        # threads and holding transcript text here would put customer speech in
        # a third place for no reason gate 12 would accept.
        self._turn_words = 0

    def submit(self, record: SegmentRecord) -> None:
        """Called synchronously from an STT thread, for either speaker."""
        try:
            with self._lock:
                should_fire = self._advance(record)
        except Exception as exc:  # noqa: BLE001 -- an STT thread must survive this
            logger.error("turn trigger failed error_type=%s", type(exc).__name__)
            return
        if not should_fire:
            return
        try:
            self._downstream(record)
        except Exception as exc:  # noqa: BLE001
            # The turn stays marked as fired: retrying a downstream that just
            # failed would spend the cycle budget this gate exists to protect.
            logger.error("turn trigger downstream failed error_type=%s", type(exc).__name__)

    def _advance(self, record: SegmentRecord) -> bool:
        """Fold one segment into the turn state. Caller holds the lock."""
        speaker = record.speaker
        end = float(record.audio_end_s or 0.0)

        if speaker != self._turn_speaker:
            # A speaker change opens a turn, and it opens where the previous
            # one ended -- `_clock`, not anything on this record.
            self._turn_speaker = speaker
            self._turn_start = self._clock
            self._fired = False
            self._turn_words = 0

        # Accumulated BEFORE the gates below, so the segment that carries the
        # turn over the floor is itself allowed to be the trigger. Counting
        # after would make every turn wait one extra segment for no reason.
        self._turn_words += len((record.text or "").split())

        if end > self._clock:
            self._clock = end

        if speaker != SPEAKER_INTERVIEWEE or self._fired:
            return False
        if end - self._turn_start < self._delay_s:
            # Still early in the turn, or the turn is shorter than the delay
            # and will end without ever firing -- which is deliberate: a cycle
            # started here could not land before the interviewer speaks.
            return False
        if self._turn_words < self._min_turn_words:
            # Long enough to fire, but nothing has been SAID yet. Deliberately
            # not a permanent skip for this turn: `_fired` stays False, so a
            # turn that starts "Um, I guess..." and then goes on to say
            # something fires on the segment that carries it over the floor.
            # The alternative -- burning the turn's one cycle on its filler --
            # is exactly the behaviour this gate was added to stop.
            return False
        self._fired = True
        return True
