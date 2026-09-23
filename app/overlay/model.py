"""The overlay's state, with no tkinter in it.

Separated from the window so every rule below is testable without opening a
window, on a machine with no display -- the same reasoning that keeps
`app/web/hub.py`'s payload builders separate from the server.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from app.web.status import (
    FAILED_SIDE_COACH,
    REASON_LAGGING,
    REASON_NO_AUDIO,
    REASON_NO_TRANSCRIPT,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_HEALTHY,
)

# Imported, never re-spelled. A local copy of the literal "degraded" would go
# on passing after the server renamed its tier, and this panel would fall
# through to the FAILURE branch -- telling the operator their recording died
# because a microphone was muted. `app/web/status.py` imports nothing but
# `dataclasses` and `typing`, so this costs the overlay no server dependency.

STATE_IDLE = "idle"
STATE_PICKING = "picking"
STATE_RECORDING = "recording"
STATE_STOPPING = "stopping"

# What the recording panel says when the coach has produced nothing yet. NOT
# "no suggestions" -- an empty coach region and a coach that has not answered
# yet are different situations, and Phase 4 criterion 6 turns on the operator
# being able to tell them apart at a glance.
COACH_RESTING_TEXT = "listening..."

# The next-question line before the first `survey` message lands. It says the
# survey is loading rather than that it is empty, for the same reason.
QUESTION_RESTING_TEXT = "loading the survey..."

# Shown when every survey question has been asked. A real state the server
# reports, not an absence.
QUESTION_EXHAUSTED_TEXT = "every survey question asked"

# What an UNCOACHED recording says in place of the two panels.
#
# Not a third resting string. Both of the ones above assert something in
# flight -- the survey is loading, the coach is listening -- and in a run with
# no coach neither will ever resolve: no `survey` and no `suggestion` message
# is ever sent, because nothing is running that could send one. A panel that
# says "loading..." for a whole interview does not merely fail to inform, it
# teaches the operator that this surface lies.
UNCOACHED_TEXT = "recording only, no coach"

# What a run says when a coach WAS asked for and did not come up.
#
# A separate sentence from `UNCOACHED_TEXT`, not a reuse of it, and the
# distinction is the entire fix here. `--no-coach` is a choice the operator
# made and "recording only, no coach" tells them their choice took effect. A
# missing or rejected `ANTHROPIC_API_KEY` is a run that is quietly worth less
# than the operator thinks it is, on a filed project, with hypotheses loaded
# and nobody watching the terminal where the child printed "Coach disabled".
# Wording the two the same would hide the second inside the first.
#
# It names the remedy because nothing else on this surface does, and the
# operator is by definition looking at a screen and not at a terminal.
COACH_UNAVAILABLE_TEXT = "no coach on this run -- check ANTHROPIC_API_KEY"

# What the next-question panel says when there is no microphone to hear a
# question being asked.
#
# Coverage is computed from the INTERVIEWER stream and nothing else, so a
# loopback-only recording can never advance the survey: it would sit on
# question one for the whole recording while looking perfectly live. That is
# the same failure as `UNCOACHED_TEXT`'s -- a panel asserting something in
# flight that will never arrive -- so it gets the same treatment, a plain
# statement of what this recording IS.
# Not "on this run": the microphone can now be released and re-taken
# mid-interview, so this has to be true in both cases.
QUESTION_NO_MIC_TEXT = "no microphone open -- the survey cannot advance"

# The microphone control, worded as a STATE rather than as a verb.
#
# "MIC ON" tells the operator what is true; "mute" would tell them what
# pressing it does. Mid-conversation, glancing at a translucent panel, the
# first is read correctly and the second is read as a claim about the present.
# The sub-label carries the verb, where there is room to be unambiguous.
MIC_ON_TEXT = "MIC ON"
MIC_OFF_TEXT = "MIC OFF"
MIC_ON_HINT = "click to release the microphone"
MIC_OFF_HINT = "device released -- click to re-open"

# --- what the capture is doing ---------------------------------------------
#
# Five situations, five sentences, and they must stay five DIFFERENT
# sentences -- gate 16 pins exactly this property on the phone surface's own
# copy, and for the reason that matters most here: `degraded` means capture is
# running and the audio reaching it is wrong, while `failed` means
# transcription has STOPPED. Wording that collapses those two sends the
# operator to restart a recording that is fine, mid-interview, with the
# customer still talking.
#
# Shorter than the phone's, because this panel is read in under a second over
# the top of a call window -- but never shorter in a way that drops the
# distinction.
STATUS_NO_AUDIO_THEIRS_TEXT = "no audio from the call -- are they still unmuted?"
STATUS_NO_AUDIO_MINE_TEXT = "no audio from your microphone -- check it is unmuted"
STATUS_LAGGING_TEXT = "audio is arriving late -- the transcript is behind"
# Sound is arriving and none of it is becoming words, while the other side is
# fine. Worded as the cause rather than the symptom: "nothing transcribed" sends
# an operator hunting through the app, and the answer is almost always that the
# call is playing through a different output than the one being captured.
STATUS_NO_TRANSCRIPT_THEIRS_TEXT = "nothing transcribed from the call -- check you are capturing the output it plays through"
STATUS_NO_TRANSCRIPT_MINE_TEXT = "nothing transcribed from your microphone -- check the right one is selected"
STATUS_COACH_DOWN_TEXT = "suggestions unavailable -- the recording continues"
STATUS_STREAM_DOWN_TEXT = "transcription stopped on the laptop"

# --- deliberate gestures ---------------------------------------------------
#
# Starting and stopping an interview both require Ctrl to be held. The overlay
# floats over the call window, always on top, and a stray click lands on it
# rather than on whatever was behind it -- so an un-modified click must never
# begin or end a recording. Ending one is the expensive direction: the customer
# is mid-sentence and the capture simply stops.
#
# The unmodified click is NOT swallowed silently. A dead control teaches the
# operator the overlay is broken; a control that says what it wants teaches
# them the gesture once. That is the whole reason these are strings rather
# than a bare `return`.
RECORD_HINT_TEXT = "hold Ctrl and click to start"
STOP_HINT_TEXT = "hold Ctrl and click to stop"
QUIT_WHILE_RECORDING_HINT_TEXT = "Ctrl+Escape to stop the interview and quit"

# How long a hint stays on screen. Long enough to read mid-conversation,
# short enough that it is gone before it becomes another thing to ignore.
HINT_VISIBLE_MS = 2600


def quit_is_allowed(phase: str) -> bool:
    """Whether Escape alone may close the overlay.

    Escape is a reflex key -- far easier to hit by accident than the button
    the Ctrl guard protects -- and quitting mid-interview is not free: it
    stops a live recording. Guarding the click and leaving Escape open would
    defeat the point of guarding the click.

    A pure function, and here rather than in `window.py`, so the rule can be
    tested without a display: `window.py` needs a real tkinter root, and every
    test in `app/tests/test_overlay_model.py` runs headless."""
    return phase not in (STATE_RECORDING, STATE_STOPPING)


@dataclass(frozen=True)
class OverlayState:
    """Immutable. Every transition returns a new value, so a render can never
    observe a half-applied change."""

    phase: str = STATE_IDLE
    project: "Optional[str]" = None
    elapsed_s: int = 0
    next_question: "Optional[str]" = None
    asked_count: int = 0
    total_count: int = 0
    coach_primary: "Optional[str]" = None
    connected: bool = False
    error: "Optional[str]" = None

    status: str = STATUS_HEALTHY
    status_side: "Optional[str]" = None
    status_reason: "Optional[str]" = None
    """The last `server_status` the interview reported: the tier, which side
    it is about, and the reason class. Three plain fields rather than the
    `ServerStatus` value itself, so this module still folds a WIRE MESSAGE in
    and owns no server type."""

    coached: bool = True
    """Whether this interview has a coach at all.

    `True` for every ordinary run, so nothing that existed before this field
    changes. `False` for an unfiled recording (`--no-coach`), where there is
    no brief, no survey and no suggestion pipeline -- so no `survey` or
    `suggestion` message will ever arrive and the two panels below have
    nothing to say for the whole interview.

    **This is the operator's INTENT, and it is not evidence.** It is derived
    at `controller.py` from how the project is FILED, before the child process
    starts, so it can only ever report what was asked for. See
    `coach_confirmed` for what actually happened."""

    loopback_only: bool = False
    """Whether this recording opened the microphone at all.

    `False` for an ordinary interview: both sides are captured, the loopback
    as `interviewee` and the microphone as `interviewer`.

    `True` for a one-sided recording -- a podcast, a webinar, anything played
    rather than spoken to. Measured on 2026-08-12 and the reason this exists:
    a podcast played through SPEAKERS put the same speech on BOTH tracks, 15
    of 21 microphone segments near-duplicating a loopback segment ~0.35s
    earlier, so 71% of the "interviewer" track was the podcast. For a
    one-sided recording the microphone has nothing to contribute and
    everything to pollute.

    It is also why the survey cannot advance in this mode -- see
    `survey_can_advance`."""

    mic_on: bool = True
    """Whether the microphone is open RIGHT NOW.

    Distinct from `loopback_only`, which is how the run STARTED and never
    changes. This one moves: the operator can release the device mid-interview
    and take it back, and the panel has to show which it currently is.

    The two interact in one direction only: a run started with no microphone
    begins with this `False`, and turning it on mid-interview builds the
    interviewer pair on demand.

    That last clause was WRONG when first written -- it claimed the child could
    already honour the request because `--only interviewee` "decided which
    devices to OPEN, not which exist". It had not built the pair at all, so the
    control fell through in silence and the operator clicked twice for nothing.
    The only test of it drove this controller against a fake session. Fixed in
    `app/main.py::_set_microphone`, and tested against a REAL one-sided run."""

    coach_confirmed: "Optional[bool]" = None
    """Whether a coach actually came up, as reported by the interview itself.

    `None` until `hello` arrives -- genuinely unknown, and deliberately not
    folded into `coached`, because "we have not heard yet" and "there is no
    coach" are different and the window words them differently.

    The defect this field exists to fix, found by the 2026-08-12 audit: with
    no `ANTHROPIC_API_KEY`, `app/main.py` prints "Coach disabled: ..." and
    runs the interview with no pipeline, while this surface showed the
    "Next question" and "Coach" panels resting and empty for the entire
    interview. The guard was already written -- `UNCOACHED_TEXT` and
    `shows_coaching` -- and was wired to the wrong signal. There is no known
    case of an operator being misled by it, because nobody has yet watched a
    full record-to-stop cycle; that is why it was found by reading rather than
    by using."""

    # --- transitions ------------------------------------------------------

    def picking(self) -> "OverlayState":
        return replace(self, phase=STATE_PICKING, error=None)

    def cancelled(self) -> "OverlayState":
        """Backing out of the picker returns to idle and forgets everything
        chosen -- a half-made selection surviving a cancel is how the wrong
        customer's knowledge ends up grounding the next interview."""
        return OverlayState()

    def recording(
        self, project: str, *, coached: bool = True, loopback_only: bool = False
    ) -> "OverlayState":
        """A FRESH state, never a `replace` of this one -- see `cancelled()`
        for why a half-made selection must not survive into an interview.

        That construction is also why `coached` has to be threaded here
        explicitly: any field not named on this line is silently reset to its
        default, and the default is `True`. An uncoached run would then render
        the coached panel, with no test anywhere noticing."""
        return OverlayState(
            phase=STATE_RECORDING, project=project, coached=coached,
            loopback_only=loopback_only,
            # A one-sided run starts with the device already not open.
            mic_on=not loopback_only,
        )

    def with_microphone(self, enabled: bool) -> "OverlayState":
        """The live toggle. `replace`, not a fresh state -- unlike
        `recording()`, this happens mid-interview and every other field
        (elapsed time, the coach line, the survey) must survive it."""
        return replace(self, mic_on=enabled)

    def ticked(self, elapsed_s: int) -> "OverlayState":
        return replace(self, elapsed_s=elapsed_s)

    def stopping(self) -> "OverlayState":
        return replace(self, phase=STATE_STOPPING)

    def failed(self, message: str) -> "OverlayState":
        return replace(self, phase=STATE_IDLE, error=message)

    # --- wire messages ----------------------------------------------------

    def applied(self, message: "dict[str, Any]") -> "OverlayState":
        """Folds one wire message in. Unknown types are IGNORED rather than
        raising: the phone surface and this one consume the same broadcast,
        and a message added for the phone must never take the overlay down
        mid-interview."""
        kind = message.get("type")
        if kind == "hello":
            # `coach_enabled` is read with an explicit `is None` test rather
            # than `bool(...)`: an older interview that does not send the key
            # leaves this UNKNOWN, and coercing a missing key to `False` would
            # make the overlay announce a broken coach on a run that has a
            # working one. Absent means unknown; only a real `false` means no.
            reported = message.get("coach_enabled")
            return replace(
                self,
                connected=True,
                project=message.get("project") or self.project,
                coach_confirmed=(None if reported is None else bool(reported)),
            )
        if kind == "survey":
            text = message.get("next_question_text")
            return replace(
                self,
                next_question=text,
                asked_count=int(message.get("asked_count") or 0),
                total_count=int(message.get("total_count") or 0),
            )
        if kind == "suggestion":
            # The primary only. The phone shows ranked alternatives; this
            # surface is read in under a second while a customer is talking,
            # and a second option is a decision to make rather than a
            # question to ask.
            #
            # The key is `primary`, and it must stay whatever
            # `app/web/hub.py::suggestion_payload` actually emits --
            # `WIRE_SUGGESTION_KEYS` pins that to {type, primary,
            # alternatives}. This read said `candidates` from the day the
            # overlay was written, so the coach panel could never populate:
            # both suites passed because each built its own dict and neither
            # crossed the seam. `test_overlay_model.py` now feeds this the
            # REAL payload builder's output.
            primary = message.get("primary")
            return replace(self, coach_primary=primary or None)
        if kind == "server_status":
            # The one message about the CAPTURE, and it was dropped by the
            # catch-all below from the day this surface was written. The phone
            # page was the only client that rendered it -- so on an
            # overlay-run interview a muted customer stayed invisible, live.
            #
            # Keys per `app/web/status.py::status_message`, which is the only
            # function permitted to build this payload. A missing `status` is
            # read as healthy: a message this surface cannot understand must
            # not become a red banner over a working recording.
            return replace(
                self,
                status=message.get("status") or STATUS_HEALTHY,
                status_side=message.get("failed_side"),
                status_reason=message.get("reason"),
            )
        return self

    # --- what the window draws -------------------------------------------

    @property
    def has_status_line(self) -> bool:
        """Whether there is anything to say about the capture at all.

        Healthy draws NOTHING. A permanently-present status line trains the
        eye to skip it -- the same argument this module already makes about
        the always-present empty coach box -- and this panel's whole job is
        being readable at a glance while someone is talking."""
        return self.status != STATUS_HEALTHY

    @property
    def status_is_failure(self) -> bool:
        """Whether the window should paint this as a failure rather than as
        something to look at.

        Degraded is deliberately NOT a failure: capture is running. Painting
        it red would be the false alarm in colour rather than in words."""
        return self.status == STATUS_FAILED

    @property
    def status_line(self) -> str:
        if self.status == STATUS_DEGRADED:
            if self.status_reason == REASON_NO_AUDIO:
                # Two remedies, not one: ask them to unmute, versus check your
                # own microphone. The operator cannot tell which from a
                # transcript that has simply stopped moving.
                return (
                    STATUS_NO_AUDIO_MINE_TEXT
                    if self.status_side == "interviewer"
                    else STATUS_NO_AUDIO_THEIRS_TEXT
                )
            if self.status_reason == REASON_NO_TRANSCRIPT:
                return (
                    STATUS_NO_TRANSCRIPT_MINE_TEXT
                    if self.status_side == "interviewer"
                    else STATUS_NO_TRANSCRIPT_THEIRS_TEXT
                )
            if self.status_reason == REASON_LAGGING:
                return STATUS_LAGGING_TEXT
        if self.status_side == FAILED_SIDE_COACH:
            # A dead reasoning layer is not a dead recording (OPS-05): the
            # transcript is fine and only the suggestions stopped.
            return STATUS_COACH_DOWN_TEXT
        return STATUS_STREAM_DOWN_TEXT

    @property
    def shows_coaching(self) -> bool:
        """Whether the window draws the next-question and coach regions at
        all.

        The rule lives here rather than in `window.py` for the reason
        `quit_is_allowed` does: the window needs a real tkinter root and
        cannot be constructed headless, so anything testable belongs in this
        module and the window's job is only to call it.

        Two signals, and the ORDER matters. `coached` is what the operator
        asked for and is known immediately; `coach_confirmed` is what the
        interview reported and arrives with `hello`. Either one saying no is
        enough to stop promising suggestions. `coach_confirmed is None` --
        the window between pressing record and the socket connecting -- keeps
        the panels up, because that gap is a second or two and blanking the
        surface for it would flicker on every single run."""
        return self.coached and self.coach_confirmed is not False

    @property
    def coaching_absence_line(self) -> str:
        """WHY there is no coaching panel, worded for the situation.

        Returned by the model rather than chosen in `window.py` so that the
        two situations cannot quietly collapse into one sentence at the point
        of drawing -- which is how the original defect stayed invisible."""
        if not self.coached:
            return UNCOACHED_TEXT
        return COACH_UNAVAILABLE_TEXT

    @property
    def question_line(self) -> str:
        if not self.survey_can_advance:
            # Said BEFORE any of the resting states below, which all describe a
            # survey that is waiting for something. This one is not waiting.
            return QUESTION_NO_MIC_TEXT
        if self.next_question:
            return self.next_question
        if self.total_count and self.asked_count >= self.total_count:
            return QUESTION_EXHAUSTED_TEXT
        if self.total_count == 0 and self.connected:
            return QUESTION_EXHAUSTED_TEXT if self.asked_count else QUESTION_RESTING_TEXT
        return QUESTION_RESTING_TEXT

    @property
    def mic_line(self) -> str:
        """What the microphone control says. The STATE, not the verb.

        "MIC ON" / "MIC OFF" reads as what is true now; "mute" would read as
        what will happen if pressed, and the two are read differently at a
        glance by someone mid-conversation. Every conferencing tool shows the
        state for the same reason."""
        return MIC_ON_TEXT if self.mic_on else MIC_OFF_TEXT

    @property
    def survey_can_advance(self) -> bool:
        """Whether the survey panel can ever move.

        `CoverageDetector` is fed the INTERVIEWER stream only -- a survey
        question is marked asked when the INTERVIEWER is heard asking it. A
        loopback-only recording never opens the microphone, so there is no
        interviewer stream and nothing can ever be detected. The panel would
        show question one for the whole recording and look live doing it.

        The coach itself is unaffected and stays on: suggestions are generated
        from the INTERVIEWEE stream, which is exactly the stream a one-sided
        recording does have."""
        return not self.loopback_only and self.mic_on

    @property
    def coach_line(self) -> str:
        return self.coach_primary or COACH_RESTING_TEXT

    @property
    def has_coach_line(self) -> bool:
        return self.coach_primary is not None

    @property
    def progress_line(self) -> str:
        return f"{self.asked_count} of {self.total_count} asked"


def format_elapsed(elapsed_s: int) -> str:
    """`MM:SS`, and `H:MM:SS` only once an hour has passed.

    Zero-padded minutes below an hour so the field does not change WIDTH as
    the interview runs -- a timer that reflows the panel beside it is
    movement in the operator's peripheral vision for no information."""
    elapsed_s = max(0, int(elapsed_s))
    hours, remainder = divmod(elapsed_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
