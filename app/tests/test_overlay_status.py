"""The overlay renders `server_status`.

`OverlayState.applied` branched on `hello`, `survey` and `suggestion` and
dropped everything else. That default is right -- an unknown wire type must
never take the overlay down mid-interview -- but it meant the one message
about the CAPTURE never reached the one surface the operator watches.

That became load-bearing when `app/web/status.py` gained its `degraded` tier:
the server can now say "no audio is arriving from the interviewee", and on an
overlay-run interview nobody would ever see it, because the phone page was the
only client that rendered `server_status`. A muted customer stays invisible,
live -- which is the exact defect the degraded tier was added to fix.

**Every test here crosses the seam.** The payload is built by
`status_message(derive_status(...))`, the REAL producer, never by hand. This
overlay has already been burned once by the other kind: the coach panel read
`candidates` while `hub.py` emitted `primary`, and both suites passed for
weeks because each built its own dict. Two suites that agree with themselves
and disagree with each other prove nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.overlay import model
from app.overlay.model import OverlayState
from app.web.status import (
    FAILED_SIDE_COACH,
    REASON_COACH_FAILURE,
    StreamHealth,
    derive_status,
    status_message,
)

_WINDOW_SOURCE = (Path(__file__).resolve().parents[1] / "overlay" / "window.py").read_text(
    encoding="utf-8"
)


def _wire(*healths: StreamHealth, coach_unhealthy: bool = False) -> "dict":
    """The real server-side derivation, projected onto the real wire. If
    either end of that pair changes shape, every test in this file learns
    about it -- which is the entire point of not hand-building the dict."""
    return status_message(derive_status(healths, coach_unhealthy=coach_unhealthy))


def _healthy(speaker: str) -> StreamHealth:
    return StreamHealth(speaker=speaker, worker_failed=False, capture_dead=False)


def _recording(*, coached: bool = True) -> OverlayState:
    return OverlayState().recording("acme", coached=coached)


# --- healthy says nothing ----------------------------------------------------------------


def test_a_healthy_status_puts_nothing_on_the_panel():
    # A permanently-present status line trains the eye to skip it -- the same
    # argument window.py already makes about the always-present empty coach
    # box. Nothing to say means nothing drawn.
    state = _recording().applied(_wire(_healthy("interviewee"), _healthy("interviewer")))
    assert state.has_status_line is False
    assert state.status_is_failure is False


def test_recovering_clears_a_status_that_was_showing():
    # The status board only broadcasts on CHANGE, so the recovery message is
    # the only thing that will ever clear this. A panel that latched on the
    # first bad status would report a muted microphone for the rest of the
    # interview after it was unmuted.
    degraded = _recording().applied(
        _wire(StreamHealth("interviewee", False, False, idle=True), _healthy("interviewer"))
    )
    assert degraded.has_status_line is True
    recovered = degraded.applied(_wire(_healthy("interviewee"), _healthy("interviewer")))
    assert recovered.has_status_line is False


# --- degraded is not failure -------------------------------------------------------------


@pytest.mark.parametrize(
    "health",
    [
        pytest.param(StreamHealth("interviewee", False, False, idle=True), id="call-muted"),
        pytest.param(StreamHealth("interviewer", False, False, idle=True), id="mic-muted"),
        pytest.param(StreamHealth("interviewee", False, False, lagging=True), id="lagging"),
    ],
)
def test_a_degraded_stream_never_claims_the_recording_stopped(health):
    # The distinction the whole status vocabulary exists for: `failed` means
    # transcription has STOPPED, `degraded` means it is running and the audio
    # reaching it is wrong. Wording that collapses the two is a false alarm
    # that would send the operator to restart a recording that is fine --
    # mid-interview, with the customer still talking.
    state = _recording().applied(_wire(health, _healthy("interviewer")))
    assert state.has_status_line is True
    assert state.status_is_failure is False, "a degraded stream must not read as a failure"
    line = state.status_line.lower()
    assert "stopped" not in line, line
    assert "failed" not in line, line


def test_a_muted_call_and_a_muted_microphone_are_told_apart():
    # Different remedies -- "ask them to unmute" versus "check your own mic"
    # -- and the operator cannot tell which from a transcript that has simply
    # stopped moving. Collapsing them would rebuild a smaller version of the
    # defect the degraded tier fixes.
    theirs = _recording().applied(
        _wire(StreamHealth("interviewee", False, False, idle=True), _healthy("interviewer"))
    )
    mine = _recording().applied(
        _wire(_healthy("interviewee"), StreamHealth("interviewer", False, False, idle=True))
    )
    assert theirs.status_line != mine.status_line


def test_no_audio_and_late_audio_are_told_apart():
    silent = _recording().applied(
        _wire(StreamHealth("interviewee", False, False, idle=True), _healthy("interviewer"))
    )
    late = _recording().applied(
        _wire(StreamHealth("interviewee", False, False, lagging=True), _healthy("interviewer"))
    )
    assert silent.status_line != late.status_line


# --- failure ------------------------------------------------------------------------------


def test_a_dead_stream_reads_as_a_failure():
    state = _recording().applied(
        _wire(StreamHealth("interviewee", True, False), _healthy("interviewer"))
    )
    assert state.has_status_line is True
    assert state.status_is_failure is True


def test_a_dead_coach_is_not_told_as_a_dead_recording():
    # OPS-05, and `app/static/app.js` had to make exactly this distinction: a
    # dead reasoning layer is not a dead recording. The transcript is fine and
    # only the suggestions stopped, so this gets the calmer sentence.
    coach = _recording().applied(_wire(_healthy("interviewee"), coach_unhealthy=True))
    stream = _recording().applied(_wire(StreamHealth("interviewee", True, False)))
    assert coach.status_line != stream.status_line
    assert "recording" in coach.status_line.lower()

    # And the seam holds in the other direction: this is genuinely the coach
    # branch of the real derivation, not some other failure that happened to
    # differ.
    payload = _wire(_healthy("interviewee"), coach_unhealthy=True)
    assert payload["failed_side"] == FAILED_SIDE_COACH
    assert payload["reason"] == REASON_COACH_FAILURE


def test_every_status_sentence_is_distinct():
    # Gate 16's property, applied to this surface: five situations, five
    # sentences. Two branches sharing a sentence makes the situations they
    # distinguish indistinguishable to the operator, and the dangerous pair --
    # a degraded stream and a dead one -- is exactly the pair that would be
    # collapsed by a tidy-up.
    lines = [
        _recording().applied(payload).status_line
        for payload in (
            _wire(StreamHealth("interviewee", False, False, idle=True)),
            _wire(StreamHealth("interviewer", False, False, idle=True)),
            _wire(StreamHealth("interviewee", False, False, lagging=True)),
            _wire(_healthy("interviewee"), coach_unhealthy=True),
            _wire(StreamHealth("interviewee", True, False)),
        )
    ]
    assert len(set(lines)) == len(lines) == 5, lines


# --- it must not be swallowed by anything else --------------------------------------------


def test_an_unknown_wire_type_is_still_ignored():
    # The default that made this whole gap possible is still right: the phone
    # surface and this one consume the same broadcast, and a message added for
    # the phone must never take the overlay down mid-interview.
    before = _recording()
    assert before.applied({"type": "something-new", "payload": 1}) == before


def test_a_new_interview_does_not_inherit_the_last_ones_failure():
    # `recording()` constructs a fresh state, which is what makes this true --
    # and the same construction is why `coached` had to be threaded through
    # explicitly. Asserted rather than assumed: a status carried over would
    # report a dead stream on a recording that has not started yet.
    failed = _recording().applied(_wire(StreamHealth("interviewee", True, False)))
    assert failed.has_status_line is True
    assert failed.recording("globex").has_status_line is False


def test_an_uncoached_recording_still_reports_a_degraded_stream():
    # The whole reason to record unfiled is that there is no coach -- and a
    # muted customer is MORE dangerous there, not less, because there is no
    # next-question panel moving on screen to suggest anything is alive.
    state = _recording(coached=False).applied(
        _wire(StreamHealth("interviewee", False, False, idle=True), _healthy("interviewer"))
    )
    assert state.shows_coaching is False
    assert state.has_status_line is True


def _draw_status_calls_in_recording() -> "tuple[int, int]":
    """`(calls, calls guarded by shows_coaching)` for `_draw_status` inside
    `window.py::_draw_recording`.

    Mutation-tested -- and the FIRST version of this guard SURVIVED both
    mutations it was written for. It scanned for reads of `state.status_line`,
    which only ever happen inside `_draw_status` itself, so it could see
    neither the call being deleted nor the call being moved into the coaching
    branch. A guard has to follow the CALL, not the thing the callee reads."""
    tree = ast.parse(_WINDOW_SOURCE)

    def _is_draw_status(inner: ast.AST) -> bool:
        return (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "_draw_status"
        )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_draw_recording"):
            continue
        guarded: "set[int]" = set()
        for branch in ast.walk(node):
            if (
                isinstance(branch, ast.If)
                and isinstance(branch.test, ast.Attribute)
                and branch.test.attr == "shows_coaching"
            ):
                for statement in list(branch.body) + list(branch.orelse):
                    for inner in ast.walk(statement):
                        if _is_draw_status(inner):
                            guarded.add(id(inner))
        calls = [inner for inner in ast.walk(node) if _is_draw_status(inner)]
        return len(calls), len([call for call in calls if id(call) in guarded])
    raise AssertionError("window.py no longer has a _draw_recording")


def test_the_window_draws_the_status_outside_the_coaching_guard():
    # The natural place to put it is beside the next-question panel, and that
    # is the one place it must not go: an unfiled recording draws no coaching
    # regions at all, so the status would be invisible on exactly the runs
    # that have nothing else moving on screen.
    calls, guarded = _draw_status_calls_in_recording()
    assert calls >= 1, "the recording panel never draws the capture status at all"
    assert guarded == 0, (
        "the capture status is drawn inside the coaching guard -- an uncoached recording "
        "would never be told its audio stopped"
    )


def test_degraded_is_not_drawn_in_the_recording_colour():
    # ACCENT is the recording dot and genuine failure. Painting a muted
    # microphone with it makes a degraded stream look like a crash, which is
    # the false alarm in colour rather than in words. ACCENT_HINT already
    # exists and already means "attention, not failure".
    recording_body = _WINDOW_SOURCE.split("def _draw_status")[1].split("\n    def ")[0]
    assert "ACCENT_HINT" in recording_body
    assert "status_is_failure" in recording_body, (
        "the status is painted one colour regardless of severity -- a degraded stream and a "
        "dead one would look identical"
    )


def _names_model_imports_from(module: str) -> "set[str]":
    source = (Path(__file__).resolve().parents[1] / "overlay" / "model.py").read_text(
        encoding="utf-8"
    )
    return {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_the_status_vocabulary_is_imported_and_never_re_spelled():
    # Mutation-tested, and the FIRST version of this guard was VACUOUS: it
    # compared `model.STATUS_HEALTHY is status_mod.STATUS_HEALTHY`, and
    # CPython interns short string literals, so a locally re-typed
    # `STATUS_HEALTHY = "healthy"` passed identity. The property is about
    # where the name comes FROM, so the check has to be about the import.
    #
    # A local copy would keep passing after the server renamed a tier, and
    # this panel would then fall through to the FAILURE branch -- telling the
    # operator their recording died because a microphone was muted.
    imported = _names_model_imports_from("app.web.status")
    assert {"STATUS_HEALTHY", "STATUS_DEGRADED", "STATUS_FAILED"} <= imported, imported
    assert {"REASON_NO_AUDIO", "REASON_LAGGING", "FAILED_SIDE_COACH"} <= imported, imported

    # And they are actually the server's values, not merely imported names.
    from app.web import status as status_mod

    assert model.STATUS_DEGRADED == status_mod.STATUS_DEGRADED
    assert model.STATUS_FAILED == status_mod.STATUS_FAILED
