"""The overlay's state rules, tested without opening a window.

Everything here is `app/overlay/model.py`, which is separated from the tkinter
code precisely so these rules can be checked on a machine with no display --
the same split `app/web/hub.py` uses for its payload builders.
"""

from __future__ import annotations

import pytest

from app.overlay.model import (
    COACH_RESTING_TEXT,
    QUESTION_EXHAUSTED_TEXT,
    QUESTION_RESTING_TEXT,
    STATE_IDLE,
    STATE_PICKING,
    STATE_RECORDING,
    OverlayState,
    format_elapsed,
)


# --- the timer ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "00:00"), (5, "00:05"), (65, "01:05"), (600, "10:00"), (3599, "59:59"),
     (3600, "1:00:00"), (3725, "1:02:05"), (-4, "00:00")],
)
def test_the_timer_reads_as_a_clock(seconds, expected):
    assert format_elapsed(seconds) == expected


def test_the_timer_keeps_a_fixed_width_below_an_hour():
    # A field that changes width as the interview runs is movement in the
    # operator's peripheral vision carrying no information, and it reflows
    # the project name beside it.
    widths = {len(format_elapsed(s)) for s in range(0, 3600, 37)}
    assert widths == {5}


# --- transitions -------------------------------------------------------------------------


def test_a_fresh_overlay_is_idle_and_knows_nothing():
    state = OverlayState()
    assert state.phase == STATE_IDLE
    assert state.project is None and state.elapsed_s == 0


def test_pressing_record_opens_the_picker_without_choosing_anything():
    state = OverlayState().picking()
    assert state.phase == STATE_PICKING and state.project is None


def test_cancelling_the_picker_forgets_everything():
    # A half-made selection surviving a cancel is how the wrong customer's
    # knowledge ends up grounding the next interview.
    state = OverlayState().picking().cancelled()
    assert state == OverlayState()


def test_choosing_a_project_starts_a_clean_recording():
    state = OverlayState(error="an old failure").picking().recording("acme/onboarding")
    assert state.phase == STATE_RECORDING
    assert state.project == "acme/onboarding"
    assert state.error is None, "a previous run's error must not persist into a new recording"
    assert state.coach_primary is None and state.elapsed_s == 0


def test_a_failure_returns_to_idle_carrying_the_reason():
    state = OverlayState().picking().recording("acme").failed("engine did not start")
    assert state.phase == STATE_IDLE and state.error == "engine did not start"


# --- wire messages -----------------------------------------------------------------------


def _recording() -> OverlayState:
    return OverlayState().recording("acme")


def test_an_unknown_message_type_is_ignored_rather_than_raising():
    # The phone surface and this one consume the SAME broadcast. A message
    # type added for the phone must never take the overlay down mid-interview.
    #
    # `server_status` used to be this test's example of an ignored type, and
    # that was the bug rather than the property: it is the one message about
    # the CAPTURE, and dropping it meant a muted customer stayed invisible on
    # the only surface the operator watches. It is handled now
    # (test_overlay_status.py), so the example here is a type that genuinely
    # does not exist -- the RULE is unchanged and still worth pinning.
    state = _recording()
    assert state.applied({"type": "pong"}) == state
    assert state.applied({"type": "transcript-replay", "segments": []}) == state
    assert state.applied({}) == state


def test_hello_marks_the_connection_and_adopts_the_project():
    state = _recording().applied({"type": "hello", "project": "acme/onboarding"})
    assert state.connected is True and state.project == "acme/onboarding"


def test_hello_without_a_project_keeps_the_one_already_chosen():
    state = _recording().applied({"type": "hello", "project": None})
    assert state.project == "acme"


def test_a_survey_message_sets_the_next_question_and_the_counts():
    state = _recording().applied({
        "type": "survey", "next_question_text": "What broke?",
        "asked_count": 2, "total_count": 5,
    })
    assert state.question_line == "What broke?"
    assert state.progress_line == "2 of 5 asked"


def test_an_exhausted_survey_says_so_rather_than_going_blank():
    # A real state the server reports, not the absence of a message -- the
    # distinction Phase 4 criterion 6 turns on.
    state = _recording().applied({
        "type": "survey", "next_question_text": None, "asked_count": 5, "total_count": 5,
    })
    assert state.question_line == QUESTION_EXHAUSTED_TEXT


def test_before_any_survey_message_the_line_says_loading_not_empty():
    assert _recording().question_line == QUESTION_RESTING_TEXT


def test_a_suggestion_shows_only_the_primary_candidate():
    # Fed the REAL payload builder's output, not a hand-built dict. This test
    # used to construct `{"candidates": [...]}` -- a shape the wire has never
    # carried -- while the hub's own tests asserted `primary`/`alternatives`.
    # Each suite agreed with itself, neither crossed the seam, and the coach
    # panel could not populate on any recording ever made.
    from app.coach.generate import Candidate
    from app.web.hub import suggestion_payload

    payload = suggestion_payload((
        Candidate(index=0, text="the primary", move="probe"),
        Candidate(index=1, text="an alternative", move="probe"),
    ))
    state = _recording().applied(payload)
    assert state.coach_line == "the primary"
    assert "alternative" not in state.coach_line


def test_the_overlay_reads_the_key_the_wire_allowlist_actually_pins():
    # The forcing function. `WIRE_SUGGESTION_KEYS` is the single source of
    # truth for what a suggestion may carry; if someone renames the field on
    # the hub side, this fails here rather than silently blanking the panel.
    from app.web.hub import WIRE_SUGGESTION_KEYS

    assert "primary" in WIRE_SUGGESTION_KEYS
    assert "candidates" not in WIRE_SUGGESTION_KEYS


def test_a_suggestion_with_no_primary_returns_to_resting():
    # `suggestion_payload` refuses to build an empty one (an empty ranked set
    # means "nothing worth showing", and D-05 says that must never blank a
    # healthy region), so this case can only arrive malformed -- which is
    # exactly why it is asserted directly rather than through the builder.
    from app.coach.generate import Candidate
    from app.web.hub import suggestion_payload

    state = _recording().applied(suggestion_payload((Candidate(index=0, text="x", move="probe"),)))
    assert state.has_coach_line is True
    state = state.applied({"type": "suggestion"})
    assert state.has_coach_line is False and state.coach_line == COACH_RESTING_TEXT


def test_the_resting_coach_line_does_not_claim_there_are_no_suggestions():
    # "listening..." and "no suggestions" are different situations. A coach
    # that has not answered YET must not read as a coach that answered with
    # nothing.
    assert COACH_RESTING_TEXT == "listening..."
    assert "no " not in COACH_RESTING_TEXT.lower()


def test_a_coach_suggestion_never_replaces_the_next_question():
    # The two regions answer different questions. Phase C's whole point is
    # that the survey line needs no model call and so can never be emptied by
    # a slow or failing one.
    state = _recording().applied({
        "type": "survey", "next_question_text": "What broke?", "asked_count": 0, "total_count": 3,
    }).applied({"type": "suggestion", "primary": "a follow-up", "alternatives": []})
    assert state.question_line == "What broke?"
    assert state.coach_line == "a follow-up"


def test_the_state_is_immutable_so_a_render_cannot_see_a_half_applied_change():
    state = _recording()
    state.applied({"type": "survey", "next_question_text": "x", "asked_count": 1, "total_count": 2})
    assert state.next_question is None, "applied() must return a new state, never mutate"


def test_ticking_only_changes_the_clock():
    before = _recording().applied({
        "type": "survey", "next_question_text": "What broke?", "asked_count": 1, "total_count": 4,
    })
    after = before.ticked(97)
    assert after.elapsed_s == 97
    assert after.question_line == before.question_line
    assert after.progress_line == before.progress_line
