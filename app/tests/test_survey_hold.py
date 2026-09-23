"""The next-question panel holds until the interviewee answers.

Every fixture here is built so it CAN exhibit the property it asserts -- the
standing trap in this repo is a fixture whose numbers cannot distinguish the
behaviour under test from its opposite. A hold test with no second question in
the survey, for instance, could never show the panel advancing wrongly, so
every case below carries at least two distinct asked-sets.

The clock is injected everywhere; nothing here sleeps.
"""

from __future__ import annotations

import pytest

from app.web.survey_hold import (
    DEFAULT_HOLD_TIMEOUT_S,
    SurveyDisplayHold,
)

SPEAKER_INTERVIEWEE = "interviewee"
SPEAKER_INTERVIEWER = "interviewer"


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _hold(*, initial=frozenset(), timeout=DEFAULT_HOLD_TIMEOUT_S):
    clock = FakeClock()
    return SurveyDisplayHold(initial_asked=initial, hold_timeout_s=timeout, clock=clock), clock


# --- the defect this module exists for -------------------------------------------


def test_a_coverage_change_alone_does_not_advance_the_panel():
    """The reported defect: the panel moved while the interviewer was still
    reading the question aloud."""
    hold, _clock = _hold()

    assert hold.note_coverage_change(frozenset({"s1"})) is None, (
        "an interviewer asking s1 must NOT immediately broadcast -- that is the "
        "question jumping out from under them mid-read"
    )
    assert hold.asked_for_display() == frozenset(), (
        "the panel must still be showing the pre-match state, i.e. s1 itself"
    )
    assert hold.is_holding


def test_an_interviewee_answer_releases_the_hold():
    hold, _clock = _hold()
    hold.note_coverage_change(frozenset({"s1"}))

    released = hold.note_segment(speaker=SPEAKER_INTERVIEWEE)

    assert released == frozenset({"s1"}), "the answer landing is what advances the panel"
    assert hold.asked_for_display() == frozenset({"s1"})
    assert not hold.is_holding


def test_the_interviewer_talking_on_does_not_release_the_hold():
    """The interviewer restating or elaborating their own question must not
    count as an answer -- otherwise the hold releases on exactly the speech it
    was added to sit through."""
    hold, _clock = _hold()
    hold.note_coverage_change(frozenset({"s1"}))

    assert hold.note_segment(speaker=SPEAKER_INTERVIEWER) is None
    assert hold.asked_for_display() == frozenset()
    assert hold.is_holding


# --- release reason 2: they moved on without waiting ------------------------------


def test_a_second_coverage_change_while_held_releases_to_the_newest_state():
    """Holding past a question the interviewer has visibly abandoned would show
    them a question they already left behind -- worse than advancing early."""
    hold, _clock = _hold()
    hold.note_coverage_change(frozenset({"s1"}))

    released = hold.note_coverage_change(frozenset({"s1", "s2"}))

    assert released == frozenset({"s1", "s2"}), (
        "the release must carry the NEWEST state, not the state that was pending"
    )
    assert not hold.is_holding
    assert hold.asked_for_display() == frozenset({"s1", "s2"})


def test_a_coverage_change_that_changes_nothing_is_not_broadcast_and_does_not_disturb_a_hold():
    hold, _clock = _hold()
    hold.note_coverage_change(frozenset({"s1"}))
    hold.note_segment(speaker=SPEAKER_INTERVIEWEE)

    assert hold.note_coverage_change(frozenset({"s1"})) is None
    assert not hold.is_holding, "a no-op change must not start a hold"


# --- release reason 3: the fallback, without which the panel sticks forever -------


def test_the_hold_expires_and_the_panel_advances_even_with_no_answer():
    hold, clock = _hold(timeout=45.0)
    hold.note_coverage_change(frozenset({"s1"}))

    clock.advance(44.0)
    assert hold.note_segment(speaker=SPEAKER_INTERVIEWER) is None, "not yet expired"

    clock.advance(2.0)
    released = hold.note_segment(speaker=SPEAKER_INTERVIEWER)

    assert released == frozenset({"s1"}), (
        "without this fallback a question nobody ever answers freezes the panel "
        "for the rest of the interview"
    )
    assert not hold.is_holding


def test_a_stale_hold_also_expires_on_a_fresh_connect():
    """`asked_for_display` is the reconnect path; it must not be the one route
    that can observe an indefinitely stale hold."""
    hold, clock = _hold(timeout=45.0)
    hold.note_coverage_change(frozenset({"s1"}))

    clock.advance(100.0)

    assert hold.asked_for_display() == frozenset({"s1"})
    assert not hold.is_holding


def test_the_timeout_is_measured_from_the_match_not_from_the_last_segment():
    """A hold refreshed by every interviewer segment would never expire during
    a long monologue -- exactly the case the fallback is for."""
    hold, clock = _hold(timeout=45.0)
    hold.note_coverage_change(frozenset({"s1"}))

    for _ in range(4):
        clock.advance(12.0)
        result = hold.note_segment(speaker=SPEAKER_INTERVIEWER)

    assert result == frozenset({"s1"}), (
        "48s of interviewer speech must have expired the hold; a timeout reset by "
        "each segment would still be holding"
    )


# --- the reconnect door the fix must not leave open -------------------------------


def test_a_fresh_connect_mid_hold_is_shown_the_held_question_not_the_newest_state():
    """The same defect through a different door: a phone waking up mid-interview
    must not be handed the advanced state and skip the held question."""
    hold, _clock = _hold(initial=frozenset({"s1"}))
    hold.note_coverage_change(frozenset({"s1", "s2"}))

    assert hold.asked_for_display() == frozenset({"s1"}), (
        "the reconnecting client must see what the held panel shows, not live "
        "coverage state"
    )


# --- resumed sessions -------------------------------------------------------------


def test_initial_state_comes_from_replayed_coverage_not_an_assumed_empty_set():
    """Coverage state is rebuilt from the durable record, so a resumed session
    starts part way down the survey. Defaulting to empty would show question one
    to an interviewer who is half way through."""
    hold, _clock = _hold(initial=frozenset({"s1", "s2", "s3"}))

    assert hold.asked_for_display() == frozenset({"s1", "s2", "s3"})
    assert not hold.is_holding


# --- the None-vs-empty distinction ------------------------------------------------


def test_holding_returns_none_and_never_an_empty_set():
    """A caller treating the hold's answer as a set would blank the panel on
    every held turn. `None` (do not broadcast) and `frozenset()` (broadcast
    that nothing is asked) must stay distinguishable."""
    hold, _clock = _hold(initial=frozenset({"s1"}))

    assert hold.note_segment(speaker=SPEAKER_INTERVIEWEE) is None, (
        "no hold in flight means nothing to broadcast"
    )

    hold.note_coverage_change(frozenset({"s1", "s2"}))
    assert hold.note_segment(speaker=SPEAKER_INTERVIEWER) is None
    assert hold.note_segment(speaker=SPEAKER_INTERVIEWEE) == frozenset({"s1", "s2"})


# --- coverage state itself is never delayed ---------------------------------------


def test_the_hold_owns_no_coverage_state_and_cannot_delay_it():
    """Structural, not behavioural: this module is handed a plain frozenset and
    has no reference to the detector, so there is no route by which a display
    hold could postpone the coach's own view of what has been asked.

    Read from the AST's IMPORT nodes, not from the source text. The first
    version of this test matched on raw source and failed on its own module
    docstring, which explains the coupling it forbids -- the standing trap in
    this repo, where a text-matching guard reads prose as if it were code.
    """
    import ast
    import inspect

    from app.web import survey_hold

    tree = ast.parse(inspect.getsource(survey_hold))
    imported: "list[str]" = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not any(name.startswith("app.coach") for name in imported), (
        f"a display concern importing the coach package is how the two get coupled: {imported}"
    )
    assert not any(name.startswith("app.session") for name in imported), (
        f"nor should it reach the durable store: {imported}"
    )


@pytest.mark.parametrize("speaker", [SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER])
def test_a_segment_with_no_hold_in_flight_is_always_a_no_op(speaker):
    hold, _clock = _hold(initial=frozenset({"s1"}))

    assert hold.note_segment(speaker=speaker) is None
    assert hold.asked_for_display() == frozenset({"s1"})
