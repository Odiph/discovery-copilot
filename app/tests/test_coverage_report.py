"""Reading survey coverage back after the interview.

The live half (`app/coach/coverage.py`) decides, once, whether a question was
asked and records it. This is the read of what it left behind, and the tests
that matter here are about the two ways that read can silently lie:

1. **Going quietly empty.** If the event name or detail key drifts, every
   session reports "no questions asked" -- which looks exactly like a real,
   plausible answer. `test_a_real_recorded_event_still_parses` pins the shape
   against an event copied verbatim off disk.
2. **Attributing today's question text to last week's answer.** An event stores
   only a `question_id`; the text has to come from somewhere, and the two
   available sources are not interchangeable.
"""

from __future__ import annotations

import pytest

from app.coach.context import SurveyItem
from app.coach.coverage import EVENT_COVERAGE_ASKED
from app.coach.coverage_report import (
    SURVEY_SOURCE_NONE,
    SURVEY_SOURCE_PROJECT,
    SURVEY_SOURCE_SESSION,
    asked_question_ids,
    coverage_for_session,
    project_survey_standing,
)


def _survey(*ids: str) -> "list[SurveyItem]":
    return [SurveyItem(id=qid, order=index, text=f"question {qid}") for index, qid in enumerate(ids)]


def _asked_event(question_id: str, *, seq: int = 1) -> dict:
    return {
        "kind": "event",
        "seq": seq,
        "speaker": "interviewer",
        "event": EVENT_COVERAGE_ASKED,
        "detail": {"question_id": question_id, "matched_seq": seq - 1},
    }


# --- the drift guard ------------------------------------------------------------------


def test_a_real_recorded_event_still_parses():
    """**Copied verbatim from `projects/_eval/sessions/20260810T111011`**, the
    only session on the maintainer's machine that had ever recorded coverage.

    The point is not the parse. It is that this reader's idea of the event shape
    is checked against a real one rather than against a fixture built from the
    same assumptions the reader was built from -- both sides of a seam agreeing
    with themselves is this project's defining defect."""
    recorded = {
        "kind": "event",
        "seq": 19,
        "speaker": "interviewer",
        "event": "coverage-asked",
        "detail": {"question_id": "s3", "matched_seq": 18, "matched_audio_end_s": 0.5},
        "at_s": 27.92200000002049,
        "written_at": "2026-08-10T11:10:39.696+03:00",
    }
    assert asked_question_ids([recorded]) == ("s3",)


def test_the_event_name_comes_from_the_writer_not_a_literal():
    """If this module respelled the event name, a rename in the writer would
    make every report read "nothing was asked" -- a plausible answer, which is
    exactly why it would survive review."""
    from app.coach import coverage, coverage_report

    assert coverage_report.EVENT_COVERAGE_ASKED is coverage.EVENT_COVERAGE_ASKED


# --- replaying the events -------------------------------------------------------------


def test_asked_ids_are_returned_in_the_order_they_happened():
    records = [_asked_event("s3", seq=5), _asked_event("s1", seq=9)]
    assert asked_question_ids(records) == ("s3", "s1")


def test_a_repeated_event_counts_once():
    """Nothing un-asks a question and the live detector treats `asked` as
    terminal. A resumed session replaying its own log must not make one question
    look twice-covered."""
    assert asked_question_ids([_asked_event("s1"), _asked_event("s1", seq=4)]) == ("s1",)


def test_other_events_and_segments_are_ignored():
    records = [
        {"kind": "segment", "seq": 1, "text": "hello"},
        {"kind": "event", "seq": 2, "event": "keepalive-warning", "detail": {"reason": "x"}},
        _asked_event("s2", seq=3),
    ]
    assert asked_question_ids(records) == ("s2",)


@pytest.mark.parametrize(
    "detail", [None, {}, {"question_id": None}, {"question_id": ""}, {"question_id": 7}, "nope"]
)
def test_a_malformed_event_is_skipped_rather_than_crashing_the_review(detail):
    """The hub is a READ of a finished interview. A single malformed record must
    not take the whole page down -- the rest of the session is still worth
    seeing. A non-string id is skipped rather than coerced, because coercing it
    would make it an orphan, which reads as "the survey was edited" and sends
    the reader after the wrong thing."""
    assert asked_question_ids([{"kind": "event", "event": EVENT_COVERAGE_ASKED, "detail": detail}]) == ()


# --- where the text comes from --------------------------------------------------------


def test_the_sessions_own_survey_is_preferred_over_the_live_one():
    """`context_survey` is copied into `session.json` at `start` precisely so a
    later edit cannot rewrite history. When it is there, it wins."""
    session_survey = [{"id": "s1", "order": 0, "text": "as asked on the night"}]
    project_survey = _survey("s1")
    coverage = coverage_for_session(
        [_asked_event("s1")], session_survey=session_survey, project_survey=project_survey
    )
    assert coverage.survey_source == SURVEY_SOURCE_SESSION
    assert coverage.asked[0].text == "as asked on the night"


def test_the_live_survey_is_a_flagged_fallback_not_a_silent_one():
    """The real case, measured: the one session on the maintainer's machine with coverage
    events predates `context_survey` and has none. Falling back is right -- the
    ids are still meaningful -- but the hub has to be able to say the text may
    have been reworded since, so the source is carried rather than assumed."""
    coverage = coverage_for_session(
        [_asked_event("s1")], session_survey=[], project_survey=_survey("s1", "s2")
    )
    assert coverage.survey_source == SURVEY_SOURCE_PROJECT
    assert [item.id for item in coverage.asked] == ["s1"]
    assert [item.id for item in coverage.unasked] == ["s2"]


def test_no_survey_anywhere_is_reported_as_not_reportable():
    """**The COMMON path, not the edge one.** Every real recording this project
    has made so far ran `--no-coach` and unfiled, so it has no survey at all. A
    session that ran without one did not "cover 0 of 0 questions", and the hub
    must word it differently."""
    coverage = coverage_for_session([], session_survey=None, project_survey=None)
    assert coverage.survey_source == SURVEY_SOURCE_NONE
    assert coverage.is_reportable is False
    assert coverage.total == 0


def test_a_survey_with_nothing_asked_IS_reportable():
    """The distinction the test above depends on: a survey existed and none of
    it was covered is a real, and quite bad, finding. It must not collapse into
    the same empty state as "there was no survey"."""
    coverage = coverage_for_session([], session_survey=None, project_survey=_survey("s1", "s2"))
    assert coverage.is_reportable is True
    assert len(coverage.unasked) == 2


def test_an_asked_id_with_no_matching_question_is_surfaced_not_dropped():
    """A question asked in an interview and since deleted from the survey. The
    tempting behaviour is to drop it -- it matches nothing, so it renders as
    nothing. But it is a fact about the record: something was covered and the
    question it answered is gone."""
    coverage = coverage_for_session(
        [_asked_event("s1"), _asked_event("deleted-since", seq=4)],
        session_survey=None,
        project_survey=_survey("s1"),
    )
    assert coverage.orphaned_question_ids == ("deleted-since",)
    assert [item.id for item in coverage.asked] == ["s1"]


def test_unasked_keeps_survey_order_while_asked_keeps_conversation_order():
    coverage = coverage_for_session(
        [_asked_event("s3"), _asked_event("s1", seq=8)],
        session_survey=None,
        project_survey=_survey("s1", "s2", "s3", "s4"),
    )
    assert [item.id for item in coverage.asked] == ["s3", "s1"]
    assert [item.id for item in coverage.unasked] == ["s2", "s4"]


# --- across the whole project ---------------------------------------------------------


def test_a_question_never_asked_in_any_interview_is_the_headline():
    """The number a per-session view structurally cannot produce, and the one
    that decides what to do in the next interview: not "we missed question 4
    tonight" but "question 4 has never been asked, in three interviews"."""
    standing = project_survey_standing(
        {"sessA": ["s1"], "sessB": ["s1", "s2"], "sessC": []}, survey=_survey("s1", "s2", "s3")
    )
    assert [item.id for item in standing.never_asked] == ["s3"]
    assert standing.times_asked("s1") == 2
    assert standing.times_asked("s3") == 0
    assert standing.sessions_read == 3


def test_the_sessions_that_asked_a_question_are_named_not_just_counted():
    """A count answers "how much"; the ids answer "which interviews", which is
    what you need to go and read them. Per-interview counting is this project's
    central rule -- a bare tally is the shape that lets one enthusiastic
    interviewee look like four."""
    standing = project_survey_standing(
        {"sessB": ["s1"], "sessA": ["s1"]}, survey=_survey("s1")
    )
    assert standing.asked_in_sessions["s1"] == ("sessA", "sessB")


def test_an_empty_project_reports_every_question_as_never_asked():
    standing = project_survey_standing({}, survey=_survey("s1", "s2"))
    assert len(standing.never_asked) == 2
    assert standing.sessions_read == 0
