"""What the survey covered, as the hub reports it.

Two properties carry most of the weight here, and neither is about arithmetic:

- **"There was no survey" must never render as "0 of 0 covered."** Every real
  recording this project has made so far ran with no survey at all, so the
  empty state is the COMMON path and the one most likely to be read as a
  failing score.
- **Where the question TEXT came from is part of the answer.** A session that
  stored no survey of its own is reported against today's `survey.json`, and
  that fallback is flagged, because today's wording attached to last month's
  answer is a silent lie rather than a rounding error.

The counting itself is tested in `test_coverage_report.py`; what is defended
here is that the hub carries those answers out intact and degrades per section.
"""

from __future__ import annotations

import json

from app.web.review import build_review

SURVEY = [
    {"id": "s1", "order": 0, "text": "how do you record"},
    {"id": "s2", "order": 1, "text": "what happened last time"},
    {"id": "s3", "order": 2, "text": "what did you do next"},
]


def _project(
    tmp_path,
    *,
    survey=None,
    sessions=None,
):
    """`sessions` is `{session_id: (asked_question_ids, context_survey_or_None)}`."""
    root = tmp_path / "projects" / "acme"
    context_dir = root / "context"
    context_dir.mkdir(parents=True)
    if survey is not None:
        (context_dir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    for session_id, (asked, context_survey) in (sessions or {}).items():
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        lines = []
        for index, question_id in enumerate(asked):
            lines.append(json.dumps({
                "kind": "event", "seq": index, "speaker": "interviewer",
                "event": "coverage-asked",
                "detail": {"question_id": question_id, "matched_seq": index},
            }))
        (session_dir / "transcript.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        metadata = {"started_at": "2026-08-01T10:00:00+03:00"}
        if context_survey is not None:
            metadata["context_survey"] = context_survey
        (session_dir / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


# --- the empty state, which is the common one ---------------------------------------------


def test_a_project_with_no_survey_reports_absence_rather_than_a_zero_score(tmp_path):
    # The failure this exists to prevent: a page that says "0 of 0 covered"
    # over an interview that never had a survey to cover.
    _project(tmp_path, sessions={"S1": ((), None)})
    coverage = build_review("acme", "S1", repo_root=tmp_path)["coverage"]
    assert coverage["session"]["reportable"] is False
    assert coverage["session"]["total"] == 0
    assert coverage["project"] is None
    assert coverage["error"] is None


def test_a_survey_nobody_answered_is_still_reportable(tmp_path):
    # The other side of the same line: a survey EXISTS and none of it was
    # asked. That is a real, actionable zero and must not be silenced along
    # with the absent-survey case.
    _project(tmp_path, survey=SURVEY, sessions={"S1": ((), SURVEY)})
    session = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["session"]
    assert session["reportable"] is True
    assert session["asked"] == [] and len(session["unasked"]) == 3


def test_the_review_carries_coverage_even_with_no_session_selected(tmp_path):
    # The project standing is a fact about the project, not about one call --
    # it has to survive a hub opened without a session.
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), SURVEY)})
    coverage = build_review("acme", None, repo_root=tmp_path)["coverage"]
    assert coverage["session"] is None
    assert [item["id"] for item in coverage["project"]["never_asked"]] == ["s2", "s3"]


# --- provenance of the question text -------------------------------------------------------


def test_a_session_that_stored_its_own_survey_is_reported_against_that_one(tmp_path):
    stored = [{"id": "s1", "order": 0, "text": "the wording that was on screen"}]
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), stored)})
    session = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["session"]
    assert session["survey_source"] == "session"
    assert session["asked"][0]["text"] == "the wording that was on screen"


def test_falling_back_to_todays_survey_is_flagged(tmp_path):
    # A session recorded before `context_survey` was persisted. The text is
    # recoverable but only from the CURRENT survey, which may have been
    # reworded since -- so the answer is carried with its source attached
    # rather than presented as though it were the record.
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), None)})
    session = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["session"]
    assert session["survey_source"] == "project"
    assert session["asked"][0]["text"] == "how do you record"


def test_a_question_asked_but_since_deleted_is_surfaced_not_dropped(tmp_path):
    # An asked question whose text is gone is a fact about the record. Dropping
    # it would quietly shrink the numerator and nothing would say why.
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1", "s9"), None)})
    session = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["session"]
    assert session["orphaned_question_ids"] == ["s9"]
    assert [item["id"] for item in session["asked"]] == ["s1"]


# --- the project standing -------------------------------------------------------------------


def test_the_standing_reads_every_session_under_the_project(tmp_path):
    _project(tmp_path, survey=SURVEY, sessions={
        "S1": (("s1",), SURVEY),
        "S2": (("s1", "s2"), SURVEY),
    })
    project = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["project"]
    assert project["sessions_read"] == 2
    assert project["times_asked"]["s1"] == 2
    assert [item["id"] for item in project["never_asked"]] == ["s3"]


def test_one_unreadable_session_does_not_cost_the_others(tmp_path):
    # Per-section degradation, applied per SESSION: the operator is most likely
    # opening this page BECAUSE something went wrong, and a standing that
    # refuses to render is worse than one that is short by a session.
    #
    # An unparseable `session.json` is one of the two cases `read_session`
    # really does raise on. A torn transcript TAIL does not -- see the test
    # below -- and neither does a missing transcript, so either of those would
    # have made this pass without exercising the guard at all.
    root = _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), SURVEY)})
    broken = root / "sessions" / "S2"
    broken.mkdir(parents=True)
    (broken / "session.json").write_text("{not json", encoding="utf-8")
    project = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["project"]
    assert project["times_asked"]["s1"] == 1
    assert project["sessions_read"] == 1


def test_a_session_with_a_torn_tail_still_contributes_what_it_recorded(tmp_path):
    # `read_session` tolerates a truncated final line by design -- a session
    # killed mid-write is the normal way a recording ends badly. The standing
    # counts what survived rather than discarding the session, and counts the
    # session as read, because it was.
    root = _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), SURVEY)})
    torn = root / "sessions" / "S2"
    torn.mkdir(parents=True)
    (torn / "transcript.jsonl").write_text(
        json.dumps({
            "kind": "event", "seq": 0, "speaker": "interviewer", "event": "coverage-asked",
            "detail": {"question_id": "s2"},
        }) + "\n{\"kind\": \"eve",
        encoding="utf-8",
    )
    project = build_review("acme", "S1", repo_root=tmp_path)["coverage"]["project"]
    assert project["sessions_read"] == 2
    assert project["times_asked"]["s2"] == 1


def test_a_stored_survey_that_no_longer_parses_is_an_error_not_a_crash(tmp_path):
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), [{"id": "s1"}])})
    coverage = build_review("acme", "S1", repo_root=tmp_path)["coverage"]
    assert coverage["error"] is not None
    assert coverage["session"] is None


def test_coverage_failing_does_not_cost_the_rest_of_the_page(tmp_path):
    _project(tmp_path, survey=SURVEY, sessions={"S1": (("s1",), [{"id": "s1"}])})
    review = build_review("acme", "S1", repo_root=tmp_path)
    assert review["coverage"]["error"] is not None
    assert review["standings"] == [] and "transcript" in review
