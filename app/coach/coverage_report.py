"""What the survey actually covered, read back AFTER the interview.

`app/coach/coverage.py` is the live half: it watches the interviewer's stream
and appends a `coverage-asked` event the moment a survey question is asked. This
is the other half -- a pure, offline read of what those events left behind, for
the hub.

**Nothing here re-infers anything.** No model call, no matching, no heuristic.
The decision was made live, once, and recorded; this replays it. A second
inference path would be a second answer to "was this asked", and the two would
disagree the first time the prompt changed.

**The event vocabulary is IMPORTED, never respelled.** `EVENT_COVERAGE_ASKED`
and the `question_id` detail key come from the module that writes them. A
coverage report that greps for its own spelling of the event name is a report
that goes quietly empty the day the writer is renamed -- and "no questions were
asked" is a perfectly plausible-looking answer, which is what makes that failure
mode worth designing against rather than testing for.

**Where the survey TEXT comes from, and why it is not a detail.** An event
records a `question_id` and nothing else, so the text has to be recovered. Two
sources, and they are not interchangeable:

- **The session's own `context_survey`**, copied into `session.json` at `start`
  precisely so a later edit cannot re-write history. This is the correct source
  and the only one that can be trusted to say what was on screen that night.
- **The project's CURRENT `context/survey.json`**, which is a fallback for
  sessions recorded before `context_survey` was persisted -- and which may have
  been reworded, reordered or partly deleted since.

Reporting the second as though it were the first would attach today's question
text to last week's answer. So the source is carried on the result
(`survey_source`), the hub says which one it used, and ids that resolve to no
question at all are surfaced as `orphaned_question_ids` rather than dropped --
an asked question whose text is gone is a fact about the record, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.coach.context import SurveyItem
from app.coach.coverage import EVENT_COVERAGE_ASKED

# The detail key the live writer uses. Imported in spirit, pinned here by a
# gate-style test rather than by import, because it is a dict key inside a
# payload rather than a module constant -- see the test that asserts a real
# recorded event still parses.
QUESTION_ID_KEY = "question_id"

SURVEY_SOURCE_SESSION = "session"
SURVEY_SOURCE_PROJECT = "project"
SURVEY_SOURCE_NONE = "none"


@dataclass(frozen=True)
class SurveyCoverage:
    """One session's survey coverage, as read from what it left on disk."""

    asked: "tuple[SurveyItem, ...]"
    unasked: "tuple[SurveyItem, ...]"
    orphaned_question_ids: "tuple[str, ...]"
    survey_source: str

    @property
    def total(self) -> int:
        return len(self.asked) + len(self.unasked)

    @property
    def is_reportable(self) -> bool:
        """Whether this coverage means anything at all.

        A session with no survey did not "cover 0 of 0 questions" -- it ran
        without a survey, which is a different statement and must read
        differently. Every one of this project's real recordings so far is this
        case (`--no-coach`, unfiled), so the empty state is the COMMON path
        here, not the edge one."""
        return self.total > 0


def asked_question_ids(records: "Iterable[Mapping[str, Any]]") -> "tuple[str, ...]":
    """The survey question ids this session recorded as asked, in order, once
    each.

    De-duplicated because nothing un-asks a question: the live detector already
    treats `asked` as terminal, and a repeated event (a resumed session
    replaying its own log) must not make a question look twice-covered."""
    seen: "list[str]" = []
    for record in records:
        if record.get("kind") != "event" or record.get("event") != EVENT_COVERAGE_ASKED:
            continue
        detail = record.get("detail")
        if not isinstance(detail, Mapping):
            continue
        question_id = detail.get(QUESTION_ID_KEY)
        # Deliberately strict about the type: a non-string id cannot match a
        # `SurveyItem.id` and would silently become an orphan, which reads as
        # "the survey was edited" rather than "this record is malformed".
        if isinstance(question_id, str) and question_id and question_id not in seen:
            seen.append(question_id)
    return tuple(seen)


def coverage_for_session(
    records: "Iterable[Mapping[str, Any]]",
    *,
    session_survey: "Optional[Sequence[Mapping[str, Any]]]",
    project_survey: "Optional[Sequence[SurveyItem]]" = None,
) -> SurveyCoverage:
    """One session's coverage.

    `session_survey` is `session.json`'s `context_survey` -- the survey as it
    was AT THE TIME. `project_survey` is the live artifact, used only when the
    session recorded none, and flagged when it is."""
    survey, source = _resolve_survey(session_survey, project_survey)
    asked_ids = asked_question_ids(records)
    by_id = {item.id: item for item in survey}

    # Ordered by the ASKED order, not the survey's -- the hub's question is
    # "what did this interview cover", and the sequence it happened in is the
    # shape of the conversation. `unasked` keeps survey order, because that is
    # the order they would be asked in next time.
    asked = tuple(by_id[qid] for qid in asked_ids if qid in by_id)
    orphaned = tuple(qid for qid in asked_ids if qid not in by_id)
    asked_set = set(asked_ids)
    unasked = tuple(
        item for item in sorted(survey, key=lambda i: i.order) if item.id not in asked_set
    )
    return SurveyCoverage(
        asked=asked, unasked=unasked, orphaned_question_ids=orphaned, survey_source=source,
    )


def _resolve_survey(
    session_survey: "Optional[Sequence[Mapping[str, Any]]]",
    project_survey: "Optional[Sequence[SurveyItem]]",
) -> "tuple[tuple[SurveyItem, ...], str]":
    if session_survey:
        # Parsed through the real model, never read as loose dicts: a stored
        # survey that no longer satisfies `SurveyItem` should fail here, loudly,
        # rather than render as a half-populated card.
        return tuple(SurveyItem(**dict(entry)) for entry in session_survey), SURVEY_SOURCE_SESSION
    if project_survey:
        return tuple(project_survey), SURVEY_SOURCE_PROJECT
    return (), SURVEY_SOURCE_NONE


@dataclass(frozen=True)
class ProjectSurveyStanding:
    """Which survey questions this project has EVER had asked, across every
    interview under it -- and which are still open.

    This is the number that decides what to do in the next interview, and it is
    the one the hub could not previously answer at all: a per-session view can
    say "we missed question 4 tonight" but never "question 4 has never been
    asked, in six interviews"."""

    never_asked: "tuple[SurveyItem, ...]"
    asked_in_sessions: "dict[str, tuple[str, ...]]"
    """question id -> the session ids that asked it, in the order encountered."""

    sessions_read: int

    def times_asked(self, question_id: str) -> int:
        return len(self.asked_in_sessions.get(question_id, ()))


def project_survey_standing(
    per_session_asked: "Mapping[str, Sequence[str]]",
    *,
    survey: "Sequence[SurveyItem]",
) -> ProjectSurveyStanding:
    """Fold many sessions' asked-ids into one standing.

    Takes an already-read mapping of `session_id -> asked question ids` rather
    than reaching for the disk itself, for the reason every loader in this
    project takes its directory: it makes the whole thing testable with a dict
    and no session tree at all."""
    asked_in: "dict[str, list[str]]" = {}
    for session_id in sorted(per_session_asked):
        for question_id in per_session_asked[session_id]:
            asked_in.setdefault(question_id, []).append(session_id)
    never = tuple(
        item for item in sorted(survey, key=lambda i: i.order) if item.id not in asked_in
    )
    return ProjectSurveyStanding(
        never_asked=never,
        asked_in_sessions={qid: tuple(sessions) for qid, sessions in asked_in.items()},
        sessions_read=len(per_session_asked),
    )
