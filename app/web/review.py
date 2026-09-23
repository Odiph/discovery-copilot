"""The hub: what you look at the moment an interview stops.

Named `review.py` rather than `hub.py` because `app/web/hub.py` is already the
WebSocket connection hub. The command is still `python -m app hub`.

**Transcript-primary, standing in the lead card.** Competitive research across
thirteen research tools found the split cleanly: research-native tools
(Dovetail, Condens, Marvin, Looppanel, Great Question) put the transcript
centre-stage with the AI summary in a collapsible rail, and only the
meeting-recorder lineage leads with a summary. Summary-first is a sales
convention. What NONE of them do -- the gap this page exists for -- is answer
"which belief did this call move, and what should I ask next time". Every one
of those screens ends at *evidence captured*.

So: hypothesis standing is the first card, the transcript is the body, and the
summary sits between them as a block of COUNTS. It is arithmetic over the
corpus -- `app/coach/interview_summary.py` computes it with no model call --
because the interview's claims are already rendered above with their quotes
and citations, and a generated paragraph restating them would be a second,
unfalsifiable account of the same evidence.

**This module itself stays read-only, and the accept/reject card does not
change that.** Moving a hypothesis's state means putting bytes into
`context/hypotheses.json`, and gate 19 pins the modules that may do that by
set equality -- detected TEXTUALLY, per FILE rather than per statement, so a
module that merely names the artifact and puts bytes anywhere trips it. The
card is therefore mounted here from `app/web/accept.py`, a module whose entire
job is that one save, rather than being implemented in this file: admitting
this file to gate 19's set would retire the gate's reach over everything else
the hub does and will do. `register_accept_routes` adds the two routes to this
app so the evidence and the decision are one sitting on one port.

**Evidence is never more than one click from a claim.** The loudest complaint
users report about post-interview screens is not hallucination, it is evidence
buried one click too deep -- they *fear* hallucination exactly when they
cannot click through to the quote. Every fact here carries its own quote and
its segment span.

**All three columns always render.** A supporting-only view of a hypothesis is
the monotonic count `Bearing` was introduced to kill. The contrary column is
drawn even when it is empty, and says so -- and so is the unclear one, because
a claim the model could not direct disappearing without trace is the same
failure wearing a politer face.

**The page shows a DELTA, not a level.** Every counted number is split into
"before this interview" and "now", and the counting itself is delegated to
`app/coach/interview_summary.py`, which is the one home for the rule. This
module used to fork it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app import projects as projects_mod
from app.coach import corpus as coach_corpus
from app.coach import interview_summary as summary_mod
from app.coach import coverage_report
from app.coach import rejection_store
from app.coach.asked_store import CRITIQUE_DOC_FILENAME
from app.coach.context import load_survey
from app.session.participant import read_participant
from app.session.reader import list_session_ids, read_session
from app.session.render import TRANSCRIPT_DOC_FILENAME
from app.web.accept import register_accept_routes
from app.web.author import _FORBIDDEN, _host_header_is_loopback, register_hypothesis_routes
from app.coach.interview_summary import people_bounds
from app.session.talk_share import talk_share
from app.web.participant import register_participant_routes

logger = logging.getLogger(__name__)

_REVIEW_HOST = "127.0.0.1"

# Interviewer turns are shown too -- the question a claim answered is part of
# reading whether the claim means what it says.
SPEAKER_LABELS = {"interviewee": "them", "interviewer": "you"}


@dataclass(frozen=True)
class _ReviewStaticAssets:
    index_html: bytes
    review_js: bytes
    review_css: bytes


def _load_static_assets(static_dir: Path) -> _ReviewStaticAssets:
    return _ReviewStaticAssets(
        index_html=(static_dir / "index.html").read_bytes(),
        review_js=(static_dir / "review.js").read_bytes(),
        review_css=(static_dir / "review.css").read_bytes(),
    )


def _entry(fact: Any, *, session_id: "Optional[str]") -> "dict[str, Any]":
    """One claim, rendered. The only place in this module that touches a
    `Fact`'s fields -- the counting happens over `interview_summary.FactRef`,
    which structurally cannot carry the quote."""
    return {
        "fact_id": fact.id,
        "claim": fact.claim,
        "quoted_text": fact.quoted_text,
        "provenance": fact.provenance,
        "session_id": fact.session_id,
        "seqs": list(fact.seqs),
        "from_this_interview": fact.session_id == session_id,
    }


def _counts(standing: Any) -> "dict[str, int]":
    return {
        "supporting_interviews": standing.supporting_interviews,
        "contrary_interviews": standing.contrary_interviews,
        "unclear_interviews": standing.unclear_interviews,
    }


def standing_for(
    hypothesis: Any,
    facts: "tuple[Any, ...]",
    *,
    session_id: "Optional[str]" = None,
    project: str = "",
    summary: "Optional[Any]" = None,
) -> "dict[str, Any]":
    """One hypothesis's standing, and what a given interview did to it.

    `summary` is the page's one `InterviewSummary`, computed once by
    `build_review` and handed down. It is optional because a caller holding a
    single hypothesis should not have to build one, and it is CHECKED against
    `session_id` rather than trusted: a summary for a different interview would
    report another call's contribution under this one's heading, and no number
    on the page would look wrong.

    `session_id` splits every counted number into "this interview" and "all
    interviews". That split is the entire point of the page: a running total
    cannot tell you whether the call you just finished changed anything. So
    `before` is the standing this hypothesis would have had had this interview
    never happened, the top-level counts are the standing it has now, and
    `this_interview` is what this call contributed.

    Every one of those numbers, and the before/now split itself, is delegated
    ENTIRELY to `app.coach.interview_summary.standing_delta`. This function is
    an ADAPTER: it re-shapes, re-labels, and re-attaches the quotes and segment
    numbers that module is deliberately not allowed to hold. `python -m app
    standing <session>` renders the same triple on a terminal and calls the
    same function, so the two surfaces cannot print two different confident
    integers. This module used to fork the rule; the fork is gone."""
    refs = tuple(summary_mod.fact_ref(fact, project=project) for fact in facts)
    by_id = {fact.id: fact for fact in facts}

    delta = summary_mod.standing_delta(
        hypothesis, refs, session_id=session_id, project=project, summary=summary
    )
    now, before = delta.now, delta.before
    contributed = {
        "supporting_facts": delta.contributed.supporting_facts,
        "contrary_facts": delta.contributed.contrary_facts,
        "unclear_facts": delta.contributed.unclear_facts,
    }

    def _entries(column: "tuple[Any, ...]") -> "list[dict[str, Any]]":
        return [_entry(by_id[ref.fact_id], session_id=session_id) for ref in column]

    return {
        "id": hypothesis.id,
        "statement": hypothesis.statement,
        "falsifier": hypothesis.falsifier,
        "state": hypothesis.state,
        "criterion": hypothesis.criterion,
        # A hypothesis with no pre-registered rule cannot be honestly scored,
        # and the page says so rather than showing a confident-looking tally
        # against a rule nobody wrote.
        "is_scorable": hypothesis.is_scorable,
        # The interviews that ALREADY EXISTED when this criterion was written,
        # carried so the page can price a criterion edit before it happens:
        # everything on disk and NOT in here is currently pre-dated by the
        # rule, and an edit would move all of it inside the boundary. Naming
        # that cost is the whole reason criterion editing is allowed on this
        # page at all -- a silent edit here is exactly the target-painting
        # `criterion_registered_after` exists to catch.
        "criterion_registered_after": list(hypothesis.criterion_registered_after),
        "supporting": _entries(now.supporting),
        "contrary": _entries(now.contrary),
        "unclear": _entries(now.unclear),
        **_counts(now),
        "total_interviews": now.total_interviews,
        "before": _counts(before),
        "this_interview": contributed,
        # TOUCHED and MOVED are two different claims and the badge makes the
        # stronger one -- see `StandingDelta`, which owns both definitions so
        # the hub and the CLI cannot disagree about what "moved" means.
        "touched_this_interview": delta.touched,
        "moved_this_interview": delta.moved,
    }


# The page's extraction vocabulary, named once here instead of being typed as
# bare literals in this function and again in `review.js`.
#
# That duplication was a real finding of the 2026-08-12 audit: the producer
# emitted four literals, `review.js` keyed a lookup table on the same four by
# hand, and NO test rendered the banner -- so a rename on either side gave
# `headline === undefined` with the whole suite green. `ALL_EXTRACTION_STATUSES`
# is what `test_web_review_extraction_status.py` and the JS suite both pin
# against, in place of two hand-typed lists that could agree with each other
# and disagree with the code.
#
# Deliberately a SUPERSET of `app/mcp/tools.py`'s three: that surface answers
# per NODE and this one per INTERVIEW, so `no-hypotheses` and
# `no-facts-from-this-interview` have no meaning there. The words they DO share
# mean the same thing, which is the property worth holding.
STATUS_EXTRACTED = "extracted"
STATUS_NOT_EXTRACTED = "not-extracted"
STATUS_NO_FACTS_FROM_THIS_INTERVIEW = "no-facts-from-this-interview"
STATUS_NO_HYPOTHESES = "no-hypotheses"
STATUS_UNREADABLE = "unreadable"
ALL_EXTRACTION_STATUSES = (
    STATUS_EXTRACTED,
    STATUS_NOT_EXTRACTED,
    STATUS_NO_FACTS_FROM_THIS_INTERVIEW,
    STATUS_NO_HYPOTHESES,
    STATUS_UNREADABLE,
)


def _extraction_status(
    facts: "Sequence[Any]",
    hypotheses: "Sequence[Any]",
    *,
    session_id: "Optional[str]",
    unreadable: bool,
    corpus_exists: bool,
) -> "dict[str, Any]":
    """Whether this page KNOWS anything about this interview yet.

    Without this the page reported four different situations identically, as
    zero counts under the prose "Nothing in the corpus supports this yet" and
    "No disconfirming evidence found -- was any sought?": extraction never run;
    extraction run with no hypotheses loaded, where a bearing is structurally
    impossible; extraction run and the model found no bearing; and nothing in
    the world bearing on the belief. Only the last is a research finding, and
    the wording was written to provoke a reaction, so the other three read as
    one.

    That is not hypothetical. On 2026-08-12 the only real corpus in this repo
    held 23 facts with zero bearings, and the page could not distinguish a
    correct answer -- a career-history interview against note-taking
    hypotheses genuinely bears on nothing -- from a broken pipeline.

    Every number here comes from reads `build_review` has already done. This
    reports; it does not compute anything new."""
    known_ids = {h.id for h in hypotheses}
    mine = [f for f in facts if f.session_id == session_id] if session_id is not None else []
    # Against hypotheses that STILL EXIST. A bearing on an id since renamed or
    # deleted is not a connection to anything the operator can see, and
    # counting it would report the corpus as more connected than the page is.
    without_bearings = sum(
        1 for f in mine if not any(b.hypothesis_id in known_ids for b in f.bearings)
    )

    if unreadable:
        status = STATUS_UNREADABLE
    elif not hypotheses:
        # Said before "not-extracted": with nothing to bear on, zero bearings
        # is a statement about the CONTEXT DIR, not about the interview, and
        # reporting it as a finding would be a plain false negative.
        status = STATUS_NO_HYPOTHESES
    elif not corpus_exists:
        status = STATUS_NOT_EXTRACTED
    elif not mine:
        # SPLIT OUT from `not-extracted` after the 2026-08-12 audit, which
        # observed that one word was covering two situations and that the
        # remedy printed underneath told the operator to run the command they
        # had just run.
        #
        # What is actually knowable here, stated exactly: the corpus EXISTS,
        # so extraction has run in this project at least once, and it holds
        # nothing from this interview. What is NOT knowable is which of two
        # reasons that is -- this session was never extracted, or it was and
        # the model found nothing worth recording. Extraction leaves no record
        # of having run, so nothing on disk separates them.
        #
        # The status is therefore named for what is OBSERVED rather than for a
        # cause, and the page words it as the open question it is. Inventing
        # the distinction would be the same failure one level down: a
        # confident sentence about something the page cannot see.
        status = STATUS_NO_FACTS_FROM_THIS_INTERVIEW
    else:
        status = STATUS_EXTRACTED

    return {
        "status": status,
        "facts_total": len(facts),
        "facts_this_interview": len(mine),
        "facts_without_bearings": without_bearings,
        "hypotheses_loaded": len(hypotheses),
    }


def _summary_payload(
    summary: "Optional[Any]", hypotheses: "Sequence[Any]"
) -> "Optional[dict[str, Any]]":
    """This interview, summarized -- counts and node names, nothing else.

    The server has been computing this since `summarize_interview` existed and
    throwing all but three numbers of it away, while the page said "not built
    yet". This is that object reaching the wire; not one number here is new.

    **Arithmetic, never prose.** A sentence about what the interviewee meant is
    a model call, and this page renders what is on disk -- deterministically,
    at no cost, with no network failure mode on a read. The claims themselves
    are already rendered under the standings above, each with its own quote and
    citation; a paragraph restating them here would be a second, unfalsifiable
    account of the same evidence.

    `hypotheses_touched` counts only bearings on hypotheses that STILL EXIST,
    the same rule `_extraction_status` applies: a bearing on an id since
    renamed or deleted is not a connection the operator can see, and counting
    it would report the interview as better connected than the page is."""
    if summary is None:
        return None
    known_ids = {h.id for h in hypotheses}
    return {
        # Always 1 here, and carried anyway -- it is the field that says these
        # numbers describe ONE interview, next to the `*_interviews` counts on
        # the standings above, which describe many.
        "interviews": summary.interviews,
        "fact_count": summary.fact_count,
        # The split the fact count alone hides: an interview whose claims are
        # mostly the model's inference is a weaker result than the same count
        # of things the interviewee actually reported.
        "evidence_facts": summary.evidence_facts,
        "inference_facts": summary.inference_facts,
        "hypotheses_touched": sum(
            1 for touch in summary.touches if touch.hypothesis_id in known_ids
        ),
        # Which nodes this interview's claims were filed at. Usually one, and
        # more than one under a subtree rollup -- which is exactly the case the
        # counting rule exists to keep at ONE interview.
        "projects": list(summary.projects),
    }


# The asked-question critique's flag vocabulary, named ONCE here instead of
# being typed as bare literals in this module and again in `review.js`.
#
# The same duplication `ALL_EXTRACTION_STATUSES` exists to kill, and with the
# same failure mode: the producer emits six keys, the page keys a label table on
# the same six by hand, and a rename on either side renders a row with no label
# and no test red. `test_web_review_question_critique.py` pins this tuple
# against `asked_critique.to_json_dict`'s own summary keys, so this list cannot
# agree with the page while disagreeing with the module that writes the file.
QUESTION_FLAGS = (
    "leading",
    "hypothetical_or_future_tense",
    "pitch",
    "opinion_seeking",
    "confirmation_only",
    "generic",
)

# The flags that are 0 in this artifact BY CONSTRUCTION rather than by merit.
#
# `asked_critique.critique_turns` sends the critic a BATCH of interviewer turns
# from different points in the interview, so there is no single preceding
# utterance that could serve them -- and `is_generic` is judged against exactly
# that utterance. The prompt handles the empty case deterministically (set the
# flag false, never guess), so the number is honestly absent rather than quietly
# invented.
#
# It travels ON THE WIRE rather than being hardcoded in the page for the same
# reason `talk_share.interviewer_key` does: which flags are unjudgeable here is
# a property of the pipeline, and a client that had guessed would go on printing
# "0" as a pass the day a trigger utterance became available. A zero that means
# "not asked" and a zero that means "never fired" must never render alike.
QUESTION_FLAGS_NOT_JUDGED = ("generic",)

CRITIQUE_NO_SESSION = "no-session"
CRITIQUE_NOT_RUN = "not-run"
CRITIQUE_UNREADABLE = "unreadable"
CRITIQUE_READ = "read"
ALL_CRITIQUE_STATUSES = (
    CRITIQUE_NO_SESSION,
    CRITIQUE_NOT_RUN,
    CRITIQUE_UNREADABLE,
    CRITIQUE_READ,
)

REJECTIONS_NOT_KEPT = "not-kept"
REJECTIONS_UNREADABLE = "unreadable"
REJECTIONS_KEPT = "kept"
ALL_REJECTION_STATUSES = (REJECTIONS_NOT_KEPT, REJECTIONS_UNREADABLE, REJECTIONS_KEPT)


def _count(value: Any) -> "Optional[int]":
    """A non-negative integer off disk, or `None` for anything else.

    `None` rather than 0 for a value this module cannot read, because the whole
    point of these two panels is that an absent measurement and a measurement of
    zero are different claims. The `bool` check is not pedantry: `True` is an
    `int` in Python and would sail through as the count 1."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _rate(value: Any) -> "Optional[float]":
    """A share in [0, 1], or `None`.

    `None` is the artifact's OWN answer for "nothing of this kind could be
    judged" -- see `AskedCritique.clean_rate` -- and it is passed through
    untouched rather than defaulted, because an interview the critic could not
    reach is not an interview that scored zero and is not one that scored
    perfectly either."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0.0 <= float(value) <= 1.0:
        return None
    return float(value)


def _question_critique(paths: Any, session_id: "Optional[str]") -> "dict[str, Any]":
    """What `python -m app critique-questions` said about the questions the
    INTERVIEWER asked -- or the reason there is nothing to say.

    A STATUS rather than a bare payload, for exactly the reason
    `_extraction_status` exists: a missing critique file and a critique that
    found nothing worth flagging are different situations with different
    remedies, and a page handed `null` for both would have to invent a sentence
    that is wrong half the time. The critique is a paid model call the operator
    has to run deliberately, so "not run" is the COMMON state and it must not
    render as a score.

    Reads only the `summary` block. The artifact also carries every judged turn
    with its rationale, which is the interviewer's own speech and is not needed
    to draw a count -- so it does not cross onto the wire.

    Degrades in its own section like everything else here: a hand-edited
    critique that will not parse must not cost the operator the transcript."""
    result: "dict[str, Any]" = {
        "status": CRITIQUE_NO_SESSION, "error": None, "summary": None
    }
    if session_id is None:
        return result

    path = paths.sessions_root / session_id / CRITIQUE_DOC_FILENAME
    if not path.is_file():
        result["status"] = CRITIQUE_NOT_RUN
        return result

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["status"] = CRITIQUE_UNREADABLE
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    summary = loaded.get("summary") if isinstance(loaded, dict) else None
    if not isinstance(summary, dict):
        # Said as UNREADABLE, never as an empty critique. A file that exists
        # without the block this panel is drawn from is a broken artifact, and
        # reporting it as "the critic found nothing" would be a confident
        # sentence about something nothing on disk supports.
        result["status"] = CRITIQUE_UNREADABLE
        result["error"] = "the critique file carries no summary block"
        return result

    result["status"] = CRITIQUE_READ
    result["summary"] = {
        "turns": _count(summary.get("turns")),
        "judged": _count(summary.get("judged")),
        # Reachability, kept beside the rates rather than folded into them. A
        # turn the critic could not be reached for is absent from every rate
        # below, and a reader who is not told that reads the rates as covering
        # the whole interview.
        "unjudged": _count(summary.get("unjudged")),
        "clean": _count(summary.get("clean")),
        "clean_rate": _rate(summary.get("clean_rate")),
        # The denominator of the marked-question rate, so the page can print
        # "of N judged" rather than a bare percentage. The unmarked denominator
        # is `judged - marked_questions` and is left to the client rather than
        # duplicated here.
        "marked_questions": _count(summary.get("marked_questions")),
        "marked_question_clean_rate": _rate(summary.get("marked_question_clean_rate")),
        "unmarked_clean_rate": _rate(summary.get("unmarked_clean_rate")),
        "flags": {name: _count(summary.get(name)) for name in QUESTION_FLAGS},
        "flags_not_judged": list(QUESTION_FLAGS_NOT_JUDGED),
    }
    return result


def _rejections(paths: Any, session_id: "Optional[str]") -> "dict[str, Any]":
    """What extraction REFUSED, by reason -- the other half of the corpus.

    `app/coach/rejection_store.py` exists because this number was famously
    unreproducible: the project's notes carry "~30% of extractions reject as
    `unlocatable-quote`" and the terminal it was printed to is gone. This panel
    is the reason that record is kept, so it reports the project-wide answer
    that question was asked about, and the interview on screen underneath it.

    **A file that exists with no rows is MEANINGFUL and must not read as a
    missing file.** `replace_session_rejections` is called unconditionally,
    including with an empty sequence, precisely so "extracted and refused
    nothing" leaves a mark -- so an existing file with zero rows is a real zero
    and an absent file is silence. That distinction is the entire reason this
    function checks for the file itself rather than reading `load_rejections`'s
    empty tuple and guessing.

    `load_rejections` is deliberately TOLERANT -- a corrupt log must never block
    extraction -- which means it answers `()` for a corrupt file just as it does
    for an empty one. This function therefore parses the file once on its own
    account to tell those two apart, and reports a corrupt log as unreadable
    rather than as a project that refused nothing."""
    result: "dict[str, Any]" = {
        "status": REJECTIONS_NOT_KEPT, "error": None, "total": 0, "reasons": {},
        "sessions_recorded": 0, "this_interview": None,
    }
    path = paths.context_dir / rejection_store.REJECTIONS_FILENAME
    if not path.is_file():
        return result

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["status"] = REJECTIONS_UNREADABLE
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if not isinstance(parsed, list):
        result["status"] = REJECTIONS_UNREADABLE
        result["error"] = "the rejection log is not a list of rejections"
        return result

    rows = rejection_store.load_rejections(paths.context_dir)
    result["status"] = REJECTIONS_KEPT
    result["total"] = len(rows)
    # Counted by the store, never re-derived here. It is the function the module
    # was written to provide, and a second tally in this file would be a second
    # answer to the question the record exists to settle.
    result["reasons"] = rejection_store.count_by_reason(paths.context_dir)
    result["sessions_recorded"] = len(
        {row.get("session_id") for row in rows if isinstance(row.get("session_id"), str)}
    )

    if session_id is not None:
        mine = [row for row in rows if row.get("session_id") == session_id]
        counts: "dict[str, int]" = {}
        for row in mine:
            reason = row.get("reason")
            if isinstance(reason, str):
                counts[reason] = counts.get(reason, 0) + 1
        # ALWAYS a dict when an interview is selected, and a zero total here is
        # genuinely AMBIGUOUS: an extraction that refused nothing writes no row
        # naming this session, and so does an extraction that never ran. The
        # page says exactly that rather than picking one -- the same rule
        # `no-facts-from-this-interview` follows one section up.
        result["this_interview"] = {
            "total": len(mine),
            "reasons": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        }
    return result


def build_review(
    project_path: str,
    session_id: "Optional[str]",
    *,
    repo_root: "Optional[Path]" = None,
) -> "dict[str, Any]":
    """Everything the page shows, assembled once.

    Degrades per section exactly as the console does: a malformed corpus must
    not make the transcript unreadable, because the operator has just finished
    an interview and the transcript is the part that cannot be regenerated."""
    paths = projects_mod.resolve(project_path, repo_root=repo_root)
    review: "dict[str, Any]" = {"project": paths.path, "session_id": session_id}

    try:
        hypotheses = coach_corpus.load_hypotheses(paths.context_dir, missing_ok=True)
        review["hypotheses_error"] = None
    except Exception as exc:  # noqa: BLE001
        hypotheses, review["hypotheses_error"] = (), f"{type(exc).__name__}: {exc}"

    facts: "tuple[Any, ...]" = ()
    # Captured rather than re-derived from `facts` being empty: an ABSENT
    # corpus and a corpus that simply holds nothing from this interview are
    # different answers with different remedies, and the emptiness of the
    # tuple cannot tell them apart.
    corpus_exists = (paths.context_dir / coach_corpus.FACTS_FILENAME).is_file()
    try:
        if corpus_exists:
            facts = coach_corpus.load_facts(paths.context_dir, sessions_root=None)
        review["facts_error"] = None
    except Exception as exc:  # noqa: BLE001
        review["facts_error"] = f"{type(exc).__name__}: {exc}"

    review["extraction"] = _extraction_status(
        facts,
        hypotheses,
        session_id=session_id,
        unreadable=review["facts_error"] is not None,
        corpus_exists=corpus_exists,
    )
    # ONE pass over the corpus for this answer, computed here and handed to
    # every reader below. `summarize_interview` looks at every fact on disk and
    # `standing_for` is called once per hypothesis, so computing it there made
    # a project with twenty hypotheses do twenty identical passes for an answer
    # that cannot differ between them.
    refs = tuple(summary_mod.fact_ref(fact, project=paths.path) for fact in facts)
    summary = None if session_id is None else summary_mod.summarize_interview(session_id, refs)

    review["standings"] = [
        standing_for(h, facts, session_id=session_id, project=paths.path, summary=summary)
        for h in hypotheses
    ]
    review["coverage"] = _coverage(paths, session_id)
    # HOW MANY PEOPLE those interviews were, which is not the same number as how
    # many interviews there were and is the number every criterion is actually
    # written about ("at least 3 of 6 interviewees"). Counts only -- no label
    # crosses this line; `people_bounds` returns integers by construction.
    review["people"] = _people(paths)
    # WHO was interviewed. Always PRESENT and often `null` -- the key is never
    # omitted, because a missing key arrives in the client as `undefined`,
    # which is how a page ends up printing an empty string into the place a
    # name belongs and calling it a name. `read_participant` answers `None` for
    # every failure it can meet (no metadata file, metadata that will not
    # parse, no key, a key holding something that is not a string), so nothing
    # about this line can take the transcript down with it.
    #
    # This is the field the product's central counting rule has been missing:
    # `interview_summary` counts SESSIONS and calls them interviews, which is
    # the same number only while nobody is re-interviewed and nobody records
    # two calls in one sitting. The label does not change the count -- it makes
    # the count checkable by a human, which is all a label can honestly do.
    review["participant"] = (
        None if session_id is None else read_participant(paths.sessions_root / session_id)
    )
    review["transcript"] = _transcript(paths.sessions_root, session_id)
    # Who did the talking. The interviewer spoke ~65% of the words in both real
    # calls and nothing computed it until now; it is the only quality signal on
    # this page that costs no model call at all.
    #
    # Computed from the ALREADY-PROJECTED transcript above rather than from a
    # second `read_session`, for two reasons. It avoids reading the same file
    # twice per page load, and -- the one that actually matters -- it makes the
    # meter's speaker keys the same strings the reader sees in the transcript
    # below it. A share attributed to `interviewer` sitting under turns labelled
    # `you` invites the reader to wonder whether they are the same person.
    #
    # `interviewer_key` is sent so the client can emphasise the interviewer's
    # own row without hardcoding a label it does not own. `SPEAKER_LABELS` is
    # this module's to change, and a client that had guessed `you` would
    # silently stop emphasising anything the day it did.
    review["talk_share"] = {
        **talk_share(review["transcript"]),
        "interviewer_key": SPEAKER_LABELS.get("interviewer", "interviewer"),
    }
    # ONE field, `summary`, holding either the counts or `null`. There used to
    # be a second, `summary_status`, carrying the words "not built yet" that no
    # client ever read -- the page read `data.summary` and printed its own
    # fallback, so the server's status string was dead on the wire and the
    # sentence a reader saw could not be changed from the server at all.
    #
    # `null` rather than zeros when the corpus could not be read: "0 claims
    # extracted" is a finding about the interview, and an unreadable file is a
    # fault in the file. The banner above already names which.
    review["summary"] = (
        None if review["facts_error"] is not None else _summary_payload(summary, hypotheses)
    )
    # HOW YOU INTERVIEWED, and WHAT EXTRACTION REFUSED. Both are already on disk
    # and neither was reaching a screen; both are read in their own try/except so
    # a hand-edited artifact costs its own panel and nothing else.
    #
    # Deliberately NOT folded into `summary`: that field is arithmetic over
    # `facts.json` and is `null` when the corpus will not parse, which has no
    # bearing on whether the interviewer's own questions were judged. Sharing a
    # field would make an unreadable corpus silently delete the question
    # critique too.
    review["question_critique"] = _question_critique(paths, session_id)
    review["rejections"] = _rejections(paths, session_id)
    return review


def _survey_item(item: Any) -> "dict[str, Any]":
    return {"id": item.id, "order": item.order, "text": item.text}


def _coverage(
    paths: Any, session_id: "Optional[str]"
) -> "dict[str, Any]":
    """What the survey covered -- this interview, and the project standing.

    Two ANSWERS, not one, because they lead to different actions: "we missed
    question 4 tonight" is a note about this call, and "question 4 has never
    been asked, in six interviews" is a decision about the next one.

    **Never re-infers.** Every judgement here was made live by
    `app/coach/coverage.py` and written to the transcript; this replays it. See
    `app/coach/coverage_report.py` for why a second inference path would be a
    second answer to the same question.

    Degrades in its own section like everything else on this page, and
    per-session inside the standing: one unreadable transcript among twenty
    must not cost the operator the other nineteen."""
    result: "dict[str, Any]" = {"session": None, "project": None, "error": None}

    try:
        survey = load_survey(paths.context_dir)
    except Exception as exc:  # noqa: BLE001 -- a project with no survey is the COMMON path
        # Not an error the operator has to act on: every real recording so far
        # ran with no survey at all. The absent survey is reported as absence
        # further down, via `reportable`, rather than as a failure here.
        logger.info("no readable survey for this project error_type=%s", type(exc).__name__)
        survey = ()

    per_session: "dict[str, tuple[str, ...]]" = {}
    if paths.sessions_root.is_dir():
        for session_dir in sorted(paths.sessions_root.iterdir()):
            if not session_dir.is_dir():
                continue
            try:
                records = read_session(session_dir).records_of_kind("event")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "skipping a session while counting coverage session=%s error_type=%s",
                    session_dir.name, type(exc).__name__,
                )
                continue
            per_session[session_dir.name] = coverage_report.asked_question_ids(records)

    if survey:
        standing = coverage_report.project_survey_standing(per_session, survey=survey)
        result["project"] = {
            "sessions_read": standing.sessions_read,
            "never_asked": [_survey_item(item) for item in standing.never_asked],
            "times_asked": {qid: standing.times_asked(qid) for qid in standing.asked_in_sessions},
        }

    if session_id is None:
        return result

    session_dir = paths.sessions_root / session_id
    if not session_dir.is_dir():
        return result
    try:
        read = read_session(session_dir)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    try:
        coverage = coverage_report.coverage_for_session(
            read.records,
            session_survey=read.metadata.get("context_survey"),
            project_survey=survey,
        )
    except Exception as exc:  # noqa: BLE001 -- a stored survey that no longer parses
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["session"] = {
        # `reportable` before anything else, because it is the field that keeps
        # "this interview ran without a survey" from rendering as "0 of 0
        # covered". They are different statements and only one of them is a
        # miss.
        "reportable": coverage.is_reportable,
        "survey_source": coverage.survey_source,
        "asked": [_survey_item(item) for item in coverage.asked],
        "unasked": [_survey_item(item) for item in coverage.unasked],
        "orphaned_question_ids": list(coverage.orphaned_question_ids),
        "total": coverage.total,
    }
    return result


def _people(paths: Any) -> "dict[str, Any]":
    """How many distinct people this project's interviews represent.

    Project-scoped rather than per-hypothesis, deliberately. The criteria are
    written as "at least N of M interviewees", and M -- the size of the pool
    somebody recruited -- is the half a reader cannot get from anywhere else on
    this page. N is per-hypothesis and is already served by the standings.

    Reads `session.json` per interview and never a transcript, exactly as
    `sessions_at` does: naming who was in the room is not displaying what they
    said, so this stays on the right side of that line and stays cheap enough
    to run on every page load.

    Degrades to an empty absence rather than raising. A project whose sessions
    directory does not exist yet is the first-run shape, not a fault."""
    if not paths.sessions_root.is_dir():
        return {"sessions": 0, "fewest": 0, "most": 0, "certain": True}
    labels = {
        name: read_participant(paths.sessions_root / name)
        for name in list_session_ids(paths.sessions_root)
    }
    bounds = people_bounds(labels)
    return {
        "sessions": bounds.sessions,
        "fewest": bounds.fewest,
        "most": bounds.most,
        # Named `certain` rather than `is_certain` to match the payload's own
        # flat style; the client branches on it before quoting either end,
        # because quoting one end of a range alone is the failure this whole
        # block exists to prevent.
        "certain": bounds.is_certain,
    }


def _transcript(sessions_root: Path, session_id: "Optional[str]") -> "list[dict[str, Any]]":
    if session_id is None:
        return []
    session_dir = sessions_root / session_id
    if not session_dir.is_dir():
        return []
    try:
        result = read_session(session_dir)
    except Exception as exc:  # noqa: BLE001 -- an unreadable transcript is a message, not a crash
        logger.warning("reading the transcript failed error_type=%s", type(exc).__name__)
        return []
    return [
        {
            "seq": record.get("seq"),
            "speaker": SPEAKER_LABELS.get(record.get("speaker", ""), record.get("speaker")),
            "text": record.get("text", ""),
        }
        for record in result.records_of_kind("segment")
    ]


def sessions_at(sessions_root: Path) -> "list[dict[str, Any]]":
    """The interviews this project has, newest first, for the in-page picker.

    `rendered` says whether `python -m app render` has been run. The picker
    shows it rather than hiding un-rendered sessions: the hub reads the
    transcript from the JSONL directly and does not need the document, so
    hiding them would drop real interviews from the only list of them.

    `participant` is who was interviewed, and it is what turns this list from a
    column of timestamps into a list a human can choose from. `None` is the
    common answer -- every recording made before the field existed carries no
    label -- so the row is present either way and the page decides how to say
    "nobody is named"; dropping the key for unlabelled interviews would make
    the absence indistinguishable from a listing that never learned about the
    field at all.

    Nothing here opens a transcript. Listing interviews is not displaying
    customer speech, and naming the person in the room is not either --
    `read_participant` reads `session.json` alone, never `transcript.jsonl`, so
    this function stays on the right side of that line and stays cheap enough
    to run once per interview on every page load."""
    return [
        {
            "id": name,
            "rendered": (sessions_root / name / TRANSCRIPT_DOC_FILENAME).is_file(),
            "participant": read_participant(sessions_root / name),
        }
        for name in list_session_ids(sessions_root)
    ]


def resolve_session(
    sessions_root: Path, requested: "Optional[str]", *, default: "Optional[str]"
) -> "tuple[Optional[str], Optional[str]]":
    """Turns a session name off the wire into one this process will open, or an
    error message.

    **The check is MEMBERSHIP of the real listing, never a pattern.** The hub
    is the first surface where a session id arrives from a browser rather than
    from the operator's own command line, and `sessions_root / requested` with
    a `..` in it resolves outside the project. An allowlist built from what is
    actually on disk cannot be talked round by traversal, by a symlink or by a
    case difference, and it needs no separate argument about which characters
    are dangerous.

    `None` means "whatever the command line chose", which is how the page
    behaves before the operator has touched the picker."""
    if requested is None:
        return default, None
    if requested not in list_session_ids(sessions_root):
        # Deliberately does not echo the requested value back into the
        # message. It is attacker-controlled text heading for a page that
        # renders it, and the operator does not need it repeated to understand
        # what happened.
        return None, "that session is not one of this project's interviews"
    return requested, None


def create_review_app(
    *,
    static_dir: Path,
    project: str,
    session_id: "Optional[str]",
    repo_root: "Optional[Path]" = None,
) -> FastAPI:
    """The hub's ASGI app. Same two guards as the authoring and console apps
    -- the loopback bind stops the LAN but not the operator's own browser, and
    this page renders verbatim customer speech."""
    assets = _load_static_assets(static_dir)
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def _guard_cross_origin_requests(request: Request, call_next):
        if not _host_header_is_loopback(request.headers.get("host", "")):
            if request.method == "POST":
                await request.body()
            return _FORBIDDEN
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                await request.body()
                return _FORBIDDEN
        return await call_next(request)

    @app.get("/")
    async def get_page() -> Response:
        return Response(
            content=assets.index_html, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/review.js")
    async def get_js() -> Response:
        return Response(
            content=assets.review_js, media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/review.css")
    async def get_css() -> Response:
        return Response(
            content=assets.review_css, media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/sessions")
    async def get_sessions(session: "Optional[str]" = None) -> JSONResponse:
        """The picker's contents, and which row it should have selected.

        Served separately from `/api/review` rather than folded into it so the
        picker survives a review that fails: the operator whose corpus will not
        load is exactly the one who needs to be able to switch to another
        interview."""
        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        selected, _ = resolve_session(paths.sessions_root, session, default=session_id)
        return JSONResponse({
            "project": paths.path,
            "sessions": sessions_at(paths.sessions_root),
            "selected": selected,
        })

    @app.get("/api/review")
    async def get_review(session: "Optional[str]" = None) -> JSONResponse:
        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        selected, error = resolve_session(paths.sessions_root, session, default=session_id)
        if error is not None:
            return JSONResponse({"error": error}, status_code=400)
        return JSONResponse(build_review(project, selected, repo_root=repo_root))

    # The accept/reject card's two routes, defined in `app/web/accept.py` and
    # mounted here rather than on a second server -- a second port would force
    # a cross-origin fetch and a CORS surface for the sake of one write.
    register_accept_routes(app, project=project, session_id=session_id, repo_root=repo_root)

    # WHO was interviewed, mounted from `app/web/participant.py` for exactly
    # the reason the accept card is mounted from `app/web/accept.py`: this
    # module stays read-only, and the set of modules that can put bytes into a
    # finished session directory stays small enough to read in a sitting. A
    # session is the one artifact here that cannot be regenerated.
    register_participant_routes(
        app, project=project, session_id=session_id, repo_root=repo_root
    )

    # Building a hypothesis and filling in a missing criterion, mounted from
    # `app/web/author.py` for the same reason the accept card is mounted from
    # `app/web/accept.py`: gate 19 pins the modules that may write
    # `context/hypotheses.json`, and this module is deliberately not one of
    # them. See `register_hypothesis_routes` for why a second save path was
    # the wrong answer.
    #
    # Resolved ONCE here rather than per request because the project is fixed
    # for this app's lifetime -- it came from the command line, and unlike the
    # session it is not selectable in the page.
    try:
        register_hypothesis_routes(
            app, context_dir=projects_mod.resolve(project, repo_root=repo_root).context_dir
        )
    except projects_mod.ProjectPathError:
        # A project path that will not resolve is already reported by every
        # other route on this app; failing to mount is better than refusing to
        # start, because the page still reads the transcript from disk.
        logger.warning("hypothesis routes not mounted: the project path does not resolve")

    return app


def build_review_server(app: FastAPI, *, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=_REVIEW_HOST, port=port, log_level="warning")
    return uvicorn.Server(config)
