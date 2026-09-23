"""Accept or reject one coach proposal -- and the ONLY other module in this
codebase that may put bytes into `context/hypotheses.json`.

**Why this is its own file rather than a few lines added somewhere sensible.**
Gate 19 (`app/tests/test_phase_gates.py`) pins, by set EQUALITY, which modules
both name the hypothesis artifact and contain a write primitive. It matches its
two regexes PER FILE, not per statement, so membership is all-or-nothing per
module: the moment a module joins that set, the gate stops discriminating
anything about the rest of what that module does. Adding the write to
`app/web/review.py` (the hub, which renders transcripts, resolves projects and
will keep growing) or to `app/web/author.py` (already in the set, but whose
job is pre-interview authoring) would each buy the same feature at the cost of
the gate's reach over a file with plenty of other reasons to change. A
dedicated single-purpose module keeps the blast radius to a file whose entire
content is that one write, and leaves gate 19 at full strength everywhere else.

**A REJECTION IS RECORDED, and this is where.** This module used to answer a
reject with `{"applied": false}` and keep nothing, on the argument that "the
proposal already lives in the transcript forever". That argument was wrong in
one specific way: the transcript records what the COACH proposed, and nothing
anywhere recorded what the HUMAN decided about it. A rejection is real
information -- someone read the evidence, considered a state change, and
judged it insufficient -- and it is arguably worth more than an acceptance,
because an acceptance leaves its trace in `state` while a rejection left none
at all. Worse, the loss was not even idempotent in the operator's experience:
the coach can propose the same (hypothesis, state) pair again in a later
interview, and with nothing on disk the operator re-adjudicates from scratch
with no memory of their own earlier reasoning. For a product whose entire
thesis is disconfirmation and pre-registration, silently dropping the one
decision that says "the evidence did not move me" is the exact wrong thing to
forget.

It goes in `context/declined_proposals.json`, an artifact of its own beside
the corpus. The two alternatives were considered and both are worse:

- **Not on the hypothesis record.** `context/hypotheses.json` is the
  user-authored corpus the COACH reads to decide what to ask; a decline is a
  fact about an operator's judgement, not about the belief. Putting it there
  means every reader of `Hypothesis` (`extra="forbid"`) must grow a field for
  adjudication history that none of them wants, the corpus grows without bound
  with clicks, and -- the one that actually decides it -- a decline would have
  to REWRITE the whole corpus. `apply_decision` below refuses to write when it
  cannot first read the file strictly, precisely so a partial corpus can never
  be put back. A rejection changes nothing about the corpus, so making it a
  corpus write would put the corpus at risk for a decision defined by not
  touching it.
- **Not in the session transcript.** A session is closed by the time the hub
  renders it, and `SessionStore`'s CONSTRUCTOR appends a `session-resumed`
  event when it reopens a directory that already has a transcript -- so
  writing there would record a fake interview restart (the trap
  `app/tests/test_web_participant.py` already documents). Beyond the
  mechanism: the question this record has to answer is "has this
  (hypothesis, state) pair already been declined, EVER, anywhere in this
  project", which is asked when a LATER interview re-proposes it. A record
  filed inside interview A's transcript could only be found by scanning every
  transcript in the project on every page load. The lookup is project-scoped
  and keyed by hypothesis, so the record belongs next to the hypotheses.

The log is APPEND-ONLY and never collapsed: two declines of the same proposal
are two judgements made at two times, and keeping one would throw away the
same information this whole section exists to stop throwing away. Records
written by a future version are appended to VERBATIM -- `record_decline`
re-serializes the existing list unread rather than re-projecting it, so a
field this version does not know about is not deleted by this version.

**What it does not do.** No second copy of the
hypothesis schema, no second validator. The payload is validated by
`app/web/author.py::_validate_hypotheses_payload` -- imported, never
re-derived, on exactly the precedent `author.py:45-47` states for the brief and
survey ("imported, never redefined, so the two halves can never drift"). A
second validator drifting from the first is the defect this avoids, and it
would drift in the worst possible direction: this module is the one that writes
the file the coach later reads.

**Read strictly, then write.** The existing corpus is read through
`app/coach/corpus.py::load_hypotheses`, which RAISES on a malformed file --
never through `author.py`'s deliberately tolerant `_read_existing`, which
reports a malformed file as an empty list. That distinction is the whole
safety property here: an accept applied on top of a tolerant read would take a
corpus the loader could not parse, silently reduce it to one hypothesis, and
put that back on disk. A file we cannot read is a file we refuse to replace.

**Proposals come from the interview that produced them.** `store.py`'s
`suggestion` records already carry `target_hypothesis_id` and `proposed_state`
(06-03-PLAN.md §H1); `render.py` already prints them as "PROPOSED (not
applied)". Nothing new is recorded to make this card work -- it reads what the
call already wrote. The card shows only the proposal; the hypothesis text, its
criterion and its supporting/undermining facts come from the hub's own
`/api/review` standing for the same id, so the evidence a reader judges by has
exactly ONE source.

**Loopback-only, like every other page here** -- and the write route checks the
Host header ITSELF rather than trusting whichever app it is mounted on to have
installed the hub's middleware. `_FORBIDDEN` and `_host_header_is_loopback` are
imported from `author.py` verbatim, as `console.py` and `review.py` already do.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import projects as projects_mod
from app.coach import corpus as coach_corpus
from app.coach.corpus import HYPOTHESES_FILENAME
from app.session.reader import list_session_ids, read_session
from app.web.author import (
    _FORBIDDEN,
    _atomic_write_json,
    _host_header_is_loopback,
    _validate_hypotheses_payload,
)

PROPOSALS_ROUTE = "/api/proposals"
DECISION_ROUTE = "/api/decision"

DECISION_ACCEPT = "accept"
DECISION_REJECT = "reject"

# The declined-proposal log. Lives in `context/` beside the corpus rather than
# inside it -- see this module's docstring for why neither the hypothesis
# record nor the session transcript is the right home. Git-ignores under the
# root `context/*` rule and the blanket `projects/` rule already in
# `.gitignore`; gate 10 asserts that over this concrete path.
DECLINED_FILENAME = "declined_proposals.json"


def _utc_now_iso() -> str:
    """The wall clock, in UTC, to the second.

    A clock value is normally something this codebase refuses to store --
    `Fact` carries none at all, and `Hypothesis.criterion_registered_after`
    deliberately records SESSION IDS rather than a timestamp so the claim it
    makes cannot be forged from a clock. Neither argument transfers here, and
    it is worth saying exactly why rather than letting the exception look like
    an oversight:

    - For a fact, the time the tool ran is provenance-of-PROCESS and says
      nothing about the claim. For a decline, WHEN A HUMAN LOOKED is the
      content: "I considered this in March" and "I considered this an hour
      ago" are different facts about how much the operator should trust their
      own earlier judgement.
    - The forgery worry behind `criterion_registered_after` was that a clock
      could manufacture a pre-registration claim. This field claims nothing
      about ordering against anything -- there is no evidence it must pre-date
      -- so there is nothing for a wrong value to unfairly earn.

    The interview the proposal came from is stored ALONGSIDE it, and that half
    is the checkable one: it names a directory under `sessions/` that either
    exists or does not."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_declines_strictly(path: Path) -> "list[Any]":
    """The log exactly as it is on disk, raising rather than guessing.

    Deliberately does NOT validate the individual records: this is the read
    that a WRITE is built on top of, and its whole job is to hand back
    something that can be appended to and put down again without losing
    anything. Validating (and therefore dropping) an item here would make the
    write path the thing that deletes records, which is the defect
    `apply_decision`'s own strict read exists to prevent one file over.

    Raises `ValueError` for a file that is not valid JSON or not a JSON array.
    An absent file is an empty log, which is the normal first case."""
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON array of declined proposals.")
    return raw


def _decline_key(record: Any) -> "Optional[tuple[str, str]]":
    """The (hypothesis, proposed state) pair a record is about, or `None` if
    this record does not carry one.

    The same pair `proposals_for` deduplicates on, for the same reason: two
    different proposed states for one hypothesis genuinely disagree, so a
    decline of one must never be reported as a decline of the other. A record
    that cannot be keyed is skipped for DISPLAY only -- it is still preserved
    on disk by `record_decline`."""
    if not isinstance(record, dict):
        return None
    hypothesis_id = record.get("hypothesis_id")
    proposed_state = record.get("proposed_state")
    if not isinstance(hypothesis_id, str) or not isinstance(proposed_state, str):
        return None
    return (hypothesis_id, proposed_state)


def declines_by_pair(context_dir: "Optional[Path]") -> "dict[tuple[str, str], dict[str, Any]]":
    """Everything the log knows, folded per (hypothesis, proposed state) pair,
    for the card that has to say "you already looked at this".

    TOLERANT, unlike `_read_declines_strictly` above, and the asymmetry is the
    same one this module already draws between `corpus.load_hypotheses` and
    `author.py::_read_existing`: a log that will not parse must cost the
    operator a BADGE, never the page. Reading is where degrading is right;
    writing is where refusing is right, because only the write can destroy
    what is already there.

    Reports `count`, `last_declined_at` and the interviews the declines were
    filed against. It does NOT report how many times the proposal was made --
    nothing here counts proposals, and a field named for one would be a
    confident number about something this log never saw."""
    if context_dir is None:
        return {}
    try:
        raw = _read_declines_strictly(Path(context_dir) / DECLINED_FILENAME)
    except (ValueError, OSError):
        return {}

    folded: "dict[tuple[str, str], dict[str, Any]]" = {}
    for record in raw:
        key = _decline_key(record)
        if key is None:
            continue
        declined_at = record.get("declined_at")
        session_id = record.get("session_id")
        entry = folded.get(key)
        if entry is None:
            entry = {"count": 0, "last_declined_at": None, "sessions": []}
            folded[key] = entry
        entry["count"] += 1
        if isinstance(declined_at, str) and declined_at:
            # LAST WRITTEN, not the lexicographic maximum. The log is
            # append-only, so its order is the order the decisions were
            # actually made; a max() over the strings would silently re-order
            # the history the moment a record arrived with a differently
            # formatted stamp, and would then report a time no human recognised
            # as the most recent one.
            entry["last_declined_at"] = declined_at
        if isinstance(session_id, str) and session_id and session_id not in entry["sessions"]:
            entry["sessions"].append(session_id)
    return folded


def record_decline(
    context_dir: Path,
    *,
    hypothesis_id: str,
    proposed_state: str,
    session_id: "Optional[str]",
    declined_at: str,
) -> "tuple[Optional[int], Optional[str]]":
    """Appends ONE decline to the log. Returns `(count_written, None)` or
    `(None, error)`; on any error NOTHING is written, the same order
    `apply_decision` uses.

    **The existing records are re-serialized UNREAD.** `_read_declines_strictly`
    hands back whatever was in the array and this function appends to it
    without projecting the items through a schema -- so a record carrying a
    field this version has never heard of survives the next decline instead of
    being quietly normalised away. That is the same property
    `apply_decision`'s strict read protects for the corpus, arrived at
    differently: there, an unparseable item aborts the write; here there is no
    per-item schema to be unparseable against, so the items are simply never
    touched.

    Never checks that `hypothesis_id` names a hypothesis that still exists,
    and that is deliberate -- unlike `apply_decision`, which must. Declining a
    proposal about a hypothesis someone has since deleted is still a real
    judgement a human made, and refusing to record it would lose exactly the
    information this log was added to stop losing."""
    path = Path(context_dir) / DECLINED_FILENAME
    try:
        existing = _read_declines_strictly(path)
    except (ValueError, OSError) as exc:
        # `OSError` as well as the malformed-file `ValueError`, so a log that
        # cannot be OPENED (a permission, a name that is now a directory)
        # produces the same refusal-to-clobber and the same reported reason
        # rather than a 500. Either way the rule is one rule: a file we cannot
        # read is a file we refuse to replace.
        return None, f"{type(exc).__name__}: {exc}"

    record: "dict[str, Any]" = {
        "hypothesis_id": hypothesis_id,
        "proposed_state": proposed_state,
        "declined_at": declined_at,
        "session_id": session_id,
    }
    updated = [*existing, record]
    _atomic_write_json(path, updated)
    return len(updated), None


def proposals_for(
    sessions_root: Path, session_id: "Optional[str]", *, context_dir: "Optional[Path]"
) -> "list[dict[str, Any]]":
    """Every hypothesis-state proposal this interview made, one per distinct
    (hypothesis, proposed state) pair.

    Deduplicated on that pair and NOT on the hypothesis alone: a coach that
    asked twice about the same belief has not doubled anything, so two
    identical cards would ask for two clicks on one decision -- but two
    DIFFERENT proposed states for one hypothesis genuinely disagree, and
    collapsing them would silently pick a winner between them, which is
    precisely the judgement this card exists to leave with a human.

    `context_dir` is a REQUIRED keyword argument that is explicitly nullable,
    on `corpus.load_facts`'s own precedent for `sessions_root`: a caller must
    consciously decide whether the declined-proposal log is consulted, because
    a DEFAULT that skipped it would silently hand the page a proposal it could
    not tell had already been declined -- which is the entire defect this
    argument exists to fix, reintroduced by an argument default.

    Every proposal carries `previously_declined`, ALWAYS present and often
    `null`. Never omitted, for the reason `review.py` states about
    `participant`: a missing key arrives in the client as `undefined`, and
    `undefined` is how a page ends up rendering nothing where a warning
    belongs. A previously-declined proposal is still returned and still
    decidable -- hiding it would replace one silent loss with another, and the
    operator who declined this in March may well accept it in June on new
    evidence.

    Degrades to no proposals rather than raising, matching the hub's
    per-section degradation: a transcript that will not parse must not take
    the standing card down with it."""
    if session_id is None:
        return []
    session_dir = Path(sessions_root) / session_id
    if not session_dir.is_dir():
        return []
    try:
        result = read_session(session_dir)
    except Exception:  # noqa: BLE001 -- an unreadable transcript is an empty card, not a crash
        return []

    declined = declines_by_pair(context_dir)

    ordered: "list[dict[str, Any]]" = []
    by_pair: "dict[tuple[str, str], dict[str, Any]]" = {}
    for record in result.records_of_kind("suggestion"):
        target = record.get("target_hypothesis_id")
        state = record.get("proposed_state")
        if not isinstance(target, str) or not isinstance(state, str):
            continue
        proposal = by_pair.get((target, state))
        if proposal is None:
            proposal = {
                "target_hypothesis_id": target,
                "proposed_state": state,
                "questions": [],
                "seqs": [],
                "previously_declined": declined.get((target, state)),
            }
            by_pair[(target, state)] = proposal
            ordered.append(proposal)
        proposal["questions"].append(record.get("text", ""))
        proposal["seqs"].append(record.get("seq"))
    return ordered


def apply_decision(
    context_dir: Path, *, hypothesis_id: str, proposed_state: str
) -> "tuple[Optional[int], Optional[str]]":
    """Moves ONE hypothesis to `proposed_state`, leaving every other
    hypothesis and every other field of that one untouched. Returns
    `(count_written, None)` or `(None, error)`; on any error NOTHING is
    written -- validation completes before the atomic write is reached, the
    same "validated, then written, never the other order" order `author.py`
    already applies to the brief and survey.

    The criterion in particular survives the move: it is the pre-registered
    rule the new state was judged against, and a write that dropped it would
    erase the only thing making the state defensible."""
    path = Path(context_dir) / HYPOTHESES_FILENAME
    try:
        existing = coach_corpus.load_hypotheses(context_dir, missing_ok=False)
    except coach_corpus.CorpusValidationError as exc:
        return None, str(exc)

    if not any(hypothesis.id == hypothesis_id for hypothesis in existing):
        return None, f"no hypothesis {hypothesis_id!r} exists in {path}"

    updated = [
        {**hypothesis.model_dump(), "state": proposed_state}
        if hypothesis.id == hypothesis_id
        else hypothesis.model_dump()
        for hypothesis in existing
    ]
    validated, error = _validate_hypotheses_payload(updated)
    if error is not None:
        return None, error
    _atomic_write_json(path, validated)
    return len(validated), None


def _decision_from(
    body: Any,
) -> "tuple[Optional[tuple[str, str, str, Optional[str]]], Optional[str]]":
    """The request body, or the reason it is not one. A verb that is neither
    exactly `accept` nor exactly `reject` is refused rather than falling
    through to either branch -- a typo must never be read as consent.

    `session` is OPTIONAL and defaults to `None`, meaning "whatever the
    command line chose" -- the same default `get_proposals` applies, and for
    the same reason: the card follows the page's picker, so a decline has to
    be filed against the interview that was actually on screen rather than the
    one this process was started with. A value that IS supplied is validated
    by the route, by membership of the real listing, never here."""
    if not isinstance(body, dict):
        return None, "the decision body must be a JSON object"
    hypothesis_id = body.get("hypothesis_id")
    proposed_state = body.get("proposed_state")
    decision = body.get("decision")
    session = body.get("session")
    if not isinstance(hypothesis_id, str) or not hypothesis_id:
        return None, "hypothesis_id must be a non-empty string"
    if not isinstance(proposed_state, str) or not proposed_state:
        return None, "proposed_state must be a non-empty string"
    if not isinstance(decision, str) or decision not in (DECISION_ACCEPT, DECISION_REJECT):
        return None, f"decision must be exactly {DECISION_ACCEPT!r} or {DECISION_REJECT!r}"
    if session is not None and (not isinstance(session, str) or not session):
        return None, "session, when given, must be a non-empty string"
    return (hypothesis_id, proposed_state, decision, session), None


def register_accept_routes(
    app: FastAPI,
    *,
    project: str,
    session_id: "Optional[str]",
    repo_root: "Optional[Path]" = None,
) -> None:
    """Mounts the two accept/reject routes onto an existing app -- the hub's,
    so the evidence and the decision are one sitting on one port. A second
    server would force a cross-origin fetch and a CORS surface for the sake of
    the single write in the product, which is a worse trade than sharing the
    page the reader is already looking at."""

    @app.get(PROPOSALS_ROUTE)
    async def get_proposals(session: "Optional[str]" = None) -> JSONResponse:
        """`?session=` FOLLOWS the hub's picker.

        Without it this card would keep answering for the session named on the
        command line while the page beside it showed a different interview --
        and the card is the one place in the product that writes, so the
        mismatch would be silently offering a decision about the wrong call.

        Validated the same way the hub validates it, by MEMBERSHIP of the real
        listing rather than by pattern, because `sessions_root / session` with
        a `..` in it leaves the project. This route re-does the check instead
        of trusting the mounting app for the same reason `post_decision`
        re-does the loopback guard below: the write path must not depend on a
        caller having remembered."""
        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if session is None:
            chosen = session_id
        elif session in list_session_ids(paths.sessions_root):
            chosen = session
        else:
            return JSONResponse(
                {"error": "that session is not one of this project's interviews"},
                status_code=400,
            )
        return JSONResponse({
            "proposals": proposals_for(
                paths.sessions_root, chosen, context_dir=paths.context_dir
            )
        })

    @app.post(DECISION_ROUTE)
    async def post_decision(request: Request):
        # This route's OWN guard, not the mounting app's. The hub installs the
        # same check in middleware, but the one write in this product must not
        # depend on a caller having remembered to.
        if not _host_header_is_loopback(request.headers.get("host", "")):
            await request.body()
            return _FORBIDDEN
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 -- malformed JSON is a 400, never a 500
            return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
        decision, error = _decision_from(body)
        if error is not None:
            return JSONResponse({"error": error}, status_code=400)
        hypothesis_id, proposed_state, verb, requested_session = decision

        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if verb == DECISION_REJECT:
            # The hypothesis artifact is NOT touched -- a rejection is defined
            # by the corpus staying where it is. What IS written is the
            # judgement itself, to its own file, because the alternative is
            # the defect this branch used to have: the operator read the
            # evidence, decided the evidence did not move them, and the product
            # kept nothing at all. See the module docstring for why that record
            # lives beside the corpus rather than inside it or in the
            # transcript.
            #
            # Validated by MEMBERSHIP of the real listing, exactly as
            # `get_proposals` does and for the same reason: this is a write
            # path, and `sessions_root / session` with a `..` in it leaves the
            # project. An unusable session id is a 400 and records nothing,
            # rather than being quietly dropped to `null` -- a decline filed
            # against no interview at all is a worse record than no decline.
            if requested_session is None:
                filed_against = session_id
            elif requested_session in list_session_ids(paths.sessions_root):
                filed_against = requested_session
            else:
                return JSONResponse(
                    {"error": "that session is not one of this project's interviews"},
                    status_code=400,
                )
            _, record_error = record_decline(
                paths.context_dir,
                hypothesis_id=hypothesis_id,
                proposed_state=proposed_state,
                session_id=filed_against,
                declined_at=_utc_now_iso(),
            )
            # Answered 200 with `recorded: false` rather than as a failure, and
            # the page says so in words. The rejection's PRIMARY effect -- the
            # hypothesis does not move -- succeeded unconditionally, so a 4xx
            # here would tell the operator their rejection did not happen, which
            # is false. Reporting success without qualification would be the
            # opposite lie. Both halves are stated, and the page prints the
            # reason.
            return JSONResponse({
                "applied": False,
                "decision": DECISION_REJECT,
                "recorded": record_error is None,
                "record_error": record_error,
            })

        written, apply_error = apply_decision(
            paths.context_dir, hypothesis_id=hypothesis_id, proposed_state=proposed_state
        )
        if apply_error is not None:
            return JSONResponse({"error": apply_error}, status_code=400)
        return JSONResponse({"applied": True, "decision": DECISION_ACCEPT, "saved": written})
