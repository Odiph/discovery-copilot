"""The critique call, and -- in the same module but an entirely separate pure
function -- `rank_candidates`, the ONLY place an order is ever decided
(04-CONTEXT.md D-05; 04-02-PLAN.md §D5).

**The model classifies. Code ranks.** This is the whole mechanism. The
critique call returns, per candidate: six named per-rule booleans, a band
(`clear_violation` | `uncertain` | `clean`), a one-line rationale, and the
index of the candidate it refers to. Nothing about ORDER is ever asked of the
model -- `critique_instruction()` in `app/coach/prompts.py` says so
explicitly, and `rank_candidates` below is a pure function over a fixed sort
key, exactly like `app/session/store.py::latency_stats`: no class, no I/O, no
client, unit-tested directly.

**The fifth flag, `is_confirmation_only` (06-02-PLAN.md §G4, D-07).** A
candidate that targets a named hypothesis and is worded so that no possible
answer could count as evidence against it -- the risk this whole pivot
invites, since a coach that knows the hypothesis is a coach that can walk the
interviewer into confirming it. Judged against the hypothesis's OWN written
`falsifier` text, given verbatim in the critique prompt, never a falsifier
the model infers itself. It is added and read exactly like the existing
four: `rank_candidates` drops it only through the band, never directly (§G2).
**What the offline stub can prove and cannot prove (A7):** every test in
this module drives `app/coach/stub.py`, which "returns whatever it was
primed with." That proves the falsifier text reaches the prompt, the schema
carries the flag, and a primed `clear_violation` drops at any rank -- it
proves NOTHING about whether a real model correctly recognizes a
confirmation-only question. Record this as **A7**, UNVERIFIED, alongside
A1/A2/A3 -- a green offline suite here must never be read as evidence the
judgment itself is reliable.

**The sixth flag, `is_generic` (2026-08-15).** A candidate that could have
been asked of any interviewee in any interview -- it does not engage with
what THIS person just said. Added because the first real coached interview
produced the verdict "the coach was not pushing quality followup and it was
always on screen -- only give it to push something if its good quality."
The root cause was structural rather than tuning: `rank_candidates` drops
only `clear_violation`, so "broke no rule" had silently become "worth
interrupting a live conversation for". This flag is the deep fix -- it lets
the critic judge WORTH inside the existing rubric, instead of a second,
parallel worth mechanism that would violate the ordering contract below.

It is judged against the trigger utterance, which `assemble_critique_request`
now requires (see there for why it rides in the user message and not the
cached system block). Like the other five it drops a candidate ONLY through
the band -- special-casing a direct drop here would be a SECOND ordering
path. **A7 applies to it exactly as it does to `is_confirmation_only`**: the
offline stub proves the trigger text reaches the prompt and that the schema
carries the flag, and proves NOTHING about whether a real model recognizes
genericness.

**`rank_candidates` must NEVER:**
- let an uncertain candidate be dropped (only `clear_violation` is dropped);
- let a `clear_violation` candidate be merely ranked down (it is dropped
  entirely, at no rank);
- ask the model for an order, or use anything other than
  `(band_rank, original_generate_index)` as the sort key;
- perform I/O, construct a client, or require anything beyond plain data.

A primary standing alone (one survivor) is a normal, correct, complete
result -- never a degraded or partial one (D-05).

Structured output via forced tool-use, mirroring `generate.py`/
`coverage.py`'s precedent (D1). Same two-tier failure discipline as
`generate.py`: any parse failure, refusal or truncation makes the WHOLE
critique response untrusted (`CritiqueFailure`), and the caller
(`app/coach/pipeline.py`) still durably records what the generate stage
proposed, per §D5/Task 3.

This module must NEVER import `anthropic` (T-04-12, gate 11) and must NEVER
place candidate/rationale text into a log line or event `detail` dict
(T-04-27).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from app.coach import generate, prompts, reasoning_client
from app.coach.generate import Candidate

logger = logging.getLogger(__name__)

BAND_CLEAR_VIOLATION = "clear_violation"
BAND_UNCERTAIN = "uncertain"
BAND_CLEAN = "clean"
ALL_BANDS = frozenset({BAND_CLEAR_VIOLATION, BAND_UNCERTAIN, BAND_CLEAN})

CRITIC_TOOL_NAME = "classify_candidates"

_UNTRUSTED_STOP_REASONS = frozenset({"refusal", "max_tokens", "pause_turn", "model_context_window_exceeded"})

# clean before uncertain -- clear_violation never appears here because it is
# dropped before a rank is ever computed (D-05).
_BAND_RANK = {BAND_CLEAN: 0, BAND_UNCERTAIN: 1}


class CritiqueFailure(Exception):
    """Raised for anything that makes the critique response untrusted: a
    timeout, a parse failure, a refusal, a truncation, or any other
    exception. NEVER carries candidate or rationale text -- only `reason`
    and the response's own `stop_reason` (or `None`)."""

    def __init__(self, *, reason: str, stop_reason: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stop_reason = stop_reason


@dataclass(frozen=True)
class Judgment:
    """One candidate's verdict. `index` refers back to `Candidate.index` --
    the generate stage's own emission order, never re-numbered here.

    `is_confirmation_only` (06-02-PLAN.md §G4): NO DEFAULT, mirroring the
    existing four -- a `Judgment` constructed without it raises, exactly
    like a missing key in the parsed response makes the whole critique
    response untrusted. This flag drops a candidate ONLY through the band
    (`rank_candidates` below reads `judgment.band` alone), exactly like the
    existing four -- special-casing a direct drop on
    `is_confirmation_only=True` here would be a SECOND ordering path and
    would violate the "must NEVER ... use anything other than
    `(band_rank, original_generate_index)` as the sort key" contract stated
    at the top of this module (§G2)."""

    index: int
    is_leading: bool
    is_hypothetical_or_future_tense: bool
    is_pitch: bool
    is_opinion_seeking: bool
    is_confirmation_only: bool
    is_generic: bool
    is_off_topic: bool
    band: str
    rationale: str


def _critic_tool_schema() -> dict:
    return {
        "name": CRITIC_TOOL_NAME,
        "description": "Classify each candidate question against the Mom Test rubric.",
        "input_schema": {
            "type": "object",
            "properties": {
                "judgments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "is_leading": {"type": "boolean"},
                            "is_hypothetical_or_future_tense": {"type": "boolean"},
                            "is_pitch": {"type": "boolean"},
                            "is_opinion_seeking": {"type": "boolean"},
                            # 06-02-PLAN.md §G4: added HERE and in `required`
                            # below -- both, in the SAME function, is the
                            # part that matters (gate 23's sibling mistake:
                            # a flag declared in properties and forgotten in
                            # `required` makes every candidate's classify
                            # response fail to parse, per §G4/§G6).
                            "is_confirmation_only": {"type": "boolean"},
                            # The SIXTH flag (2026-08-15). Same two-place rule
                            # as the fifth: properties AND `required` below.
                            "is_generic": {"type": "boolean"},
                            # The SEVENTH flag (2026-08-19). Same two-place rule
                            # again: properties AND `required` below.
                            "is_off_topic": {"type": "boolean"},
                            "band": {"type": "string", "enum": sorted(ALL_BANDS)},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "index",
                            "is_leading",
                            "is_hypothetical_or_future_tense",
                            "is_pitch",
                            "is_opinion_seeking",
                            "is_confirmation_only",
                            "is_generic",
                            "is_off_topic",
                            "band",
                            "rationale",
                        ],
                    },
                },
            },
            "required": ["judgments"],
        },
    }


def assemble_critique_request(
    *,
    candidates: Sequence[Candidate],
    hypotheses: Sequence[Any],
    survey: Sequence[Any],
    trigger_text: str,
    model: str,
    max_tokens: int,
) -> dict:
    """Pure: no I/O, no client. Every candidate the generate stage produced
    is embedded here -- including the ones that will end up dropped -- so a
    candidate the critic never saw is impossible by construction.

    `hypotheses` (06-02-PLAN.md §G3/§G4) is a REQUIRED keyword, never
    defaulted: a call site that forgets it must fail loudly rather than
    silently critique a `test_hypothesis` candidate with no falsifier text
    to judge it against.

    `trigger_text` (2026-08-15) is REQUIRED for the same reason and carries
    the same lesson: until the sixth flag, this request contained NO
    conversational context at all -- only candidate texts and hypotheses.
    `is_generic` asks whether a candidate engages with what THIS interviewee
    just said, which is unanswerable without the utterance it is supposed to
    engage with. A defaulted empty string would let a call site silently ask
    the model to judge engagement against nothing, and the model would
    answer anyway.

    `survey` (2026-08-19) is REQUIRED on the same reasoning, arriving with
    the seventh flag. `is_off_topic` asks whether a candidate serves what
    this interview is FOR, and until now this request carried no statement
    of that at all -- the critic saw candidate texts, hypotheses and one
    utterance, so "relevant to the research" was a question it had no
    material to answer. A defaulted empty survey would let a call site ask
    it anyway, and the model would answer anyway: it would read an empty
    plan as "nothing is on the plan" and could only ever flag everything or
    nothing.

    **It goes in the SYSTEM block, and that placement is load-bearing for
    the opposite reason `trigger_text`'s is.** The survey is the
    interviewer's plan for the call: fixed for the life of the session,
    exactly like the hypotheses it sits beside. Only the ASKED/UNASKED state
    moves during a call, and that state is deliberately NOT sent here --
    this flag asks what a question is FOR, never whether its item has been
    reached yet, and including it would rewrite the cached prefix on every
    cycle for information the flag does not use.

    **It goes in the USER message, and that placement is load-bearing.** The
    system block below carries the cache breakpoint and must be
    byte-identical on every cycle of an interview; the trigger changes every
    cycle by definition. Putting it above the breakpoint would silently
    return the whole caching win described in the next paragraph -- no
    error, just `cache_creation_input_tokens: 0` and a third of the coach
    bill back.

    **The system block carries a cache breakpoint, and it is load-bearing.**
    Every field in it -- the rubric, the hypothesis falsifiers, the injection
    guard, the worked examples -- is fixed for the life of a session, while
    the candidates it judges live in the user message below. So the prefix is
    byte-identical on every cycle of an interview: written once, read on every
    call after that.

    It was a plain string before, which meant no breakpoint and no caching at
    all -- 184 uncached calls in a measured 26-minute interview, ~30% of the
    whole coach bill. The reason it went unnoticed is that this model's
    minimum cacheable prefix (4,096 tokens) sat above what the instruction
    reached on its own, so adding a breakpoint alone would have changed
    nothing; `prompts._CRITIQUE_WORKED_EXAMPLES` is what carries it over the
    floor, and the two changes are only useful together."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": prompts.critique_instruction(
                    tool_name=CRITIC_TOOL_NAME, hypotheses=hypotheses, survey=survey
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": prompts.critique_user_block(
                    candidates, trigger_text=trigger_text
                ),
            }
        ],
        "tools": [_critic_tool_schema()],
        "tool_choice": {"type": "tool", "name": CRITIC_TOOL_NAME},
    }


def _parse_critique_response(response: Any) -> "list[Judgment]":
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason in _UNTRUSTED_STOP_REASONS:
        raise CritiqueFailure(reason="untrusted-stop-reason", stop_reason=stop_reason)
    # Guarded, not read unconditionally: `stop_details`/refusal detail is
    # populated only on a refusal and absent on the common path (§D5) -- a
    # normal completion must never raise reading it.
    _ = getattr(response, "stop_details", None)

    for block in getattr(response, "content", None) or ():
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == CRITIC_TOOL_NAME:
            try:
                raw_judgments = block.input["judgments"]
                if not isinstance(raw_judgments, list):
                    raise ValueError("judgments must be an array")
                judgments: "list[Judgment]" = []
                seen_indices: "set[int]" = set()
                for entry in raw_judgments:
                    band = entry["band"]
                    if band not in ALL_BANDS:
                        raise ValueError(f"unrecognized band {band!r}")
                    # Type-check what the MODEL supplied. The band is validated
                    # above, but an `index` of "0" (a string) would parse fine
                    # and then never match a Candidate's int index -- every
                    # candidate would look unvetted, `rank_candidates` would
                    # drop them all, and the cycle would record SUCCESS with an
                    # empty result. The surface freezes on stale content while
                    # health says everything is fine, indistinguishable from
                    # "all candidates were genuinely bad". A stub returning
                    # well-typed blocks can never produce this; a real model can.
                    # `bool` is a subclass of int, so exclude it explicitly.
                    if not isinstance(entry["index"], int) or isinstance(entry["index"], bool):
                        raise ValueError("judgment index must be an integer")
                    if not isinstance(entry["rationale"], str):
                        raise ValueError("judgment rationale must be a string")
                    # Contradictory output (two judgments claiming the same
                    # index, one clean and one clear_violation) must make the
                    # WHOLE response untrusted, per D-05. Silently last-winning
                    # would let emission order decide whether a violation is
                    # shown to the user.
                    if entry["index"] in seen_indices:
                        raise ValueError(f"duplicate judgment index {entry['index']!r}")
                    seen_indices.add(entry["index"])
                    judgments.append(
                        Judgment(
                            index=entry["index"],
                            is_leading=bool(entry["is_leading"]),
                            is_hypothetical_or_future_tense=bool(entry["is_hypothetical_or_future_tense"]),
                            is_pitch=bool(entry["is_pitch"]),
                            is_opinion_seeking=bool(entry["is_opinion_seeking"]),
                            # 06-02-PLAN.md §G4: a missing key here raises
                            # KeyError, caught below and turned into a
                            # parse-error that makes the WHOLE response
                            # untrusted -- exactly like a missing existing
                            # flag does today, never a silent False default.
                            is_confirmation_only=bool(entry["is_confirmation_only"]),
                            # Same rule as every flag above: a missing key
                            # raises KeyError and makes the WHOLE response
                            # untrusted, never a silent False default. A
                            # defaulted-False `is_generic` would read as "the
                            # model judged this engaged with what was said"
                            # when in fact it never answered.
                            is_generic=bool(entry["is_generic"]),
                            # Same rule again, and it matters more here than
                            # anywhere: a defaulted-False `is_off_topic` would
                            # read as "the model judged this relevant to the
                            # research" on a response where it never looked --
                            # which is precisely the silence the flag exists to
                            # break.
                            is_off_topic=bool(entry["is_off_topic"]),
                            band=band,
                            rationale=entry["rationale"],
                        )
                    )
                return judgments
            except (KeyError, TypeError, ValueError) as exc:
                raise CritiqueFailure(reason="parse-error", stop_reason=stop_reason) from exc

    raise CritiqueFailure(reason="no-tool-use-block", stop_reason=stop_reason)


def run_critique(
    client: Any,
    *,
    candidates: Sequence[Candidate],
    hypotheses: Sequence[Any],
    survey: Sequence[Any],
    trigger_text: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> "list[Judgment]":
    """Drives ONE critique call through the shared client seam. Raises
    `CritiqueFailure` for anything that makes the response untrusted --
    never returns a partial judgment list.

    `hypotheses`, `survey` and `trigger_text` are REQUIRED keywords, never
    defaulted -- threaded straight into `assemble_critique_request`, which
    explains why each must fail loudly rather than default."""
    request = assemble_critique_request(
        candidates=candidates,
        hypotheses=hypotheses,
        survey=survey,
        trigger_text=trigger_text,
        model=model,
        max_tokens=max_tokens,
    )
    bound = reasoning_client.with_timeout(client, timeout_s)
    try:
        response = bound.messages.create(**request)
    except CritiqueFailure:
        raise
    except Exception as exc:  # noqa: BLE001 -- a timeout, a connection error, anything --
        raise CritiqueFailure(reason="exception", stop_reason=None) from exc
    # Two exemptions, applied in a fixed order and both narrow. Each clears
    # `is_generic` where the flag has no subject; neither one ever sets it.
    # Order is irrelevant to the result (their move sets are disjoint --
    # `SURVEY_COVERAGE_MOVES` cannot contain `MOVE_NO_QUESTION`), and is fixed
    # anyway so a rationale annotated by both is impossible to produce by
    # accident.
    return enforce_off_topic(
        exempt_declines_from_generic(
            exempt_survey_coverage_from_generic(
                _parse_critique_response(response), candidates
            ),
            candidates,
        ),
        candidates,
    )


# The flags that describe a question being WRONG, as opposed to `is_generic`,
# which describes it being unresponsive. Derived from the judgment's own
# fields rather than hand-listed twice, so a seventh flag joins it by default.
_VIOLATION_FLAGS = (
    "is_leading",
    "is_hypothetical_or_future_tense",
    "is_pitch",
    "is_opinion_seeking",
    "is_confirmation_only",
)


# --- The rationale annotation (2026-08-16) ---------------------------------------------
#
# **The contradiction this fixes.** Both exemptions below clear `is_generic`
# and neither touches `rationale`, because `rationale` is the MODEL'S prose.
# So a reviewer reading a rendered transcript sees a candidate with
# `is_generic=false` whose rationale reads "...making it generic in this
# context". Two artifacts of the same judgment, on the same row, disagreeing.
# The reviewer cannot tell from the row whether the critic changed its mind,
# whether the flag failed to parse, or whether something downstream rewrote it
# -- and the answer (a deterministic post-step overrode the flag on purpose)
# is nowhere on the page.
#
# **Options considered, and why this one.**
#
# 1. Rewrite the rationale to drop the word. Rejected outright. The rationale
#    is a record of what the model said; editing it to say something the model
#    did not say destroys the only evidence anyone has that the exemption was
#    ever needed, and would make the next person's "is this exemption still
#    earning its keep?" question unanswerable.
# 2. Blank the rationale on an exempted judgment. Rejected: strictly worse
#    than option 1 -- it destroys the same evidence AND leaves the reviewer
#    with nothing.
# 3. Ask the model not to write the word (a prompt fix). Rejected on the
#    precedent already recorded in `exempt_survey_coverage_from_generic`: the
#    instruction to not flag these is ALREADY in the prompt and the model
#    overrode it. A prompt clause is also, by construction, not a guarantee --
#    the reviewer would still meet the contradiction on the day it is not
#    honoured, which is the day it matters.
# 4. Annotate: leave the model's sentence byte-for-byte intact and append a
#    clearly-marked, machine-authored note saying what was overridden and why.
#    Chosen. The reviewer gets both halves and can see which is which, and the
#    cost is one bracketed clause on the minority of rows that were exempted.
#
# The notes are written in the third person and open with "automatic note" so
# they cannot be misread as the critic's own voice. They are appended, never
# prepended: the model's own sentence stays the first thing read.
#
# Renderers must not need to know about this. Nothing downstream parses a
# rationale -- `app/coach/render.py` and the hub page display it as text --
# so a longer string is the whole blast radius. If anything ever DOES parse
# rationales, the marker below is the stable prefix to split on.
_ANNOTATION_MARKER = " [automatic note, not the critic's words:"

_GENERIC_CLEARED_SURVEY_NOTE = (
    _ANNOTATION_MARKER
    + " is_generic was cleared for this candidate because its move covers the"
    " survey. A survey question is written before the interview and cannot"
    " engage with an utterance that had not happened yet. Any wording above"
    " calling it generic is the critic's own and is left exactly as written.]"
)

_GENERIC_CLEARED_DECLINE_NOTE = (
    _ANNOTATION_MARKER
    + " is_generic was cleared for this candidate because it is a no_question"
    " decline, which carries no question text -- there is nothing for the flag"
    " to be about. Any wording above calling it generic is the critic's own and"
    " is left exactly as written.]"
)


_OFF_TOPIC_CLEARED_NOTE = (
    _ANNOTATION_MARKER
    + " is_off_topic was cleared for this candidate because its move is not"
    " deepen. advance and fill_gap serve a survey item by definition and"
    " test_hypothesis names a hypothesis; each is on-topic by construction, so"
    " the flag has no subject. Any wording above calling it off-topic is the"
    " critic's own and is left exactly as written.]"
)

_OFF_TOPIC_ENFORCED_NOTE = (
    _ANNOTATION_MARKER
    + " the band was set to clear_violation because is_off_topic is true. A"
    " question that serves no survey item and no hypothesis does not earn an"
    " interruption however well it is worded, so this does not depend on the"
    " critic's own band. Any band wording above is the critic's own and is left"
    " exactly as written.]"
)


def _annotate(rationale: str, note: str) -> str:
    """Appends `note` to the model's own `rationale`, exactly once.

    Idempotent on the marker rather than on the whole note, so re-running an
    already-exempted judgment through either exemption cannot stack two
    bracketed clauses onto one line. An empty rationale (the model is required
    to supply one, but a whitespace-only string parses fine) still gets the
    note -- the note is the more useful half in that case, not the less."""
    if _ANNOTATION_MARKER in rationale:
        return rationale
    return f"{rationale.rstrip()}{note}" if rationale.strip() else note.lstrip()


def exempt_survey_coverage_from_generic(
    judgments: "list[Judgment]", candidates: Sequence[Candidate]
) -> "list[Judgment]":
    """`is_generic` is not a meaningful question about a survey-coverage move.

    **The measurement that forced this.** In the first live coached interview,
    13 of 29 suggestions were flagged `is_generic` and every other flag fired
    ZERO times -- so this one flag was doing 100% of the suppression. Reviewing
    the survey afterwards showed why: 8 of those 13 were `advance` or
    `fill_gap` moves, and five were the operator's own survey questions, word
    for word. "What is on your second screen while you play, and how did it
    end up there?" is survey item s3 verbatim.

    A survey question is written BEFORE the interview. It cannot engage with an
    utterance that had not happened yet, and it was never supposed to -- an
    `advance` exists precisely to leave the current subject. So the flag was
    punishing the coach for doing the job the move is for.

    **Why this is code and not a prompt fix.** `prompts._GENERIC_INSTRUCTION`
    ALREADY says it -- "an advance move is expected to introduce something the
    interviewee has not just been talking about" -- and `candidates_block`
    already puts `move=` in front of the model. The instruction is present, the
    move is visible, and the model overrode both. It also contains a competing
    clause ("the question restates the survey item verbatim..." -> TRUE), and
    the model resolved the tension the wrong way. Saying it a third time, more
    loudly, is not a different experiment.

    **What it does NOT do.** It does not promote anything to `clean`. A
    `clear_violation` whose ONLY basis was `is_generic` is demoted to
    `uncertain`, which is still shown -- so the operator sees the question and
    the critic's doubt survives. A candidate carrying any real violation flag
    is left exactly as the model banded it: this clears a category error, it
    does not overrule the critic.

    **It does not silently edit the critic either (2026-08-16).** The model's
    `rationale` is left byte-for-byte intact and an `_ANNOTATION_MARKER` note
    is APPENDED saying what was overridden and why -- see the comment above
    that marker for the four options weighed and why annotating beat
    rewriting. Without it a reviewer meets a row flagged not-generic whose
    rationale says "generic", with nothing on the page to say which artifact
    to believe.
    """
    move_by_index = {candidate.index: candidate.move for candidate in candidates}
    adjusted: "list[Judgment]" = []
    for judgment in judgments:
        move = move_by_index.get(judgment.index)
        if move not in generate.SURVEY_COVERAGE_MOVES or not judgment.is_generic:
            adjusted.append(judgment)
            continue
        others = any(getattr(judgment, flag) for flag in _VIOLATION_FLAGS)
        band = judgment.band
        if band == BAND_CLEAR_VIOLATION and not others:
            band = BAND_UNCERTAIN
        adjusted.append(
            replace(
                judgment,
                is_generic=False,
                band=band,
                rationale=_annotate(judgment.rationale, _GENERIC_CLEARED_SURVEY_NOTE),
            )
        )
    return adjusted


def exempt_declines_from_generic(
    judgments: "list[Judgment]", candidates: Sequence[Candidate]
) -> "list[Judgment]":
    """`is_generic` is not a meaningful question about a DECLINE either.

    A `no_question` candidate (`generate.MOVE_NO_QUESTION`) means "nothing
    should be asked right now" and carries an empty `text` by construction --
    `prompts._SYSTEM_PREAMBLE` asks for exactly that. `is_generic` asks whether
    a QUESTION engages with what this interviewee just said; asked of the empty
    string the answer is neither true nor false, because there is no question
    there to have the property. The critic returned `is_generic=true` for
    declines anyway, which is a number that looks like a measurement and is not.

    **Why this is code as well as a prompt clause.** `prompts._GENERIC_
    INSTRUCTION` now says it outright, and `prompts.candidates_block` has always
    rendered `move=` in front of the model -- but that is precisely the
    combination the exemption above already watched the model override, and its
    "saying it a third time, more loudly, is not a different experiment" applies
    here unchanged. The prompt clause makes the model likelier to be right; this
    makes the recorded flag correct either way.

    **What it deliberately does NOT do: how declines are counted and shown is
    untouched.** Only `is_generic` moves, and the band is left exactly as the
    model set it. That restraint is load-bearing rather than merely tidy.
    `pipeline.py` counts declines by walking `rank_candidates`' OUTPUT, which
    has already dropped every `clear_violation`; demoting a decline's band to
    `uncertain` the way the survey-coverage exemption does would push declines
    the critic condemned back into that list and inflate the
    `suggestion-declined` event's own count -- silently moving a measurement
    while claiming only to fix a flag. A decline is never shown regardless
    (`generate.SHOWABLE_MOVES` excludes it), so there is nothing to gain from
    touching the band and one real thing to break.

    Like the exemption above, the model's `rationale` is left intact and a
    marked note is appended rather than the prose being edited.
    """
    move_by_index = {candidate.index: candidate.move for candidate in candidates}
    adjusted: "list[Judgment]" = []
    for judgment in judgments:
        move = move_by_index.get(judgment.index)
        if move != generate.MOVE_NO_QUESTION or not judgment.is_generic:
            adjusted.append(judgment)
            continue
        adjusted.append(
            replace(
                judgment,
                is_generic=False,
                rationale=_annotate(judgment.rationale, _GENERIC_CLEARED_DECLINE_NOTE),
            )
        )
    return adjusted


def enforce_off_topic(
    judgments: "list[Judgment]", candidates: Sequence[Candidate]
) -> "list[Judgment]":
    """The seventh flag's two deterministic halves (2026-08-19). Unlike the two
    exemptions above, this one can TIGHTEN -- and both halves are needed.

    **First half: the flag has no subject except on `deepen`.** `advance` and
    `fill_gap` ARE the survey, and `test_hypothesis` names a hypothesis in
    `target_hypothesis_id`; each is on-topic by construction, and `no_question`
    carries no question at all. So `deepen` is the only move that can wander,
    which is exactly what it did -- it is the one move whose own definition
    points at the interviewee's last utterance and nothing else. Clearing the
    flag elsewhere mirrors `is_confirmation_only`'s "for a candidate with no
    target_hypothesis_id, this is always false", for the same reason: a flag
    recorded true where it cannot apply is a number that looks like a
    measurement and is not.

    **Second half: where it survives, the band goes to `clear_violation`.**
    `rank_candidates` reads the band alone (§G2's sort-key contract), so this
    is the ONLY way to make an off-topic candidate not appear without
    introducing a second ordering path -- the same reasoning `Judgment`'s own
    docstring gives for why `is_confirmation_only` drops through the band
    rather than directly. Making it deterministic rather than trusting the
    model's band is the point: the interviewer's complaint was that sharp,
    well-formed questions about irrelevant material kept appearing, and a
    well-formed question is exactly the case where a model reaching for a band
    is most likely to say `clean`.

    **What it deliberately does NOT do: it never RELAXES a band.** A judgment
    the critic already called `clear_violation` stays there whatever this flag
    says, so the step can only ever remove a suggestion from the panel, never
    add one. That asymmetry is what makes a wrong flag cost a missed suggestion
    instead of a shown violation -- the same cost asymmetry the rest of this
    module is built on.

    Like both exemptions, the model's `rationale` is left intact and a marked
    note is appended rather than the prose being edited.
    """
    move_by_index = {candidate.index: candidate.move for candidate in candidates}
    adjusted: "list[Judgment]" = []
    for judgment in judgments:
        if not judgment.is_off_topic:
            adjusted.append(judgment)
            continue
        if move_by_index.get(judgment.index) != generate.MOVE_DEEPEN:
            adjusted.append(
                replace(
                    judgment,
                    is_off_topic=False,
                    rationale=_annotate(judgment.rationale, _OFF_TOPIC_CLEARED_NOTE),
                )
            )
            continue
        adjusted.append(
            replace(
                judgment,
                band=BAND_CLEAR_VIOLATION,
                rationale=_annotate(judgment.rationale, _OFF_TOPIC_ENFORCED_NOTE),
            )
        )
    return adjusted


def rank_candidates(
    candidates: Sequence[Candidate],
    judgments: Sequence[Judgment],
    *,
    max_shown: int = 3,
) -> "list[Candidate]":
    """D-05, made code. Drops every `clear_violation` candidate entirely --
    it appears at no rank. Orders the survivors clean before uncertain, and
    within a band preserves the generate stage's OWN emission order (the
    order the interviewer directly experiences, used as the tie-break). At
    most `max_shown` survivors; a single survivor is a complete result, never
    a partial one.

    Pure: no I/O, no client, no randomness -- deterministic across repeated
    calls with identical inputs. A candidate with no matching judgment (the
    critic never classified it, which should not happen given
    `assemble_critique_request` embeds every candidate, but a defensive
    guard is cheap) is treated as unvetted and dropped, exactly like a
    `clear_violation` -- never shown without a verdict.

    This function must NEVER ask the model for an order, and must NEVER let
    an uncertain candidate be dropped or a clear violation be merely ranked
    down. Both are enforced here by construction, not by convention.
    """
    judgment_by_index = {judgment.index: judgment for judgment in judgments}

    # Iterate `candidates` in ITS OWN order first, so the stable sort below
    # preserves emission order within a band (Python's sort is stable).
    ranked: "list[tuple[int, Candidate]]" = []
    for candidate in candidates:
        judgment = judgment_by_index.get(candidate.index)
        if judgment is None or judgment.band == BAND_CLEAR_VIOLATION:
            continue
        ranked.append((_BAND_RANK.get(judgment.band, 1), candidate))

    ranked.sort(key=lambda pair: pair[0])
    return [candidate for _, candidate in ranked[:max_shown]]
