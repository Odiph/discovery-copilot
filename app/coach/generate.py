"""The generate stage: a finalized INTERVIEWEE segment becomes a set of
candidate next-questions, each one carrying exactly one move (04-CONTEXT.md
D-02, D-03, D-04; 04-02-PLAN.md §D3; 06-02-PLAN.md §G1).

**A small, closed decision space, not open-ended invention (§D3).** Every
candidate is `advance` (the next unasked survey question), `deepen` (dig into
what was just said), `fill_gap` (a survey item that was asked but whose
answer the transcript since then never actually obtained), `test_hypothesis`
(a question that would test a named, user-authored hypothesis, 06-02-PLAN.md
§G1), or `no_question` (2026-08-15). `ALL_MOVES` is the complete, pinned set
-- a new move cannot be introduced without a test asserting set equality
against the schema failing first, the same discipline that grew this set from
three to four and then to five.

**`no_question`, and why its absence was a defect rather than a missing
feature.** This stage could not decline. An empty candidate array is a parse
failure counted against coach health (see `_parse_generate_response` below),
so every cycle had to propose SOMETHING -- while the rubric it carries says
"talk less, listen more". A coach structurally incapable of silence will fill
the panel whether or not there is anything worth saying, and in the first
real coached interview it did exactly that.

The empty array stays a failure, deliberately. "The model returned nothing"
and "the model judged that nothing should be asked" are different events and
must not arrive through the same channel: the first is a broken response, the
second is a considered answer that should be recorded as such. So declining
is a normal candidate carrying `no_question`, which costs nothing downstream
beyond keeping it out of the shown set (`SHOWABLE_MOVES`), and leaves a
durable trace -- a cycle that declined must not look identical to a cycle
that never ran.

**Structured output via forced tool-use, not `.parse()`/JSON-mode prompting**
(D1, matching `app/coach/coverage.py`'s own precedent from 04-01, which found
this "a more universally stable JSON-forcing mechanism across SDK versions").
`_generate_tool_schema()` is the schema; it is a fixed module-level shape,
never built per-call from anything that varies, because the API caches a
compiled schema for about a day after first use.

**On any parse failure, refusal or truncation, the WHOLE cycle contributes
nothing (§D5, extended here to generate too).** `GenerateFailure` is raised
and never carries candidate or transcript text -- only a reason string and
the response's own `stop_reason` (or `None`). `app/coach/pipeline.py` is the
only catcher; it records a named event and moves on, exactly like
`app/coach/coverage.py`'s own degrade-and-continue except-block.

This module must NEVER import `anthropic` (T-04-12, enforced by gate 11 in
`app/tests/test_phase_gates.py`) and must NEVER log or place candidate text
into an event `detail` dict (T-04-27) -- full candidate text belongs ONLY in
the durable suggestion record `app/coach/pipeline.py` writes via
`SessionStore.append_suggestion`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from app.coach import prompts, reasoning_client
from app.coach.context import FACT_ID_PREFIX, ContextItem, SurveyItem
# 06-02-PLAN.md §G5: module-scope, deliberately -- this direction carries no
# cycle (corpus.py imports context.py and app.session.reader, never
# generate.py), unlike context.py's OWN function-local import of corpus.py
# (06-01-PLAN.md §F2). 06-01's function-local-import rule applies to
# context.py alone and must not be over-applied here.
from app.coach.corpus import ALL_HYPOTHESIS_STATES
from app.session.store import SegmentRecord

logger = logging.getLogger(__name__)

MOVE_ADVANCE = "advance"
MOVE_DEEPEN = "deepen"
MOVE_FILL_GAP = "fill_gap"
MOVE_TEST_HYPOTHESIS = "test_hypothesis"
# 2026-08-15: the move that lets a cycle DECLINE. Everything above answers
# "what should the interviewer ask next"; this one answers "nothing right
# now, let them keep listening". See the module docstring on why the absence
# of it was a structural defect rather than a missing feature.
MOVE_NO_QUESTION = "no_question"
# 06-02-PLAN.md §G1: deliberately, consciously grown here in the SAME commit
# that grows the schema/prompt prose to match -- the docstring above already
# predicted this moment, and `test_coach_generate.py`'s own set-equality
# assertion was the test that failed first, exactly as designed.
ALL_MOVES = frozenset({
    MOVE_ADVANCE, MOVE_DEEPEN, MOVE_FILL_GAP, MOVE_TEST_HYPOTHESIS, MOVE_NO_QUESTION,
})

# The moves that produce a question a human could be shown. `no_question` is
# the only one that does not, and several places downstream need to say so
# without hand-typing the exclusion: `app/coach/pipeline.py` must keep it out
# of the ranked set, out of the repetition window and out of `shown_indices`.
# Derived rather than listed, so a sixth move joins the shown set by default
# -- the safe direction, since a move nobody remembered to add here would
# otherwise silently never reach the interviewer.
SHOWABLE_MOVES = ALL_MOVES - {MOVE_NO_QUESTION}

# The moves whose JOB is to leave the current subject -- `advance` moves the
# survey on, `fill_gap` reaches for a survey item nothing has covered yet.
#
# Named here, beside the moves themselves, because `critic.py` needs it to
# decide where `is_generic` is a meaningful question and where it is a
# category error. A survey question was written BEFORE the interview and
# therefore cannot engage with an utterance that had not happened yet; asking
# whether it does is like asking whether a hammer is a good screwdriver. See
# `critic.exempt_survey_coverage_from_generic` for what is done about it and
# the measurement that forced it.
SURVEY_COVERAGE_MOVES = frozenset({MOVE_ADVANCE, MOVE_FILL_GAP})

GENERATE_TOOL_NAME = "propose_candidates"

# Stop reasons that make the whole response untrusted, never a partial salvage
# (§D5's rule, applied identically to generate). `tool_use` and `end_turn` are
# the normal-completion shapes; everything else here is a reason to show
# nothing this cycle.
_UNTRUSTED_STOP_REASONS = frozenset({"refusal", "max_tokens", "pause_turn", "model_context_window_exceeded"})


class GenerateFailure(Exception):
    """Raised for anything that makes the generate response untrusted:
    a timeout, a parse failure, a refusal, a truncation, or any other
    exception from the call itself. NEVER carries candidate or transcript
    text -- only `reason` (a short label) and `stop_reason` (the response's
    own value, or `None` when the failure happened before any response came
    back, e.g. a network timeout)."""

    def __init__(self, *, reason: str, stop_reason: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stop_reason = stop_reason


@dataclass(frozen=True)
class AskedAt:
    """The sequence number and time of the interviewer segment that asked a
    survey item (§D3/§D4) -- sourced from the `coverage-asked` event's own
    `matched_seq`/`matched_audio_end_s` detail, never re-derived."""

    seq: int
    audio_end_s: float


@dataclass(frozen=True)
class Candidate:
    """One proposed next-question. `index` is the candidate's position in
    the generate stage's OWN emission order -- the tie-break `rank_candidates`
    (app/coach/critic.py) uses within a band (D-05), and the identifier the
    critique response references each candidate by.

    `target_hypothesis_id` (06-02-PLAN.md §G3) is populated only for a
    `test_hypothesis` candidate -- the id of the hypothesis it targets, so
    the critique stage can fetch that hypothesis's own written `falsifier`
    text rather than being asked to infer one. A model-reported id matching
    no loaded hypothesis is dropped by `_parse_generate_response` before a
    `Candidate` is ever built here -- the candidate itself always survives
    with no target, never raised, never logged by value (§G5's drop-and-warn
    shape, applied here one field early).

    `grounding_ids` (06-02-PLAN.md §G5, GC-02): which retrieved-fact ids (the
    ones actually rendered with an id in the prompt -- `FACT_ID_PREFIX`-
    prefixed items only; a hand-written brief item renders no id and is
    deliberately outside this mechanism) the model says backed this
    candidate. Default `()` is an ABSENCE, not an assertion that nothing
    grounded it -- the same reasoning `corpus.py:112-119` gives for
    `Hypothesis.state`'s own default. An invented id is dropped and counted,
    never trusted, never raised.

    `proposed_state` (06-02-PLAN.md §G5, GC-05): the hypothesis state this
    candidate's question would propose, validated against
    `ALL_HYPOTHESIS_STATES`. A PROPOSAL only -- nothing on the live path ever
    writes it to `context/hypotheses.json` (§G7); the interviewer reads it
    and separately, explicitly accepts or rejects it after the call (D-01).
    An unrecognized value is dropped and counted, never trusted, never
    raised. Neither field is in the generate tool schema's `required` list:
    a candidate grounded in nothing and proposing nothing is a normal,
    complete candidate."""

    index: int
    text: str
    move: str
    target_hypothesis_id: Optional[str] = None
    grounding_ids: "tuple[str, ...]" = ()
    proposed_state: Optional[str] = None


def _generate_tool_schema() -> dict:
    """Fixed, module-level shape (D4) -- never built per-call from anything
    that varies, since the API caches a compiled schema for about a day."""
    return {
        "name": GENERATE_TOOL_NAME,
        "description": (
            "Propose one or more next-question candidates for the interviewer, "
            "each tagged with exactly one of the four allowed moves."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "move": {"type": "string", "enum": sorted(ALL_MOVES)},
                            # 06-02-PLAN.md §G3/§G5: all three below are
                            # deliberately NOT in `required` below -- a
                            # candidate targeting no hypothesis, grounded in
                            # nothing, and proposing nothing is a normal,
                            # complete candidate.
                            "target_hypothesis_id": {"type": "string"},
                            "grounding_ids": {"type": "array", "items": {"type": "string"}},
                            "proposed_state": {"type": "string"},
                        },
                        "required": ["text", "move"],
                    },
                },
            },
            "required": ["candidates"],
        },
    }


def assemble_generate_request(
    *,
    brief: Sequence[ContextItem],
    survey: Sequence[SurveyItem],
    asked: Mapping[str, AskedAt],
    transcript: Sequence[SegmentRecord],
    trigger: SegmentRecord,
    detected_language: str,
    hypotheses: Sequence[Any],
    recently_shown: Sequence[str],
    transcript_cache_chunk: int = prompts.TRANSCRIPT_CACHE_CHUNK,
    model: str,
    max_tokens: int,
) -> dict:
    """Builds the exact kwargs for `client.messages.create(...)`. Pure: no
    I/O, no client, fully unit-testable by inspecting the returned dict
    directly (test_coach_generate.py). Three cache breakpoints (D4), in
    order: the static block (rarely-changing rubric+brief+hypotheses), the
    survey block (changes on a coverage transition), the transcript block
    (changes every turn) -- the trigger is deliberately NOT cached, since it
    is unique per call.

    `hypotheses` is a REQUIRED keyword (06-02-PLAN.md §G1/§G3), never
    defaulted -- a call site that forgets it must fail loudly rather than
    silently generate `test_hypothesis` candidates with no falsifier text to
    ground them in. Folded into the FIRST cache breakpoint via
    `prompts.static_block`, never a fourth breakpoint (§G3, T-06-16): a
    hypothesis set is user-authored and fixed for the life of the session,
    exactly like the hand-written brief.

    `recently_shown` is a REQUIRED keyword for the same reason `hypotheses` is:
    defaulting it would make an un-updated call site silently produce the
    pre-fix behaviour -- and the pre-fix behaviour is a coach that proposed one
    question 67 times in a 26-minute interview (see `app/coach/repetition.py`).
    A silent regression to that is indistinguishable from the feature working.

    It goes in the `messages` user content, NOT a fourth cache breakpoint and
    NOT folded into an existing one: it changes on almost every cycle, so
    caching it would invalidate the breakpoint every call. The three system
    blocks keep their byte-stability contract untouched."""
    already_proposed = prompts.already_proposed_block(recently_shown)
    # D4's three breakpoints, with the third one moved so it can actually be
    # READ. Previously the transcript was one block with the breakpoint at its
    # end: the block grew every cycle, so the entry written by cycle N never
    # matched cycle N+1 and every call paid the 1.25x cache-WRITE premium on
    # the whole prompt while never once reading it back.
    #
    # Splitting the transcript at a quantized boundary fixes both halves of
    # that: the cached prefix now holds still for a run of cycles (so it is
    # read), and folding the stable transcript in behind the same breakpoint
    # carries the prefix well past this model's 4,096-token cache floor --
    # which `static_block` alone sits uncomfortably close to. The module
    # docstring in prompts.py records that the floor was never measured
    # against the live API; this layout stops that open question from being
    # load-bearing.
    cached_count = prompts.cached_transcript_count(
        len(transcript), chunk=transcript_cache_chunk
    )
    stable = prompts.transcript_stable_block(transcript, cached_count)
    tail = prompts.transcript_tail_block(transcript, cached_count)
    system: "list[dict]" = [
        {
            "type": "text",
            "text": prompts.static_block(brief=brief, hypotheses=hypotheses),
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": prompts.survey_block(survey, asked),
            "cache_control": {"type": "ephemeral"},
        },
    ]
    if stable:
        system.append(
            {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}}
        )
    # NEVER cached: it changes on every cycle by construction, so a breakpoint
    # here would reintroduce exactly the defect this split removes.
    system.append({"type": "text", "text": tail})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": (
                    prompts.trigger_block(trigger)
                    # Omitted entirely when there is nothing to list, rather
                    # than sent as an empty heading -- see already_proposed_block.
                    + (("\n\n" + already_proposed) if already_proposed else "")
                    + "\n\n"
                    + prompts.language_instruction(detected_language)
                    + "\n\n"
                    + prompts.decision_instruction(tool_name=GENERATE_TOOL_NAME)
                ),
            }
        ],
        "tools": [_generate_tool_schema()],
        "tool_choice": {"type": "tool", "name": GENERATE_TOOL_NAME},
    }


def _parse_generate_response(
    response: Any,
    *,
    known_hypothesis_ids: "frozenset[str]",
    known_grounding_ids: "frozenset[str]",
) -> "list[Candidate]":
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason in _UNTRUSTED_STOP_REASONS:
        raise GenerateFailure(reason="untrusted-stop-reason", stop_reason=stop_reason)

    for block in getattr(response, "content", None) or ():
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == GENERATE_TOOL_NAME:
            try:
                raw_candidates = block.input["candidates"]
                if not isinstance(raw_candidates, list) or not raw_candidates:
                    raise ValueError("candidates must be a non-empty array")
                candidates: "list[Candidate]" = []
                # 06-02-PLAN.md §G5's drop-and-warn shape, applied to all
                # three self-reported fields: counts only, never the invented
                # values themselves -- gate 12 fails the whole suite for
                # transcript-derived text in a log line, and a user-authored
                # hypothesis id is free-form text this project does not
                # control.
                dropped_target_count = 0
                dropped_grounding_count = 0
                dropped_state_count = 0
                for index, entry in enumerate(raw_candidates):
                    move = entry["move"]
                    if move not in ALL_MOVES:
                        raise ValueError(f"unrecognized move {move!r}")
                    # A non-str `text` from a schema-violating response would
                    # otherwise flow into the critic prompt, the wire payload
                    # and `append_suggestion` -- whose debug log evaluates
                    # `len(text)` EAGERLY (arguments are computed before the
                    # call, whatever the log level), raising TypeError deep
                    # inside the worker thread. Reject it here, where it is a
                    # named parse failure the health path already handles.
                    if not isinstance(entry["text"], str):
                        raise ValueError("candidate text must be a string")
                    target_hypothesis_id = entry.get("target_hypothesis_id")
                    if target_hypothesis_id is not None:
                        if not isinstance(target_hypothesis_id, str):
                            raise ValueError("target_hypothesis_id must be a string")
                        if target_hypothesis_id not in known_hypothesis_ids:
                            dropped_target_count += 1
                            target_hypothesis_id = None

                    # 06-02-PLAN.md §G5 (GC-02): follows extract.py:324-330's
                    # shape verbatim -- keep what matches, drop what does not,
                    # never raise on an invented id. A hallucinated citation
                    # must never cost the interviewer a suggestion mid-call.
                    raw_grounding_ids = entry.get("grounding_ids") or []
                    if not isinstance(raw_grounding_ids, list):
                        raise ValueError("grounding_ids must be an array")
                    grounding_ids: "list[str]" = []
                    for grounding_id in raw_grounding_ids:
                        if not isinstance(grounding_id, str):
                            raise ValueError("grounding_ids entries must be strings")
                        if grounding_id in known_grounding_ids:
                            grounding_ids.append(grounding_id)
                        else:
                            dropped_grounding_count += 1

                    # 06-02-PLAN.md §G5 (GC-05): validated against the pinned
                    # hypothesis state set -- an invented value is dropped,
                    # never trusted, never raised. This is a PROPOSAL only;
                    # nothing on the live path ever applies it (§G7, D-01).
                    proposed_state = entry.get("proposed_state")
                    if proposed_state is not None:
                        if not isinstance(proposed_state, str):
                            raise ValueError("proposed_state must be a string")
                        if proposed_state not in ALL_HYPOTHESIS_STATES:
                            dropped_state_count += 1
                            proposed_state = None

                    candidates.append(
                        Candidate(
                            index=index,
                            text=entry["text"],
                            move=move,
                            target_hypothesis_id=target_hypothesis_id,
                            grounding_ids=tuple(grounding_ids),
                            proposed_state=proposed_state,
                        )
                    )
                if dropped_target_count:
                    logger.warning(
                        "generate response referenced %d target_hypothesis_id(s) matching no "
                        "loaded hypothesis -- dropped, candidate(s) kept with no target",
                        dropped_target_count,
                    )
                if dropped_grounding_count or dropped_state_count:
                    logger.warning(
                        "generate response self-report drops -- grounding_ids: %d, proposed_state: %d "
                        "(counts only; a user-authored id/value is never logged by its text)",
                        dropped_grounding_count,
                        dropped_state_count,
                    )
                return candidates
            except (KeyError, TypeError, ValueError) as exc:
                raise GenerateFailure(reason="parse-error", stop_reason=stop_reason) from exc

    raise GenerateFailure(reason="no-tool-use-block", stop_reason=stop_reason)


def run_generate(
    client: Any,
    *,
    brief: Sequence[ContextItem],
    survey: Sequence[SurveyItem],
    asked: Mapping[str, AskedAt],
    transcript: Sequence[SegmentRecord],
    trigger: SegmentRecord,
    detected_language: str,
    hypotheses: Sequence[Any],
    recently_shown: Sequence[str],
    transcript_cache_chunk: int = prompts.TRANSCRIPT_CACHE_CHUNK,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> "list[Candidate]":
    """Drives ONE generate call through the shared client seam (D1/D2). Takes
    `client` by injection (real or `app.coach.stub.StubAnthropicClient`).
    Raises `GenerateFailure` for anything that makes the cycle contribute
    nothing -- never returns a partial or best-effort candidate list.

    `hypotheses` (06-02-PLAN.md §G1) is a REQUIRED keyword, never defaulted,
    threaded into `assemble_generate_request` and into the known-id set
    `_parse_generate_response` validates a self-reported
    `target_hypothesis_id` against. `brief` (already required) is the same
    source `_parse_generate_response` validates a self-reported
    `grounding_ids` against -- restricted to the `FACT_ID_PREFIX`-prefixed
    items only, since those are the only ones rendered with an id in the
    prompt (§G5)."""
    request = assemble_generate_request(
        brief=brief,
        survey=survey,
        asked=asked,
        transcript=transcript,
        trigger=trigger,
        detected_language=detected_language,
        hypotheses=hypotheses,
        recently_shown=recently_shown,
        transcript_cache_chunk=transcript_cache_chunk,
        model=model,
        max_tokens=max_tokens,
    )
    bound = reasoning_client.with_timeout(client, timeout_s)
    try:
        response = bound.messages.create(**request)
    except GenerateFailure:
        raise
    except Exception as exc:  # noqa: BLE001 -- a timeout, a connection error, anything --
        # the whole cycle contributes nothing (§D5); never carries candidate text.
        raise GenerateFailure(reason="exception", stop_reason=None) from exc
    known_hypothesis_ids = frozenset(h.id for h in hypotheses)
    known_grounding_ids = frozenset(item.id for item in brief if item.id.startswith(FACT_ID_PREFIX))
    return _parse_generate_response(
        response, known_hypothesis_ids=known_hypothesis_ids, known_grounding_ids=known_grounding_ids
    )
