"""The knowledge-base schema (05-CONTEXT.md D-01/D-04/D-05/D-06/D-07/D-08;
05-01-PLAN.md §K2/§K3/§K7). `Fact` and `Hypothesis`, `CorpusValidationError`,
and the two loaders -- the same shape `app/coach/context.py` already
establishes for the brief/survey, extended to a second pair of artifacts.

**STATE.md § FIRST REAL-SPEECH EVIDENCE, FINDING 1 (two real bilingual calls,
2026-08-09): a fact's citation is a SPAN of one or more segments, not a
single `seq`.** Real speech measured a 5-word median segment with a ~20-word
ceiling -- against that, a single-segment citation cannot carry a claim.
`Fact.seqs` replaces the original `Fact.seq` (05-01-PLAN.md §K2's own
single-segment design) with exactly this fix, while preserving §K2's original
concern in full: a quote verified against a careless concatenation can match
across a join that never existed in the interview. `validate_citation_span`
(below) is the mechanism that keeps that true for a span instead of
discarding the requirement -- see its own docstring.

**This module never writes.** `app/coach/corpus_store.py` is the one writer
of `context/facts.json`; the (05-02/05-03) authoring path is the one writer
of `context/hypotheses.json`. Never guesses a default `provenance` -- the
same reasoning `context.py`'s own docstring already states applies here
unchanged (D-07/CTX-02): a validator that coerced a missing or malformed
value would defeat the requirement more thoroughly than having no validator
at all. Never guesses a default `falsifier` either, on the identical rule
(D-04). Never stores a time of any kind on a `Fact` -- see the field-count
note on the model itself. Never the second place either artifact is parsed.

`provenance`'s allowed values are IMPORTED from `app.coach.context`, never
copied or re-derived -- `app/web/author.py:45-47` set this precedent
explicitly ("imported, never redefined, so the two halves can never drift");
a second literal pair of values here is exactly the drift D-07 exists to
prevent.

Validated with `pydantic`, arriving as `anthropic`'s own transitive
dependency, exactly as `context.py`'s own docstring already justifies (not a
separately pinned package).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app.coach.context import PROVENANCE_VALUES
from app.session.reader import read_session

FACTS_FILENAME = "facts.json"
HYPOTHESES_FILENAME = "hypotheses.json"

HYPOTHESIS_STATE_UNTESTED = "untested"
HYPOTHESIS_STATE_SUPPORTED = "supported"
HYPOTHESIS_STATE_CONTRADICTED = "contradicted"
# The fourth state, added deliberately and with the equality test below
# updated in the same commit -- which is the friction the pinned frozenset
# exists to create.
#
# WHY A FOURTH STATE. Without it, a hypothesis with real evidence on BOTH
# sides has nowhere to sit. It either gets forced to a winner, or it stays
# `untested` -- where it is indistinguishable from a hypothesis nobody has
# looked at yet. Those are opposite situations: one needs an interview, the
# other needs a sharper question. Competitive research found this is a named
# failure across the category (experiment platforms that render "inconclusive"
# and "well-powered evidence of no effect" identically), and that Strategyzer
# is close to alone in giving inconclusive a first-class resting place.
#
# `contested` is a legal RESTING state, not a way station. Nothing may
# auto-resolve it to a winner; resolving it is a human deciding the question
# was wrong, or gathering evidence that settles it.
HYPOTHESIS_STATE_CONTESTED = "contested"
# A pinned module-level frozenset, in `app/coach/generate.py:45-48`'s own
# shape, for the same reason `generate.py:8-10` gives: a fifth state cannot
# be introduced without an equality test failing first.
ALL_HYPOTHESIS_STATES = frozenset(
    {
        HYPOTHESIS_STATE_UNTESTED,
        HYPOTHESIS_STATE_SUPPORTED,
        HYPOTHESIS_STATE_CONTRADICTED,
        HYPOTHESIS_STATE_CONTESTED,
    }
)


class CorpusValidationError(Exception):
    """Raised for a malformed `context/facts.json` or `context/hypotheses.json`
    -- a missing file, invalid JSON, a missing/misspelled/empty `provenance`,
    an absent or whitespace-only `falsifier`, an invalid citation span (empty,
    over the segment cap, unresolvable, mixed-speaker, or non-contiguous), or
    a duplicate id. The message always names the offending item's id (or the
    file, when the error is not item-scoped)."""


# STATE.md § FIRST REAL-SPEECH EVIDENCE, FINDING 1: real median segment length
# is ~5 words with a ~20-word ceiling (two real bilingual calls, 2026-08-09).
# Six segments is therefore roughly 30 words -- comfortably enough to carry a
# full claim, while still stopping a model from citing half an interview and
# finding its quote somewhere inside the concatenation (05-01-PLAN.md §K2's
# original join-risk concern, now bounded by size as well as by contiguity).
MAX_SPAN_SEGMENTS = 6


class SpanValidityError(Exception):
    """Raised by `validate_citation_span` for exactly one of the three
    session-dependent span-validity conditions -- never for the two
    structural conditions (non-empty, at-most-`MAX_SPAN_SEGMENTS`), which
    `Fact`'s own field validator enforces without needing any session data.
    `code` is a short, stable, machine-readable label so a caller (this
    module's `load_facts`, or `app.coach.extract`'s per-fact rejection
    pipeline) can report distinctly WHICH condition failed, mirroring
    `app.coach.extract`'s own `Rejection.reason` convention."""

    SPAN_UNRESOLVED = "unresolved-citation"
    SPAN_MIXED_SPEAKER = "mixed-speaker-span"
    SPAN_NONCONTIGUOUS = "noncontiguous-span"

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class Fact(BaseModel):
    """One extracted, cited claim (D-06/D-07). Carries EXACTLY SEVEN fields:
    an id, the session id, the cited `seqs` (a span of one or more segment
    numbers, STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1), the claim, the
    provenance, the quoted text, and the related hypothesis ids. There is no
    eighth, and in particular NO EXTRACTION TIMESTAMP -- this is the single
    most obvious field a later contributor will want to add back, so the
    reason it is deliberately absent is written down here rather than merely
    in the plan that specified it (05-01-PLAN.md §K2):

    - An extraction timestamp is provenance-of-PROCESS (when the tool ran),
      not provenance-of-CLAIM (what the customer said and when the interview
      that captured it happened -- both already carried by `session_id`).
    - The moment a fact record has somewhere to put a time, a later edit that
      populates it from a segment rather than from the clock becomes a
      plausible mistake nothing in the schema prevents. `audio_start_s` is
      FABRICATED (`app/stt/live.py:194-196` sets it to the end of the same
      speaker's PREVIOUS segment) -- keeping this record empty of time is
      what keeps that fabrication from ever having anywhere to leak into.
    - A per-fact clock value interacts badly with fact identity (below): the
      same claim re-extracted a day later would differ in a field that says
      nothing about the claim itself.

    Fact identity: `id` is a fresh `uuid4` assigned when the record is built.
    D-08 (re-runnable without duplicating) is satisfied entirely by
    05-02-PLAN.md's slice-replacement rule -- by SESSION, never by matching
    fact identities across runs. A re-run therefore produces NEW ids for that
    session's facts by design. Nothing compares fact ids or claim text across
    extraction runs, and nothing may start to -- no dedup key, no content
    hash. A fact id is a handle into the CURRENT corpus and is explicitly NOT
    stable across a re-extraction, unlike `SurveyItem.id`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    session_id: str
    seqs: "tuple[int, ...]"
    claim: str
    provenance: str  # "evidence" | "inference" -- no default, ever (D-07)
    quoted_text: str
    bearings: "tuple[Bearing, ...]" = ()
    """Which hypotheses this claim bears on, and in which direction. Still the
    SEVENTH field -- it replaces `related_hypothesis_ids` rather than joining
    it, so §K2's "exactly seven, and no eighth" is untouched."""

    @property
    def related_hypothesis_ids(self) -> "tuple[str, ...]":
        """Every hypothesis this claim bears on, direction discarded.

        A derived property, never a stored field, so there is ONE source of
        truth. Retrieval scoring in `app/coach/context.py` asks only "is this
        fact linked to a live hypothesis" and genuinely does not care which
        way -- a fact that DEMOLISHES an open hypothesis is exactly as worth
        retrieving as one that supports it. Storing both would let the two
        drift, which is the mistake the alias table's own docstring warns
        about."""
        return tuple(bearing.hypothesis_id for bearing in self.bearings)

    @field_validator("provenance")
    @classmethod
    def _provenance_must_be_recognized(cls, value: str) -> str:
        if value not in PROVENANCE_VALUES:
            raise ValueError(
                f"provenance must be one of {PROVENANCE_VALUES!r}, got {value!r} -- "
                "never silently defaulted to 'evidence' (D-07)"
            )
        return value

    @field_validator("seqs")
    @classmethod
    def _seqs_must_be_a_structurally_valid_span(cls, value: "tuple[int, ...]") -> "tuple[int, ...]":
        """The two STRUCTURAL span conditions -- non-empty, and at most
        `MAX_SPAN_SEGMENTS` -- checked here because neither needs any session
        data, unlike the three SESSION-DEPENDENT conditions
        `validate_citation_span` checks (resolution, same-speaker,
        contiguity). A fact with an empty span cites nothing at all; never
        silently coerced to some default segment (D-07's own "never guess"
        rule, applied to the citation itself, not just to `provenance`)."""
        if not value:
            raise ValueError("seqs must be non-empty -- a fact must cite at least one segment")
        if len(value) > MAX_SPAN_SEGMENTS:
            raise ValueError(
                f"seqs cites {len(value)} segments, more than the {MAX_SPAN_SEGMENTS}-segment "
                "cap (STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1) -- a span this wide risks "
                "finding its quote somewhere inside the concatenation rather than actually "
                "supporting the claim"
            )
        return value


STANCE_SUPPORTS = "supports"
STANCE_UNDERMINES = "undermines"
STANCE_UNCLEAR = "unclear"
# Pinned by equality like every other closed set in this module. `unclear` is
# not padding: forcing a direction on a claim that genuinely bears on a
# hypothesis without settling it MANUFACTURES evidence, and the manufactured
# direction will almost always be `supports` -- the model is completing a
# sentence about a hypothesis someone wrote down because they believed it.
ALL_STANCES = frozenset({STANCE_SUPPORTS, STANCE_UNDERMINES, STANCE_UNCLEAR})


class Bearing(BaseModel):
    """How one extracted claim bears on one hypothesis -- and in WHICH
    DIRECTION.

    This type exists because the field it replaces did not carry a direction.
    `related_hypothesis_ids` recorded that a claim "bears on" a hypothesis and
    nothing else, so every count built from it could only go UP: a claim that
    demolished a hypothesis and a claim that confirmed it were the same
    integer. Competitive research found that exact shape shipped in four
    discovery tools (airfocus, ProductPlan, ProdPad, Vistaly), where
    disconfirming evidence is literally unrepresentable in the data model.
    A monotonic evidence count is a confirmation machine.

    Changed rather than added, and the window for that was narrow: no
    facts.json existed anywhere yet, so there was no corpus to migrate. Once
    a real extraction has run, this becomes a migration instead of a fix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hypothesis_id: str
    stance: str

    @field_validator("stance")
    @classmethod
    def _stance_must_be_recognized(cls, value: str) -> str:
        if value not in ALL_STANCES:
            raise ValueError(f"stance must be one of {sorted(ALL_STANCES)!r}, got {value!r}")
        return value


class Hypothesis(BaseModel):
    """A user-authored hypothesis (D-03/D-04/D-05): a statement, a REQUIRED
    falsification condition, and a validation state. `state` MAY default to
    untested, and this asymmetry against `falsifier`'s no-default rule is
    deliberate: a brand-new hypothesis IS untested by definition -- that is
    not a guess about the world -- whereas a defaulted falsifier would assert
    something a human never confirmed, exactly as a defaulted provenance
    would."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    statement: str
    falsifier: str  # no default, ever -- a hypothesis without one is INVALID, not incomplete (D-04)
    state: str = HYPOTHESIS_STATE_UNTESTED
    criterion: str = ""
    """The PRE-REGISTERED decision rule: "we are right if 3 of 10 unprompted
    mention X". Written before the interviewing starts and not edited to fit
    what arrived -- otherwise you shoot the arrow and paint the target around
    it, which is the failure Strategyzer's Test Card ("We are right if...")
    and every pre-registration practice exists to prevent.

    Competitive research across ~27 discovery, experimentation, repository and
    argumentation systems found NO shipped product with this field. The
    experimentation platforms have its equivalent -- GrowthBook's target MDE
    and decision criteria, chosen at creation -- and the discovery tools do
    not.

    Defaults to EMPTY rather than being required, and that is a compatibility
    decision, not an endorsement: hypotheses authored before this field
    existed are still valid and must keep loading. `is_scorable` below is
    where the absence bites instead."""

    criterion_registered_after: "tuple[str, ...]" = ()
    """The session ids that ALREADY EXISTED at this node when the criterion
    text above was written -- the evidence for the ONE claim `criterion` makes
    and cannot back up on its own.

    A criterion says "we are right if 3 of 10 unprompted mention X". The worth
    of that sentence is entirely in WHEN it was written: a rule fixed before
    the interviewing is a pre-registration, and the identical sentence typed
    afterwards is a description of what already arrived. Nothing about the
    text distinguishes the two, so without this field the pre-registration is
    a claim the record cannot check.

    **Why session ids and NOT a timestamp.** `Fact` carries no time value at
    all, and its own field-count note gives three reasons; two of them do not
    transfer here (for a fact, when the tool ran is process noise -- for a
    criterion, "this rule pre-dates the evidence" is the entire claim). The
    third transfers exactly, and lands harder: the moment this record has
    somewhere to put a time, a later edit that fills it from the wrong source
    becomes a plausible mistake nothing in the schema prevents -- and here the
    wrong source would FORGE the only thing this field exists to prove. A
    tuple of session ids cannot be populated from a clock. They are names of
    directories under `sessions/` that either exist or do not, so a wrong
    value is a value naming an interview that never happened.

    **An empty tuple is a CLAIM, not an absence** -- and the strongest one
    available: written before any interview existed at this node, so it
    pre-dates every interview, including ones not yet run. It is also the
    default, for the identical compatibility reason `criterion`'s own default
    is empty: every hypotheses.json already on disk lacks this field, and
    requiring it would render all of them malformed, which in this codebase
    RAISES. That collision between "earned empty" and "defaulted empty" is
    deliberate and is exactly why `pre_registered_for` below refuses to answer
    for a hypothesis with no criterion -- see its own docstring.

    **The boundary MOVES when the criterion text is edited**, and moves for
    nothing else. `app/web/author.py` is where that rule lives and is argued;
    the short version is that an edited criterion is a different rule written
    later, and freezing the boundary at the original write would hand a
    rewritten-to-fit rule the credential the original one earned."""

    @field_validator("falsifier")
    @classmethod
    def _falsifier_must_be_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "falsifier must be a non-empty, non-whitespace string -- a hypothesis "
                "without a falsification condition is invalid, not incomplete (D-04)"
            )
        return value

    @field_validator("state")
    @classmethod
    def _state_must_be_recognized(cls, value: str) -> str:
        if value not in ALL_HYPOTHESIS_STATES:
            raise ValueError(f"state must be one of {sorted(ALL_HYPOTHESIS_STATES)!r}, got {value!r}")
        return value

    @property
    def is_scorable(self) -> bool:
        """Whether this hypothesis has a rule to be judged against.

        A PROPERTY, not a validator, and the distinction is the whole point.
        Making an absent criterion invalid would turn every hypothesis
        authored before the field existed into a malformed file -- and in
        this codebase malformed RAISES, which would take the coach down over
        a field added later. So a criterion-less hypothesis stays perfectly
        valid and merely cannot be honestly SCORED, which is a fact about the
        judgement rather than about the file."""
        return bool(self.criterion.strip())

    def pre_registered_for(self, session_id: str) -> bool:
        """Whether this hypothesis's criterion was fixed BEFORE the named
        interview happened -- the question `criterion_registered_after` exists
        so a consumer can ASK rather than assume.

        A method on `is_scorable`'s own stated model (a derived answer, never a
        validator): a `False` here is a fact about the JUDGEMENT, never about
        the file. A boundary naming a session since deleted is still a valid
        record of what was there when the rule was written, and must not raise.

        Answered by SET MEMBERSHIP, and an ordering comparison is NOT an
        equivalent shortcut. Session ids sort chronologically today
        (`%Y%m%dT%H%M%S`), so `session_id > max(boundary)` agrees with
        membership for every id the boundary happens to contain and looks
        correct in a test written only from those. It diverges precisely where
        it matters: a session id that is NOT in the boundary and sorts BELOW
        one that is -- an interview at a sibling node (cross-node reads are
        legal within one hierarchy, `app/projects.py::share_a_root`), or one
        filed out of order -- would be reported as arriving AFTER a rule that
        never accounted for it. The boundary is the RECORD of what existed, not
        a rule for extrapolating past it, and only membership reads it that way.

        Requires a criterion, and that clause is the whole reason this is a
        method rather than one `in` at the call site. A hypothesis with no rule
        has an empty boundary too, so membership ALONE would report every
        hypothesis nobody wrote a criterion for as pre-registered for every
        interview ever run -- the exact unearned claim this field was added to
        make checkable. No rule cannot pre-date anything."""
        return self.is_scorable and session_id not in self.criterion_registered_after


def _item_label(entry: object, index: int) -> str:
    if isinstance(entry, dict) and "id" in entry:
        return repr(entry["id"])
    return f"<item at index {index}>"


def _real_segment_speakers(sessions_root: "Path | str", session_id: str) -> "dict[int, str]":
    """Every real segment `seq` for one session, mapped to its own `speaker`
    -- read the ONLY way any consumer in this codebase may
    (`app.session.reader.read_session`), never a second, direct read of
    `transcript.jsonl` (§K7). `validate_citation_span` needs the speaker, not
    merely the set of valid seqs, to check the two speaker-aware conditions
    below."""
    session_dir = Path(sessions_root) / session_id
    result = read_session(session_dir)
    return {record["seq"]: record["speaker"] for record in result.records_of_kind("segment")}


def validate_citation_span(seqs: "Sequence[int]", *, speaker_by_seq: "Mapping[int, str]") -> None:
    """The three SESSION-DEPENDENT span-validity conditions (STATE.md § FIRST
    REAL-SPEECH EVIDENCE FINDING 1), shared verbatim between this module's
    own `load_facts` and `app.coach.extract`'s per-fact rejection pipeline --
    a second, drifted copy of this logic is exactly the risk the rest of this
    codebase's "imported, never redefined" precedent
    (`app/web/author.py:45-47`, `corpus.py:35`) exists to prevent. Pure: no
    I/O, no session read -- `speaker_by_seq` is the CALLER's own
    already-resolved `seq -> speaker` map for one session (built via
    `_real_segment_speakers` here, or from the caller's own in-memory
    transcript in `extract.py`), so this function never re-reads anything.

    Raises `SpanValidityError`, naming exactly ONE of:

    1. `SPAN_UNRESOLVED` -- a cited seq does not resolve to a real segment in
       that session (D-06's original single-segment rule, unchanged in
       spirit for a span: EVERY cited seq must resolve, not just one).
    2. `SPAN_MIXED_SPEAKER` -- the cited seqs do not all belong to the SAME
       speaker.
    3. `SPAN_NONCONTIGUOUS` -- the cited seqs, once confirmed same-speaker,
       are not a CONSECUTIVE RUN within THAT SPEAKER'S OWN ordered sequence
       of segments in the session. Other speakers' segments interleaving
       between the cited ones is fine and expected -- that is just the other
       person talking, and `speaker_by_seq` positions are computed only over
       the CITED speaker's own segments for exactly this reason. What is
       forbidden is skipping one of the SAME speaker's own segments, because
       that would join text across a gap that was never spoken contiguously
       -- 05-01-PLAN.md §K2's original single-segment design existed to
       prevent precisely this ("a quote verified against a concatenation can
       match across a join that never existed in the interview"); contiguity
       WITHIN one speaker's own turn is the mechanism that keeps that true
       for a multi-segment span instead of discarding the requirement. A
       naive check of the cited seqs' own raw numeric spread (`max(seqs) -
       min(seqs) + 1 == len(seqs)`) is NOT this condition and must never be
       substituted for it: it cannot tell "another speaker's turn sits
       between these two" (valid) apart from "this speaker's OWN segment
       sits between these two, uncited" (a real join -- invalid) without
       ever asking who actually occupies the gap, which is exactly what
       resolving positions within the cited speaker's OWN ordered list does
       and a raw-seq arithmetic shortcut does not.

    The two STRUCTURAL conditions (non-empty, at-most-`MAX_SPAN_SEGMENTS`)
    are NOT checked here -- `Fact`'s own field validator enforces both
    without needing any session data at all."""
    for seq in seqs:
        if seq not in speaker_by_seq:
            raise SpanValidityError(
                SpanValidityError.SPAN_UNRESOLVED,
                f"cites seq={seq}, which does not resolve to a real segment in that "
                "session's own transcript (D-06)",
            )

    speakers = {speaker_by_seq[seq] for seq in seqs}
    if len(speakers) > 1:
        raise SpanValidityError(
            SpanValidityError.SPAN_MIXED_SPEAKER,
            f"cites seqs {sorted(seqs)!r} across more than one speaker ({sorted(speakers)!r}) "
            "-- a span may only cite ONE speaker's own segments",
        )
    speaker = next(iter(speakers))

    speaker_seqs_sorted = sorted(s for s, sp in speaker_by_seq.items() if sp == speaker)
    index_of_seq = {s: position for position, s in enumerate(speaker_seqs_sorted)}
    positions = sorted(index_of_seq[seq] for seq in seqs)
    span_width = positions[-1] - positions[0] + 1
    if span_width != len(set(seqs)):
        raise SpanValidityError(
            SpanValidityError.SPAN_NONCONTIGUOUS,
            f"cites seqs {sorted(seqs)!r}, which skip one or more of speaker {speaker!r}'s "
            "OWN segments in between -- other speakers' segments interleaving is fine, but a "
            "gap in this speaker's own sequence would join text across a break that was "
            "never spoken contiguously",
        )


def load_facts(context_dir: "Path | str", *, sessions_root: "Optional[Path | str]") -> "tuple[Fact, ...]":
    """Reads `context/facts.json`. `sessions_root` is a REQUIRED keyword
    argument that is explicitly nullable, so a caller must consciously decide
    whether the D-06 span-validity check runs -- a default that skipped it
    would be the same class of mistake as a defaulting provenance. When
    given, every fact's citation span is validated via
    `validate_citation_span` against that session's own real segments (via
    `read_session`), and a failure raises naming the fact's id and the
    specific span-validity condition that failed. Raises
    `CorpusValidationError` naming the offending item if the file is missing,
    unreadable, not a JSON array, any item is malformed, or an id is
    duplicated -- the same four-way ladder `context.py`'s own loaders use."""
    path = Path(context_dir) / FACTS_FILENAME
    if not path.is_file():
        raise CorpusValidationError(f"{path} does not exist -- no facts have been extracted yet.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CorpusValidationError(f"{path} must be a JSON array of facts.")

    items: "list[Fact]" = []
    seen_ids: "set[str]" = set()
    speakers_by_session: "dict[str, dict[int, str]]" = {}
    for index, entry in enumerate(raw):
        try:
            fact = Fact.model_validate(entry)
        except ValidationError as exc:
            raise CorpusValidationError(
                f"{path} item {_item_label(entry, index)} is invalid: {exc}"
            ) from exc
        if fact.id in seen_ids:
            raise CorpusValidationError(f"{path} item {fact.id!r} is a duplicate id.")
        seen_ids.add(fact.id)
        if sessions_root is not None:
            if fact.session_id not in speakers_by_session:
                speakers_by_session[fact.session_id] = _real_segment_speakers(sessions_root, fact.session_id)
            try:
                validate_citation_span(fact.seqs, speaker_by_seq=speakers_by_session[fact.session_id])
            except SpanValidityError as exc:
                raise CorpusValidationError(f"{path} item {fact.id!r} {exc}") from exc
        items.append(fact)
    return tuple(items)


def load_hypotheses(context_dir: "Path | str", *, missing_ok: bool) -> "tuple[Hypothesis, ...]":
    """Reads `context/hypotheses.json`. `missing_ok` is a REQUIRED keyword
    argument for the same reason `load_facts`'s `sessions_root` is: an absent
    file is legitimately empty for extraction (D-01 makes an empty corpus the
    normal first case) and is a hard error for anything that expected the
    user to have authored one. Raises `CorpusValidationError` naming the
    offending item if the file is unreadable, not a JSON array, any item is
    malformed, or an id is duplicated."""
    path = Path(context_dir) / HYPOTHESES_FILENAME
    if not path.is_file():
        if missing_ok:
            return ()
        raise CorpusValidationError(f"{path} does not exist -- author a hypothesis before this.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CorpusValidationError(f"{path} must be a JSON array of hypotheses.")

    items: "list[Hypothesis]" = []
    seen_ids: "set[str]" = set()
    for index, entry in enumerate(raw):
        try:
            hypothesis = Hypothesis.model_validate(entry)
        except ValidationError as exc:
            raise CorpusValidationError(
                f"{path} item {_item_label(entry, index)} is invalid: {exc}"
            ) from exc
        if hypothesis.id in seen_ids:
            raise CorpusValidationError(f"{path} item {hypothesis.id!r} is a duplicate id.")
        seen_ids.add(hypothesis.id)
        items.append(hypothesis)
    return tuple(items)
