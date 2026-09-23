"""The ONE home for the counting rule: what an interview did to a hypothesis,
and what a hypothesis's standing is once several interviews have landed.

**Counts are per-INTERVIEW, never per-fact, and that is the whole reason this
module exists as a module.** A talkative interviewee who makes the same point
five times must move a hypothesis's evidence count by ONE. Counting facts lets
one person who repeated themselves outweigh four separate people -- the
vote-counting failure evidence-synthesis methodologists abandoned decades ago,
and the one `app/web/review.py::standing_for` already refuses. Two surfaces
each re-deriving that rule is the hazard: both would render a confident-looking
integer and only one of them would be right, and nothing would say which.

**The de-duplication key is `session_id` ALONE.** Not `(project, session_id)`.
This is the single subtlest line in the module, so it is stated here rather
than left in the code: when a subtree is rolled up, one interview's facts may
be filed at a parent node AND at a child node. `(project, session_id)` LOOKS
like the careful key -- it is scoped, it is explicit -- and it counts that one
interview twice. An interview is an event that happened once; where its
extracted claims were filed afterwards is bookkeeping.

**Deterministic. No model call, ever.** The model judgement was already bought
and paid for upstream, in `Bearing.stance` -- whether a claim supports or
undermines a hypothesis is exactly what extraction decided. Re-asking a model
here would make the same corpus yield two different standings on two runs, and
would put a per-token cost and a network failure mode on a read. Everything
below is arithmetic over data already on disk.

**Why NOT in `app/web/review.py`.** That module is inside gate 20's pinned set
of modules permitted to read `Fact.quoted_text`, so summary logic written
there could read a verbatim quote and trip nothing. Written HERE, outside that
set, gate 20's equality assertion becomes the structural proof that this logic
cannot leak a quote: the field is unreadable from this file without failing a
gate first. `FactRef` below has no field one could be parked in either.

**`app/web/review.py` has been migrated onto this rule and no longer forks
it.** `standing_for` there is now an adapter: it projects `Fact`s onto
`FactRef`s, calls `standing()` twice -- once over the whole corpus and once
over the corpus minus the interview just finished, which is how the hub gets a
DELTA rather than a level -- and re-attaches the quote and segment numbers this
module is not allowed to hold. The agreement test that held the two forks
together while they both existed survives as
`app/tests/test_interview_summary.py::test_the_hub_counts_through_this_module_and_the_two_answers_agree`,
because an adapter can still drift even when the arithmetic underneath it
cannot.

**The before/now/contributed triple lives here too, as `standing_delta`.** It
was `review.py`'s alone until `python -m app standing <session>` needed the
same answer on a terminal. Re-deriving it in `app/main.py` would have been the
exact fork the paragraph above describes being paid off -- and worse in one
way, because the CLI would then have had to hold `Fact`s to do it. Written
here, the CLI reads `FactRef`s and never holds a quote at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Iterable, Mapping, Optional

from app.coach import corpus as coach_corpus
from app.coach.context import PROVENANCE_EVIDENCE, PROVENANCE_INFERENCE

# Dispatch through a MAP rather than an `if/elif/else` chain, deliberately. A
# fourth stance added to `ALL_STANCES` would be swept into whichever branch the
# final `else` happened to be, and every "each stance lands in exactly one
# column" test would still pass, because it does -- the wrong one. A map's key
# set can be pinned against `ALL_STANCES` by equality, which is the only shape
# that fails on that addition.
COLUMN_BY_STANCE: "Mapping[str, str]" = {
    coach_corpus.STANCE_SUPPORTS: "supporting",
    coach_corpus.STANCE_UNDERMINES: "contrary",
    coach_corpus.STANCE_UNCLEAR: "unclear",
}


@dataclass(frozen=True)
class FactRef:
    """One extracted claim, reduced to the fields a COUNT is allowed to see,
    plus the node it was filed at.

    Six fields, and the two that are absent are the point:

    - **No `quoted_text`.** Gate 20 makes reading it from this module a
      failing test; this type makes it impossible to park one here later
      under a different name without the type itself changing.
    - **No `seqs`.** Segment numbers are a handle into the transcript, and
      the transcript is exactly what does not leave the operator's machine.

    `project` is the node whose `context/facts.json` this fact was loaded
    from. Carried on every reference rather than alongside the collection: a
    rolled-up answer that cannot be traced back to a node is worse than no
    rollup, and attribution kept beside the list gets lost the first time
    someone filters it."""

    fact_id: str
    session_id: str
    claim: str
    provenance: str
    project: str
    bearings: "tuple[tuple[str, str], ...]"
    """`(hypothesis_id, stance)` pairs, flattened out of `Bearing` so this
    module holds no reference to a pydantic model that has other fields."""


def fact_ref(fact: Any, *, project: str) -> FactRef:
    """Projects one `Fact` onto a reference, FIELD BY FIELD.

    Never `fact.model_dump()`: `Fact`'s sixth field is `quoted_text`, and a
    wholesale dump would carry it into this module while naming it nowhere --
    passing gate 20's AST scan, passing gate 32's, and passing any reviewer
    grepping for the string. An explicit projection is the only form of this
    function whose safety is visible."""
    return FactRef(
        fact_id=fact.id,
        session_id=fact.session_id,
        claim=fact.claim,
        provenance=fact.provenance,
        project=project,
        bearings=tuple((bearing.hypothesis_id, bearing.stance) for bearing in fact.bearings),
    )


def count_interviews(refs: "Iterable[FactRef]") -> int:
    """The rule, in one line, in one place. Keyed on `session_id` alone -- see
    the module docstring for why adding the node to this key is the plausible
    mistake that double-counts one interview filed at two nodes."""
    return len({ref.session_id for ref in refs})


def _contributing_projects(refs: "Iterable[FactRef]") -> "tuple[str, ...]":
    return tuple(sorted({ref.project for ref in refs}))


@dataclass(frozen=True)
class PeopleBounds:
    """How many PEOPLE a set of interviews represents -- as a range, because
    with any unlabelled interview it is genuinely not a single number.

    **The gap this closes.** Every count in this module is keyed on
    `session_id`, and `count_interviews` is right to be: an interview is an
    event that happened once. But the criteria those counts are judged against
    are written about PEOPLE -- the demo project's own h1 says "At least 3 of 6
    interviewees describe, unprompted, a specific question they meant to look
    up". Three sessions satisfy that rule only if they are three different
    interviewees. Two calls with the same person, or one person recorded twice
    in a sitting, satisfies it on the arithmetic and not in fact, and until now
    nothing in the product could tell the difference or even raise the
    question. Session counting was never wrong; it was being read as an answer
    to a question it does not answer.

    **A range, not an estimate, and never a single guess.** Given some
    interviews labelled with who was in the room and some not:

    - `fewest` assumes every unlabelled interview was with somebody already
      named. That is the floor: an unlabelled call cannot reduce the number of
      distinct people below the number already identified.
    - `most` assumes every unlabelled interview was a different person nobody
      has named yet.

    The truth is somewhere in between and the product cannot know where. A
    midpoint would be a fabrication, and picking either end would be a claim --
    the low end flatters nothing but understates recruitment, and the high end
    would let an unlabelled corpus satisfy any criterion it liked. Reporting
    the interval is the only honest option, and it collapses to one number the
    moment every interview carries a label, which is exactly the incentive the
    label needs.

    **Counts only, never names.** Nothing here returns or stores a label. A
    participant label is a real person's name; this module is deliberately
    outside gate 20's pinned set of modules allowed to read verbatim speech,
    and a name deserves the same treatment even though no gate names it yet.
    The caller hands in labels and gets back integers.
    """

    sessions: int
    named: int
    """Distinct labels. Two interviews with the same name count once -- which
    is the entire point, and the case that makes `sessions` and this number
    disagree."""

    unlabelled: int

    @property
    def fewest(self) -> int:
        if self.sessions == 0:
            return 0
        # Not `named` alone: with no labels at all but interviews on disk,
        # somebody was in the room. Zero people would be a claim that the
        # interviews did not happen.
        return self.named or 1

    @property
    def most(self) -> int:
        return self.named + self.unlabelled

    @property
    def is_certain(self) -> bool:
        """True only when every interview is labelled. A caller that renders
        `fewest` as "the" number of people must branch on this first -- the
        whole value of the range is lost the moment one end is quoted alone."""
        return self.unlabelled == 0


def people_bounds(participants_by_session: "Mapping[str, Any]") -> PeopleBounds:
    """How many distinct people the given interviews represent.

    Takes a mapping of session id to whatever the participant reader answered
    for it -- a label, or `None` for the overwhelmingly common case of an
    interview recorded before anyone could name who was in it. A non-string, or
    a string that is blank once stripped, is treated as unlabelled rather than
    as a person: `read_participant` already answers `None` for every failure it
    can meet, and a second interpretation of "not a name" here would be a place
    for the two to disagree.

    Labels are compared after stripping, so a trailing space typed into the box
    on one night does not manufacture a second person. Case is deliberately NOT
    folded: "dana" and "Dana" may well be one person, but they may not, and
    silently merging two names on a guess is a worse failure than reporting a
    range one wider than it needed to be. The operator can see both rows in the
    picker and fix it; nothing here can."""
    named: "set[str]" = set()
    unlabelled = 0
    for label in participants_by_session.values():
        if not isinstance(label, str) or not label.strip():
            unlabelled += 1
            continue
        named.add(label.strip())
    return PeopleBounds(
        sessions=len(participants_by_session),
        named=len(named),
        unlabelled=unlabelled,
    )


@dataclass(frozen=True)
class Standing:
    """Where one hypothesis stands, given every fact in scope.

    Every count that is about INTERVIEWS is named `*_interviews`; every count
    that is about claims is named `*_facts`. The two are different numbers and
    the naming is what stops a reader treating them as the same one."""

    hypothesis_id: str
    statement: str
    falsifier: str
    state: str
    criterion: str
    is_scorable: bool

    project: str
    """The node that AUTHORED this hypothesis."""

    projects: "tuple[str, ...]"
    """The nodes whose facts CONTRIBUTED to it. Different question from
    `project`, and under a subtree rollup the answers routinely differ: a
    hypothesis written at the parent is usually evidenced from the children."""

    supporting: "tuple[FactRef, ...]"
    contrary: "tuple[FactRef, ...]"
    unclear: "tuple[FactRef, ...]"

    supporting_interviews: int
    contrary_interviews: int
    unclear_interviews: int

    total_interviews: int
    """Distinct interviews touching this hypothesis AT ALL -- never the sum of
    the three columns above, which counts an interview that both supported and
    undermined the hypothesis twice."""

    @property
    def supporting_facts(self) -> int:
        return len(self.supporting)

    @property
    def contrary_facts(self) -> int:
        return len(self.contrary)

    @property
    def unclear_facts(self) -> int:
        return len(self.unclear)


def standing(hypothesis: Any, refs: "Iterable[FactRef]", *, project: str) -> Standing:
    """One hypothesis's standing against every fact in scope.

    `refs` may span several nodes of one subtree; this function does not care
    which, and that is deliberate -- the merge decision belongs to the caller
    that resolved the scope, and the counting rule belongs here."""
    columns: "dict[str, list[FactRef]]" = {"supporting": [], "contrary": [], "unclear": []}
    touched: "list[FactRef]" = []

    for ref in refs:
        for hypothesis_id, stance in ref.bearings:
            if hypothesis_id != hypothesis.id:
                continue
            # An unrecognized stance cannot reach here through a validated
            # `Fact` -- `Bearing` rejects one. The fallback exists so a corpus
            # widened before this map is does not silently vanish; the
            # equality test on `COLUMN_BY_STANCE` is what makes the widening
            # visible.
            columns[COLUMN_BY_STANCE.get(stance, "unclear")].append(ref)
            touched.append(ref)

    return Standing(
        hypothesis_id=hypothesis.id,
        statement=hypothesis.statement,
        falsifier=hypothesis.falsifier,
        state=hypothesis.state,
        criterion=hypothesis.criterion,
        is_scorable=hypothesis.is_scorable,
        project=project,
        projects=_contributing_projects(touched),
        supporting=tuple(columns["supporting"]),
        contrary=tuple(columns["contrary"]),
        unclear=tuple(columns["unclear"]),
        supporting_interviews=count_interviews(columns["supporting"]),
        contrary_interviews=count_interviews(columns["contrary"]),
        unclear_interviews=count_interviews(columns["unclear"]),
        total_interviews=count_interviews(touched),
    )


@dataclass(frozen=True)
class HypothesisTouch:
    """What ONE interview did to one hypothesis.

    Every field here counts FACTS, and says so in its name. Inside a single
    interview that is the only meaningful count -- the interview count is 1 by
    definition -- but the names still carry it, because these numbers get
    rendered next to `*_interviews` numbers and a reader must never have to
    remember which kind they are looking at."""

    hypothesis_id: str
    supporting_facts: int
    contrary_facts: int
    unclear_facts: int


@dataclass(frozen=True)
class InterviewSummary:
    """One interview, summarized deterministically from its extracted claims."""

    session_id: str

    interviews: int
    """Always 1. Carried as a FIELD rather than left implicit so that a caller
    aggregating summaries adds interviews rather than facts, and so the
    invariant has somewhere to be asserted."""

    projects: "tuple[str, ...]"
    """Nodes whose `context/facts.json` holds a claim from this interview.
    Routinely more than one under a subtree rollup, which is exactly the case
    that must still count as ONE interview."""

    fact_count: int
    evidence_facts: int
    inference_facts: int
    touches: "tuple[HypothesisTouch, ...]"


def summarize_interview(session_id: str, refs: "Iterable[FactRef]") -> InterviewSummary:
    """Everything deterministic that can be said about one interview from the
    corpus alone. Facts from other interviews in `refs` are ignored rather
    than rejected, so a caller may pass a whole subtree's facts in."""
    mine = tuple(ref for ref in refs if ref.session_id == session_id)

    touches: "dict[str, dict[str, int]]" = {}
    for ref in mine:
        for hypothesis_id, stance in ref.bearings:
            column = COLUMN_BY_STANCE.get(stance, "unclear")
            bucket = touches.setdefault(
                hypothesis_id, {"supporting": 0, "contrary": 0, "unclear": 0}
            )
            bucket[column] += 1

    return InterviewSummary(
        session_id=session_id,
        interviews=1,
        projects=_contributing_projects(mine),
        fact_count=len(mine),
        evidence_facts=sum(1 for ref in mine if ref.provenance == PROVENANCE_EVIDENCE),
        inference_facts=sum(
            1 for ref in mine if ref.provenance == PROVENANCE_INFERENCE
        ),
        touches=tuple(
            HypothesisTouch(
                hypothesis_id=hypothesis_id,
                supporting_facts=counts["supporting"],
                contrary_facts=counts["contrary"],
                unclear_facts=counts["unclear"],
            )
            for hypothesis_id, counts in sorted(touches.items())
        ),
    )


NO_CONTRIBUTION = HypothesisTouch(
    hypothesis_id="", supporting_facts=0, contrary_facts=0, unclear_facts=0
)
"""What an interview that touched a hypothesis not at all put into it. A shared
zero rather than a literal at each call site, so "this interview did nothing
here" is one value a caller can compare against."""


def contribution(hypothesis_id: str, summary: "Optional[InterviewSummary]") -> HypothesisTouch:
    """What ONE interview put into one hypothesis, in FACTS.

    Reads an already-computed `InterviewSummary` rather than making one: the
    touches for every hypothesis are already in it, so a caller with twenty
    hypotheses makes one pass over the corpus instead of twenty identical
    ones."""
    if summary is None:
        return dataclass_replace(NO_CONTRIBUTION, hypothesis_id=hypothesis_id)
    for touch in summary.touches:
        if touch.hypothesis_id == hypothesis_id:
            return touch
    return dataclass_replace(NO_CONTRIBUTION, hypothesis_id=hypothesis_id)


@dataclass(frozen=True)
class StandingDelta:
    """Where one hypothesis stands, where it stood before one interview, and
    what that interview contributed.

    The three together are what turns a running total into an answer to "did
    the call I just finished change anything" -- a level alone cannot say."""

    now: Standing
    before: Standing
    """The standing this hypothesis would have had had the interview never
    happened. Equal to `now` when no interview was named."""

    contributed: HypothesisTouch
    """FACTS from the named interview. Zero throughout when none was named."""

    @property
    def touched(self) -> bool:
        """TOUCHED, not moved: an `unclear` bearing means the question was put
        and did not settle, which is a real outcome of an interview and would
        be hidden by reporting the hypothesis as untouched."""
        return bool(
            self.contributed.supporting_facts
            or self.contributed.contrary_facts
            or self.contributed.unclear_facts
        )

    @property
    def moved(self) -> bool:
        """The stronger claim. A claim the model could not direct is precisely
        the absence of direction, so `unclear` must not light this."""
        return bool(self.contributed.supporting_facts or self.contributed.contrary_facts)


def standing_delta(
    hypothesis: Any,
    refs: "Iterable[FactRef]",
    *,
    session_id: "Optional[str]" = None,
    project: str = "",
    summary: "Optional[InterviewSummary]" = None,
) -> StandingDelta:
    """One hypothesis's standing, and what a given interview did to it.

    The ONE home for the before/now/contributed triple, for the same reason
    this module is the one home for the counting rule underneath it: the hub
    and the CLI both render these numbers, and two derivations would render two
    confident-looking answers with nothing to say which was right.

    **`before` is RE-COUNTED over the corpus minus this session, never derived
    from what this interview contributed.** The tempting derivation subtracts
    the CONTRIBUTION -- and the contribution counts FACTS while the standing
    counts INTERVIEWS, so one talkative interviewee makes the "before" go
    negative. Re-counting cannot express that mistake.

    `summary` is an already-computed `InterviewSummary` a caller may hand in to
    avoid a pass over the corpus per hypothesis. It is CHECKED against
    `session_id` rather than trusted: a summary for a different interview would
    report another call's contribution under this one's heading, and no number
    rendered from it would look wrong."""
    refs = tuple(refs)
    now = standing(hypothesis, refs, project=project)
    before = (
        now
        if session_id is None
        else standing(
            hypothesis,
            tuple(ref for ref in refs if ref.session_id != session_id),
            project=project,
        )
    )

    mine = summary if summary is not None and summary.session_id == session_id else None
    if mine is None and session_id is not None:
        mine = summarize_interview(session_id, refs)

    return StandingDelta(
        now=now, before=before, contributed=contribution(hypothesis.id, mine)
    )
