"""`app/coach/interview_summary.py` -- the ONE home for the counting rule.

**The rule this file exists to hold still: counts are per-INTERVIEW, never
per-fact.** A talkative interviewee who makes the same point five times must
move a hypothesis's evidence count by ONE. `app/web/review.py::standing_for`
already states why (vote-counting is the failure evidence-synthesis
methodologists abandoned decades ago), and this module is where that rule now
lives so a second surface cannot fork it.

**Why the tests below hammer the SUBTREE case specifically.** A rollup across
a project subtree is exactly where per-fact counting silently reappears,
because the union genuinely IS a longer list of facts, and because the
tempting de-duplication key -- `(node, session_id)` -- looks careful while
being wrong: one interview whose facts were filed at two nodes would count as
two interviews. The load-bearing test is
`test_one_interview_filed_at_two_nodes_counts_once`, and it is written so that
BOTH wrong answers (N, and 2) fail it.

**Quote-freedom is structural here, not asserted.** This module is outside
gate 20's pinned `quoted_text` reader set, so gate 20's equality assertion is
the proof that summary logic written here cannot read a verbatim quote. The
tests below add the behavioural half: a sentinel quote on the input `Fact`
appears nowhere in the serialized output.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.coach import corpus as coach_corpus
from app.coach import interview_summary as summary_mod

SENTINEL_QUOTE = "summary-privacy-sentinel-quote-6cf30b12"

SESSION_A = "20260812-101500-aaaaaa"
SESSION_B = "20260813-090000-bbbbbb"


def _hypothesis(hypothesis_id="h1", **overrides):
    payload = {
        "id": hypothesis_id,
        "statement": "Activation stalls on the invite step",
        "falsifier": "three interviewees breeze through invites",
    }
    payload.update(overrides)
    return coach_corpus.Hypothesis(**payload)


def _fact(fact_id, session_id, *, stance="supports", hypothesis_id="h1", provenance="evidence"):
    return coach_corpus.Fact(
        id=fact_id,
        session_id=session_id,
        seqs=[1],
        claim=f"claim behind {fact_id}",
        provenance=provenance,
        quoted_text=SENTINEL_QUOTE,
        bearings=[{"hypothesis_id": hypothesis_id, "stance": stance}],
    )


def _refs(*pairs):
    """`(fact, node)` pairs projected onto the module's own reference type."""
    return tuple(summary_mod.fact_ref(fact, project=node) for fact, node in pairs)


# --- THE COUNTING TRAP ----------------------------------------------------------------


def test_one_interview_filed_at_two_nodes_counts_once():
    """THE failure this module exists to prevent, in its subtree form.

    Five facts, ONE interview, filed at TWO nodes of the same subtree. The
    honest answer is 1. The two dishonest answers are both reachable by a
    plausible implementation and both are asserted against by name:

    - **5** -- counting facts, the vote-counting failure. Reappears here
      because the merged list genuinely is longer.
    - **2** -- de-duplicating on `(node, session_id)`, which LOOKS more
      careful than de-duplicating on `session_id` and is the sneakier of the
      two. A test that only proved "not 5" passes this bug cheerfully.
    """
    refs = _refs(
        (_fact("f1", SESSION_A), "acme"),
        (_fact("f2", SESSION_A), "acme"),
        (_fact("f3", SESSION_A), "acme"),
        (_fact("f4", SESSION_A), "acme/onboarding"),
        (_fact("f5", SESSION_A), "acme/onboarding"),
    )
    standing = summary_mod.standing(_hypothesis(), refs, project="acme")

    assert standing.supporting_interviews == 1, (
        "one interview counted more than once -- 5 means facts were counted, "
        "2 means the de-duplication key included the node"
    )
    assert standing.supporting_interviews != 5
    assert standing.supporting_interviews != 2
    # The facts themselves are still all there. The rollup loses nothing; it
    # merely refuses to let volume masquerade as corroboration.
    assert standing.supporting_facts == 5
    assert summary_mod.count_interviews(refs) == 1


def test_a_talkative_interviewee_moves_the_count_by_one():
    refs = _refs(*[(_fact(f"f{index}", SESSION_A), "acme") for index in range(5)])
    assert summary_mod.standing(_hypothesis(), refs, project="acme").supporting_interviews == 1


def test_two_separate_interviews_count_as_two():
    """The other direction, and the mutation the test above cannot catch on
    its own: an implementation that hard-coded `1`, or that de-duplicated on
    something constant, passes every "not more than one" assertion."""
    refs = _refs(
        (_fact("f1", SESSION_A), "acme"),
        (_fact("f2", SESSION_B), "acme/onboarding"),
    )
    assert summary_mod.standing(_hypothesis(), refs, project="acme").supporting_interviews == 2


def test_the_two_directions_are_counted_separately_and_one_interview_can_be_in_both():
    """A monotonic count is a confirmation machine -- `Bearing` exists to stop
    one. An interview that produced both a supporting and an undermining claim
    counts once in EACH column, and the columns never merge."""
    refs = _refs(
        (_fact("f1", SESSION_A, stance="supports"), "acme"),
        (_fact("f2", SESSION_A, stance="undermines"), "acme/onboarding"),
        (_fact("f3", SESSION_A, stance="unclear"), "acme"),
        (_fact("f4", SESSION_B, stance="undermines"), "acme"),
    )
    standing = summary_mod.standing(_hypothesis(), refs, project="acme")
    assert standing.supporting_interviews == 1
    assert standing.contrary_interviews == 2
    assert standing.unclear_interviews == 1
    # Distinct interviews touching this hypothesis AT ALL -- not the sum of
    # the three columns above, which double-counts session A three times.
    assert standing.total_interviews == 2


def test_a_fact_bearing_on_another_hypothesis_is_not_counted():
    refs = _refs(
        (_fact("f1", SESSION_A, hypothesis_id="h1"), "acme"),
        (_fact("f2", SESSION_B, hypothesis_id="h2"), "acme"),
    )
    standing = summary_mod.standing(_hypothesis("h1"), refs, project="acme")
    assert standing.supporting_interviews == 1
    assert [ref.fact_id for ref in standing.supporting] == ["f1"]


# --- node attribution -----------------------------------------------------------------


def test_every_counted_fact_says_which_node_it_came_from():
    """A rolled-up answer that cannot be traced back to a node is worse than
    no rollup: the reader cannot tell whether "3 interviews" means three
    interviews about THIS sub-question or three about a sibling."""
    refs = _refs(
        (_fact("f1", SESSION_A), "acme"),
        (_fact("f2", SESSION_B), "acme/onboarding"),
    )
    standing = summary_mod.standing(_hypothesis(), refs, project="acme")
    assert {ref.fact_id: ref.project for ref in standing.supporting} == {
        "f1": "acme",
        "f2": "acme/onboarding",
    }
    assert standing.projects == ("acme", "acme/onboarding")


def test_the_hypothesis_own_node_is_carried_separately_from_the_contributing_nodes():
    """Under a subtree rollup a hypothesis authored at the parent is scored
    against facts filed at a child. Which node AUTHORED it and which nodes
    CONTRIBUTED to it are different questions and get different fields."""
    refs = _refs((_fact("f1", SESSION_A), "acme/onboarding"))
    standing = summary_mod.standing(_hypothesis(), refs, project="acme")
    assert standing.project == "acme"
    assert standing.projects == ("acme/onboarding",)


# --- one interview's summary ----------------------------------------------------------


def test_an_interview_summary_counts_itself_as_exactly_one_interview():
    """Inside ONE interview, counting claims is the only meaningful count --
    so every such field is named `*_facts`, and the interview count is 1 by
    definition rather than by arithmetic that could drift."""
    refs = _refs(
        (_fact("f1", SESSION_A, provenance="evidence"), "acme"),
        (_fact("f2", SESSION_A, provenance="evidence"), "acme"),
        (_fact("f3", SESSION_A, provenance="inference", stance="undermines"), "acme/onboarding"),
    )
    summary = summary_mod.summarize_interview(SESSION_A, refs)
    assert summary.interviews == 1
    assert summary.fact_count == 3
    assert summary.evidence_facts == 2
    assert summary.inference_facts == 1
    assert summary.projects == ("acme", "acme/onboarding")
    touches = {touch.hypothesis_id: touch for touch in summary.touches}
    assert touches["h1"].supporting_facts == 2
    assert touches["h1"].contrary_facts == 1


def test_an_interview_summary_ignores_facts_from_other_interviews():
    refs = _refs(
        (_fact("f1", SESSION_A), "acme"),
        (_fact("f2", SESSION_B), "acme"),
    )
    summary = summary_mod.summarize_interview(SESSION_A, refs)
    assert summary.fact_count == 1
    assert summary.session_id == SESSION_A


def test_an_interview_with_no_facts_summarizes_to_zero_rather_than_raising():
    summary = summary_mod.summarize_interview(SESSION_A, ())
    assert summary.fact_count == 0
    assert summary.touches == ()
    assert summary.interviews == 1


# --- privacy --------------------------------------------------------------------------


def test_the_reference_type_has_no_field_that_could_hold_a_quote():
    """Gate 20 is the structural proof that this module cannot READ
    `Fact.quoted_text`. This is the shape proof one layer down: the type the
    module passes around has no field a quote could be parked in later."""
    names = {field.name for field in dataclasses.fields(summary_mod.FactRef)}
    assert names == {"fact_id", "session_id", "claim", "provenance", "project", "bearings"}
    assert "quoted_text" not in names
    assert "seqs" not in names, (
        "segment numbers are a handle into the transcript, and the transcript is "
        "exactly what does not leave this machine"
    )


def test_no_serialized_summary_output_contains_the_verbatim_quote():
    """Asserted over the WHOLE serialized structure -- never a chosen
    sub-part -- because the interesting leak is a quote arriving in a
    differently-named field."""
    assert len(SENTINEL_QUOTE) > 16, "a short sentinel proves nothing either way"
    refs = _refs((_fact("f1", SESSION_A), "acme"))
    # The mutation this test cannot survive without: if the fixture stopped
    # carrying a quote, the assertions below would pass vacuously forever.
    assert SENTINEL_QUOTE in _fact("f1", SESSION_A).quoted_text

    for produced in (
        summary_mod.standing(_hypothesis(), refs, project="acme"),
        summary_mod.summarize_interview(SESSION_A, refs),
    ):
        blob = json.dumps(dataclasses.asdict(produced), default=str)
        assert SENTINEL_QUOTE not in blob, f"{type(produced).__name__} leaked the verbatim quote"


# --- the migration, and the agreement that outlives it ---------------------------------


def test_the_hub_counts_through_this_module_and_the_two_answers_agree():
    """`app/web/review.py::standing_for` no longer computes standings itself.
    The debt this test was written against is paid: the constant that named it,
    `MODULE_AWAITING_MIGRATION_ONTO_THIS_RULE`, is gone, and its absence is
    asserted so that re-introducing the fork cannot be done quietly.

    The agreement assertions SURVIVE the migration rather than being deleted
    with it. The hub is now an adapter around `standing()` -- it re-shapes,
    re-labels and re-attaches -- and an adapter can drift from the thing it
    adapts even when the arithmetic underneath is shared. This still fails the
    day the two surfaces render two different confident integers.
    """
    from app.web import review as review_mod

    assert not hasattr(summary_mod, "MODULE_AWAITING_MIGRATION_ONTO_THIS_RULE"), (
        "the migration constant is back -- either the fork returned or the debt "
        "was re-declared rather than paid"
    )

    facts = (
        _fact("f1", SESSION_A, stance="supports"),
        _fact("f2", SESSION_A, stance="supports"),
        _fact("f3", SESSION_B, stance="undermines"),
    )
    hypothesis = _hypothesis()
    theirs = review_mod.standing_for(hypothesis, facts)
    ours = summary_mod.standing(
        hypothesis, _refs(*[(fact, "acme") for fact in facts]), project="acme"
    )
    assert ours.supporting_interviews == theirs["supporting_interviews"] == 1
    assert ours.contrary_interviews == theirs["contrary_interviews"] == 1
    assert ours.state == theirs["state"]
    assert ours.is_scorable == theirs["is_scorable"]


def test_a_hypothesis_with_no_pre_registered_criterion_is_not_scorable():
    """Carried through the summary rather than left for the caller to infer
    from an empty string: an agent that tallied evidence against a hypothesis
    with no rule would be painting the target around the arrow."""
    refs = _refs((_fact("f1", SESSION_A), "acme"))
    assert summary_mod.standing(_hypothesis(), refs, project="acme").is_scorable is False
    scorable = _hypothesis(criterion="we are right if 3 of 10 raise it unprompted")
    assert summary_mod.standing(scorable, refs, project="acme").is_scorable is True


def test_the_module_is_deterministic_and_makes_no_model_call():
    """The model judgement was already bought and paid for in `Bearing.stance`.
    Re-asking a model here would make the same corpus produce two different
    standings on two runs, and would put a cost on a read."""
    import app.coach.interview_summary as module

    source = module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    for forbidden in ("anthropic", "reasoning_client", "httpx", "requests"):
        assert forbidden not in text, f"the summary module reached for {forbidden}"

    refs = _refs(*[(_fact(f"f{i}", SESSION_A), "acme") for i in range(4)])
    first = summary_mod.standing(_hypothesis(), refs, project="acme")
    second = summary_mod.standing(_hypothesis(), refs, project="acme")
    assert first == second, "two runs over the same input disagreed"


def test_the_stance_to_column_map_covers_exactly_the_corpus_stance_set():
    """The mutation an `if/elif/else` over stances cannot fail on.

    A fourth stance added to `ALL_STANCES` -- say `qualifies` -- would be
    swept into whichever column the final `else` happens to be, and every
    "each stance lands in exactly one column" test would still pass, because
    it DOES land in one: the wrong one. Pinning the map's key set against
    `ALL_STANCES` by EQUALITY is the only shape that fails on that addition,
    which is why the module dispatches through a map at all."""
    assert set(summary_mod.COLUMN_BY_STANCE) == coach_corpus.ALL_STANCES, (
        "a stance the corpus allows has no column, or a column exists for a stance "
        "the corpus does not allow -- decide where it belongs rather than letting a "
        "fallback decide"
    )


@pytest.mark.parametrize("stance", sorted(coach_corpus.ALL_STANCES))
def test_every_stance_the_corpus_allows_lands_in_its_named_column(stance):
    refs = _refs((_fact("f1", SESSION_A, stance=stance), "acme"))
    standing = summary_mod.standing(_hypothesis(), refs, project="acme")
    columns = {
        "supporting": standing.supporting,
        "contrary": standing.contrary,
        "unclear": standing.unclear,
    }
    expected = summary_mod.COLUMN_BY_STANCE[stance]
    assert [ref.fact_id for ref in columns[expected]] == ["f1"]
    assert sum(len(entries) for entries in columns.values()) == 1
    assert standing.total_interviews == 1
