"""Tests for the critic judge harness (`benchmark/eval/judge_critic.py`).

**What these tests are and are not.** They prove the HARNESS is sound: that
the labels are well-formed, that every rule flag is actually covered by a
case, that a failed critique call is recorded as unjudged rather than
silently passing, and that the two asymmetric outcomes are counted the way
the report claims. They prove NOTHING about the critic's judgement -- that is
what running the harness against a real model produces, and it cannot be
asserted offline.

That distinction is the whole reason this file exists separately from
`test_coach_critic.py`. A green run here means the measuring instrument is
not lying; it does not mean the thing being measured is good. Per the
standing note in `app/coach/critic.py`, a test that exercises one side of a
seam proves INTENT, not BEHAVIOUR.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.coach import critic
from app.coach.stub import StubAnthropicClient, tool_use_response
from benchmark.eval import critic_gold, judge_critic
from benchmark.eval.critic_gold import (
    ALL_FLAGS,
    GOLD_CASES,
    GOLD_SURVEY,
    LANG_EN,
    LANG_HE,
    LANG_MIXED,
    TIER_BAD,
    TIER_CONTESTED,
    TIER_GOOD,
    GoldCase,
    cases_by_language,
    cases_by_tier,
    scored_cases,
)


# --- the labels themselves ------------------------------------------------------


def test_all_flags_matches_the_judgment_dataclass():
    """The one test that catches a flag added to `Judgment` and forgotten here.

    A sixth rule would otherwise be completely unmeasured by this harness
    while every report it prints still looked complete -- exactly the failure
    `_critic_tool_schema` guards against with its properties/required pair."""
    judgment_flags = {
        field.name
        for field in dataclasses.fields(critic.Judgment)
        if field.type in ("bool", bool)
    }
    assert set(ALL_FLAGS) == judgment_flags


def test_every_rule_flag_has_at_least_one_bad_case():
    """A rule with no known-bad case is a rule this harness cannot detect a
    miss on, and its row in the report would read as a clean zero."""
    covered: "set[str]" = set()
    for case in cases_by_tier(TIER_BAD):
        covered |= set(case.must_flag)
    assert covered == set(ALL_FLAGS)


def test_good_cases_constrain_every_flag():
    """Negative controls are the only thing that makes over-flagging visible,
    and they only work if every flag is pinned to False on them."""
    good = cases_by_tier(TIER_GOOD)
    assert good, "there must be negative controls"
    for case in good:
        assert case.no_other_flags, f"{case.case_id} does not constrain its flags"
        assert not case.must_flag


def test_contested_cases_are_never_scored():
    """The tier exists so a judgement call is not promoted to ground truth.
    If a contested case leaked into `scored_cases()` the report would measure
    the critic's agreement with the label author instead of its correctness."""
    scored_ids = {case.case_id for case in scored_cases()}
    for case in cases_by_tier(TIER_CONTESTED):
        assert case.case_id not in scored_ids
    assert scored_ids == {c.case_id for c in GOLD_CASES if c.tier in (TIER_BAD, TIER_GOOD)}


def test_case_ids_are_unique():
    ids = [case.case_id for case in GOLD_CASES]
    assert len(ids) == len(set(ids))


# Bad cases that are NOT lifted from the synthetic interview script, each with
# the reason stated in its own `note`. Keep this list as short as the evidence
# allows: every id here is a case whose label is authored rather than observed.
_AUTHORED_BAD_CASES = frozenset({"bad-generic", "bad-off-topic"})


def test_english_bad_cases_are_verbatim_from_the_script():
    """The labels' provenance is the script's own authorial intent. If the
    script is reworded and these are not, the cases quietly stop being the
    thing that was declared bad in advance and become sentences I wrote.

    English only: a translation cannot be verbatim in a language the script was
    never written in. Its provenance runs through `translation_of` instead, and
    the test below is what enforces it.

    `_AUTHORED_BAD_CASES` is an explicit, per-id exemption rather than a
    loosened assertion. `bad-generic` was added when `is_generic` was, and it
    could not have come from the script: the script's failures were planted
    before that flag existed, and the script is not tracked in this
    repository, so planting one there would produce a case whose provenance
    nobody else could check. The case says all of this in its own `note`. The
    exemption is a list of ids so that adding a second authored case is a
    visible, deliberate act -- the guard stays sharp for every case that does
    claim script provenance, which is the whole point of it.

    `bad-off-topic` (2026-08-19) is the second, added on exactly that
    deliberate act and for the identical reason: `is_off_topic` postdates the
    script's planted failures too. Its own `note` carries the narrower claim,
    and the same discount applies to a failure on it."""
    from benchmark.eval import claudictive_script

    script_text = {
        text for speaker, text in claudictive_script.TURNS if speaker == "interviewer"
    }
    english_bad = [c for c in cases_by_tier(TIER_BAD) if c.language == LANG_EN]
    assert english_bad, "the English bad cases are the provenance anchor"
    anchored = [c for c in english_bad if c.case_id not in _AUTHORED_BAD_CASES]
    assert anchored, "at least one bad case must still be anchored to the script"
    for case in anchored:
        assert case.text in script_text, f"{case.case_id} no longer matches the script"
    for case_id in _AUTHORED_BAD_CASES:
        case = next((c for c in english_bad if c.case_id == case_id), None)
        assert case is not None, f"{case_id} is exempted but no longer exists"
        assert "PROVENANCE" in case.note, (
            f"{case_id} is exempt from the script anchor, so its own note must say "
            "where its label actually comes from -- an exemption with no stated "
            "basis is just an unlabelled sentence somebody wrote"
        )


def test_every_translation_carries_its_source_labels_unchanged():
    """The multilingual half is an EXPERIMENT, and it only is one while language
    is the sole variable. A translation whose labels drifted from its source
    would confound "the critic is worse in Hebrew" with "these are different
    questions" -- and it would do so invisibly, because both cases would still
    look internally consistent."""
    by_id = {case.case_id: case for case in GOLD_CASES}
    translations = [case for case in GOLD_CASES if case.translation_of]
    assert translations, "there must be translated cases"
    for case in translations:
        source = by_id.get(case.translation_of)
        assert source is not None, f"{case.case_id} names a source that does not exist"
        assert source.language == LANG_EN, f"{case.case_id} must translate an English case"
        assert case.tier == source.tier, f"{case.case_id} changed tier"
        assert case.must_flag == source.must_flag, f"{case.case_id} changed must_flag"
        assert case.no_other_flags == source.no_other_flags
        assert case.target_hypothesis_id == source.target_hypothesis_id
        assert case.move == source.move, f"{case.case_id} changed move"


def test_a_non_english_case_must_name_its_source():
    with pytest.raises(ValueError, match="must name the English case"):
        GoldCase(case_id="x", tier=TIER_BAD, text="שאלה", move="deepen",
                 must_flag=frozenset({"is_leading"}), language=LANG_HE)


def test_the_hard_register_is_actually_represented():
    """Hebrew alone would not measure the real case. `PROJECT.md:17-19` names
    code-switching, not Hebrew, as the defining constraint -- and the switch
    happens mid-sentence, which is what makes an English-written rubric hard to
    apply. A set with clean Hebrew and no mixed cases would look multilingual
    and still miss it."""
    mixed = cases_by_language(LANG_MIXED)
    assert len(mixed) >= 4
    for case in mixed:
        has_hebrew = any("֐" <= ch <= "׿" for ch in case.text)
        has_latin = any(ch.isascii() and ch.isalpha() for ch in case.text)
        assert has_hebrew and has_latin, (
            f"{case.case_id} is not actually code-switched: {case.text!r}"
        )


def test_every_language_covers_both_directions():
    """A language represented only by bad cases would measure catching and be
    blind to over-flagging, which is the failure this file exists to make
    visible in the first place."""
    for language in (LANG_EN, LANG_HE, LANG_MIXED):
        cases = cases_by_language(language)
        assert any(c.tier == TIER_BAD for c in cases), f"{language} has no bad cases"
        assert any(c.tier == TIER_GOOD for c in cases), f"{language} has no controls"


def test_a_gold_case_rejects_an_unknown_flag():
    with pytest.raises(ValueError, match="unknown flag"):
        GoldCase(case_id="x", tier=TIER_BAD, text="t", move="deepen",
                 must_flag=frozenset({"is_rude"}))


def test_a_good_case_may_not_assert_a_flag():
    with pytest.raises(ValueError, match="must assert no flags"):
        GoldCase(case_id="x", tier=TIER_GOOD, text="t", move="deepen",
                 must_flag=frozenset({"is_leading"}))


# --- ordering -------------------------------------------------------------------


def test_interleaving_preserves_every_case():
    ordered = judge_critic.interleaved(GOLD_CASES)
    assert sorted(c.case_id for c in ordered) == sorted(c.case_id for c in GOLD_CASES)


def test_interleaving_breaks_up_the_run_of_bad_questions():
    """The five bad cases are consecutive in the script. Left consecutive
    here, a model shown four violations in a row has been told something
    about the fifth, and the run measures priming as much as judgement."""
    ordered = judge_critic.interleaved(GOLD_CASES)
    tiers = [case.tier for case in ordered]
    longest = run = 1
    for previous, current in zip(tiers, tiers[1:]):
        run = run + 1 if current == previous == TIER_BAD else 1
        longest = max(longest, run)
    # Not merely "shorter than the run in the script" -- that would still pass
    # with four of the five adjacent. No two bad cases may touch at all.
    assert longest == 1


# --- the harness ----------------------------------------------------------------


def _judgment(index: int, *, band: str, **flags) -> dict:
    payload = {flag: False for flag in ALL_FLAGS}
    payload.update(flags)
    payload.update({"index": index, "band": band, "rationale": "because"})
    return payload


def _response(count: int, *, band: str = critic.BAND_CLEAN) -> object:
    return tool_use_response(
        critic.CRITIC_TOOL_NAME,
        {"judgments": [_judgment(i, band=band) for i in range(count)]},
    )


def test_judge_all_returns_a_judgment_per_case():
    cases = list(GOLD_CASES[:5])
    client = StubAnthropicClient(responses=[_response(2), _response(2), _response(1)])
    judged = judge_critic.judge_all(
        client, cases, hypotheses=(), model="m", batch_size=2,
        max_tokens=64, timeout_s=1.0,
    )
    assert set(judged) == {case.case_id for case in cases}
    assert all(judgment is not None for judgment in judged.values())


def test_a_failed_batch_records_none_and_does_not_drop_the_cases(capsys):
    """"The critic could not be reached" and "the critic said clean" must
    never look the same. A dropped case would simply vanish from the report's
    denominators, making a run that reached nothing look partially clean."""
    cases = list(GOLD_CASES[:4])
    # The stub raises on every call, which is `run_critique`'s "exception"
    # path -- the same shape a timeout takes.
    client = StubAnthropicClient(raises=RuntimeError("vendor down"))
    judged = judge_critic.judge_all(
        client, cases, hypotheses=(), model="m", batch_size=2,
        max_tokens=64, timeout_s=1.0,
    )
    assert set(judged) == {case.case_id for case in cases}
    assert all(judgment is None for judgment in judged.values())
    assert "CRITIQUE FAILED" in capsys.readouterr().out


def test_each_batch_indexes_candidates_from_zero():
    """Per-batch indexing, matching the live path. If the harness numbered
    candidates globally, every judgment after the first batch would fail to
    match and the whole set would read as unjudged."""
    cases = list(GOLD_CASES[:4])
    client = StubAnthropicClient(responses=[_response(2), _response(2)])
    judged = judge_critic.judge_all(
        client, cases, hypotheses=(), model="m", batch_size=2,
        max_tokens=64, timeout_s=1.0,
    )
    assert all(judgment is not None for judgment in judged.values())


def test_to_candidate_carries_the_hypothesis_target():
    case = next(c for c in GOLD_CASES if c.target_hypothesis_id)
    candidate = judge_critic.to_candidate(case, 3)
    assert candidate.index == 3
    assert candidate.text == case.text
    assert candidate.move == case.move
    assert candidate.target_hypothesis_id == case.target_hypothesis_id


def test_every_targeted_hypothesis_exists_in_the_demo_project():
    """`is_confirmation_only` is judged against a hypothesis's OWN falsifier.
    A case targeting an id that is not loaded would be critiqued with no
    falsifier at all, and the report would still look complete."""
    from app.coach import corpus

    if not (judge_critic.PROJECT / "context").is_dir():
        pytest.skip("the demo-claudictive project is local data, not shipped with the repository")
    hypotheses = corpus.load_hypotheses(judge_critic.PROJECT / "context", missing_ok=False)
    known = {h.id for h in hypotheses}
    targeted = {c.target_hypothesis_id for c in GOLD_CASES if c.target_hypothesis_id}
    assert targeted <= known


# --- the scoring ----------------------------------------------------------------


def _all_judged(band_by_tier: dict) -> dict:
    return {
        case.case_id: critic.Judgment(
            index=0, band=band_by_tier[case.tier], rationale="r",
            **{flag: False for flag in ALL_FLAGS},
        )
        for case in GOLD_CASES
    }


def test_a_perfect_run_reports_no_dangerous_outcomes(capsys):
    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAR_VIOLATION,
        TIER_GOOD: critic.BAND_CLEAN,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    assert judge_critic.report(judged, band_history={}, runs=1) == 0
    capsys.readouterr()


def test_a_good_question_banded_clear_violation_is_dangerous(capsys):
    """This is the direction nothing else in the project can see: the
    candidate is dropped by `rank_candidates` and the interviewer never
    learns it existed."""
    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAR_VIOLATION,
        TIER_GOOD: critic.BAND_CLEAR_VIOLATION,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    assert judge_critic.report(judged, band_history={}, runs=1) == len(cases_by_tier(TIER_GOOD))
    assert "SILENTLY DROPPED" in capsys.readouterr().out


def test_a_bad_question_banded_clean_is_dangerous(capsys):
    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAN,
        TIER_GOOD: critic.BAND_CLEAN,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    assert judge_critic.report(judged, band_history={}, runs=1) == len(cases_by_tier(TIER_BAD))
    assert "SHOWN AS ADVICE" in capsys.readouterr().out


def test_uncertain_is_not_dangerous_in_either_direction(capsys):
    """`rank_candidates` shows uncertain candidates -- only `clear_violation`
    is dropped. So an uncertain band is a hedge, not a failure, and counting
    it as one would push the critic toward overconfident bands."""
    judged = _all_judged({tier: critic.BAND_UNCERTAIN
                          for tier in (TIER_BAD, TIER_GOOD, TIER_CONTESTED)})
    assert judge_critic.report(judged, band_history={}, runs=1) == 0
    capsys.readouterr()


def test_contested_cases_never_count_as_dangerous(capsys):
    """Whatever the critic says about a contested case, the count stays 0.
    If this failed, the harness would be gating on an opinion."""
    for band in (critic.BAND_CLEAR_VIOLATION, critic.BAND_CLEAN, critic.BAND_UNCERTAIN):
        judged = _all_judged({
            TIER_BAD: critic.BAND_CLEAR_VIOLATION,
            TIER_GOOD: critic.BAND_CLEAN,
            TIER_CONTESTED: band,
        })
        assert judge_critic.report(judged, band_history={}, runs=1) == 0
    capsys.readouterr()


def test_an_unjudged_case_is_not_counted_as_a_pass(capsys):
    """A run where every call failed must not print a clean bill of health.
    The dangerous count is 0 -- nothing was judged -- so the UNJUDGED line is
    the only thing standing between that and a false green."""
    judged = {case.case_id: None for case in GOLD_CASES}
    assert judge_critic.report(judged, band_history={}, runs=1) == 0
    out = capsys.readouterr().out
    assert "UNJUDGED (the call failed -- NOT a pass)" in out


def test_a_partial_run_says_so_before_any_number(capsys):
    """A measured run judged 10 of 17 and every table still read as complete.
    The coverage line must come FIRST, so a partial run cannot be quoted as a
    score for the whole set."""
    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAR_VIOLATION,
        TIER_GOOD: critic.BAND_CLEAN,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    dropped = [c.case_id for c in GOLD_CASES[:4]]
    for case_id in dropped:
        judged[case_id] = None

    judge_critic.report(judged, band_history={}, runs=1)
    out = capsys.readouterr().out
    expected = len(GOLD_CASES) - len(dropped)
    assert f"{expected} of {len(GOLD_CASES)} cases judged" in out
    assert "PARTIAL run" in out
    # Before the flag table, not buried under it.
    assert out.index("COVERAGE") < out.index("FLAG ACCURACY")


def test_a_complete_run_does_not_warn_about_partial_coverage(capsys):
    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAR_VIOLATION,
        TIER_GOOD: critic.BAND_CLEAN,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    judge_critic.report(judged, band_history={}, runs=1)
    out = capsys.readouterr().out
    assert f"{len(GOLD_CASES)} of {len(GOLD_CASES)} cases judged" in out
    assert "PARTIAL run" not in out


def test_stability_section_names_cases_that_changed_band(capsys):
    from collections import Counter

    judged = _all_judged({
        TIER_BAD: critic.BAND_CLEAR_VIOLATION,
        TIER_GOOD: critic.BAND_CLEAN,
        TIER_CONTESTED: critic.BAND_UNCERTAIN,
    })
    history = {
        "good-last-time": Counter({critic.BAND_CLEAN: 2, critic.BAND_UNCERTAIN: 1}),
        "bad-pitch": Counter({critic.BAND_CLEAR_VIOLATION: 3}),
    }
    judge_critic.report(judged, band_history=history, runs=3)
    out = capsys.readouterr().out
    assert "good-last-time" in out.split("STABILITY")[1]
    assert "bad-pitch" not in out.split("STABILITY")[1]


# --- the gold survey (2026-08-19) -----------------------------------------------------


def test_no_survey_line_restates_a_case():
    """The fixture defect this pins actually happened, twice, and neither time
    was caught by review.

    `GOLD_SURVEY` is derived from the same script the GOOD cases are turns of,
    so phrasing its lines as questions made several of them near-verbatim
    copies of those cases. `is_generic`'s own rubric penalises a question that
    "restates the survey item verbatim" -- so the fixture taught the critic to
    condemn a good case for matching a line this repository had copied out of
    it, and the measured run duly over-flagged `good-last-time`.

    A similarity bound rather than an inequality: the lines SHOULD cover the
    same subjects (that is the whole point of deriving them from the script),
    and only near-identical WORDING is the defect."""
    from difflib import SequenceMatcher

    for line in GOLD_SURVEY:
        for case in GOLD_CASES:
            ratio = SequenceMatcher(None, line.lower(), case.text.lower()).ratio()
            assert ratio < 0.8, (
                f"survey line {line!r} restates {case.case_id} ({case.text!r}) at "
                f"{ratio:.2f} similarity -- a survey written in the cases' own words "
                "makes is_generic fire on them, which is a fixture defect wearing a "
                "measurement's clothes"
            )


def test_the_off_topic_case_is_outside_every_survey_line():
    """`bad-off-topic` is the ONLY positive evidence this harness carries for
    the seventh flag -- the good cases are turns from the interview itself and
    so can only ever be a negative control. If a survey line ever grows to
    cover commuting, the case stops testing anything and the row would still
    read as a pass."""
    case = next(c for c in GOLD_CASES if c.case_id == "bad-off-topic")
    haystack = " ".join(GOLD_SURVEY).lower()
    for word in ("commute", "commuting", "drive", "traffic", "car"):
        assert word not in haystack, (
            f"{word!r} appears in GOLD_SURVEY, so bad-off-topic may no longer be "
            "off topic and its row would pass for the wrong reason"
        )
    assert "is_off_topic" in case.must_flag
