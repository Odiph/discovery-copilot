"""The repetition guard and the anti-repeat prompt block.

These exist because of a MEASURED defect, not a hypothetical one: the first
scored eval run showed one question 64 times, and 32% of everything the
operator saw was an exact repeat. The cases below are drawn from that run's
actual text, so a regression fails against the thing that really happened.
"""

from __future__ import annotations

import pytest

from app.coach import prompts
from app.coach.repetition import (
    DEFAULT_THRESHOLD,
    RecentSuggestions,
    normalise,
    similarity,
)

# Verbatim from session 20260813T213559 -- the x64 offender and its x17 variant.
OFFENDER = "Walk me through the last time you removed something from your gaming machine. What made you get rid of it?"
VARIANT = "Walk me through the last time you removed something from that machine. What made you get rid of it?"
UNRELATED = "What is on your second screen while you play, and how did it end up there?"


def test_normalise_collapses_case_and_punctuation_so_two_spellings_are_one_question():
    assert normalise("Walk me through it -- please!") == "walk me through it please"


def test_the_real_x64_offender_matches_itself_across_a_cycle():
    recent = RecentSuggestions().remember(OFFENDER)
    assert recent.contains_near(OFFENDER)


def test_the_real_near_variant_is_caught_which_is_the_whole_point_of_not_using_equality():
    """The x17 variant differs from the x64 offender by two words. Equality
    calls them different questions; an operator would not."""
    assert OFFENDER != VARIANT
    assert similarity(OFFENDER, VARIANT) >= DEFAULT_THRESHOLD
    recent = RecentSuggestions().remember(OFFENDER)
    assert recent.contains_near(VARIANT)


def test_a_genuinely_different_question_is_not_suppressed():
    recent = RecentSuggestions().remember(OFFENDER)
    assert not recent.contains_near(UNRELATED)
    assert similarity(OFFENDER, UNRELATED) < DEFAULT_THRESHOLD


def test_the_window_forgets_and_a_question_older_than_it_may_be_asked_again():
    recent = RecentSuggestions(window=2).remember(OFFENDER)
    recent = recent.remember(UNRELATED).remember("A third entirely separate question here")
    assert not recent.contains_near(OFFENDER), "the window should have dropped the oldest"
    assert recent.contains_near(UNRELATED)


def test_remember_never_mutates_the_instance_it_was_called_on():
    original = RecentSuggestions()
    updated = original.remember(OFFENDER)
    assert original.texts == ()
    assert updated.texts == (OFFENDER,)
    assert original is not updated


def test_an_empty_candidate_is_never_treated_as_a_repeat_because_that_is_a_different_defect():
    recent = RecentSuggestions().remember(OFFENDER)
    assert not recent.contains_near("")
    assert not recent.contains_near("   ")
    # ...and remembering one must not poison the window into matching everything.
    assert recent.remember("").texts == (OFFENDER,)


def test_for_prompt_returns_the_most_recent_and_caps_them():
    recent = RecentSuggestions().remember_all([f"question number {n}" for n in range(10)])
    assert recent.for_prompt(3) == ("question number 7", "question number 8", "question number 9")
    assert recent.for_prompt(0) == ()


# --- the prompt block ------------------------------------------------------------------


def test_the_block_is_empty_when_nothing_has_been_proposed_yet():
    """An empty 'you already proposed:' heading tells the model something
    untrue at the start of an interview, so the caller omits it entirely."""
    assert prompts.already_proposed_block(()) == ""


def test_the_block_lists_what_was_shown_and_forbids_rewording_not_just_repeating():
    block = prompts.already_proposed_block((OFFENDER, UNRELATED))
    assert OFFENDER in block and UNRELATED in block
    assert "reworded" in block.lower(), (
        "forbidding only exact repeats leaves the x17 near-variant permitted"
    )


@pytest.mark.parametrize("threshold", [0.80, 0.90])
def test_the_measured_offenders_are_caught_at_both_thresholds_that_were_measured(threshold):
    recent = RecentSuggestions(threshold=threshold).remember(OFFENDER)
    assert recent.contains_near(VARIANT)
