"""Tests for the rehearsal harness's two load-bearing pure pieces.

The harness itself drives real model calls and is exercised by running it. What
is tested here is the part that decides whether its headline numbers mean
anything: fragmentation sets the cycle count, and the synthetic clock sets the
"per spoken minute" denominator. Both are simulations, and a simulation that is
wrong in the flattering direction would report a suggestion rate the product
does not have.
"""

from __future__ import annotations

from benchmark.eval.rehearse import Clock, fragments, scripted_questions


def test_text_is_fragmented_the_way_the_engine_finalizes():
    """One cycle runs per submitted segment, and the real STT emits roughly
    five-word finalizations. Submitting whole turns would under-count cycles by
    an order of magnitude and report a comfortable rate the product lacks."""
    text = "one two three four five six seven eight nine ten eleven"
    assert fragments(text, 5) == ["one two three four five", "six seven eight nine ten", "eleven"]


def test_fragmenting_loses_no_words():
    text = "tell me about the last time you needed something from outside the game"
    assert " ".join(fragments(text, 4)) == text


def test_empty_text_produces_no_segments():
    assert fragments("", 5) == []
    assert fragments("   ", 5) == []


def test_the_clock_advances_with_speech_length():
    clock = Clock(2.5)
    start, end = clock.advance("one two three four five")
    assert start == 0.0
    assert end == 2.0
    later_start, _ = clock.advance("six")
    assert later_start == end, "segments must be contiguous, never overlapping"


def test_a_very_short_fragment_still_takes_time():
    """A zero-length segment would make two segments share an audio position,
    and every window computed from those positions would be degenerate."""
    clock = Clock(2.5)
    start, end = clock.advance("ok")
    assert end > start


def test_the_scripted_interviewer_is_the_scripts_own_turns():
    """Reused rather than invented, so an unattended run asks questions somebody
    already judged competent -- and so `--auto` measures the coach rather than
    measuring an improvised interview."""
    questions = scripted_questions()
    assert len(questions) > 20
    assert all(isinstance(q, str) and q for q in questions)
    assert "Walk me through the last time you sat down to play, from turning the machine on to the game actually starting." in questions
