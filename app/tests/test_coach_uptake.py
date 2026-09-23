"""Tests for the P1 uptake measurement (`app/coach/uptake.py`).

What these defend is not the arithmetic but the direction of every error. The
metric bounds P1 from above, so the dangerous bugs are the ones that make the
number LOOK BETTER than reality: a shrinking denominator, an empty suggestion
matching everything, a match counted before the interviewer could have seen the
suggestion. Each has a test aimed at it specifically.

What they cannot prove is stated in the module docstring and restated here: no
test in this file establishes that a match means the coach CAUSED the question.
That distinction is not available from the data at all, at any threshold.
"""

from __future__ import annotations

import pytest

from app.coach.uptake import DEFAULT_THRESHOLD, best_match, uptake

SUGGESTION = "Tell me about the last time you needed something from outside the game"


def _segment(seq, speaker, text, end_s):
    return {"kind": "segment", "seq": seq, "speaker": speaker,
            "text": text, "audio_end_s": end_s}


def _suggestion(seq, text, trigger_seq, *, shown=True):
    return {"kind": "suggestion", "seq": seq, "text": text,
            "trigger_seq": trigger_seq, "shown": shown}


# --- the denominator, which must never shrink -----------------------------------


def test_a_suggestion_nobody_asked_counts_against_the_rate():
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
    ]
    result = uptake(records)
    assert result.shown == 1
    assert result.taken == 0
    assert result.rate == 0.0


def test_a_suggestion_with_no_trigger_segment_is_unmatched_not_dropped():
    """Dropping it would shrink the denominator, which RAISES the rate -- the
    one direction this metric must never move by accident."""
    records = [_suggestion(2, SUGGESTION, trigger_seq=999)]
    result = uptake(records)
    assert result.shown == 1
    assert result.taken == 0


def test_unshown_candidates_are_excluded_entirely():
    """A candidate the critic dropped was never in front of the interviewer, so
    it would deflate the rate with material P1 makes no claim about."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1, shown=False),
    ]
    assert uptake(records).shown == 0


def test_no_suggestions_shown_is_none_rather_than_zero():
    """An interview where the coach proposed nothing is not one where the
    interviewer ignored it. A 0.0 would put those in the same column."""
    result = uptake([_segment(1, "interviewee", "hello", 1.0)])
    assert result.shown == 0
    assert result.rate is None


# --- matching --------------------------------------------------------------------


def test_the_interviewer_asking_it_almost_verbatim_counts():
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer",
                 "tell me about the last time you needed something from outside the game", 20.0),
    ]
    result = uptake(records)
    assert result.taken == 1
    assert result.rate == 1.0
    assert result.matches[0].matched_seqs == (3,)


def test_a_question_split_across_fragments_still_matches():
    """The STT emits ~5-word finalizations, so a whole question routinely spans
    several segments. Comparing against one fragment at a time would read ~0%
    on a coach that was working perfectly -- the defect that would have made
    this whole measurement useless."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer", "tell me about the last", 20.0),
        _segment(4, "interviewer", "time you needed something", 22.0),
        _segment(5, "interviewer", "from outside the game", 24.0),
    ]
    result = uptake(records)
    assert result.taken == 1
    assert result.matches[0].matched_seqs == (3, 4, 5)


def test_an_unrelated_question_does_not_match():
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer", "how much would you expect to pay for this", 20.0),
    ]
    assert uptake(records).taken == 0


def test_the_interviewees_own_words_never_count():
    """Only the interviewer can take a suggestion. Matching interviewee speech
    would score uptake on the customer repeating the question back."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewee",
                 "tell me about the last time you needed something from outside the game", 20.0),
    ]
    assert uptake(records).taken == 0


# --- the window ------------------------------------------------------------------


def test_speech_after_the_window_does_not_count():
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer",
                 "tell me about the last time you needed something from outside the game", 200.0),
    ]
    assert uptake(records, window_s=90.0).taken == 0


def test_speech_before_the_suggestion_never_counts():
    """The interviewer cannot have taken advice they had not been given. This
    is the false-positive direction, so it is closed by construction rather
    than by threshold."""
    records = [
        _segment(1, "interviewer",
                 "tell me about the last time you needed something from outside the game", 5.0),
        _segment(2, "interviewee", "we went back to paper", 10.0),
        _suggestion(3, SUGGESTION, 2),
    ]
    assert uptake(records).taken == 0


def test_min_delay_excludes_speech_the_interviewer_could_not_yet_have_seen():
    """A suggestion is written after its cycle completes, so the earliest part
    of the window can hold speech produced before it ever appeared. The knob
    trades that false positive against a false negative; it defaults to 0.0
    because a metric that only bounds from above must not be quietly tightened
    in the flattering direction."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer",
                 "tell me about the last time you needed something from outside the game", 10.5),
    ]
    assert uptake(records, min_delay_s=0.0).taken == 1
    assert uptake(records, min_delay_s=2.0).taken == 0


# --- the flattering bugs ---------------------------------------------------------


def test_an_empty_suggestion_matches_nothing():
    """An empty suggestion matching everything would be the single most
    flattering possible bug in this module.

    Also passes with the explicit guard removed, because `similarity` already
    answers 0.0 for an empty left side -- verified by mutation. Pinned anyway:
    the property must survive a future change to `similarity`, and this is the
    test that would notice."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, "   ", 1),
        _segment(3, "interviewer", "tell me about the last time", 20.0),
    ]
    assert uptake(records).taken == 0


def test_a_long_unrelated_stretch_cannot_accumulate_a_match():
    """A talkative interviewer must not score uptake by sheer volume.

    This passes with `max_span_ratio` removed entirely -- verified by mutation.
    The property is real but it comes from `SequenceMatcher.ratio()` dividing by
    the total length of both sequences, not from the span cap, which is a cost
    bound. The test is kept because the PROPERTY is what matters and it must
    hold however it is delivered; the module docstring records which mechanism
    actually delivers it."""
    filler = [
        _segment(10 + i, "interviewer",
                 "so anyway the thing about that is you needed the last time outside", 20.0 + i)
        for i in range(40)
    ]
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        *filler,
    ]
    assert uptake(records).taken == 0


def test_the_threshold_is_actually_applied():
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer", "tell me about the last time you needed a thing", 20.0),
    ]
    assert uptake(records, threshold=0.99).taken == 0
    assert uptake(records, threshold=0.3).taken == 1


def test_best_match_reports_its_score_even_when_it_misses():
    """A near miss is the case a human most wants to adjudicate. Reporting 0.0
    for everything below threshold would hide 0.59 and 0.01 in one bucket."""
    score, seqs = best_match(
        SUGGESTION,
        [_segment(3, "interviewer", "tell me about the last time you needed a thing", 20.0)],
        threshold=0.99,
    )
    assert seqs == ()
    assert score > 0.5


def test_no_text_is_ever_returned():
    """Sequence numbers only, exactly as `interview_summary.FactRef` avoids
    carrying `quoted_text`. A reporting layer fetches the words itself."""
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _segment(3, "interviewer", SUGGESTION, 20.0),
    ]
    result = uptake(records)
    blob = repr(result)
    assert "paper" not in blob
    assert "outside the game" not in blob


@pytest.mark.parametrize("threshold", [DEFAULT_THRESHOLD, 0.5, 0.8])
def test_the_rate_never_exceeds_one(threshold):
    records = [
        _segment(1, "interviewee", "we went back to paper", 10.0),
        _suggestion(2, SUGGESTION, 1),
        _suggestion(3, SUGGESTION, 1),
        _segment(4, "interviewer", SUGGESTION, 20.0),
    ]
    result = uptake(records, threshold=threshold)
    assert result.rate is not None and 0.0 <= result.rate <= 1.0
