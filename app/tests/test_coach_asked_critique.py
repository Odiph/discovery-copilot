"""Tests for banding the interviewer's own turns (`app/coach/asked_critique.py`).

The one thing this layer must never do is let "the critic could not be reached"
and "the critic said clean" look the same. A post-call score is read as a
verdict on the interviewer, and a batch that silently failed would flatter them
by removing the questions most likely to have been bad -- there is no reason a
failure correlates with a clean band, but there is no reason it correlates with
a violation either, and a missing row is invisible where a `None` is not.

That is the same rule `benchmark/eval/judge_critic.judge_all` already follows,
for the same reason.
"""

from __future__ import annotations

import pytest

from app.coach.asked import InterviewerTurn
from app.coach.asked_critique import (
    MOVE_ASKED,
    critique_turns,
    to_json_dict,
)
from app.coach.critic import BAND_CLEAN, BAND_CLEAR_VIOLATION, CritiqueFailure, Judgment
from app.coach.generate import ALL_MOVES


def _turn(index, text, *, seqs=None, at_s=10.0):
    return InterviewerTurn(
        index=index, seqs=seqs or (index,), text=text, at_s=at_s
    )


def _judgment(index, *, band=BAND_CLEAN, leading=False):
    return Judgment(
        index=index,
        is_leading=leading,
        is_hypothetical_or_future_tense=False,
        is_pitch=False,
        is_opinion_seeking=False,
        is_confirmation_only=False, is_generic=False, is_off_topic=False,
        band=band,
        rationale="because",
    )


class _Critic:
    """Stands in for `critic.run_critique`, recording what it was asked."""

    def __init__(self, *, responses=None, fail_on=None):
        self.batches = []
        self.trigger_texts = []
        self._responses = responses or {}
        self._fail_on = fail_on or set()

    def __call__(self, client, *, candidates, hypotheses, trigger_text, model, max_tokens, timeout_s):
        texts = tuple(c.text for c in candidates)
        self.batches.append(candidates)
        # Recorded so a test can assert this path passes an EMPTY trigger --
        # a batch spans turns from different points in the interview, so
        # there is no single utterance `is_generic` could be judged against.
        self.trigger_texts.append(trigger_text)
        if len(self.batches) - 1 in self._fail_on:
            raise CritiqueFailure(reason="timeout", stop_reason=None)
        return self._responses.get(
            texts, [_judgment(c.index) for c in candidates]
        )


def _run(turns, critic_fn, **kwargs):
    return critique_turns(
        object(),
        turns,
        hypotheses=(),
        model="test-model",
        run_critique=critic_fn,
        **kwargs,
    )


# --- a failed batch is recorded, never dropped ---------------------------------


def test_a_failed_batch_leaves_its_turns_unjudged_not_absent():
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(4))
    result = _run(turns, _Critic(fail_on={0}), batch_size=4)
    assert len(result.judged) == 4
    assert all(j.judgment is None for j in result.judged)
    assert all(j.failure_reason == "timeout" for j in result.judged)


def test_a_failed_batch_does_not_stop_the_others():
    """One unreachable call must not cost the whole interview."""
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(8))
    result = _run(turns, _Critic(fail_on={0}), batch_size=4)
    assert [j.judgment is None for j in result.judged] == [True] * 4 + [False] * 4


def test_unjudged_turns_are_excluded_from_the_rate_not_counted_as_clean():
    """Counting an unreachable turn as clean is the single most flattering
    possible bug in this file."""
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(8))
    result = _run(turns, _Critic(fail_on={0}), batch_size=4)
    assert result.judged_count == 4
    assert result.unjudged_count == 4
    assert result.clean_count == 4


def test_the_rate_is_none_when_nothing_could_be_judged():
    """An interview the critic could not reach is not an interview with a
    perfect score, and a 0.0 or 1.0 would put those in the same column."""
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(4))
    result = _run(turns, _Critic(fail_on={0}), batch_size=4)
    assert result.clean_rate is None


def test_no_turns_at_all_yields_no_calls_and_no_rate():
    critic_fn = _Critic()
    result = _run((), critic_fn, batch_size=4)
    assert critic_fn.batches == []
    assert result.clean_rate is None


# --- the index mapping, which puts a band on a question -------------------------


def test_indices_are_per_batch_and_mapped_back_to_the_right_turn():
    """`Candidate.index` is per-BATCH, matching the live path. Mapping it back
    by anything other than position would put batch 2's first band on batch
    1's first question."""
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(6))
    critic_fn = _Critic()
    result = _run(turns, critic_fn, batch_size=4)
    assert [c.index for c in critic_fn.batches[0]] == [0, 1, 2, 3]
    assert [c.index for c in critic_fn.batches[1]] == [0, 1]
    assert [j.turn.index for j in result.judged] == [0, 1, 2, 3, 4, 5]


def test_a_judgment_for_an_index_the_batch_never_sent_is_ignored():
    """A response is only trusted for the candidates that were actually in the
    request; an out-of-range index would otherwise write onto a neighbour."""
    turns = (_turn(0, "what happens when you start reading?"),)
    critic_fn = _Critic(responses={
        ("what happens when you start reading?",): [_judgment(7, band=BAND_CLEAR_VIOLATION)],
    })
    result = _run(turns, critic_fn, batch_size=4)
    assert result.judged[0].judgment is None


# --- what the critic is told ----------------------------------------------------


def test_an_asked_question_carries_no_fabricated_move():
    """The interviewer never declared a move. Borrowing one from `ALL_MOVES`
    would tell the critic something about intent that nobody said, and the
    move is rendered verbatim into the prompt."""
    turns = (_turn(0, "would you pay for something like this?"),)
    critic_fn = _Critic()
    _run(turns, critic_fn, batch_size=4)
    assert critic_fn.batches[0][0].move == MOVE_ASKED
    assert MOVE_ASKED not in ALL_MOVES


def test_the_turn_text_reaches_the_critic_verbatim():
    turns = (_turn(0, "So you're protecting the playtime from the work."),)
    critic_fn = _Critic()
    _run(turns, critic_fn, batch_size=4)
    assert critic_fn.batches[0][0].text == "So you're protecting the playtime from the work."


def test_no_asked_question_targets_a_hypothesis():
    turns = (_turn(0, "would you pay for something like this?"),)
    critic_fn = _Critic()
    _run(turns, critic_fn, batch_size=4)
    assert critic_fn.batches[0][0].target_hypothesis_id is None


# --- serialisation --------------------------------------------------------------


def test_the_json_carries_seqs_so_the_words_can_be_found_again():
    turns = (_turn(0, "what happens when you start reading?", seqs=(93, 95), at_s=116.64),)
    result = _run(turns, _Critic(), batch_size=4)
    payload = to_json_dict(result, session_id="20260813T213559")
    assert payload["session_id"] == "20260813T213559"
    assert payload["turns"][0]["seqs"] == [93, 95]
    assert payload["turns"][0]["at_s"] == 116.64


def test_an_unjudged_turn_serialises_as_null_with_its_reason():
    turns = (_turn(0, "what happens when you start reading?"),)
    result = _run(turns, _Critic(fail_on={0}), batch_size=4)
    payload = to_json_dict(result, session_id="s")
    assert payload["turns"][0]["judgment"] is None
    assert payload["turns"][0]["failure_reason"] == "timeout"
    assert payload["summary"]["unjudged"] == 1


def test_the_summary_rate_matches_the_counts():
    turns = tuple(_turn(i, f"question number {i} here?") for i in range(4))
    critic_fn = _Critic(responses={
        tuple(f"question number {i} here?" for i in range(4)): [
            _judgment(0, band=BAND_CLEAN),
            _judgment(1, band=BAND_CLEAR_VIOLATION, leading=True),
            _judgment(2, band=BAND_CLEAN),
            _judgment(3, band=BAND_CLEAR_VIOLATION, leading=True),
        ],
    })
    result = _run(turns, critic_fn, batch_size=4)
    payload = to_json_dict(result, session_id="s")
    assert payload["summary"]["judged"] == 4
    assert payload["summary"]["clean"] == 2
    assert payload["summary"]["clean_rate"] == 0.5
    assert payload["summary"]["leading"] == 2


# --- the split that stops one number answering two questions -----------------------------


def _critique(pairs):
    """`pairs` is (text, band) -- built through `AskedCritique` directly so the
    rates are exercised without a critic round trip."""
    from app.coach.asked_critique import AskedCritique, JudgedTurn

    return AskedCritique(
        judged=tuple(
            JudgedTurn(
                turn=_turn(index, text),
                judgment=None if band is None else _judgment(index, band=band),
            )
            for index, (text, band) in enumerate(pairs)
        )
    )


def test_a_marked_question_and_an_unmarked_turn_score_separately():
    critique = _critique([
        ("What did you do next?", BAND_CLEAN),
        ("And then what happened?", BAND_CLEAN),
        ("So you're protecting the playtime from the work.", BAND_CLEAR_VIOLATION),
        ("That sounds really frustrating for the whole team.", BAND_CLEAR_VIOLATION),
    ])
    assert critique.marked_question_clean_rate == 1.0
    assert critique.unmarked_clean_rate == 0.0
    # The combined rate is the thing that was hard to read: it reports 50% and
    # says nothing true about either population.
    assert critique.clean_rate == 0.5


def test_the_unmarked_bucket_still_contains_the_leading_statements():
    """The point of splitting rather than filtering. A `?` filter would have
    dropped this turn, and a leading statement is the most damaging rubric
    violation there is."""
    critique = _critique([
        ("What did you do next?", BAND_CLEAN),
        ("So you're protecting the playtime from the work.", BAND_CLEAR_VIOLATION),
    ])
    assert critique.judged_count == 2, "a turn was filtered out instead of banded"
    texts = [item.turn.text for item in critique.judged]
    assert "So you're protecting the playtime from the work." in texts


def test_an_unreachable_turn_is_excluded_from_both_rates_not_counted_clean():
    """The module's founding rule, applied to the new rates: `None` must not
    become a denominator entry that reads as a pass."""
    critique = _critique([
        ("What did you do next?", BAND_CLEAN),
        ("And after that?", None),
        ("You must have been annoyed.", None),
    ])
    assert critique.marked_question_clean_rate == 1.0
    assert critique.unmarked_clean_rate is None
    assert critique.unjudged_count == 2


@pytest.mark.parametrize("pairs", [[], [("And after that?", None)]])
def test_a_population_with_nothing_judged_rates_none_rather_than_zero(pairs):
    critique = _critique(pairs)
    assert critique.marked_question_clean_rate is None
    assert critique.unmarked_clean_rate is None


def test_both_rates_reach_the_durable_artifact():
    critique = _critique([
        ("What did you do next?", BAND_CLEAN),
        ("You must have been annoyed.", BAND_CLEAR_VIOLATION),
    ])
    summary = to_json_dict(critique, session_id="s1")["summary"]
    assert summary["marked_questions"] == 1
    assert summary["marked_question_clean_rate"] == 1.0
    assert summary["unmarked_clean_rate"] == 0.0
    # The combined rate stays -- removing it would break every existing reader
    # of this artifact to fix a readability problem.
    assert summary["clean_rate"] == 0.5


def test_a_question_mark_anywhere_in_a_joined_turn_marks_it():
    """Turns are joined from STT fragments, and on a split question the `?`
    lands on the last fragment only -- so the test must be over the joined
    text, never the first fragment."""
    assert _turn(0, "And after that? Walk me through it.").is_marked_question
    assert _turn(0, "So you're protecting the playtime.").is_marked_question is False
