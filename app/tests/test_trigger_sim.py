"""Tests for the triggering-rule simulator (`benchmark/eval/trigger_sim.py`).

The simulator's first version reported 0% coverage for every rule at every
latency -- a broken denominator being polite about it -- because it built turn
starts from `audio_start_s`, which `app/stt/live.py` fabricates. These tests
exist so that class of failure is loud instead of plausible.
"""

from __future__ import annotations

from benchmark.eval.trigger_sim import (
    rule_every_segment, rule_min_interval, rule_turn_end, rule_turn_plus,
    simulate, turns_of,
)


def _seg(seq, speaker, end_s, start_s=None):
    record = {"kind": "segment", "seq": seq, "speaker": speaker, "audio_end_s": end_s}
    if start_s is not None:
        record["audio_start_s"] = start_s
    return record


def _conversation():
    """Interviewee talks 0-20s, interviewer 20-25s, interviewee 25-45s."""
    return [
        _seg(0, "interviewee", 10.0), _seg(1, "interviewee", 20.0),
        _seg(2, "interviewer", 25.0),
        _seg(3, "interviewee", 35.0), _seg(4, "interviewee", 45.0),
    ]


def test_turns_group_consecutive_segments_by_speaker():
    turns = turns_of(_conversation())
    assert [t.speaker for t in turns] == ["interviewee", "interviewer", "interviewee"]
    assert turns[0].segments == (0, 1)


def test_a_turn_starts_where_the_previous_segment_ENDED():
    """Not at its own `audio_start_s`, which is fabricated as the end of the
    SAME SPEAKER's previous segment and therefore points back past the whole of
    the other speaker's turn. Turns built from it overlap and every
    'was it ready in time' comparison silently answers no."""
    records = [
        _seg(0, "interviewee", 20.0, start_s=0.0),
        _seg(1, "interviewer", 25.0, start_s=0.0),   # fabricated: points to 0.0
        _seg(2, "interviewee", 45.0, start_s=20.0),  # fabricated: skips the interviewer
    ]
    turns = turns_of(records)
    assert turns[1].start_s == 20.0, "the interviewer turn began when the interviewee stopped"
    assert turns[2].start_s == 25.0
    # And the timeline is strictly ordered, which the fabricated version was not.
    assert all(a.end_s <= b.start_s for a, b in zip(turns, turns[1:]))


# --- the rules -------------------------------------------------------------------


def test_every_segment_fires_only_on_the_interviewee():
    """The coach is hooked to the interviewee's stream; the interviewer's drives
    coverage instead. A rule that fired on both would model a product that does
    not exist."""
    assert rule_every_segment(_conversation(), turns_of(_conversation())) == [10.0, 20.0, 35.0, 45.0]


def test_turn_plus_skips_a_turn_too_short_to_contain_the_delay():
    """A four-second answer usually needs no coached follow-up, and firing
    anyway spends a cycle that cannot land in time."""
    records = [_seg(0, "interviewee", 2.0), _seg(1, "interviewer", 6.0),
               _seg(2, "interviewee", 30.0)]
    fired = rule_turn_plus(3.0)(records, turns_of(records))
    assert fired == [9.0], "only the long turn fires, 3s after it began"


def test_min_interval_thins_todays_triggers_without_reordering_them():
    records = _conversation()
    fired = rule_min_interval(15.0)(records, turns_of(records))
    assert fired == [10.0, 35.0]


# --- the queue and the scoring ---------------------------------------------------


def test_the_queue_drops_the_oldest_waiting_trigger():
    """`pipeline.submit`'s actual behaviour: one worker, one slot, and a full
    slot discards the OLDEST waiter. Modelling it as unbounded would make every
    rule look better than it is -- today's rule drops 183 of 184 on the real
    transcript, which is what saturation looks like."""
    records = [_seg(i, "interviewee", float(i)) for i in range(10)]
    result = simulate(records, rule_every_segment, name="x", cycle_s=5.0)
    assert result.dropped > 0
    assert result.cycles < 10


def test_turn_end_never_covers_a_turn_when_the_cycle_is_slow():
    """The result this file exists to establish. Firing when the interviewee
    stops means the suggestion lands INSIDE the interviewer's next turn, after
    the moment it was built for. It is the rule most people would pick."""
    result = simulate(_conversation(), rule_turn_end, name="turn-end", cycle_s=7.9)
    assert result.covered == 0


def test_turn_plus_covers_the_turn_the_slow_cycle_would_have_missed():
    result = simulate(_conversation(), rule_turn_plus(3.0), name="turn+3", cycle_s=7.9)
    assert result.covered >= 1


def test_a_suggestion_from_before_the_previous_answer_is_stale_not_covering():
    """Freshness is the whole point. A suggestion computed from speech the
    interviewer has already responded to is scrollback, not help, and counting
    it would let a rule that fires once at minute zero score full marks."""
    # TWO interviewer turns are required for staleness to be expressible at all:
    # the first is legitimately covered by an early suggestion, and only the
    # second can find that same suggestion out of date. The first version of
    # this test used a one-interviewer-turn fixture and asserted a property its
    # own data could not exhibit.
    records = _conversation() + [_seg(5, "interviewer", 50.0)]
    result = simulate(records, lambda r, t: [1.0], name="one-shot", cycle_s=2.0)
    assert result.turns == 2
    assert result.covered == 1, "the first turn is genuinely served"
    assert result.stale_only == 1, "the second finds only a suggestion it has moved past"


def test_coverage_is_a_fraction_of_interviewer_turns():
    result = simulate(_conversation(), rule_turn_plus(3.0), name="x", cycle_s=2.0)
    assert result.turns == 1
    assert 0.0 <= result.coverage <= 1.0
