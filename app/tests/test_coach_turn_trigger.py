"""`turn+3s` as a live gate, held to what `trigger_sim.py` actually simulated.

The simulation fired at `turn.start_s + 3.0`, where a turn's start is the END
of the previous segment BY EITHER SPEAKER. Live there is no such clock tick to
fire on -- the gate only wakes when a segment finalizes -- so the shipped rule
fires on the first interviewee segment that ENDS at least 3s into its turn.
That is a strictly later moment than the simulation's, and these tests pin the
difference rather than paper over it.

The fixture is built so the three plausible WRONG implementations each produce
a different answer from the right one:

  * turn start from `audio_start_s` (fabricated by `app/stt/live.py` as the
    previous SAME-speaker end) -- would fire on seg2, which is 2s into its turn;
  * `>= 3s` measured as the SEGMENT's own duration -- would miss seg3 entirely;
  * `>= 3s` measured since the last FIRE -- would fire on seg2, there being no
    previous fire to measure from.

Only the correct rule yields exactly {seg3, seg8}.
"""

from __future__ import annotations

import threading

import pytest

from app.coach import trigger as trigger_mod
from app.coach.trigger import DEFAULT_TURN_DELAY_S, TurnTrigger
from app.session.store import SegmentRecord

INTERVIEWER = "interviewer"
INTERVIEWEE = "interviewee"


def seg(seq: int, speaker: str, *, end: float, start: float) -> SegmentRecord:
    """`start` is passed explicitly and is meant to be WRONG.

    `audio_start_s` is fabricated in production; every fixture here sets it to
    a value that would change the outcome if the gate ever read it."""
    return SegmentRecord(
        kind="segment", seq=seq, speaker=speaker, text=f"s{seq}",
        audio_start_s=start, audio_end_s=end, ready_wall_s=end, latency_s=0.4,
        detected_language="en", engine="test", written_at="2026-08-14T00:00:00Z",
    )


# turn boundaries fall at the end of the previous segment, whoever spoke it:
#   seg1 interviewer  ..10.0     turn A
#   seg2 interviewee  10.0..12.0 turn B, 2.0s in  -> no
#   seg3 interviewee       ..14.5 turn B, 4.5s in -> FIRE
#   seg4 interviewee       ..17.0 turn B, already fired -> no
#   seg5 interviewer  17.0..20.0 turn C
#   seg6 interviewee  20.0..21.0 turn D, 1.0s in  -> no (turn ends short)
#   seg7 interviewer  21.0..22.0 turn E
#   seg8 interviewee  22.0..26.0 turn F, 4.0s in  -> FIRE
TIMELINE = [
    seg(1, INTERVIEWER, end=10.0, start=0.0),
    seg(2, INTERVIEWEE, end=12.0, start=5.0),
    seg(3, INTERVIEWEE, end=14.5, start=12.0),
    seg(4, INTERVIEWEE, end=17.0, start=14.5),
    seg(5, INTERVIEWER, end=20.0, start=10.0),
    seg(6, INTERVIEWEE, end=21.0, start=17.0),
    seg(7, INTERVIEWER, end=22.0, start=20.0),
    seg(8, INTERVIEWEE, end=26.0, start=21.0),
]


def run(records, **kwargs) -> "list[int]":
    """Drives the trigger with the word floor OFF unless a test asks for it.

    Every fixture above carries one-word text (`s3`), because these tests were
    written to pin the TIMING rule and the timeline is expressed in seconds.
    Letting the shipped floor apply here would make all of them assert the
    floor instead -- the same rule tested twice, and the delay rule tested
    nowhere. The floor has its own section at the bottom of this file, which
    includes one test that the shipped DEFAULT is actually in force."""
    kwargs.setdefault("min_turn_words", 0)
    fired: "list[int]" = []
    trigger = TurnTrigger(downstream=lambda r: fired.append(r.seq), **kwargs)
    for record in records:
        trigger.submit(record)
    return fired


def test_fires_once_per_turn_at_the_first_segment_past_the_delay():
    assert run(TIMELINE) == [3, 8]


def test_short_turn_never_fires():
    """seg6's whole turn is 1.0s. The simulation skipped such turns rather than
    spending a cycle that could not land in time; so does this."""
    assert 6 not in run(TIMELINE)


def test_turn_start_ignores_the_fabricated_audio_start_s():
    """seg2 sits 2.0s into its turn but claims `audio_start_s=5.0`, which would
    make it look 7.0s long. Reading that field fires here; reading segment ENDS
    does not."""
    assert 2 not in run(TIMELINE)


def test_delay_is_measured_into_the_turn_not_since_the_last_fire():
    """With no previous fire, a since-last-fire rule has nothing to wait for and
    fires on the turn's first segment."""
    fired = run(TIMELINE)
    assert fired[0] == 3, "first fire must be the first segment 3s INTO turn B"


def test_delay_is_not_the_segments_own_duration():
    """seg3 runs 12.0->14.5, only 2.5s of speech, but lands 4.5s into its turn."""
    assert 3 in run(TIMELINE)


def test_interviewer_segments_never_reach_the_downstream():
    assert not (set(run(TIMELINE)) & {1, 5, 7})


def test_a_new_interviewee_turn_re_arms_after_the_interviewer_speaks():
    """Turn B fired; turn F must be able to fire again. Without re-arming, the
    coach goes silent for the rest of the interview -- a failure that a
    fires-at-most-once bug would otherwise pass every other test here."""
    assert run(TIMELINE) == [3, 8]


def test_consecutive_interviewee_segments_after_a_fire_are_suppressed():
    assert run(TIMELINE).count(3) == 1
    assert 4 not in run(TIMELINE)


def test_delay_is_configurable_and_changes_the_answer():
    """A guard that reports the same thing at every setting is not measuring the
    setting. At 5s, seg3 (4.5s in) no longer qualifies and seg4 (7.0s in) does."""
    assert run(TIMELINE, delay_s=5.0) == [4]
    assert DEFAULT_TURN_DELAY_S == 3.0


def test_downstream_failure_never_propagates_to_the_stt_thread():
    """`StreamWorker` wraps the hook, but this gate sits BETWEEN the worker and
    the pipeline and must not rely on that: a raising downstream must not stop
    later turns from firing."""
    seen: "list[int]" = []

    def downstream(record):
        seen.append(record.seq)
        raise RuntimeError("downstream is down")

    trigger = TurnTrigger(downstream=downstream, min_turn_words=0)
    for record in TIMELINE:
        trigger.submit(record)  # must not raise
    assert seen == [3, 8], "a failed fire must not disarm the following turns"


def test_out_of_order_arrival_across_the_two_stt_threads_does_not_invert_a_turn():
    """The two speakers finalize on separate threads, so a segment can arrive
    after one that ends later. The turn's start must not be dragged forward by
    a late arrival from before it."""
    late = seg(9, INTERVIEWER, end=9.0, start=0.0)
    records = [TIMELINE[0], late] + TIMELINE[1:]
    assert run(records) == [3, 8]


def test_concurrent_submits_do_not_double_fire_one_turn():
    fired: "list[int]" = []
    lock = threading.Lock()

    def downstream(record):
        with lock:
            fired.append(record.seq)

    trigger = TurnTrigger(downstream=downstream, min_turn_words=0)
    trigger.submit(TIMELINE[0])
    trigger.submit(TIMELINE[1])
    threads = [threading.Thread(target=trigger.submit, args=(TIMELINE[2],)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fired == [3], f"one turn fired {len(fired)} times"


@pytest.mark.parametrize("speaker", [INTERVIEWEE, INTERVIEWER])
def test_an_interview_that_opens_with_either_speaker_starts_its_first_turn_at_zero(speaker):
    first = seg(1, speaker, end=4.0, start=0.0)
    follow = seg(2, INTERVIEWEE, end=8.0, start=4.0)
    fired = run([first, follow])
    # Opening interviewee turn starts at 0.0, so seg1 is already 4.0s in.
    assert fired == ([1] if speaker == INTERVIEWEE else [2])


# --- the word floor: don't spend a cycle on a turn that has not said anything ------------
#
# Added after the first live coached interview, where the coach fired on turns
# reading 'Um, I guess' and 'So, uh, I' and the critic then flagged the
# resulting questions `is_generic` -- correctly, since a question cannot engage
# with an utterance that carries no content.


def worded(seq: int, speaker: str, *, end: float, text: str) -> SegmentRecord:
    return SegmentRecord(
        kind="segment", seq=seq, speaker=speaker, text=text,
        audio_start_s=0.0, audio_end_s=end, ready_wall_s=end, latency_s=0.4,
        detected_language="en", engine="test", written_at="2026-08-14T00:00:00Z",
    )


def test_a_turn_that_is_long_enough_but_says_nothing_does_not_fire():
    records = [
        worded(1, INTERVIEWER, end=10.0, text="so tell me about that"),
        # 5s into the turn -- comfortably past the delay -- and three words in.
        worded(2, INTERVIEWEE, end=15.0, text="Um, I guess"),
    ]
    assert run(records, min_turn_words=12) == []


def test_the_same_turn_fires_later_once_it_actually_says_something():
    """The floor DELAYS, it does not veto the turn. This is the behaviour the
    whole change is for: the live run's 'Um, I guess' turn went on to say
    something substantive, and the cycle should land there instead."""
    records = [
        worded(1, INTERVIEWER, end=10.0, text="so tell me about that"),
        worded(2, INTERVIEWEE, end=15.0, text="Um, I guess"),
        worded(3, INTERVIEWEE, end=18.0, text="a patch drops and you cannot tell whether "
                                              "your saved loadout still works"),
    ]
    assert run(records, min_turn_words=12) == [3]


def test_the_floor_counts_the_WHOLE_TURN_not_the_triggering_segment():
    """A segment is a ~5-word engine finalization. Counting per segment would
    stall a genuinely long answer that happened to be split small -- the exact
    opposite of the intent."""
    records = [
        worded(1, INTERVIEWER, end=10.0, text="go on"),
        worded(2, INTERVIEWEE, end=11.0, text="we onboard a new customer"),
        worded(3, INTERVIEWEE, end=12.0, text="in about three weeks"),
        # This segment is three words, but the TURN is now thirteen.
        worded(4, INTERVIEWEE, end=15.0, text="mostly by email"),
    ]
    assert run(records, min_turn_words=12) == [4]


def test_the_words_that_carry_the_turn_over_the_floor_can_themselves_fire():
    """Counted before the gate, so the segment that clears the floor is
    eligible immediately. Counting after would make every turn wait one extra
    segment for nothing."""
    records = [
        worded(1, INTERVIEWER, end=10.0, text="go on"),
        worded(2, INTERVIEWEE, end=15.0, text="one two three four five six seven eight"),
    ]
    assert run(records, min_turn_words=8) == [2]
    assert run(records, min_turn_words=9) == []


def test_the_floor_resets_between_turns():
    """Words do not carry across a speaker change -- otherwise a long first
    answer would leave every later one-word turn permanently over the floor."""
    records = [
        worded(1, INTERVIEWER, end=5.0, text="go on"),
        worded(2, INTERVIEWEE, end=10.0,
               text="one two three four five six seven eight nine ten eleven twelve"),
        worded(3, INTERVIEWER, end=12.0, text="and then?"),
        worded(4, INTERVIEWEE, end=17.0, text="Maybe"),
    ]
    assert run(records, min_turn_words=12) == [2]


def test_a_floor_of_zero_restores_the_pre_floor_behaviour_exactly():
    records = [
        worded(1, INTERVIEWER, end=10.0, text="go on"),
        worded(2, INTERVIEWEE, end=15.0, text="Maybe"),
    ]
    assert run(records, min_turn_words=0) == [2]


def test_the_shipped_default_floor_is_actually_applied():
    """`run` disables the floor for the timing fixtures, so without this the
    default could be anything and every test here would still pass."""
    fired: "list[int]" = []
    trigger = TurnTrigger(downstream=lambda r: fired.append(r.seq))
    for record in [
        worded(1, INTERVIEWER, end=10.0, text="go on"),
        worded(2, INTERVIEWEE, end=15.0, text="Um, I guess"),
    ]:
        trigger.submit(record)
    assert fired == [], (
        f"a three-word turn fired under the shipped default "
        f"({trigger_mod.DEFAULT_MIN_TURN_WORDS})"
    )
