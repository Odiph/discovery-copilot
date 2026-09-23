"""Tests for interviewer-turn reconstruction (`app/coach/asked.py`).

What these defend is the UNIT OF JUDGEMENT. The critic bands a question; if the
thing handed to it is not what the interviewer actually asked, every band it
returns is about text nobody said. Two errors do that, and they fail in
opposite directions:

  * SPLITTING one question across two turns hands the critic half a question.
    "Walk me through the last time you sat" is unimpeachable on its own -- the
    leading half is in the fragment that got separated from it. This direction
    manufactures CLEAN bands.
  * MERGING two questions into one turn hands the critic something nobody
    asked, and the band belongs to neither.

Both are invisible in the output -- a band is a band -- which is why they are
tested here at the reconstruction layer rather than downstream.

The fragment shapes below are taken from the real interview
`20260813T213559` (102 interviewer segments, 72 turns, 23 of them
multi-segment), not invented: seq 20/21 really is one question split by the
STT, and `'Right.'` really is a whole turn.
"""

from __future__ import annotations

from app.coach.asked import (
    DEFAULT_MIN_WORDS,
    INTERVIEWER,
    interviewer_turns,
)


def _segment(seq, speaker, text, end_s):
    return {"kind": "segment", "seq": seq, "speaker": speaker,
            "text": text, "audio_end_s": end_s}


# --- the split direction: never hand the critic half a question -----------------


def test_a_question_split_across_fragments_is_one_turn():
    """Verbatim from the real interview, seq 20/21. The STT ended a segment
    mid-clause; the two halves are one question and must be judged as one."""
    records = [
        _segment(20, INTERVIEWER, "Walk me through the last time you sat", 26.64),
        _segment(21, INTERVIEWER,
                 "down to play, from turning the machine onto the game actually starting.",
                 30.96),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert turns[0].text == (
        "Walk me through the last time you sat down to play, from turning the "
        "machine onto the game actually starting."
    )
    assert turns[0].seqs == (20, 21)


def test_a_within_turn_gap_does_not_split_it():
    """The largest within-run gap measured across the whole real corpus is
    5.94s, and 0 of 30 exceeded 8s -- so a gap rule would be inert on every
    session that exists. A turn ends when the other speaker talks, which is
    the same definition `TurnTrigger` already uses. This pins that: a 5.9s
    pause mid-question must not become two questions."""
    records = [
        _segment(20, INTERVIEWER, "Walk me through the last time you sat", 26.64),
        _segment(21, INTERVIEWER, "down to play, from turning the machine on", 32.58),
    ]
    assert len(interviewer_turns(records)) == 1


# --- the merge direction: never invent a question ------------------------------


def test_the_interviewee_speaking_ends_the_turn():
    records = [
        _segment(1, INTERVIEWER, "How long is that wait, usually?", 71.4),
        _segment(2, "interviewee", "about ten minutes, maybe more", 80.0),
        _segment(3, INTERVIEWER, "What do you do during it?", 90.12),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 2
    assert turns[0].text == "How long is that wait, usually?"
    assert turns[1].text == "What do you do during it?"


def test_interviewee_speech_never_enters_a_turn():
    """The whole point is judging what the INTERVIEWER asked. Leaking the
    answer into the question would let the interviewee's words earn the
    interviewer a band."""
    records = [
        _segment(1, INTERVIEWER, "What happens when you do start reading?", 116.64),
        _segment(2, "interviewee", "I lose the thread of the game entirely", 130.0),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert "lose the thread" not in turns[0].text


# --- what is worth judging at all ----------------------------------------------


def test_a_bare_acknowledgement_is_not_a_turn_worth_judging():
    """`'Right.'` is a whole interviewer turn in the real interview. Banding it
    spends a call to learn nothing, and a corpus padded with them makes the
    clean rate look better than the interviewer's actual questioning."""
    records = [
        _segment(1, INTERVIEWER, "Right.", 150.0),
        _segment(2, "interviewee", "so I stopped bothering", 160.0),
        _segment(3, INTERVIEWER, "What happens when you do start reading?", 170.0),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert turns[0].text == "What happens when you do start reading?"


def test_a_leading_statement_is_judged_even_though_it_is_not_a_question():
    """`"So you're protecting the playtime from the work."` is a real
    interviewer turn with no question mark. It is a textbook leading
    reflection -- exactly what the rubric exists to catch -- so filtering on
    `?` would discard the most incriminating material in the transcript.

    Only about half the interviewer segments in the real interview carry a `?`
    at all, and on a split question it lands on the last fragment only."""
    records = [
        _segment(1, INTERVIEWER, "So you're protecting the playtime from the work.", 153.6),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert "?" not in turns[0].text


def test_the_word_floor_counts_the_joined_turn_not_the_fragment():
    """Each fragment here is under the floor; the question they form is not.
    Applying the floor per-segment would drop real questions wholesale --
    the median real fragment is 7 words and the floor is well under that,
    so this only bites on genuinely split short questions."""
    records = [
        _segment(1, INTERVIEWER, "How long", 47.2),
        _segment(2, INTERVIEWER, "is that", 47.4),
        _segment(3, INTERVIEWER, "wait, usually?", 47.6),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert turns[0].text == "How long is that wait, usually?"


def test_the_floor_is_configurable_and_actually_applied():
    records = [_segment(1, INTERVIEWER, "one two three", 10.0)]
    assert interviewer_turns(records, min_words=3) != ()
    assert interviewer_turns(records, min_words=4) == ()


def test_a_short_follow_up_question_survives_the_word_floor():
    """All four of these are verbatim turns from the real interview that the
    word floor ALONE discarded. They are the good kind of question -- short,
    unleading, purely probing -- and losing them would bias the resulting
    score against exactly the skill being measured.

    The floor was checked against invented fixtures first and every one of
    them passed. The real transcript is what exposed this."""
    for text in ("And after that?", "And then?", "Between rounds specifically?", "Which two?"):
        records = [_segment(1, INTERVIEWER, text, 10.0)]
        assert interviewer_turns(records) != (), text


def test_a_short_nudge_with_no_question_mark_is_still_dropped():
    """`'Go on.'` was the fifth turn the floor dropped, and the only one it was
    right about. The question-mark rescue must not become a rescue of
    everything."""
    records = [_segment(1, INTERVIEWER, "Go on.", 10.0)]
    assert interviewer_turns(records) == ()


def test_the_question_mark_rescues_but_never_excludes():
    """The asymmetry, stated as a test. A long turn with no `?` is kept on
    length; a short turn with `?` is kept on punctuation. Neither rule can
    veto the other, because a `?` filter would discard leading statements --
    the most incriminating material there is."""
    long_statement = _segment(1, INTERVIEWER,
                              "So you're protecting the playtime from the work.", 10.0)
    short_question = _segment(1, INTERVIEWER, "Which two?", 10.0)
    assert interviewer_turns([long_statement]) != ()
    assert interviewer_turns([short_question]) != ()


# --- timing, ordering, indices --------------------------------------------------


def test_the_turn_is_stamped_from_audio_end_s():
    """`audio_start_s` is FABRICATED -- `app/stt/live.py` sets it to the same
    speaker's previous `audio_end_s`, so any time derived from it is fiction.
    The stamp is the end of the turn's last fragment."""
    records = [
        _segment(20, INTERVIEWER, "Walk me through the last time you sat", 26.64),
        _segment(21, INTERVIEWER, "down to play, from turning the machine on", 30.96),
    ]
    assert interviewer_turns(records)[0].at_s == 30.96


def test_records_are_ordered_by_seq_not_by_arrival():
    """The two speakers finalize on separate STT threads, so file order is not
    speech order. Reconstructing in arrival order would join fragments that
    were never adjacent."""
    records = [
        _segment(21, INTERVIEWER, "down to play.", 30.96),
        _segment(20, INTERVIEWER, "Walk me through the last time you sat", 26.64),
    ]
    turns = interviewer_turns(records)
    assert turns[0].text == "Walk me through the last time you sat down to play."


def test_indices_are_contiguous_over_the_kept_turns():
    """The index is what a judgment is mapped back through. A gap in it means
    a band lands on the wrong question."""
    records = [
        _segment(1, INTERVIEWER, "What happens when you do start reading?", 10.0),
        _segment(2, "interviewee", "answer", 20.0),
        _segment(3, INTERVIEWER, "Right.", 30.0),
        _segment(4, "interviewee", "answer", 40.0),
        _segment(5, INTERVIEWER, "What do you do during it?", 50.0),
    ]
    turns = interviewer_turns(records)
    assert [turn.index for turn in turns] == [0, 1]


def test_non_segment_records_are_ignored():
    records = [
        {"kind": "suggestion", "seq": 1, "text": "Tell me about the last time", "shown": True},
        {"kind": "event", "seq": 2, "speaker": INTERVIEWER, "text": "coverage"},
        _segment(3, INTERVIEWER, "What do you do during it?", 50.0),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert turns[0].seqs == (3,)


def test_no_interviewer_speech_yields_nothing():
    records = [_segment(1, "interviewee", "I lose the thread entirely", 10.0)]
    assert interviewer_turns(records) == ()


def test_missing_text_does_not_raise():
    """The tree is on disk and a crashed write can leave anything there; one
    stray record must not make the whole command unusable."""
    records = [
        {"kind": "segment", "seq": 1, "speaker": INTERVIEWER, "audio_end_s": 10.0},
        _segment(2, INTERVIEWER, "What do you do during it?", 20.0),
    ]
    turns = interviewer_turns(records)
    assert len(turns) == 1
    assert turns[0].text == "What do you do during it?"


def test_default_floor_keeps_the_median_real_turn():
    """Median words per real interviewer turn is 9. A floor anywhere near that
    would silently discard most of the interview."""
    assert DEFAULT_MIN_WORDS < 9
