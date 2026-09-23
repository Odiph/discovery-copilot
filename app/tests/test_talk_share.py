"""Tests for `app/session/talk_share.py`.

The arithmetic is trivial and is not what these guard. What they guard is the
set of decisions that make the number honest: that nothing is silently
dropped, that an empty transcript is an ABSENCE rather than a zero, and that
a silent speaker and an unmeasurable session stay distinguishable.
"""

from __future__ import annotations

import pytest

from app.session.talk_share import share_of, talk_share


def _segment(speaker, text):
    return {"kind": "segment", "speaker": speaker, "text": text}


# --- the absence case -----------------------------------------------------------


def test_no_segments_is_unmeasurable_not_fifty_fifty():
    """An interview with no speech must not render as a balanced conversation.
    A 50/50 default would be a claim about a conversation that never happened."""
    result = talk_share([])
    assert result["measurable"] is False
    assert result["total_words"] == 0
    assert result["speakers"] == {}


def test_segments_with_no_words_are_unmeasurable():
    result = talk_share([_segment("interviewer", ""), _segment("interviewee", "   ")])
    assert result["measurable"] is False


def test_the_unmeasurable_constant_cannot_be_mutated_by_a_caller():
    """`UNMEASURABLE` is module state and is returned on the commonest path.
    Handing out the shared dict would let one caller's mutation change what
    every later caller sees."""
    first = talk_share([])
    first["speakers"]["interviewer"] = {"words": 999, "share": 1.0}
    second = talk_share([])
    assert second["speakers"] == {}


# --- the counting ---------------------------------------------------------------


def test_shares_reflect_who_did_the_talking():
    result = talk_share([
        _segment("interviewer", "one two three four five six"),
        _segment("interviewee", "seven two three four"),
    ])
    assert result["measurable"] is True
    assert result["total_words"] == 10
    assert result["speakers"]["interviewer"]["words"] == 6
    assert result["speakers"]["interviewer"]["share"] == pytest.approx(0.6)
    assert result["speakers"]["interviewee"]["share"] == pytest.approx(0.4)


def test_a_speakers_segments_accumulate_across_the_interview():
    result = talk_share([
        _segment("interviewer", "a b"),
        _segment("interviewee", "c"),
        _segment("interviewer", "d e f"),
    ])
    assert result["speakers"]["interviewer"]["words"] == 5
    assert result["speakers"]["interviewee"]["words"] == 1


def test_runs_of_whitespace_do_not_inflate_the_count():
    """`split()` with no argument collapses runs; `split(' ')` would not, and
    would count every double space as an extra empty word -- inflating
    whichever speaker's transcript happens to be messier."""
    result = talk_share([_segment("interviewer", "  a   b \n c\t d  ")])
    assert result["speakers"]["interviewer"]["words"] == 4


def test_hebrew_text_is_counted_like_any_other_whitespace_delimited_language():
    """The corpus is English and Hebrew. A word counter that only worked on
    ASCII would under-count one of the two speakers systematically."""
    result = talk_share([
        _segment("interviewee", "שלום מה שלומך היום"),
        _segment("interviewer", "one two"),
    ])
    assert result["speakers"]["interviewee"]["words"] == 4
    assert result["speakers"]["interviewer"]["words"] == 2


def test_speakers_are_ordered_loudest_first():
    result = talk_share([
        _segment("interviewee", "a"),
        _segment("interviewer", "b c d"),
    ])
    assert list(result["speakers"]) == ["interviewer", "interviewee"]


def test_more_than_two_speakers_are_all_reported():
    """Nothing here assumes an interview has exactly two people in it. A
    hardcoded pair would drop a third speaker's words from the shares while
    leaving them in the total, or worse, drop them from both."""
    result = talk_share([
        _segment("interviewer", "a b"),
        _segment("interviewee", "c d"),
        _segment("observer", "e f"),
    ])
    assert set(result["speakers"]) == {"interviewer", "interviewee", "observer"}
    assert result["total_words"] == 6


# --- what must never be silently dropped ----------------------------------------


def test_unlabelled_speech_is_counted_not_dropped():
    """Dropping it would leave the shares summing to 1.0 while describing less
    speech than occurred -- the numbers would still LOOK right, which is what
    makes silent dropping the dangerous option rather than merely a lossy one."""
    result = talk_share([
        _segment("interviewer", "a b c"),
        _segment(None, "d e"),
    ])
    assert result["total_words"] == 5
    assert result["unattributed_words"] == 2
    assert "interviewer" in result["speakers"]
    assert result["speakers"]["interviewer"]["share"] == pytest.approx(0.6)


def test_shares_do_not_sum_to_one_when_speech_is_unattributed():
    """The visible symptom of a transcript with unlabelled speech in it.
    Normalising it away would hide the defect and make the arithmetic prettier."""
    result = talk_share([
        _segment("interviewer", "a b"),
        _segment("", "c d"),
    ])
    assert sum(entry["share"] for entry in result["speakers"].values()) < 1.0


@pytest.mark.parametrize("speaker", [None, "", "   ", 7, {"name": "x"}])
def test_an_unusable_speaker_becomes_unattributed_rather_than_a_bucket(speaker):
    """A non-string speaker must not become a dict key -- it would render as a
    speaker named `7` in the hub, which looks like a real participant."""
    result = talk_share([_segment(speaker, "a b")])
    assert result["speakers"] == {}
    assert result["unattributed_words"] == 2


@pytest.mark.parametrize("text", [None, 7, {"a": 1}, []])
def test_a_malformed_text_field_counts_zero_and_does_not_raise(text):
    """Called on whatever is on disk. One bad record must not take out the
    whole figure -- an exception here would blank the meter for the session."""
    result = talk_share([_segment("interviewer", text), _segment("interviewee", "a b")])
    assert result["measurable"] is True
    assert result["total_words"] == 2


def test_whitespace_around_a_speaker_name_does_not_split_them_in_two():
    result = talk_share([
        _segment("interviewer", "a b"),
        _segment(" interviewer ", "c d"),
    ])
    assert result["speakers"]["interviewer"]["words"] == 4


# --- share_of -------------------------------------------------------------------


def test_share_of_returns_the_speakers_share():
    result = talk_share([
        _segment("interviewer", "a b c"),
        _segment("interviewee", "d"),
    ])
    assert share_of(result, "interviewer") == pytest.approx(0.75)


def test_share_of_is_none_when_nothing_was_measured():
    assert share_of(talk_share([]), "interviewer") is None


def test_share_of_is_none_for_a_speaker_who_never_spoke():
    """`None`, never `0.0`. "This person was silent" is a finding about the
    interview; "there is no transcript" is a finding about the recording. A
    bare 0.0 cannot tell the caller which one it is looking at."""
    result = talk_share([_segment("interviewer", "a b")])
    assert share_of(result, "interviewee") is None
