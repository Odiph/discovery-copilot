"""Tests for `app/coach/language.py` -- D-06's script-ratio detection with
hysteresis: unaffected by the vendor's per-segment `detected_language` tag
(the measured cross-engine finding this module exists to protect against),
character-weighted (not segment-weighted), windowed by elapsed interviewee
speech time, and gated by a flip margin rather than a bare majority.
Behavioural test names, per this project's own convention."""

from __future__ import annotations

from types import SimpleNamespace

from app.coach.language import LANGUAGE_ENGLISH, LANGUAGE_HEBREW, detect_language

WINDOW_S = 90.0
FLIP_MARGIN = 0.65
MIN_CHARS = 40


def _seg(text: str, audio_end_s: float, detected_language: str = "en") -> SimpleNamespace:
    # A plain namespace, not a real SegmentRecord -- detect_language must
    # never read `.detected_language` at all, so giving it a deliberately
    # WRONG tag is exactly the test that proves it.
    return SimpleNamespace(text=text, audio_end_s=audio_end_s, detected_language=detected_language)


def _detect(segments, *, previous_language=LANGUAGE_ENGLISH, window_s=WINDOW_S, flip_margin=FLIP_MARGIN, min_chars=MIN_CHARS):
    return detect_language(
        segments, window_s=window_s, flip_margin=flip_margin, min_chars=min_chars, previous_language=previous_language
    )


_HEBREW_SENTENCE = "שלום מה שלומך היום אני רוצה לדבר על הריאיון הזה ולהבין יותר לעומק"
_ENGLISH_SENTENCE = "hello how are you today I want to talk about this interview in more depth"


# --- unaffected by the vendor's per-segment tag ---------------------------------------


def test_returns_hebrew_for_predominantly_hebrew_text_regardless_of_wrong_tags():
    segments = [_seg(_HEBREW_SENTENCE, 1.0, detected_language="en"), _seg(_HEBREW_SENTENCE, 2.0, detected_language="de")]
    assert _detect(segments) == LANGUAGE_HEBREW


def test_returns_english_for_predominantly_english_text_regardless_of_wrong_tags():
    segments = [_seg(_ENGLISH_SENTENCE, 1.0, detected_language="he"), _seg(_ENGLISH_SENTENCE, 2.0, detected_language="he")]
    assert _detect(segments) == LANGUAGE_ENGLISH


def test_answer_is_identical_no_matter_what_the_segment_tags_say():
    plain = [_seg(_HEBREW_SENTENCE, 1.0, detected_language="en"), _seg(_HEBREW_SENTENCE, 2.0, detected_language="en")]
    wrong = [_seg(_HEBREW_SENTENCE, 1.0, detected_language="el"), _seg(_HEBREW_SENTENCE, 2.0, detected_language="sl")]
    assert _detect(plain) == _detect(wrong) == LANGUAGE_HEBREW


# --- character-weighted, not segment-weighted -----------------------------------------


def test_a_single_short_english_interjection_inside_a_hebrew_window_does_not_flip():
    segments = [
        _seg(_HEBREW_SENTENCE, 1.0),
        _seg(_HEBREW_SENTENCE, 2.0),
        _seg(_HEBREW_SENTENCE, 3.0),
        _seg("okay", 4.0),  # one short English interjection
    ]
    assert _detect(segments, previous_language=LANGUAGE_HEBREW) == LANGUAGE_HEBREW


# --- hysteresis: a margin is required, not a bare majority -----------------------------


def test_flip_requires_the_margin_a_bare_majority_does_not_flip():
    # Construct hebrew_share just over 50% but under the 0.65 margin.
    hebrew_text = "א" * 55
    latin_text = "a" * 45
    segments = [_seg(hebrew_text, 1.0), _seg(latin_text, 2.0)]
    assert _detect(segments, previous_language=LANGUAGE_ENGLISH) == LANGUAGE_ENGLISH, (
        "55% Hebrew must not flip English -> Hebrew against a 0.65 margin"
    )


def test_flip_happens_once_the_margin_is_clearly_exceeded():
    hebrew_text = "א" * 80
    latin_text = "a" * 20
    segments = [_seg(hebrew_text, 1.0), _seg(latin_text, 2.0)]
    assert _detect(segments, previous_language=LANGUAGE_ENGLISH) == LANGUAGE_HEBREW


def test_flip_back_to_english_also_requires_the_margin():
    # Established Hebrew; a window that is 45% English (55% Hebrew) must not
    # flip back to English since English hasn't cleared its own margin.
    hebrew_text = "א" * 55
    latin_text = "a" * 45
    segments = [_seg(hebrew_text, 1.0), _seg(latin_text, 2.0)]
    assert _detect(segments, previous_language=LANGUAGE_HEBREW) == LANGUAGE_HEBREW


# --- too little signal defaults to English ----------------------------------------------


def test_too_little_accumulated_speech_returns_the_english_default():
    segments = [_seg("א", 1.0)]  # one character -- far below MIN_CHARS
    assert _detect(segments) == LANGUAGE_ENGLISH


def test_too_little_signal_holds_the_established_language_rather_than_defaulting_to_english():
    """REVERSED 2026-08-06 after an independent review. This test previously
    asserted that too-little-signal returns English even when Hebrew was
    already established.

    That followed 04-02-PLAN.md's "default to English when the window has too
    little signal ... or in a genuinely ambiguous middle", but it contradicts
    04-CONTEXT.md D-06, which forbids the flip in as many words: "a one-word
    English interjection mid-Hebrew must not flip the suggestion language."
    CONTEXT is the locked decision; the plan is downstream of it.

    The plan's reasoning is sound about AMBIGUOUS signal -- but this branch
    fires on INSUFFICIENT signal, which is a different thing, and genuine
    ambiguity is already handled by the flip margin below. Holding the
    established language is not a guess; it is the status quo."""
    segments = [_seg("א", 1.0)]
    assert _detect(segments, previous_language=LANGUAGE_HEBREW) == LANGUAGE_HEBREW
    assert _detect(segments, previous_language=LANGUAGE_ENGLISH) == LANGUAGE_ENGLISH


# --- windowed by elapsed interviewee speech time, old segments age out -----------------


def test_window_is_bounded_by_elapsed_interviewee_speech_and_old_segments_age_out():
    early_hebrew = [_seg(_HEBREW_SENTENCE, t) for t in (1.0, 2.0, 3.0)]
    # A long silence/English run pushes the window's "now" far past the
    # early Hebrew segments -- more than WINDOW_S seconds later.
    late_english = [_seg(_ENGLISH_SENTENCE, 200.0), _seg(_ENGLISH_SENTENCE, 201.0)]
    all_segments = early_hebrew + late_english
    assert _detect(all_segments, previous_language=LANGUAGE_HEBREW) == LANGUAGE_ENGLISH, (
        "the early Hebrew segments must have aged out of the window entirely"
    )


# --- both knobs come from config (the caller), never a literal in the module -----------


def test_window_bound_and_flip_margin_are_caller_supplied_not_hardcoded():
    hebrew_text = "א" * 55
    latin_text = "a" * 45
    segments = [_seg(hebrew_text, 1.0), _seg(latin_text, 2.0)]
    # With a looser margin (0.5), the same 55% Hebrew window DOES flip --
    # proving the margin is read from the argument, not baked into the module.
    assert _detect(segments, previous_language=LANGUAGE_ENGLISH, flip_margin=0.5) == LANGUAGE_HEBREW

    # With a much shorter window, the early Hebrew segment ages out sooner.
    segments2 = [_seg(_HEBREW_SENTENCE, 1.0), _seg(_ENGLISH_SENTENCE, 50.0)]
    result_wide = _detect(segments2, previous_language=LANGUAGE_HEBREW, window_s=90.0)
    result_narrow = _detect(segments2, previous_language=LANGUAGE_HEBREW, window_s=10.0)
    assert result_wide != result_narrow, "a narrower window must age the early segment out sooner"


def test_no_segments_returns_english():
    assert _detect([]) == LANGUAGE_ENGLISH


# --- Too little signal must never FLIP the language (review finding, 2026-08-06) ---


def test_a_short_interjection_after_long_hebrew_does_not_flip_the_suggestion_language():
    """D-06 forbids this in as many words: "a one-word English interjection
    mid-Hebrew must not flip the suggestion language."

    The path that broke it was the too-little-signal guard, which returned
    English unconditionally instead of holding position. Concrete case: the
    interviewee speaks Hebrew for twenty minutes, the interviewer then talks
    for two, and the interviewee's next segment is "okay". Their prior Hebrew
    has aged out of the window, so the window holds four Latin characters --
    below the floor -- and every suggestion arrives in English, mid-Hebrew
    conversation, until their next full answer flips it back."""
    assert _detect([_seg("okay", 1300.0)], previous_language=LANGUAGE_HEBREW) == LANGUAGE_HEBREW


def test_an_empty_or_unscripted_segment_holds_the_established_language():
    """Empty, punctuation-only and digits-only segments all reach the same
    too-little-signal guard, as does the first trigger after a crash-resume
    (the window rebuilds from empty). None of them is evidence of a language
    change, so none may cause one."""
    for text in ("", "   ", "...", "12345", "!!!"):
        assert _detect([_seg(text, 1300.0)], previous_language=LANGUAGE_HEBREW) == LANGUAGE_HEBREW, text
        assert _detect([_seg(text, 1300.0)], previous_language=LANGUAGE_ENGLISH) == LANGUAGE_ENGLISH, text


def test_cold_start_with_no_signal_still_resolves_to_english():
    """Holding position must not break the genuine cold start: with no prior
    language established, `previous_language` already defaults to English, so
    the first-ever call still resolves to English without special-casing."""
    assert _detect([]) == LANGUAGE_ENGLISH
    assert _detect([_seg("hi", 1.0)]) == LANGUAGE_ENGLISH


# --- A Hebrew suggestion in an English interview: detection was NOT the cause ---------
#
# On 2026-08-15 a coached, English-only interview showed the interviewer one
# Hebrew suggestion out of 29 (session 20260815T222827, suggestion seq=57,
# trigger_seq=54, move=deepen, band=clean, shown=true). This block is the
# evidence that ruled this module out, kept as a regression test rather than
# left in a commit message -- because the tempting fix for "the coach answered
# in Hebrew" is to reach for the language detector first, and that would be
# the wrong file.
#
# The interviewee's segments in that session are ASCII English throughout. The
# HEBREW in that interview belongs to the interviewer, whose asides the engine
# renders in Hebrew script (a Hebrew filler followed by English, segment
# seq=49) -- and this function never sees the interviewer at all, by design.
# The transcript block the model receives does, which is where the real cause
# lives; see `prompts._LANGUAGE_OVERRIDE_NOTE`.
#
# The interviewee texts below are SYNTHETIC stand-ins, per this repo's "no
# verbatim transcript content in any committed file" rule. They keep what the
# test depends on from that session: roughly its timing, the same
# segment count, ~5-word engine fragments cut mid-word, two short non-English
# Latin-script segments up front (the engine's mishearings of background
# audio), and zero Hebrew characters, up to and including the trigger segment.

_INTERVIEWEE_WINDOW_TO_THE_TRIGGER = (
    ("Evet.", 2.4),
    ("Dobro, dobro. Da.", 6.1),
    ("Right.", 8.7),
    ("So, um, the", 31.9),
    ("most recent one was", 34.2),
    ("a Friday, the console sits", 36.8),
    ("idle all week, so I", 39.1),
    (", um, just pick up the pad", 41.3),
    ("and right away the store", 43.9),
    ("demands an update, you kn", 46.2),
    ("ow, and you watch the bar crawl for a wh", 48.6),
    ("ile. After", 50.3),
    ("it is done,", 52.0),
    ("I choose a mode,", 54.5),
    ("press start, sort of. Sit through, through the intro,", 58.1),
    ("sort of.", 60.0),
    ("Pretty much.", 61.2),
    ("Mm, right,", 69.4),
    ("it happens most weekends, since", 72.3),
)


def test_the_real_interview_that_produced_a_hebrew_suggestion_detects_english():
    segments = [_seg(text, end) for text, end in _INTERVIEWEE_WINDOW_TO_THE_TRIGGER]
    assert _detect(segments) == LANGUAGE_ENGLISH


def test_that_same_window_cannot_be_pushed_into_hebrew_even_from_a_hebrew_prior():
    """The strongest form of the ruling-out. Even if some earlier cycle had
    somehow left the pipeline holding `he`, this window's own evidence -- zero
    Hebrew characters, far above the min-chars floor -- clears the flip margin
    in the English direction and returns English anyway. There is no
    `previous_language` value that makes this session's real input produce a
    Hebrew instruction, so the instruction the model received said English."""
    segments = [_seg(text, end) for text, end in _INTERVIEWEE_WINDOW_TO_THE_TRIGGER]
    assert _detect(segments, previous_language=LANGUAGE_HEBREW) == LANGUAGE_ENGLISH


def test_a_hebrew_speaking_interviewer_cannot_reach_this_function_at_all():
    """The complement of the test above, stated as a property rather than as a
    replay: this function is documented as taking the interviewee's OWN
    segments, and the caller (`pipeline._process`) appends only the trigger --
    always an interviewee segment. Interleaving the interviewer's Hebrew into
    the same list is therefore not a state the live path can reach; if it ever
    became one, the detector WOULD flip, and this test says so out loud so the
    next person changing `pipeline._interviewee_segments` sees the cost."""
    interviewee_only = [_seg(text, end) for text, end in _INTERVIEWEE_WINDOW_TO_THE_TRIGGER]
    with_the_interviewer = interviewee_only + [
        _seg("אוקיי, so walk me through", 61.5),
        _seg("— when was the last time the launcher made you wait before you could even queue?", 67.44),
        _seg("זה ייקח בערך חמש דקות. רגע, אני בודק משהו. טוב, נמשיך. אוקיי.", 20.94),
    ]
    assert _detect(interviewee_only) == LANGUAGE_ENGLISH
    assert _detect(with_the_interviewer) == LANGUAGE_ENGLISH, (
        "even polluted with the interviewer's turns this window is majority Latin -- "
        "the point of this test is the contract, not the arithmetic"
    )
