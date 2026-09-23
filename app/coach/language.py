"""Decides the LANGUAGE a suggestion comes back in, from the interviewee's
own transcript text (04-CONTEXT.md D-06).

**Measured finding this module responds to, not an assumption:**
`.planning/research/stt.md` lines 490-496 record a three-for-three
cross-engine observation on identical audio -- Whisper base said
`de, es, he, he, he`; the ivrit.ai finetune said `he` five times; Soniox said
`el, el, el, he` on one fragment and mostly `he` (plus a stray `sl`) on
clean slices. Three engines, three different answers, several plainly wrong.
Short-segment per-segment language identification is unreliable across
engines generally -- this is NOT a Whisper-specific quirk. Trusting a
per-segment `detected_language` tag as authoritative for which language a
LIVE SUGGESTION should be written in would misbehave on exactly this
project's own measured evidence.

**The design that responds to it:** count SCRIPT in the interviewee's own
transcript TEXT -- a character-level fact, not an inference, needing no
dependency beyond the standard library. The vendor's tag is never read here
at all; this module does not even accept it as an argument. Weight by
CHARACTER COUNT, not segment count, over a window bounded by ELAPSED
INTERVIEWEE SPEECH TIME (via each segment's own `audio_end_s`), with a
HYSTERESIS MARGIN required to flip -- a bare majority flaps around 50% and
would change the suggestion's language mid-conversation for no reason the
user can see. Both the window bound and the flip margin are `DEFAULT_*`
config knobs (`app/config.py`), never literals here, because
`04-RESEARCH.md`'s Open Question 2 is honest that no measurement stands
behind the starting values.

**Do not "simplify" this back to trusting `detected_language`.** That is the
exact regression this module exists to prevent, and the evidence against it
is already in this repository (see the citation above) -- re-adding a
dependency on the vendor tag here would silently reintroduce a measured,
three-for-three-observed failure mode.

This module must NEVER read a segment's `detected_language` field, and must
NEVER log or place transcript text into an event `detail` dict.
"""

from __future__ import annotations

from typing import Sequence

LANGUAGE_ENGLISH = "en"
LANGUAGE_HEBREW = "he"

_HEBREW_RANGE = (0x0590, 0x05FF)


def _char_script(ch: str) -> "str | None":
    codepoint = ord(ch)
    if _HEBREW_RANGE[0] <= codepoint <= _HEBREW_RANGE[1]:
        return "hebrew"
    if ch.isascii() and ch.isalpha():
        return "latin"
    return None


def _count_chars(text: str) -> "tuple[int, int]":
    """Returns `(hebrew_chars, latin_chars)` for one string. Punctuation,
    digits and whitespace count toward neither -- they carry no script
    signal either way."""
    hebrew = 0
    latin = 0
    for ch in text:
        script = _char_script(ch)
        if script == "hebrew":
            hebrew += 1
        elif script == "latin":
            latin += 1
    return hebrew, latin


def detect_language(
    interviewee_segments: Sequence["object"],
    *,
    window_s: float,
    flip_margin: float,
    min_chars: int,
    previous_language: str = LANGUAGE_ENGLISH,
) -> str:
    """Character-weighted script-ratio detection with hysteresis (D-06).

    `interviewee_segments` -- `Sequence[SegmentRecord]`, the interviewee's
    OWN segments only, any order, each exposing `.text` and `.audio_end_s`.
    Only script counted from `.text` is ever read; `.detected_language` (if
    present on the object) is never inspected -- this function is provably
    unaffected by whatever the vendor tagged a segment with.

    `window_s` bounds the window by ELAPSED INTERVIEWEE SPEECH TIME: only
    segments within `window_s` of the latest segment's own `audio_end_s` are
    counted, so a language established early stops counting once it has
    aged out.

    `previous_language` is the caller's current state (an explicit argument,
    not module state, so this function stays pure and directly testable) --
    hysteresis compares the NEW window's Hebrew share against `flip_margin`
    only when it would change `previous_language`; staying the same never
    requires clearing the margin.

    With fewer than `min_chars` total script-bearing characters in the
    window, HOLDS POSITION and returns `previous_language` -- "too little
    signal" is deliberately not "flip to English", since a mid-conversation
    flip on the evidence of a four-character interjection is exactly what D-06
    forbids. (This paragraph said the opposite until 2026-08-16: it described
    the pre-fix behaviour the 2026-08-06 review finding removed, and the
    inline comment at the guard itself has been the accurate account since.
    A reader trusting the stale text would conclude this function can flip
    language on thin evidence, which is precisely the wrong place to start
    looking when a suggestion comes back in the wrong language.)
    """
    if not interviewee_segments:
        return LANGUAGE_ENGLISH

    latest_end = max(segment.audio_end_s for segment in interviewee_segments)
    windowed = [
        segment for segment in interviewee_segments if (latest_end - segment.audio_end_s) <= window_s
    ]

    hebrew_chars = 0
    latin_chars = 0
    for segment in windowed:
        seg_hebrew, seg_latin = _count_chars(segment.text)
        hebrew_chars += seg_hebrew
        latin_chars += seg_latin

    total_chars = hebrew_chars + latin_chars
    if total_chars < min_chars:
        # Too little signal to decide. STAY WHERE WE WERE -- returning English
        # unconditionally here is a mid-conversation flip, not a cold start.
        # Concrete case D-06 forbids in as many words: the interviewee speaks
        # Hebrew for twenty minutes, the interviewer then talks for two, and
        # the interviewee's next segment is "okay" (4 Latin chars). Their prior
        # Hebrew has now aged out of the window, so the window holds only
        # "okay", 4 < 40 -- and the next suggestion arrives in English, mid
        # Hebrew conversation, until their next full answer flips it back.
        # Empty, punctuation-only and digits-only segments hit this same path,
        # as does the first trigger after a crash-resume (the window rebuilds
        # from empty). `previous_language` already defaults to English, so the
        # genuine cold start still resolves to English without special-casing.
        return previous_language

    hebrew_share = hebrew_chars / total_chars

    if previous_language == LANGUAGE_HEBREW:
        # Stay Hebrew unless English's share has CLEARLY overtaken it --
        # i.e. Hebrew's own share has dropped below (1 - flip_margin).
        return LANGUAGE_ENGLISH if hebrew_share < (1.0 - flip_margin) else LANGUAGE_HEBREW
    return LANGUAGE_HEBREW if hebrew_share > flip_margin else LANGUAGE_ENGLISH
