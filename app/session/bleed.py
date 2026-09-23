"""Did the microphone hear the loopback?

**The failure this detects, measured 2026-08-12 on the first real overlay
recording anyone made.** A Hebrew/English podcast played through SPEAKERS
landed on BOTH tracks: the loopback captured it directly, and the webcam
microphone captured it again out of the air a fraction of a second later. 15 of
21 microphone segments were near-duplicates of a loopback segment. 71% of the
"interviewer" track was the podcast, attributed to a person who had not spoken.

`app/overlay/` can now record one side only, which makes this impossible for a
podcast. It cannot help a REAL interview, where the microphone must stay open
and the interviewee is a person in the room or on speakers. There, detection is
the only defence there is.

**Why it matters more than duplicate text.** `CoverageDetector` is fed the
INTERVIEWER stream and marks a survey question ASKED when it hears the
interviewer ask it. Speech bleeding onto that stream can tick off questions
nobody asked, and a coverage match is irreversible for the rest of the run. The
transcript is merely wrong; the survey is wrong AND unrecoverable.

**The thing this must not do is cry wolf, and the reason is a real interviewing
technique.** Good interviewers REFLECT -- "so you went back to paper?" after
"we went back to paper". A naive text-similarity check flags that as bleed and
teaches the operator to ignore the warning, which is worse than not warning.

Three properties separate bleed from reflection, and all three are required:

1. **Simultaneity.** Bleed is the same sound reaching two devices at once, so
   the copies land within a fraction of a second. A reflection is a separate
   conversational turn, arriving after the other person stopped. This is the
   strongest discriminator and the tightest constraint.
2. **Verbatim overlap.** Bleed is the same audio, so a long run of characters
   is identical. A reflection is rephrased -- pronouns move, tense changes.
   Measured by longest common substring rather than an edit-distance ratio,
   because the two streams segment at DIFFERENT boundaries: the same sentence
   is chopped differently on each side, which drags a whole-string ratio down
   while leaving the shared run intact.
3. **Prevalence.** One or two coincidental overlaps happen in any long
   conversation. Bleed is not occasional -- if the microphone can hear the
   speakers it hears them all the time.

**This module decides nothing about what to DO.** It reports. Whether a
recording is unusable is the operator's call, and this deliberately never
edits, filters or drops a segment: a transcript that quietly deleted lines it
suspected would be unauditable, and the suspicion is a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Optional, Sequence

# How far apart two copies of the same sound may land and still be one event.
#
# Bleed is simultaneous; what varies is where each stream chose to cut a
# segment. Measured on the 2026-08-12 podcast: every matched pair sat within
# 0.5s, and the offsets ran in BOTH directions (-0.42s to +0.06s) because
# segmentation differs per stream -- so this is deliberately not a signed
# "the microphone copy arrives later" rule, which the data does not support.
#
# Two seconds is wide enough to absorb that, and far tighter than a
# conversational turn: reflecting a question back takes longer than two
# seconds from the end of the other person's sentence.
SIMULTANEITY_WINDOW_S = 2.0

# How much of the shorter segment must be one identical run of characters.
#
# Not an edit-distance ratio over the whole strings. The two streams cut
# segments at different points, so the same sentence appears as
# "...building the biggest infrastructure" against "we are building the biggest
# infra..." -- a whole-string ratio reads that as 0.6 and a shared-run measure
# reads it as what it is.
MIN_VERBATIM_OVERLAP = 0.6

# Below this, treat a match as coincidence rather than evidence. Two people
# discussing one subject reuse each other's words; a microphone that can hear
# the speakers hears them constantly.
SUSPICION_FRACTION = 0.30

# Segments shorter than this are excluded from BOTH sides. "yeah", "OK" and
# "right" collide across any two tracks in any conversation and would produce
# a confident-looking bleed report for a healthy recording.
MIN_SEGMENT_CHARS = 12

# Below this many comparable microphone segments the fraction is not a
# measurement. Three of four is 75% and means nothing.
MIN_SEGMENTS_FOR_A_VERDICT = 8


@dataclass(frozen=True)
class BleedReport:
    """What was found, and enough of the working to argue with it.

    `suspected` is the only judgement. Every other field is a count, published
    so a reader can disagree with the thresholds without re-running anything --
    the same reason `interview_summary` puts `*_interviews` and `*_facts` side
    by side."""

    suspected: bool
    duplicate_pairs: int
    comparable_interviewer_segments: int
    comparable_interviewee_segments: int
    fraction: float
    examples: "tuple[tuple[str, str], ...]"

    @property
    def percent(self) -> int:
        return round(self.fraction * 100)


def _verbatim_overlap(a: str, b: str) -> float:
    """Longest identical run of characters, over the length of the shorter
    string. 1.0 means one string contains the other outright."""
    shorter = min(len(a), len(b))
    if shorter == 0:
        return 0.0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return match.size / shorter


def _text(segment: "Any") -> str:
    value = segment.get("text") if isinstance(segment, dict) else getattr(segment, "text", None)
    return (value or "").strip()


def _end_s(segment: "Any") -> "Optional[float]":
    value = (
        segment.get("audio_end_s")
        if isinstance(segment, dict)
        else getattr(segment, "audio_end_s", None)
    )
    return None if value is None else float(value)


def _speaker(segment: "Any") -> str:
    value = (
        segment.get("speaker") if isinstance(segment, dict) else getattr(segment, "speaker", None)
    )
    return value or ""


def detect_bleed(
    segments: "Sequence[Any]",
    *,
    interviewer: str = "interviewer",
    interviewee: str = "interviewee",
    window_s: float = SIMULTANEITY_WINDOW_S,
    min_overlap: float = MIN_VERBATIM_OVERLAP,
    suspicion_fraction: float = SUSPICION_FRACTION,
) -> BleedReport:
    """Whether one track appears to be hearing the other.

    Takes segments rather than a path so it can be run over a live buffer, a
    stored transcript, or a hand-built list in a test -- and so this module
    opens no files, which keeps it out of every artifact-writer gate in the
    project.

    Speaker names are parameters because this compares TWO STREAMS; it has no
    opinion about which is which, and hard-coding the pair here would make the
    function untestable against a synthetic one.
    """
    mine = [
        s for s in segments
        if _speaker(s) == interviewer and len(_text(s)) >= MIN_SEGMENT_CHARS
        and _end_s(s) is not None
    ]
    theirs = [
        s for s in segments
        if _speaker(s) == interviewee and len(_text(s)) >= MIN_SEGMENT_CHARS
        and _end_s(s) is not None
    ]

    pairs = 0
    examples: "list[tuple[str, str]]" = []
    for a in mine:
        end_a = _end_s(a) or 0.0
        text_a = _text(a)
        for b in theirs:
            if abs((_end_s(b) or 0.0) - end_a) > window_s:
                continue
            if _verbatim_overlap(text_a, _text(b)) < min_overlap:
                continue
            pairs += 1
            if len(examples) < 3:
                examples.append((text_a[:80], _text(b)[:80]))
            # One match per microphone segment. Counting every loopback
            # segment it resembles would let a single sentence chopped into
            # four on the other side report 400%.
            break

    fraction = pairs / len(mine) if mine else 0.0
    # An honest verdict needs both a rate AND enough to rate. A recording with
    # four comparable segments cannot support this claim, and saying "75%
    # bleed" off three of four is the vote-counting failure this project
    # already refuses elsewhere.
    suspected = (
        len(mine) >= MIN_SEGMENTS_FOR_A_VERDICT
        and fraction >= suspicion_fraction
    )
    return BleedReport(
        suspected=suspected,
        duplicate_pairs=pairs,
        comparable_interviewer_segments=len(mine),
        comparable_interviewee_segments=len(theirs),
        fraction=fraction,
        examples=tuple(examples),
    )
