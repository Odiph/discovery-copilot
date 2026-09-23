"""What the INTERVIEWER actually asked, reconstructed from the transcript.

The critic has only ever judged questions the coach PROPOSED. The interviewer's
own questions -- the ones that were really asked, that really shaped the
answers, and that are the only ones the interview's findings rest on -- have
never been judged by anything. This module is the missing half: it turns raw
STT fragments back into the turns a human took, so the same rubric can be
pointed at them after the call.

Pure: no I/O, no client, no clock, deterministic.

**A turn ends when the other speaker talks, and nothing else ends it.** That is
the definition `app/coach/trigger.py` already uses, and it is the one the data
supports. Measured across the whole real corpus, the largest gap between two
consecutive fragments WITHIN one interviewer run is 5.94s and none of the 30
such gaps exceeded 8s -- so a silence-based split rule would never once have
fired. Adding one would have meant shipping a branch no session can reach and
no test over real data can exercise.

**Timing comes from `audio_end_s` only.** `app/stt/live.py` fabricates
`audio_start_s` from the same speaker's previous end, so a span computed from
it is fiction that happens to typecheck.

**The unit of judgement is a TURN, not a question.** No filter on `?` is
applied, for two reasons the transcript settles: only about half the real
interviewer segments carry one at all, and on a split question it lands on the
last fragment only -- but more importantly, an interviewer's leading
STATEMENT ("So you're protecting the playtime from the work.") is exactly what
the Mom Test rubric exists to catch, and a `?` filter would discard the most
incriminating material in the transcript. Bare acknowledgements are excluded by
a word floor instead, which is a claim about size rather than about grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

INTERVIEWER = "interviewer"

SEGMENT_KIND = "segment"

# Drops `'Right.'`, `'Yes.'`, `'Go on.'` -- real interviewer turns that a band
# would say nothing about -- while sitting well under the median real turn of 9
# words, so nothing substantive is filtered on size alone.
DEFAULT_MIN_WORDS = 4

QUESTION_MARK = "?"


@dataclass(frozen=True)
class InterviewerTurn:
    """One thing the interviewer said, reassembled from however many fragments
    the STT split it into.

    `index` is contiguous over the KEPT turns, because it is what a judgment is
    mapped back through -- a gap in it puts a band on the wrong question.
    `seqs` is how a reporting layer finds the words again in the transcript it
    already has."""

    index: int
    seqs: "tuple[int, ...]"
    text: str
    at_s: float

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_marked_question(self) -> bool:
        """Whether this turn is PUNCTUATED as a question.

        Named for what it measures, not for what a reader would rather have.
        The `?` asymmetry stated in `_worth_judging` applies here unchanged:
        presence is good evidence a turn is a question, absence is NOT
        evidence it is not one -- only about half the real interviewer
        segments carry one, and a leading statement carries none by
        construction. So `True` means "a question"; `False` means "unmarked",
        a mixed population of real questions and real statements, and calling
        that group "statements" would be a claim this cannot support."""
        return QUESTION_MARK in self.text


def _segments_in_speech_order(
    records: "Iterable[Mapping[str, Any]]",
) -> "list[Mapping[str, Any]]":
    """Segments only, ordered by `seq`.

    File order is not speech order: the two speakers finalize on separate STT
    threads, so reconstructing in arrival order would join fragments that were
    never adjacent."""
    segments = [
        record
        for record in records
        if record.get("kind") == SEGMENT_KIND and record.get("seq") is not None
    ]
    return sorted(segments, key=lambda record: record["seq"])


def _runs(
    segments: "list[Mapping[str, Any]]",
) -> "list[list[Mapping[str, Any]]]":
    """Every maximal contiguous run of interviewer segments."""
    runs: "list[list[Mapping[str, Any]]]" = []
    current: "list[Mapping[str, Any]]" = []
    for segment in segments:
        if segment.get("speaker") == INTERVIEWER:
            current.append(segment)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _joined_text(run: "list[Mapping[str, Any]]") -> str:
    """A missing or null `text` contributes nothing rather than raising: the
    tree is on disk and a crashed write can leave anything there, and one
    stray record must not make the whole command unusable."""
    parts = [(segment.get("text") or "").strip() for segment in run]
    return " ".join(part for part in parts if part)


def _at_s(run: "list[Mapping[str, Any]]") -> float:
    ends = [
        segment.get("audio_end_s")
        for segment in run
        if segment.get("audio_end_s") is not None
    ]
    return float(ends[-1]) if ends else 0.0


def _worth_judging(text: str, *, min_words: int) -> bool:
    """A turn is worth a band if it is long enough to say something, OR if it
    is punctuated as a question however short.

    **The `?` test is an INCLUDE signal only, and the asymmetry is the whole
    point.** Presence of a question mark is evidence a turn is a question;
    absence is not evidence it is not one, because only about half the real
    interviewer segments carry one and a leading statement carries none by
    construction. Used to EXCLUDE, it would discard the most incriminating
    material in the transcript; used to INCLUDE, it can only rescue.

    What it rescues is not marginal. Running the word floor alone over the real
    interview dropped five turns, and four of them were genuine follow-up
    questions -- `'And after that?'`, `'And then?'`, `'Between rounds
    specifically?'`, `'Which two?'` -- while only `'Go on.'` was the nudge the
    floor was aimed at. Short probing follow-ups are what good interviewing is
    largely made of, so a bare floor cuts against the grain of the very skill
    being measured. The tests over invented fixtures all passed before this was
    found; the real transcript is what found it."""
    if QUESTION_MARK in text:
        return True
    return len(text.split()) >= min_words


def interviewer_turns(
    records: "Iterable[Mapping[str, Any]]",
    *,
    min_words: int = DEFAULT_MIN_WORDS,
) -> "tuple[InterviewerTurn, ...]":
    """Every interviewer turn worth judging, in speech order.

    Takes the plain records `app.session.reader.read_session` yields.

    The word floor is applied to the JOINED turn, never to a fragment: the
    median real fragment is 5-7 words, so a per-fragment floor would discard
    most of the interview one piece at a time."""
    kept: "list[InterviewerTurn]" = []
    for run in _runs(_segments_in_speech_order(records)):
        text = _joined_text(run)
        if not _worth_judging(text, min_words=min_words):
            continue
        kept.append(
            InterviewerTurn(
                index=len(kept),
                seqs=tuple(segment["seq"] for segment in run),
                text=text,
                at_s=_at_s(run),
            )
        )
    return tuple(kept)
