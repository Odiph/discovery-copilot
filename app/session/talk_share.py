"""Who did the talking -- counted in WORDS, deliberately never in seconds.

The interviewer spoke roughly 65% of the words in both real calls, and nothing
in this project computed that. It is the cheapest quality signal available: no
model call, no network, no cost, no latency budget, just arithmetic over
records already on disk. Every other quality measure here needs a model.

**Why words and not talking TIME, which is the obvious thing to want.**
A segment's duration looks like `audio_end_s - audio_start_s`, and that
subtraction is worthless here. `app/stt/live.py` fabricates `audio_start_s`
as the end of the SAME speaker's previous segment, so the value silently
absorbs however long that speaker was quiet (`app/web/hub.py`'s docstring
documents it as unsafe, and excludes it from the wire allowlist by
construction; `app/coach/corpus.py` refuses to give it anywhere to leak).
A talk-share built on it would count each speaker's SILENCE as their speaking
time, and would inflate whoever paused most -- which in an interview is
usually the interviewee, i.e. it would be biased in exactly the direction that
makes the interviewer look better. A wrong number pointing the wrong way is
worse than no number.

`audio_end_s` alone is trustworthy, but consecutive end-times give the gap
between turns, not the length of one, so no honest per-speaker duration can
be recovered from what is stored. If talking time is ever wanted, the fix is
upstream in what `live.py` records, not a cleverer sum here.

**No threshold, and that is a decision.** This returns the share and stops.
It does not say 65% is too much, does not colour anything, and does not
define a healthy range -- there is no measurement behind such a number, and
inventing one would dress a guess as a finding. What the ratio MEANS is a
product judgement for the person reading it, who knows whether they were
explaining a consent form or steamrolling a customer.

**What counts as a word.** `str.split()` on whitespace. That is correct for
English and for Hebrew (both whitespace-delimited, and the corpus is both),
and it is WRONG for a language that does not put spaces between words --
Chinese, Japanese and Thai would each read as a handful of enormous "words"
and the share would be meaningless rather than merely imprecise. Recorded
here rather than guarded against, because the guard would need a language
model and the project's own language detection is per-segment and advisory.
If a CJK interview is ever recorded, this number must be rebuilt, not
adjusted.

Pure: no I/O, no client, no clock, deterministic. Modelled on
`app/session/store.py::latency_stats`, and for the same reason -- a figure
computed live and the same figure recomputed from disk later must be the same
number by construction, never two implementations that can drift.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

def unmeasurable() -> "dict[str, Any]":
    """The result when there is nothing to measure. `measurable` is the field
    callers must branch on: an interview with no speech is an ABSENCE, and
    rendering it as "50/50" or "0%" would be a claim about a conversation that
    did not happen. This is the same discipline `coverage.is_reportable`
    applies to "there was no survey" -- an unasked question and a missing
    survey are different statements.

    **A function, not a module-level constant, and that is not style.** It was
    a constant returned via `dict(...)`, which is a SHALLOW copy: every caller
    got its own outer dict but they all shared the same inner `speakers` dict,
    so one caller mutating its result would change what every later caller
    saw. This is the commonest path in the whole module -- every unrecorded
    interview takes it -- so the aliasing would have been widespread and
    would have looked like data corruption rather than a copy bug. Building a
    fresh nested structure per call makes it impossible rather than merely
    unlikely."""
    return {
        "measurable": False,
        "total_words": 0,
        "speakers": {},
        "unattributed_words": 0,
    }


def _word_count(text: Any) -> int:
    """Whitespace-delimited words. A non-string (a malformed record, a `None`
    where text was expected) counts zero rather than raising: this function is
    called on whatever is on disk, and one bad record must not take out the
    whole figure."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def talk_share(segments: "Iterable[Mapping[str, Any]]") -> "dict[str, Any]":
    """Per-speaker word counts and shares over `segment` records.

    Takes plain mappings -- what `app.session.reader.read_session` yields --
    rather than `SegmentRecord` instances, so a session re-read from disk and
    a live one both work without a conversion step that could differ.

    A record with no usable speaker is counted in `unattributed_words` and in
    `total_words`, never silently dropped and never folded into a real
    speaker. Dropping it would make the shares sum to 1.0 while describing
    less speech than actually occurred -- the shares would still LOOK right,
    which is what makes silent dropping the dangerous option here.

    Shares are computed over `total_words` INCLUDING unattributed speech, so
    they sum to 1.0 only when everything was attributed. That is deliberate:
    a set of shares summing to less than 1.0 is the visible symptom of a
    transcript with unlabelled speech in it, and normalising it away would
    hide the defect while making the arithmetic prettier.
    """
    words_by_speaker: "dict[str, int]" = {}
    unattributed = 0
    total = 0

    for segment in segments:
        count = _word_count(segment.get("text"))
        if count == 0:
            # An empty or unparseable segment contributes nothing to anyone.
            # Not skipped before this point, because a record with text but no
            # speaker still needs to reach the branch below.
            continue
        total += count
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            unattributed += count
            continue
        key = speaker.strip()
        words_by_speaker[key] = words_by_speaker.get(key, 0) + count

    if total == 0:
        return unmeasurable()

    return {
        "measurable": True,
        "total_words": total,
        "speakers": {
            speaker: {"words": words, "share": words / total}
            for speaker, words in sorted(
                words_by_speaker.items(), key=lambda kv: (-kv[1], kv[0])
            )
        },
        "unattributed_words": unattributed,
    }


def share_of(result: "Mapping[str, Any]", speaker: str) -> "Optional[float]":
    """One speaker's share, or `None` when there is nothing to report.

    `None` rather than `0.0` for an absent speaker, and the distinction is the
    whole point: a speaker who said nothing and a session that measured
    nothing both produce no bar, but "this person was silent" is a finding
    about the interview while "there is no transcript" is a finding about the
    recording. A caller that wants to render 0% for a silent speaker can ask
    for it; one that gets a bare `0.0` back cannot tell which case it is in."""
    if not result.get("measurable"):
        return None
    entry = result.get("speakers", {}).get(speaker)
    if entry is None:
        return None
    return entry["share"]
