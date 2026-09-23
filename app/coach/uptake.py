"""Did the interviewer actually ask what the coach suggested?

**The gap this closes.** `.planning/PRODUCT-HYPOTHESES.md` P1 claims the coach's
suggestions are usable in a live conversation, and until now nothing in this
product could link a shown suggestion to what the interviewer said next. Without
that link the first coached interview produces an anecdote about the one thing
most worth a number, and the anecdote will be favourable, because the person
reading it built the coach.

**It reuses the dedup definition rather than inventing a second one.**
`repetition.similarity` already decides whether two questions "say the same
thing" for the anti-repeat guard. The same computation, pointed the other way --
suggestion against later interviewer speech -- is what uptake is. A second
notion of sameness here could disagree with the guard, and then one of the two
numbers would be wrong with nothing to say which.

**This metric BOUNDS P1 FROM ABOVE. It can falsify, it cannot prove.** A match
cannot distinguish "the coach suggested it and the interviewer took it" from
"the coach and the interviewer independently arrived at the obvious next
question" -- and on a well-run interview those coincide often, which is exactly
when the coach is least needed. So a LOW rate is evidence against P1; a high
rate is not evidence for it. Every caller must carry that asymmetry; it is the
whole reason this docstring leads with it.

**Two further sources of false positives, both deliberate and both stated.**
A suggestion is timestamped by its TRIGGER segment, because that is the only
audio-domain time a `SuggestionRecord` carries -- but the suggestion is written
after its cycle completes, a median of ~0.64s and occasionally several seconds
later. So the earliest part of each window can contain speech the interviewer
produced before ever seeing the suggestion. `min_delay_s` exists to trade that
against false negatives; it defaults to 0.0 because a metric that only bounds
from above should not be quietly tightened in the flattering direction.
Separately, the interviewer may repeat their OWN earlier question, which the
coach also proposed; that is counted as uptake and is not distinguishable here.

**Segments are fragments, so runs are compared, not segments.** The STT emits
~5-word finalizations; a whole question routinely spans several. Comparing a
15-word suggestion against one 5-word fragment scores near zero no matter how
exactly the interviewer asked it, which would make the metric read ~0% on a
coach that was working perfectly. So every contiguous RUN of interviewer
segments in the window is considered.

**`max_span_ratio` is a COST bound, not a correctness guard, and the difference
was established by mutation rather than assumed.** It was written believing it
prevented a long stretch of unrelated speech from accumulating a match by sheer
length. Removing it entirely changes no test outcome: `SequenceMatcher.ratio()`
divides by the total length of both sequences, so a 15-word suggestion against
480 words of monologue scores near zero on its own. The protection is real but
it comes from the metric, not from this cap. What the cap actually buys is
avoiding a quadratic blow-up in the number of runs on a long window. Stated
accurately here because a comment claiming a safety property the code does not
provide is worse than no comment: the next reader deletes the cap for being
redundant and gets the performance cliff, or keeps it believing they are
protected against something else entirely.

Pure: no I/O, no client, no clock, deterministic. Never logs text, and never
returns it -- matches are reported as sequence numbers, exactly as
`interview_summary.FactRef` avoids carrying `quoted_text`, so a reporting layer
that wants the words fetches them itself from the transcript it already has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from app.coach.repetition import normalise, similarity

INTERVIEWER = "interviewer"

# The pre-registered values (PRODUCT-HYPOTHESES.md P1). Named here so the
# criterion and the code cannot drift apart, and so changing one is a visible
# edit to a constant rather than a different number passed at a call site.
DEFAULT_WINDOW_S = 90.0
DEFAULT_THRESHOLD = 0.6
DEFAULT_MAX_SPAN_RATIO = 3.0


@dataclass(frozen=True)
class Match:
    """One shown suggestion and the best thing the interviewer said after it.

    `matched_seqs` is the contiguous run that scored best -- empty when nothing
    in the window came close. Sequence numbers only, never text."""

    suggestion_seq: int
    trigger_seq: Optional[int]
    at_s: float
    score: float
    matched_seqs: "tuple[int, ...]"

    @property
    def taken(self) -> bool:
        return bool(self.matched_seqs)


@dataclass(frozen=True)
class Uptake:
    """P1's number, with the denominator it was computed over.

    `rate` is `None` rather than 0.0 when nothing was shown: an interview where
    the coach proposed nothing is not an interview where the interviewer ignored
    it, and a 0.0 would put those two in the same column."""

    shown: int
    taken: int
    matches: "tuple[Match, ...]"
    window_s: float
    threshold: float

    @property
    def rate(self) -> "Optional[float]":
        if self.shown == 0:
            return None
        return self.taken / self.shown


def _runs(segments: "Sequence[Mapping[str, Any]]", *, budget_words: int):
    """Every contiguous run of `segments` whose word count fits the budget.

    Anchored at each start index and extended while it fits, so a question split
    across five fragments is reachable and a two-minute monologue is not."""
    for start in range(len(segments)):
        words = 0
        for end in range(start, len(segments)):
            words += len(normalise(segments[end].get("text") or "").split())
            if words > budget_words:
                break
            yield segments[start:end + 1]


def best_match(
    suggestion_text: str,
    segments: "Sequence[Mapping[str, Any]]",
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_span_ratio: float = DEFAULT_MAX_SPAN_RATIO,
) -> "tuple[float, tuple[int, ...]]":
    """The best-scoring contiguous run, and its score.

    The empty-suggestion early exit is BELT AND BRACES, and mutation says so:
    removing it changes no test outcome, because `similarity` already answers
    0.0 for an empty left side. It is kept as an explicit statement that "an
    empty suggestion matches everything" is a known-dangerous shape -- it would
    be the single most flattering possible bug here -- and because relying on a
    downstream function's behaviour for a safety property is how that property
    disappears the day the function is optimised."""
    wanted = normalise(suggestion_text).split()
    if not wanted or not segments:
        return 0.0, ()

    budget = max(len(wanted), int(len(wanted) * max_span_ratio))
    best_score, best_seqs = 0.0, ()
    for run in _runs(segments, budget_words=budget):
        joined = " ".join((segment.get("text") or "") for segment in run)
        score = similarity(suggestion_text, joined)
        if score > best_score:
            best_score = score
            best_seqs = tuple(
                segment["seq"] for segment in run if segment.get("seq") is not None
            )
    if best_score < threshold:
        return best_score, ()
    return best_score, best_seqs


def uptake(
    records: "Iterable[Mapping[str, Any]]",
    *,
    window_s: float = DEFAULT_WINDOW_S,
    threshold: float = DEFAULT_THRESHOLD,
    min_delay_s: float = 0.0,
    max_span_ratio: float = DEFAULT_MAX_SPAN_RATIO,
) -> Uptake:
    """How many SHOWN suggestions the interviewer went on to ask.

    Takes the plain records `app.session.reader.read_session` yields.

    Only `shown` suggestions count. A candidate the critic dropped was never in
    front of the interviewer, so counting it would measure the generate stage
    against speech it never influenced -- and would deflate the rate with
    material P1 makes no claim about.

    The durable record carries no "primary" flag: `shown` is the whole of what
    reached the operator, up to `max_shown` per cycle. P1 is therefore measured
    over shown suggestions, not over primaries, and the pre-registration says so.
    """
    records = list(records)
    segments = [r for r in records if r.get("kind") == "segment"]
    by_seq = {r.get("seq"): r for r in segments}
    interviewer = sorted(
        (r for r in segments if r.get("speaker") == INTERVIEWER),
        key=lambda r: r.get("audio_end_s") or 0.0,
    )

    matches: "list[Match]" = []
    for record in records:
        if record.get("kind") != "suggestion" or not record.get("shown"):
            continue
        trigger = by_seq.get(record.get("trigger_seq"))
        if trigger is None:
            # A suggestion whose trigger segment is not in this transcript
            # cannot be placed in time. Recorded as unmatched rather than
            # skipped: dropping it would shrink the DENOMINATOR, which quietly
            # raises the rate -- the one direction this metric must never move
            # by accident.
            matches.append(Match(
                suggestion_seq=record.get("seq"), trigger_seq=record.get("trigger_seq"),
                at_s=0.0, score=0.0, matched_seqs=(),
            ))
            continue

        at_s = trigger.get("audio_end_s") or 0.0
        window = [
            segment for segment in interviewer
            if at_s + min_delay_s < (segment.get("audio_end_s") or 0.0) <= at_s + window_s
        ]
        score, seqs = best_match(
            record.get("text") or "", window,
            threshold=threshold, max_span_ratio=max_span_ratio,
        )
        matches.append(Match(
            suggestion_seq=record.get("seq"), trigger_seq=record.get("trigger_seq"),
            at_s=at_s, score=score, matched_seqs=seqs,
        ))

    return Uptake(
        shown=len(matches),
        taken=sum(1 for match in matches if match.taken),
        matches=tuple(matches),
        window_s=window_s,
        threshold=threshold,
    )
