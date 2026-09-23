"""The extraction stage: a finished session's transcript becomes a set of
cited, provenance-tagged facts (05-CONTEXT.md D-06/D-07/D-08;
05-01-PLAN.md §K1-§K7).

**One forced-tool-use call, offline-testable by construction (§K6).**
`extract_facts(client, *, ...)` takes its client by injection, structurally
identical to `app/coach/generate.py::run_generate` -- it never constructs a
client and never imports the vendor SDK. `assemble_extract_request` is split
out as a pure function so prompt tests need no client at all, per
`generate.py:124-176`'s own precedent.

**This module must NEVER read a transcript file, and must NEVER import
`anthropic` (gate 11, `app/tests/test_phase_gates.py`).** It receives
already-parsed `SegmentRecord`s from its caller (`app/main.py`, via
`app.session.reader.read_session` -- the ONE reader of `transcript.jsonl`,
which inherits the `74cbd6f` byte-decode fix). Reading the file a second time
here would risk reintroducing the Hebrew mid-codepoint crash-tail bug this
project already paid to fix once (§K7).

**The deterministic validation pipeline runs entirely in plain Python, with
no second model call (§K5).** Every returned fact's citation SPAN (STATE.md §
FIRST REAL-SPEECH EVIDENCE FINDING 1 -- `Fact.seqs`, not a single `seq`,
replacing 05-01-PLAN.md §K2's original single-segment design) is validated
against the session's real segments before it is ever accepted: every cited
seq must resolve, all cited seqs must share one speaker, they must form a
consecutive run within that speaker's own ordered segments (never skipping
one of that speaker's own turns -- `app.coach.corpus.validate_citation_span`,
shared with that module's own loader so this logic is never duplicated), and
the span may not exceed `app.coach.corpus.MAX_SPAN_SEGMENTS`. A span that
fails any of these is refused, named with its claim text and the failing
condition, and never reaches the corpus. The `quoted_text` substring backstop
(D-07's evidence check, §K4) now checks the space-joined text of the CITED
SPAN, in seq order, and is a further, separate rejection reason applied in
the same pipeline.

**Two-tier logging (§K5's rule, gate 12).** This module logs counts, ids and
reason codes only -- fact text is transcript-derived, exactly what gate 12
exists to contain. The human-readable report, with claim text, is printed by
`app/main.py` to stdout, which is not a log record.

## ⚠️ A2 IS FALSIFIED, AND THE ASYMMETRY BELOW IS WHY (2026-08-15)

The model labels merely-implied claims `evidence`. First seen 3/3 on a hand
check; the corpus-level measurement is stronger and reproducible: across the
two real fact corpora on the maintainer's machine, **32 of 34 facts are `evidence` and 2
are `inference` (94%)**. For discovery interviews that ratio is not credible,
and `interview_summary` reports those counts, so the headline "N grounded
facts" inherits the error.

**It is not an instruction problem, and that is the useful part.**
`prompts.extraction_rubric_block` already teaches the distinction at length,
names the hedge/future/conditional traps, gives four worked examples -- of
which example 4 IS this exact failure -- and states the tie-break ("when
unsure, choose inference"). The model is told thoroughly and gets it wrong
anyway. **So do not "fix" this by adding more prose to the same block.** That
has already been tried by construction; more of it is not a different
experiment.

**The structural defect is that this pipeline verifies the easy half.** An
`evidence` label triggers a deterministic check -- `REASON_UNLOCATABLE_QUOTE`
-- that the QUOTE really appears in the cited span. Nothing anywhere checks
that the CLAIM follows from the quote. So a claim that is an inference,
dressed with a genuinely verbatim quote, passes every gate this module has.
The guard that exists LOOKS like it covers mislabelling, and does not, which
is why the mislabelling survived a pipeline that appears well defended.

**A lexical speculation check does not work -- measured, not assumed.** The
obvious next move is to flag `evidence` facts whose quote carries a hedge
("I think", "it feels like"), a future ("we will"), or a conditional ("if").
Run over the real corpora before writing it: 3/12 flagged in one and 0/20 in
the other, and **all three flags were the bare word "if" occurring
incidentally inside quoted speech**, on claims whose content is plainly
manifest ("He describes wanting to see an answer without typing"). 100% false
positives on the only real data available. Recorded here so the next person
does not rebuild it and trust it.

**Still open.** What would actually measure this is a re-extraction over a
known transcript with the labels adjudicated by hand -- a paid call, not run.
The one change made on this pass is cheap and unverified: `provenance` was
the ONLY field in the tool schema with no `description` at all, while the
rubric explaining it lived in the system block; the tie-break is now restated
on the field itself, at the point where the model fills it in. That is a
plausible mechanism, not a demonstrated fix, and the 94% figure above is the
baseline to re-measure it against.

## THE HAND ADJUDICATION WAS RUN, AND ONE MODE IS NOW GUARDED (2026-08-16)

The paid re-extraction above was run, and all 18 of its facts were read by
hand against their own quotes. **3 are clearly mislabelled (17%), up to ~35%
on a strict reading of hedges.** (A 94%-evidence ratio is a smell, not a
measurement -- do not quote it as an error rate.) Two distinct modes:

**Mode 1, the dominant one -- the YES/NO ANSWER, and this module now guards
it.** Claim *"I keep a spare laptop beside the console."* cited to the quote
**"Yep."** Claim *"I have not changed my setup this year."* cited to
**"Not really. Uh, no."** The quotes are genuinely verbatim, so the
locatability backstop passes them; the CONTENT of the claim came from the
INTERVIEWER'S QUESTION, and the interviewee only confirmed it. The quote is
real, and states nothing.

`quote_states_nothing` (below) flags exactly this, and the fact is **DEMOTED
to `inference`, never rejected** -- the claim is usually TRUE, and rejecting
it would delete a real finding to punish a labelling error. Demotions are
reported in their own tuple, apart from rejections, and do NOT affect the
exit code.

**Mode 2, NOT guarded -- the hedged-and-disclaimed answer.** An utterance of
the shape *"Um, I think it was the settings page. I couldn't tell you for
sure."* -- a filler, a hedge, a named object, and an explicit disclaimer --
recorded as fact. The rubric already says hedges mean inference and the model
ignored it; the lexical hedge scan that would catch it is measured above at
100% false positives.

(Every illustrative UTTERANCE in this docstring is synthesized to the same
shape as the real one it stands for, per this repo's "no verbatim transcript
content in any committed file -- metrics only" rule. The two bare particles
"Yep." and "Not really. Uh, no." are the sole exceptions: they ARE the
subject of the guard below, they carry no information, and they identify
nobody. The CLAIMS are synthetic too. Every number, ratio and conclusion
here is the real measurement.)

## THE MODEL SWAP WAS RUN. MODE 2 IS A MODEL PROBLEM; MODE 1 IS NOT (2026-08-16)

Same transcript (`20260815T222827`), same prompt, same pipeline,
`EXTRACT_MODEL=claude-opus-5`. The two effects are separable because the
guard shipped FIRST, on purpose.

| | before | Opus 5 |
|---|---|---|
| facts accepted | 18 | 12 |
| provenance | **18 evidence / 0 inference** | **6 / 6** |
| mode 2 (hedged) | labelled `evidence` | **labelled `inference`** |
| mode 1 (yes/no) | labelled `evidence` | **STILL `evidence`** (guard demoted) |
| rejected | 3/21 | 9/21 |

**Mode 2 is fixed by the model.** The hedged-and-disclaimed utterance above
now comes back `inference`, and the claim itself is rewritten to carry the
hedge ("He THINKS it was the settings page, but is not sure"). Three more
hedged claims likewise.

**Mode 1 is NOT.** Opus 5 makes both yes/no errors the weaker model made --
"He keeps a spare laptop beside the console." still cited to "Yep." and still labelled
`evidence`. The guard caught both, on fresh output from a model that had
never seen it. So the two fixes are complementary and neither replaces the
other; a stronger model does not retire this guard.

**⚠️ The cost is a REJECTION SPIKE, and its cause is a defect in THIS
module, not in the model.** 9/21 rejected, 8 of them `unlocatable-quote`.
The transcriber splits segments MID-WORD -- in the shape of seq 40 ending
`'and then it starts to down'` and seq 41 beginning `'load in the
background'` -- and `_joined_segment_text` SPACE-joins them, producing
`'down load'`. A genuinely verbatim quote of `'download'` can then never
match its own span. Proven: with the join seam closed, the same quote
matches. Opus 5 quotes
longer, more natural spans, so it hits these seams far more often -- it did
not get worse at quoting, it got better, and this module punishes it for
that.

**This also corrects the earlier `MAX_SPAN_SEGMENTS = 6` diagnosis.** The cap
is real but is not the main driver: these rejections sit at or under the cap
and fail on the seam. A fix must decide what to do about word-fragment joins
WITHOUT weakening the substring check into a fuzzy one (see
`_normalize_for_quote_match`'s own reasoning on that asymmetry) -- NOT YET
DECIDED, and deliberately not done in the same pass as the measurement.
"""

from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

from app.coach import prompts, reasoning_client
from app.coach.context import PROVENANCE_EVIDENCE, PROVENANCE_INFERENCE, PROVENANCE_VALUES
from app.coach.corpus import (
    ALL_STANCES,
    MAX_SPAN_SEGMENTS,
    Bearing,
    Fact,
    Hypothesis,
    SpanValidityError,
    validate_citation_span,
)
from app.session.store import SegmentRecord

logger = logging.getLogger(__name__)

EXTRACT_TOOL_NAME = "extract_facts"

# Same untrusted-stop-reason set `generate.py`/`critic.py` already use (§D5's
# rule, applied identically here): the WHOLE response contributes nothing.
_UNTRUSTED_STOP_REASONS = frozenset(
    {"refusal", "max_tokens", "pause_turn", "model_context_window_exceeded"}
)

REASON_UNLOCATABLE_QUOTE = "unlocatable-quote"
# STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: the three named span
# errors `app.coach.corpus.validate_citation_span` can raise, and the local
# span-length cap check below it -- IMPORTED from `SpanValidityError.code`
# verbatim (never a second, literal copy of the same string) so this
# pipeline's `Rejection.reason` can never drift from the SAME conditions
# `corpus.load_facts` checks, on the identical "imported, never redefined"
# precedent `app/web/author.py:45-47`/`corpus.py:27-31` already establish.
REASON_UNRESOLVED_CITATION = SpanValidityError.SPAN_UNRESOLVED
REASON_MIXED_SPEAKER_SPAN = SpanValidityError.SPAN_MIXED_SPEAKER
REASON_NONCONTIGUOUS_SPAN = SpanValidityError.SPAN_NONCONTIGUOUS
REASON_SPAN_TOO_LONG = "span-too-long"

# NOT a rejection reason. The one DEMOTION reason (see `Demotion`, and the A2
# section of this module's docstring): a quote that states nothing cannot
# support a claim whose content is not in it, so the fact is relabelled
# `inference` and kept.
REASON_CONTENTLESS_QUOTE = "contentless-quote"

_CLAIM_TRUNCATE_CHARS = 160

# Affirmation, negation, disfluency, and the closed-class function words that
# carry no propositional content on their own. A quote made of NOTHING but
# these says nothing WITHOUT the question it answered -- and the question was
# the interviewer's, which is exactly why the claim's content is not in the
# quote.
#
# ⚠️ MEASURED, not guessed, and the measurement is pinned by
# `test_the_guard_flags_only_the_two_known_facts_across_every_real_corpus_on_this_machine`:
# over BOTH real fact corpora on the maintainer's machine (50 evidence facts) this set
# flags 2 and produces 0 false positives, and the 2 are exactly the pair
# hand-adjudicated as mislabelled. Every nearest miss left alone carried
# exactly ONE content word inside a run of particles -- "That can be
# annoying.", and two more of that same shape (synthesized here as "and then
# I had my laptop." and "So then I just rebooted.", because the real two are
# personal) -- one content word is enough, by design. Do not extend this set
# without re-running that measurement; the last lexical heuristic tried here
# (a hedge/conditional scan) was 100% false positives on the same data.
#
# WIDENED 2026-08-16 (audit finding 4), by the SAME procedure. The first
# version missed eight pure yes/no answers, each on exactly ONE absent token:
# "Uh-huh." (huh), "For sure." (for), "Pretty much." (pretty), "Not at all."
# (at, all), "No way." (way), "Sure thing." (thing), "More or less." (more,
# less), "A little bit." (little). Every one of those is mode 1 -- a claim
# whose content came from the interviewer's question and was only confirmed --
# and every one was being ACCEPTED AS EVIDENCE. Re-measured over both corpora
# after adding the ten tokens: still 46 facts, still exactly 2 flagged, still
# the same 2. The added words are the risky kind -- "way", "thing", "all" and
# "little" are content words in most sentences -- which is precisely why the
# corpus was re-run rather than reasoned about, and why the near-miss pins
# ("That can be annoying.", "I think so.") exist as their own tests.
_CONTENTLESS_TOKENS = frozenset(
    {
        # affirmation
        "yes", "yeah", "yea", "yep", "yup", "ya", "aye", "ok", "okay", "sure",
        "right", "correct", "exactly", "absolutely", "definitely", "totally",
        "true", "indeed", "certainly", "obviously", "course", "huh",
        # negation
        "no", "nope", "nah", "not", "never", "none", "neither", "nor",
        # disfluency and filler
        "uh", "um", "umm", "uhh", "mm", "mmm", "mhm", "hmm", "hm", "er", "erm",
        "ah", "oh", "eh", "well", "like", "so", "just", "really", "actually",
        "kind", "sort", "bit", "quite", "very", "much", "too", "also",
        # the degree/quantity hedges an answer can be made ENTIRELY of --
        # "pretty much", "more or less", "a little bit", "not at all". Alone
        # they scale an answer without stating what is being scaled; the thing
        # scaled was in the interviewer's question.
        "pretty", "more", "less", "little", "all", "at", "for",
        # the nouns that only ever complete a fixed affirmation/negation
        # idiom in these quotes -- "no way", "sure thing". Both are genuine
        # content words elsewhere, and both were re-measured over the real
        # corpora before being added.
        "way", "thing",
        # closed-class function words and the fragments contractions leave
        # behind once punctuation is stripped ("don't" -> "don", "t")
        "i", "you", "we", "they", "he", "she", "it", "that", "this", "there",
        "do", "does", "did", "don", "doesn", "didn", "is", "was", "are", "were",
        "am", "be", "been", "have", "has", "had", "haven", "hasn", "hadn",
        "can", "cant", "could", "couldn", "will", "won", "would", "wouldn",
        "should", "shouldn", "t", "s", "m", "re", "ve", "ll", "d",
        "a", "an", "the", "of", "to", "and", "but", "or", "if", "then",
        "my", "your", "our", "their", "his", "her", "its",
    }
)

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def quote_states_nothing(quoted_text: str) -> bool:
    """True when `quoted_text`, stripped of punctuation and cased down, holds
    no token outside `_CONTENTLESS_TOKENS` -- "Yep.", "Not really. Uh, no.",
    "Yeah, exactly.", "Uh-huh.", "Not at all.", "More or less." One content
    token is enough to be content: "That can be annoying." is not flagged, and
    neither is the hedge "I think so." (mode 2, out of scope by design).

    Pure, and deliberately importable: the corpus measurement that justifies
    the token set runs THIS function, never a copy of it, on the standing rule
    that a harness measuring a version of the system that no longer exists is
    this project's recurring failure.

    ⚠️ Non-Latin scripts: `\\w` keeps Hebrew and every other alphabetic token,
    and no Hebrew word is in the set, so a Hebrew quote can never be flagged.
    That is a conservative failure (never demoting) rather than a wrong one,
    and it matches the 100%-English decision of 2026-08-14."""
    normalized = unicodedata.normalize("NFC", quoted_text).casefold()
    without_punctuation = _NON_WORD.sub(" ", normalized)
    return not any(token not in _CONTENTLESS_TOKENS for token in without_punctuation.split())


class ExtractFailure(Exception):
    """Raised for anything that makes the WHOLE extraction response
    untrusted: a timeout, a parse failure, a refusal, a truncation, or any
    other exception from the call itself. NEVER carries claim or transcript
    text -- only `reason` (a short label) and `stop_reason` (the response's
    own value, or `None`)."""

    def __init__(self, *, reason: str, stop_reason: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stop_reason = stop_reason


@dataclass(frozen=True)
class _RawFact:
    """One fact as the model returned it, before the deterministic
    validation pipeline validates its citation span (STATE.md § FIRST
    REAL-SPEECH EVIDENCE FINDING 1) and (§K4) its quote."""

    claim: str
    seqs: "tuple[int, ...]"
    provenance: str
    quoted_text: str
    bearings: "tuple[tuple[str, str], ...]"


@dataclass(frozen=True)
class Rejection:
    """One fact the validation pipeline refused, named for the CLI report
    (§K5's "an ERROR naming the fact"). `claim` is truncated for
    readability; `reason` is a short, machine-readable code that keeps every
    rejection class (span-too-long, unresolved citation, mixed-speaker span,
    non-contiguous span, unlocatable quote) from ever being conflated in the
    report. `seqs` is the fact's own cited span, preserved exactly as
    returned (never sorted or deduplicated -- this is a report field, not a
    stored citation)."""

    claim: str
    reason: str
    seqs: "Optional[tuple[int, ...]]" = None
    quoted_text: Optional[str] = None


@dataclass(frozen=True)
class Demotion:
    """One fact whose `provenance` this pipeline OVERRODE, from `evidence` to
    `inference`, and then ACCEPTED anyway.

    Deliberately NOT a `Rejection`, and deliberately not folded into that
    tuple: a rejection deletes a fact and drives the non-zero exit code, and
    a demoted claim is usually TRUE -- what was wrong is the label. Rejecting
    it would throw away a real finding to punish a labelling error, which is
    the more expensive of the two mistakes. Reported separately so a run's
    demotion count is legible on its own; it is a quality signal about the
    EXTRACTION MODEL, not about the interview."""

    claim: str
    reason: str
    seqs: "Optional[tuple[int, ...]]" = None
    quoted_text: Optional[str] = None


@dataclass(frozen=True)
class ExtractionResult:
    """Accepted facts and rejections, kept separate (§K5): accepted facts
    are written to the corpus; rejections are printed and drive the non-zero
    exit code. `dropped_hypothesis_warnings` names any `related_hypothesis_id`
    that matched no current hypothesis -- a WARNING, never a rejection (§K8):
    a user deleting a hypothesis must not invalidate facts already
    extracted.

    `usage_input_tokens`/`usage_output_tokens` (05-02-PLAN.md §K8) are read
    OFF THE RESPONSE'S OWN usage figures, defensively -- `StubAnthropicClient`
    carries no `.usage` at all, and every offline test in the suite drives
    this path, so a missing usage object must never turn a successful
    extraction into a crash. `None` means "this response did not report it",
    never "zero tokens were used". 05-01-PLAN.md §K10's projected
    per-interview cost is arithmetic, not a measurement -- these are how it
    gets checked against reality, on every real run."""

    accepted: "tuple[Fact, ...]"
    rejections: "tuple[Rejection, ...]"
    dropped_hypothesis_warnings: "tuple[str, ...]" = ()
    demotions: "tuple[Demotion, ...]" = ()
    usage_input_tokens: Optional[int] = None
    usage_output_tokens: Optional[int] = None


def _truncate_claim(claim: str) -> str:
    if len(claim) <= _CLAIM_TRUNCATE_CHARS:
        return claim
    return claim[:_CLAIM_TRUNCATE_CHARS].rstrip() + "..."


def _extract_tool_schema() -> dict:
    """Fixed, module-level shape (`generate.py:95-121`'s own precedent) --
    never built per-call from anything that varies. The provenance enum
    comes from the IMPORTED tuple, sorted, never a literal pair (§K3).

    STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: `seq` (one integer)
    became `seqs` (an array, `minItems: 1`/`maxItems: MAX_SPAN_SEGMENTS`,
    imported from `app.coach.corpus` rather than a second literal `6`) --
    gate 18 (`app/tests/test_phase_gates.py`) pins this schema's provenance
    enum and required-field set, updated in the SAME commit as this change."""
    return {
        "name": EXTRACT_TOOL_NAME,
        "description": (
            "Extract every distinct claim the interviewee made in this transcript, each "
            "cited to the segment(s) that support it, tagged evidence or inference, and "
            "linked to any hypothesis it bears on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "seqs": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 1,
                                "maxItems": MAX_SPAN_SEGMENTS,
                                "description": (
                                    "The seq(s) of the interviewee transcript segment(s) this claim is "
                                    "cited to, in increasing order. Cite the SMALLEST span that contains "
                                    "your quoted_text -- usually just one segment, but the transcription "
                                    "engine sometimes splits a single continuous utterance into several "
                                    "short segments from the SAME speaker said back to back; when that "
                                    "happens, cite every segment your quote actually spans, never a wider "
                                    f"span than it needs, and never more than {MAX_SPAN_SEGMENTS}."
                                ),
                            },
                            "provenance": {
                                "type": "string",
                                "enum": sorted(PROVENANCE_VALUES),
                                # Every OTHER field in this schema carries a
                                # description; this one carried none, and it is
                                # the field the model gets wrong. The full rule
                                # is taught in the system block
                                # (`prompts.extraction_rubric_block`, worked
                                # example 4) -- this is the tie-break restated
                                # at the point of decision, where the model is
                                # actually filling the field in.
                                "description": (
                                    "'evidence' ONLY if the cited segments' own words directly "
                                    "state this claim -- a statement about what IS. 'inference' "
                                    "for anything you concluded, combined from separate moments, "
                                    "or that is speculative, predictive, hypothetical or "
                                    "conditional -- even when it was said outright, in one "
                                    "segment, and you could quote it word for word. Being "
                                    "quotable is NOT what makes a claim evidence. When the rules "
                                    "disagree, or you are unsure, choose 'inference': a claim "
                                    "wrongly marked inference costs a little confidence, and a "
                                    "claim wrongly marked evidence is a guess wearing the clothes "
                                    "of a fact, and it will be acted on."
                                ),
                            },
                            "quoted_text": {
                                "type": "string",
                                "description": (
                                    "For an evidence claim, a snippet copied verbatim from the cited "
                                    "segment(s)' own text, joined in seq order. For an inference claim, "
                                    "may be empty."
                                ),
                            },
                            "bearings": {
                                "type": "array",
                                "description": (
                                    "Any current hypothesis this claim bears on, WITH THE "
                                    "DIRECTION. May be empty. Judge the direction against the "
                                    "hypothesis's own falsifier, not against whether the claim "
                                    "sounds positive. Use 'unclear' when the claim is genuinely "
                                    "relevant but does not settle anything -- do NOT guess a "
                                    "direction to be helpful."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "hypothesis_id": {"type": "string"},
                                        "stance": {
                                            "type": "string",
                                            "enum": ["supports", "undermines", "unclear"],
                                            "description": (
                                                "'undermines' if the claim makes the hypothesis "
                                                "less likely or meets its falsifier; 'supports' if "
                                                "more likely; 'unclear' otherwise."
                                            ),
                                        },
                                    },
                                    "required": ["hypothesis_id", "stance"],
                                },
                            },
                        },
                        "required": ["claim", "seqs", "provenance", "quoted_text", "bearings"],
                    },
                },
            },
            "required": ["facts"],
        },
    }


def assemble_extract_request(
    *,
    transcript: Sequence[SegmentRecord],
    hypotheses: Sequence[Hypothesis],
    model: str,
    max_tokens: int,
) -> dict:
    """Builds the exact kwargs for `client.messages.create(...)`. Pure: no
    I/O, no client, fully unit-testable by inspecting the returned dict
    directly (`generate.py:124-176`'s own precedent)."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": prompts.extraction_rubric_block(hypotheses=hypotheses),
        "messages": [
            {
                "role": "user",
                "content": (
                    prompts.extraction_transcript_block(transcript)
                    + "\n\n"
                    + prompts.extraction_decision_instruction(tool_name=EXTRACT_TOOL_NAME)
                ),
            }
        ],
        "tools": [_extract_tool_schema()],
        "tool_choice": {"type": "tool", "name": EXTRACT_TOOL_NAME},
    }


def _parse_extract_response(response: Any) -> "list[_RawFact]":
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason in _UNTRUSTED_STOP_REASONS:
        raise ExtractFailure(reason="untrusted-stop-reason", stop_reason=stop_reason)

    for block in getattr(response, "content", None) or ():
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == EXTRACT_TOOL_NAME:
            try:
                raw_facts = block.input["facts"]
                if not isinstance(raw_facts, list):
                    raise ValueError("facts must be an array")
                parsed: "list[_RawFact]" = []
                for entry in raw_facts:
                    claim = entry["claim"]
                    if not isinstance(claim, str):
                        raise ValueError("claim must be a string")
                    seqs_raw = entry["seqs"]
                    # A SHAPE check only (mirrors the pre-existing single-`seq`
                    # int check this replaces) -- non-empty and every element an
                    # integer. The span-LENGTH cap
                    # (`app.coach.corpus.MAX_SPAN_SEGMENTS`) is deliberately NOT
                    # checked here: an over-long span is a per-fact quality
                    # judgment (REASON_SPAN_TOO_LONG, below), not a malformed
                    # response, so one bad fact's oversized span must not sink
                    # every OTHER fact in the same extraction call.
                    if not isinstance(seqs_raw, list) or not seqs_raw:
                        raise ValueError("seqs must be a non-empty array of integers")
                    seqs: "list[int]" = []
                    for seq_item in seqs_raw:
                        # bool is a subclass of int -- exclude it explicitly, matching
                        # critic.py's own guard against a well-typed-looking bad value.
                        if not isinstance(seq_item, int) or isinstance(seq_item, bool):
                            raise ValueError("seqs must be an array of integers")
                        seqs.append(seq_item)
                    provenance = entry["provenance"]
                    if provenance not in PROVENANCE_VALUES:
                        raise ValueError(f"unrecognized provenance {provenance!r}")
                    quoted_text = entry["quoted_text"]
                    if not isinstance(quoted_text, str):
                        raise ValueError("quoted_text must be a string")
                    raw_bearings = entry["bearings"]
                    if not isinstance(raw_bearings, list):
                        raise ValueError("bearings must be an array")
                    bearings: "list[tuple[str, str]]" = []
                    for bearing in raw_bearings:
                        if not isinstance(bearing, dict):
                            raise ValueError("each bearing must be an object")
                        hypothesis_id = bearing.get("hypothesis_id")
                        stance = bearing.get("stance")
                        if not isinstance(hypothesis_id, str) or not isinstance(stance, str):
                            raise ValueError("a bearing needs a string hypothesis_id and stance")
                        if stance not in ALL_STANCES:
                            raise ValueError(f"unrecognized stance {stance!r}")
                        bearings.append((hypothesis_id, stance))
                    parsed.append(
                        _RawFact(
                            claim=claim,
                            seqs=tuple(seqs),
                            provenance=provenance,
                            quoted_text=quoted_text,
                            bearings=tuple(bearings),
                        )
                    )
                return parsed
            except (KeyError, TypeError, ValueError) as exc:
                raise ExtractFailure(reason="parse-error", stop_reason=stop_reason) from exc

    raise ExtractFailure(reason="no-tool-use-block", stop_reason=stop_reason)


def _normalize_for_quote_match(text: str) -> str:
    """One small pure function, applied identically to both sides before
    comparison (§K4). Unicode NFC composition, removal of the invisible bidi
    control marks that can appear in mixed Hebrew/English text without being
    visible, collapse of every run of Unicode whitespace to a single space,
    strip, casefold. NOTHING ELSE -- no punctuation stripping, no fuzzy
    threshold, no edit-distance tolerance.

    The reason is an asymmetry of costs (§K4): a false ACCEPT -- a quote that
    "nearly" matches -- silently promotes an inference to evidence, which is
    the one outcome D-07 exists to prevent. A false REJECT costs the user one
    named, visible line in the extraction report about a fact they can re-run
    or hand-check. Phase 1's benchmark (how much transcription noise a real
    Hebrew interview carries) is still unrun, so a fuzzy tolerance chosen now
    would be a number with no evidence behind it -- strict now, with the
    failure visible; loosened later against measured evidence, if ever
    needed."""
    composed = unicodedata.normalize("NFC", text)
    # Bidi control marks (LRM, RLM, ALM, and the embedding/override/isolate
    # family) are invisible in a rendered UI but are real characters in the
    # string -- strip them before comparing, on both sides, identically.
    _BIDI_MARKS = (
        "\u200e\u200f"  # LRM, RLM
        "\u202a\u202b\u202c\u202d\u202e"  # embeddings and overrides
        "\u2066\u2067\u2068\u2069"  # isolates
        "\u061c"  # ALM
    )
    stripped = "".join(ch for ch in composed if ch not in _BIDI_MARKS)
    collapsed = " ".join(stripped.split())
    return collapsed.strip().casefold()


def _joined_segment_text(seqs: Sequence[int], segments_by_seq: Mapping[int, SegmentRecord]) -> str:
    """The evidence-quote comparison text for a citation span (STATE.md §
    FIRST REAL-SPEECH EVIDENCE FINDING 1): the cited segments' own `.text`,
    SPACE-JOINED, IN SEQ ORDER -- never in the order the model happened to
    list them. Callers must only pass `seqs` that already passed
    `validate_citation_span` (every seq resolves), so every lookup here is
    safe by construction. A single-seq span (`len(seqs) == 1`) reduces to
    exactly that one segment's own text, unchanged from the pre-span
    behaviour this replaces."""
    return " ".join(segments_by_seq[seq].text for seq in sorted(seqs))


def _validate_and_build(
    raw_facts: Sequence[_RawFact],
    *,
    session_id: str,
    segments_by_seq: Mapping[int, SegmentRecord],
    known_hypothesis_ids: "set[str]",
) -> ExtractionResult:
    """The deterministic validation pipeline (§K5/§K4/STATE.md § FIRST
    REAL-SPEECH EVIDENCE FINDING 1), no second model call: validate every
    citation SPAN against the session's real segments (span-length cap,
    resolution, same-speaker, contiguity-within-speaker -- the last three via
    `app.coach.corpus.validate_citation_span`, shared with that module's own
    loader); for an evidence claim, require `quoted_text` to be a locatable
    substring of the CITED SPAN's own space-joined text, in seq order, after
    normalization; drop and name anything that fails. Related-hypothesis ids
    that match nothing current are dropped from the tuple and named as a
    warning -- the fact itself is still accepted (§K8)."""
    accepted: "list[Fact]" = []
    rejections: "list[Rejection]" = []
    warnings: "list[str]" = []
    demotions: "list[Demotion]" = []

    speaker_by_seq = {seq: segment.speaker for seq, segment in segments_by_seq.items()}

    for raw in raw_facts:
        truncated_claim = _truncate_claim(raw.claim)

        # The one structural condition NOT covered by `validate_citation_span`
        # (that function only checks the three SESSION-DEPENDENT conditions;
        # the cap is structural, per `corpus.Fact`'s own field validator) --
        # checked here, per-fact, so one oversized span never sinks every
        # OTHER fact in the same extraction call (§K5's own "name and drop
        # only the offending fact" rule).
        if len(raw.seqs) > MAX_SPAN_SEGMENTS:
            rejections.append(
                Rejection(claim=truncated_claim, reason=REASON_SPAN_TOO_LONG, seqs=raw.seqs)
            )
            continue

        try:
            validate_citation_span(raw.seqs, speaker_by_seq=speaker_by_seq)
        except SpanValidityError as exc:
            rejections.append(Rejection(claim=truncated_claim, reason=exc.code, seqs=raw.seqs))
            continue

        provenance = raw.provenance

        if provenance == PROVENANCE_EVIDENCE:
            quote_normalized = _normalize_for_quote_match(raw.quoted_text)
            span_normalized = _normalize_for_quote_match(_joined_segment_text(raw.seqs, segments_by_seq))
            if not quote_normalized or quote_normalized not in span_normalized:
                rejections.append(
                    Rejection(
                        claim=truncated_claim,
                        reason=REASON_UNLOCATABLE_QUOTE,
                        seqs=raw.seqs,
                        quoted_text=raw.quoted_text,
                    )
                )
                continue

            # ORDER IS LOAD-BEARING: locatability is checked FIRST, against
            # the model's OWN label, and only a quote that really is in the
            # cited span can reach this line. Demoting before that check
            # would launder a FABRICATED "Yep." into an accepted inference --
            # the demotion path performs no quote check of its own, because
            # `inference` facts are exempt from it by design.
            if quote_states_nothing(raw.quoted_text):
                provenance = PROVENANCE_INFERENCE
                demotions.append(
                    Demotion(
                        claim=truncated_claim,
                        reason=REASON_CONTENTLESS_QUOTE,
                        seqs=raw.seqs,
                        quoted_text=raw.quoted_text,
                    )
                )

        related = tuple(
            Bearing(hypothesis_id=hid, stance=stance)
            for hid, stance in raw.bearings
            if hid in known_hypothesis_ids
        )
        for dropped_id, _stance in raw.bearings:
            if dropped_id not in known_hypothesis_ids:
                warnings.append(
                    f"fact {truncated_claim!r} referenced hypothesis "
                    f"{dropped_id!r}, which matches no current hypothesis -- dropped"
                )

        accepted.append(
            Fact(
                id=str(uuid.uuid4()),
                session_id=session_id,
                seqs=raw.seqs,
                claim=raw.claim,
                provenance=provenance,
                quoted_text=raw.quoted_text,
                bearings=related,
            )
        )

    return ExtractionResult(
        accepted=tuple(accepted),
        rejections=tuple(rejections),
        dropped_hypothesis_warnings=tuple(warnings),
        demotions=tuple(demotions),
    )


def extract_facts(
    client: Any,
    *,
    session_id: str,
    transcript: Sequence[SegmentRecord],
    hypotheses: Sequence[Hypothesis],
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> ExtractionResult:
    """Drives ONE extraction call through the shared client seam (§K6). Takes
    `client` by injection (real or `app.coach.stub.StubAnthropicClient`).
    `transcript` is the session's own segment records, already read through
    `app.session.reader.read_session` by the CALLER -- this function never
    reads a file itself. Raises `ExtractFailure` for anything that makes the
    WHOLE response untrusted; otherwise returns an `ExtractionResult` whose
    `accepted` facts have already passed the deterministic validation
    pipeline (§K5)."""
    request = assemble_extract_request(
        transcript=transcript, hypotheses=hypotheses, model=model, max_tokens=max_tokens
    )
    bound = reasoning_client.with_timeout(client, timeout_s)
    try:
        response = bound.messages.create(**request)
    except ExtractFailure:
        raise
    except Exception as exc:  # noqa: BLE001 -- a timeout, a connection error, anything --
        raise ExtractFailure(reason="exception", stop_reason=None) from exc
    raw_facts = _parse_extract_response(response)

    segments_by_seq = {segment.seq: segment for segment in transcript}
    known_hypothesis_ids = {hypothesis.id for hypothesis in hypotheses}
    result = _validate_and_build(
        raw_facts,
        session_id=session_id,
        segments_by_seq=segments_by_seq,
        known_hypothesis_ids=known_hypothesis_ids,
    )
    # §K8: read defensively -- `response.usage` may be absent entirely (the
    # offline stub carries none), and a missing usage object must never turn
    # a successful extraction into a crash. `getattr` chains, never `[...]`.
    usage = getattr(response, "usage", None)
    usage_input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    usage_output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    result = replace(result, usage_input_tokens=usage_input_tokens, usage_output_tokens=usage_output_tokens)
    logger.info(
        "extraction complete session_id=%s accepted=%d rejected=%d demoted=%d warnings=%d "
        "usage_input_tokens=%s usage_output_tokens=%s",
        session_id, len(result.accepted), len(result.rejections), len(result.demotions),
        len(result.dropped_hypothesis_warnings),
        usage_input_tokens, usage_output_tokens,
    )
    return result
