"""The artifact a human actually reads after the call (phase_specification U2,
SESS-03). Built on `app.session.reader` -- it never parses the transcript
file itself, so the tolerant-parsing behaviour (a truncated tail, a
never-closed session) is implemented exactly once.

The bidi rule (U2, and the single most important rule in this module): order
BLOCKS by session time; never reorder or annotate the TEXT inside one block.
Hebrew and English mixed inside one utterance already resolve correctly under
the reader's own bidi algorithm -- any attempt at "fixing" direction in plain
text corrupts exactly the code-switched line this product exists to capture.
Every place this module touches segment text, it is reproduced verbatim.

04-03-PLAN.md §E6 (COACH-04): this module is EXTENDED, not duplicated, to
render `SuggestionRecord`s and the coach's coverage-transition events inline,
in the SAME session-time order as everything else -- a second report would be
a second thing to keep ordered, and ordering is this module's single most
emphasized invariant. Every suggestion the coach ever produced is shown --
including the ones dropped or ranked down -- because `reasoning.md` risk 2 is
that critic over-rejection is invisible and looks identical to a weak
generator, and the per-rule flags on the dropped ones are the only signal
that tells the two apart. The FULL coverage transition history renders too,
never just a final per-item status, because `04-RESEARCH.md`'s Pitfall 1 is a
false-positive match silently removing a survey item for the rest of the
interview, catchable only here.

06-03-PLAN.md §H4 (GC-02/GC-05/GC-08): extended again, still never duplicated,
to resolve each suggestion's `grounding_ids` back to the past-interview facts
that grounded it, and to render a coach-proposed hypothesis-state change as an
explicit, un-applied proposal. `render_session`/`write_rendered_transcript`
gain a REQUIRED, explicitly nullable `context_dir` keyword -- the same
precedent `corpus.load_facts(sessions_root=...)` set -- so every caller
consciously decides whether grounding resolution runs rather than inheriting
it from a default. Resolution goes through `corpus.load_facts`/
`load_hypotheses` directly, never a second parser, and NEVER raises: a missing
or malformed corpus, or no `context_dir` at all, is tolerated here and named
with a note, because nothing downstream of a post-call read acts on the
resolved data. **This is the ONE place in this codebase a past interview's
verbatim `quoted_text` is ever printed** (gate 20/21 keep it off every live
prompt and the phone wire) -- offline, after the call, on the operator's own
machine, to the person who ran the interview and authored the corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from app.coach import corpus as coach_corpus_mod
from app.coach.context import FACT_ID_PREFIX
from app.session.bleed import detect_bleed
from app.session.reader import SessionReadResult, read_session
from app.session.mic_skew import INTERVIEWER, correct_interviewer_latencies
from app.session.store import latency_stats, latency_stats_with_modes

TRANSCRIPT_DOC_FILENAME = "transcript.md"

EVENT_COVERAGE_ASKED = "coverage-asked"
EVENT_COVERAGE_CONSIDERED = "coverage-considered"

# The named per-rule flags every suggestion record carries (04-02-PLAN.md
# §D3/COACH-04; grown to five by 06-03-PLAN.md §H1/§H2, GC-04, and to six by
# `is_generic` on 2026-08-15, and to seven by `is_off_topic` on
# 2026-08-19) -- listed once
# here so the render/summary code below never hand-types the set twice and
# risks the two drifting apart. Pinned, alongside four sibling surfaces
# (the critique schema's properties and its hand-typed required list, the
# `Judgment` dataclass, `SuggestionRecord`'s own fields), by gate 24's
# cross-surface set-equality assertion (`test_phase_gates.py`).
_SUGGESTION_FLAG_FIELDS = (
    "is_leading",
    "is_hypothetical_or_future_tense",
    "is_pitch",
    "is_opinion_seeking",
    "is_confirmation_only",
    "is_generic",
    "is_off_topic",
)


def _build_session_time_fn(records: "tuple[dict[str, Any], ...]") -> "Callable[[dict[str, Any]], float]":
    """Builds the session-time function for ONE render pass. A suggestion
    record carries neither `audio_end_s` nor `at_s` -- it is anchored to the
    audio_end_s of the INTERVIEWEE segment that triggered it (`trigger_seq`),
    since a suggestion is generated immediately after that utterance
    finishes. Ties (a suggestion sharing its trigger's exact audio_end_s) are
    broken by the record's own `seq` in the caller's sort key -- the trigger
    segment's `seq` is always lower, since it was durably written first."""
    audio_end_by_seq: "dict[int, float]" = {
        r["seq"]: float(r["audio_end_s"]) for r in records if r.get("kind") == "segment"
    }

    def _session_time(record: dict[str, Any]) -> float:
        """A single, comparable "elapsed since session start" number across
        all three record kinds.

        Segments order on `audio_end_s`, NOT `audio_start_s`. That is
        deliberate and load-bearing: the engines' `Segment` carries only an
        end time, so `StreamWorker._persist` fabricates a start as the end of
        the SAME speaker's previous segment -- a value that silently absorbs
        however long that speaker stayed quiet. Ordering across speakers on
        it misorders every session with normal turn-taking: while one speaker
        delivers a long answer, the other's stale start stays far in the
        past, so their next question sorts before the answer it followed.
        `audio_end_s` is engine-reported and real, monotone within a stream
        and truthful across streams.

        Events carry `at_s` (wall-clock elapsed since the store opened). Per
        phase_specification S3 the capture pacer keeps the two clocks
        aligned, so ordering by whichever one a record carries produces the
        true session-time order regardless of which speaker's worker
        happened to finish sending the segment to the store first (append
        order is NOT session-time order under concurrent stream workers)."""
        kind = record.get("kind")
        if kind == "segment":
            return float(record.get("audio_end_s", 0.0))
        if kind == "suggestion":
            trigger_seq = record.get("trigger_seq")
            if trigger_seq in audio_end_by_seq:
                return audio_end_by_seq[trigger_seq]
            return 0.0  # defensive only -- every suggestion has a real trigger in practice
        return float(record.get("at_s", 0.0))

    return _session_time


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _speaker_label(speaker: Optional[str]) -> str:
    if speaker is None:
        return "session"
    return speaker


def _render_latency_section(result: SessionReadResult) -> list[str]:
    lines = ["## Latency summary (end-of-utterance to segment-ready)", ""]
    latencies: dict[str, list[float]] = {}
    for record in result.records_of_kind("segment"):
        latencies.setdefault(record["speaker"], []).append(float(record["latency_s"]))

    if not latencies:
        lines.append("No segments were captured, so no latency figures exist.")
        return lines

    # Releasing the microphone freezes the interviewer's AUDIO clock while the
    # wall clock keeps running, so every later interviewer `latency_s` on disk
    # is inflated by the whole off-duration -- a plausible-looking float in a
    # field that was trustworthy all session. Corrected here, from the
    # `microphone` events in the same transcript, rather than trusting the
    # stored value. Said out loud when it fires, because a reader comparing this
    # document against `session.json`'s own summary would otherwise find two
    # different numbers for one session and no explanation.
    correction = correct_interviewer_latencies(result.records)
    if correction.applied:
        latencies[INTERVIEWER] = list(correction.corrected)
        lines.append(
            f"The microphone was released {correction.windows} time(s) for "
            f"{correction.total_off_s:.0f}s in total. Interviewer figures below are "
            f"CORRECTED for that: the audio clock stops while the device is off, so "
            f"the stored `latency_s` (and `session.json`'s summary) are inflated by it."
        )
        lines.append("")

    bimodal_seen = False
    for speaker in sorted(latencies):
        stats = latency_stats_with_modes(latencies[speaker])
        if stats["count"] < 2:
            lines.append(
                f"- **{speaker}**: only {stats['count']} segment(s) -- too few for a "
                f"median/p95 to mean anything."
            )
            continue
        lines.append(
            f"- **{speaker}**: count={stats['count']}, "
            f"median={stats['median_s']:.2f}s, p95={stats['p95_s']:.2f}s "
            f"(min={stats['min_s']:.2f}s, max={stats['max_s']:.2f}s)"
        )
        # Only when the session actually contains both, so an ordinary
        # single-mode recording reads exactly as it always has.
        if stats.get("is_bimodal"):
            bimodal_seen = True
            lines.append(
                f"  - finalized at a pause: {stats['endpoint_final_count']} segment(s), "
                f"median {stats['endpoint_final_median_s']:.2f}s"
            )
            lines.append(
                f"  - finalized mid-speech: {stats['window_final_count']} segment(s), "
                f"median {stats['window_final_median_s']:.2f}s"
            )
    if bimodal_seen:
        # Written out in full because the single median above is the number a
        # reader will quote, and on a bimodal session it describes neither
        # population. A "2.6x regression" was investigated on 2026-08-12 before
        # anyone noticed the distribution had two humps in it.
        lines.append("")
        lines.append(
            "> **On the two figures above.** Latency here is bimodal, and which mode a segment "
            "lands in is decided by the speech, not by this machine. A pause lets the engine "
            "close off what it is holding, so that segment is ready almost at once. Continuous "
            "speech gives it no such boundary, and tokens are finalized only as they age out of "
            "a rolling window -- a steady offset behind the live audio, which recovers the "
            "instant someone stops for breath. **A session's single median therefore mostly "
            "reports how much the speakers paused.** Compare like with like before concluding "
            "anything has got slower: the two per-mode medians are the comparable numbers."
        )
    return lines


def _load_fact_lookup(context_dir: "Optional[Path]") -> "tuple[dict[str, Any], Optional[str]]":
    """06-03-PLAN.md §H4: resolves `context_dir`'s facts corpus into an
    `id -> Fact` lookup keyed by the SAME `fact:`-prefixed id a suggestion's
    `grounding_ids` carries (`app.coach.context.FACT_ID_PREFIX` + `Fact.id`)
    -- never a second parse of the artifact, and never the citation resolver
    (`sessions_root=None`), which would make a post-call read fail because a
    cited session was archived or moved.

    NEVER raises. `context_dir` being `None` (the caller consciously chose
    not to resolve) and a missing/malformed corpus are both tolerated here,
    each with its own explanatory note -- the ONE place in this codebase a
    malformed corpus does not raise, because nothing downstream of a
    post-call read acts on the resolved data."""
    if context_dir is None:
        return {}, "no context directory was provided for this render"
    try:
        facts = coach_corpus_mod.load_facts(Path(context_dir), sessions_root=None)
    except coach_corpus_mod.CorpusValidationError as exc:
        return {}, f"the facts corpus could not be read ({exc})"
    return {f"{FACT_ID_PREFIX}{fact.id}": fact for fact in facts}, None


def _load_hypothesis_lookup(context_dir: "Optional[Path]") -> "tuple[dict[str, Any], Optional[str]]":
    """The same tolerant shape as `_load_fact_lookup`, for the hypotheses
    corpus (`corpus.load_hypotheses`) -- resolves a proposal's
    `target_hypothesis_id` to its statement text. `missing_ok=True`: an
    absent hypotheses artifact is the normal D-02 empty-corpus case, not an
    error."""
    if context_dir is None:
        return {}, "no context directory was provided for this render"
    try:
        hypotheses = coach_corpus_mod.load_hypotheses(Path(context_dir), missing_ok=True)
    except coach_corpus_mod.CorpusValidationError as exc:
        return {}, f"the hypotheses corpus could not be read ({exc})"
    return {hypothesis.id: hypothesis for hypothesis in hypotheses}, None


def render_session(session_dir: Path, *, context_dir: "Optional[Path]") -> str:
    """Builds the full rendered document as text. Never writes anything --
    see `write_rendered_transcript` for the file-writing entry point used by
    `python -m app render`.

    06-03-PLAN.md §H4: `context_dir` is a REQUIRED, explicitly nullable
    keyword -- omitting it raises rather than silently skipping grounding-id
    resolution, mirroring `corpus.load_facts(sessions_root=...)`'s own
    precedent. Pass `None` for a session where resolution is not wanted or
    not possible; every render still produces the whole document either
    way."""
    result = read_session(session_dir)
    fact_lookup, fact_note = _load_fact_lookup(context_dir)
    hypothesis_lookup, hypothesis_note = _load_hypothesis_lookup(context_dir)

    lines: list[str] = [f"# Transcript -- {session_dir.name}", ""]

    if not result.closed:
        lines.append(
            "**This session was not closed cleanly** (a crash, or it is still "
            "running) -- everything captured up to the interruption is below."
        )
        lines.append("")
    bleed = detect_bleed([r for r in result.records if r.get("kind") == "segment"])
    if bleed.suspected:
        # ABOVE the transcript, not in a footer. Every attribution below this
        # line is in question, and a reader who has already read the transcript
        # has already believed it.
        lines.append(
            f"**The two tracks appear to be hearing each other.** "
            f"{bleed.duplicate_pairs} of {bleed.comparable_interviewer_segments} interviewer "
            f"segments ({bleed.percent}%) are near-verbatim copies of an interviewee segment "
            f"recorded at the same moment. The usual cause is the interviewee playing through "
            f"SPEAKERS rather than headphones, so the microphone hears them too -- which means "
            f"speech attributed to the interviewer below may not be theirs. It also feeds "
            f"survey coverage, so questions may be marked asked that nobody asked."
        )
        lines.append("")
        for mine, theirs in bleed.examples:
            lines.append(f"> interviewer: {mine}")
            lines.append(f"> interviewee: {theirs}")
            lines.append("")

    if result.truncated_tail is not None:
        lines.append(
            "**Note:** the transcript file's final line was truncated (an "
            "in-progress write interrupted by a crash) and has been dropped; "
            "every complete record before it is included below."
        )
        lines.append("")

    session_time = _build_session_time_fn(result.records)
    ordered = sorted(result.records, key=lambda r: (session_time(r), r.get("seq", 0)))

    if not ordered:
        lines.append("No segments or events were captured in this session.")
        lines.append("")
    else:
        lines.extend(
            _render_timeline(
                ordered, session_time, metadata=result.metadata,
                fact_lookup=fact_lookup, fact_note=fact_note,
                hypothesis_lookup=hypothesis_lookup, hypothesis_note=hypothesis_note,
            )
        )

    lines.extend(_render_latency_section(result))
    lines.append("")
    lines.extend(_render_coaching_summary_section(result))
    lines.append("")
    return "\n".join(lines)


def _survey_text_by_id(metadata: dict[str, Any]) -> "dict[str, str]":
    """Survey question text, for the coverage-transition lines below, read
    from the session's OWN metadata (`app/main.py` copies the brief/survey
    in at `start` for exactly this kind of post-call provenance, 04-01-PLAN.md
    §C6) -- never a second read of `context/survey.json`, which may have
    since been edited for a LATER session."""
    return {item["id"]: item["text"] for item in metadata.get("context_survey", []) if "id" in item}


def _segment_text_by_seq(records: "tuple[dict[str, Any], ...]") -> "dict[int, str]":
    return {r["seq"]: r["text"] for r in records if r.get("kind") == "segment"}


def _format_coverage_event_line(
    record: dict[str, Any], *, elapsed: str, survey_text_by_id: "dict[str, str]", segment_text_by_seq: "dict[int, str]"
) -> str:
    """04-RESEARCH.md Pitfall 1: a false-positive coverage match silently
    removes a survey item from attention for the rest of the interview, and
    is catchable only in this post-call read -- and only if it shows WHAT
    matched against WHAT, not merely that the item ended up asked. Segment
    and survey-question text are reproduced verbatim (same discipline as the
    transcript itself); nothing here is ever reordered or reflowed."""
    detail = record.get("detail") or {}
    if record["event"] == EVENT_COVERAGE_ASKED:
        question_id = detail.get("question_id")
        question_text = survey_text_by_id.get(question_id, "<unknown survey item>")
        matched_seq = detail.get("matched_seq")
        matched_text = segment_text_by_seq.get(matched_seq, "<segment not found>")
        return (
            f"> [{elapsed}] coverage: MATCHED survey item {question_id!r} "
            f"({question_text!r}) -- interviewer said ({matched_text!r})"
        )
    # coverage-considered: a non-match, equally worth showing (the "near
    # misses" half of the full transition history).
    matched_seq = detail.get("seq")
    considered_text = segment_text_by_seq.get(matched_seq, "<segment not found>")
    return f"> [{elapsed}] coverage: considered ({considered_text!r}) -- no survey item matched"


def _format_grounding_lines(
    record: dict[str, Any], *, fact_lookup: "dict[str, Any]", fact_note: "Optional[str]"
) -> "list[str]":
    """06-03-PLAN.md §H4 (GC-02, GC-08): resolves each of a suggestion's
    self-reported `grounding_ids` to the past-interview fact that grounded
    it. Three cases, all of which render rather than raise:

    - Resolves: the fact's claim and -- HERE ONLY, nowhere else in this
      codebase -- its verbatim `quoted_text` (06-RESEARCH.md §6; gate 20
      permits this module to read `Fact.quoted_text` by name).
    - Does not resolve: the id, with a note that fact ids are re-minted on
      every re-extraction (`corpus.py:82-89`) -- the NORMAL state of any
      session read after a re-run, not a defect.
    - No corpus at all, or one that failed to parse: the ids raw, with one
      note (`fact_note`) explaining why, still never blocking the rest of
      the document."""
    grounding_ids = record.get("grounding_ids") or []
    if not grounding_ids:
        return []
    if fact_note is not None:
        return [f">   grounded in: {', '.join(grounding_ids)} (unresolved -- {fact_note})"]
    lines: "list[str]" = []
    for grounding_id in grounding_ids:
        fact = fact_lookup.get(grounding_id)
        if fact is None:
            lines.append(
                f">   grounded in: {grounding_id} -- matches no fact in the current corpus "
                "(fact ids are re-minted on every re-extraction; this is expected after a "
                "re-run, not a defect)"
            )
        else:
            lines.append(f">   grounded in: {fact.claim!r} (quote: {fact.quoted_text!r})")
    return lines


def _format_proposal_line(
    record: dict[str, Any], *, hypothesis_lookup: "dict[str, Any]", hypothesis_note: "Optional[str]"
) -> "Optional[str]":
    """06-03-PLAN.md §H4 (GC-05, D-01, T-06-18): a record carrying a
    `proposed_state` renders the target hypothesis and the proposed state
    with an explicit statement that it was NOT applied -- the hypothesis set
    changes only when the interviewer, separately, changes it on the
    authoring page. A line that read like a record of something that
    happened would invert D-01's whole meaning, so the "not applied" clause
    is never trimmed for brevity."""
    proposed_state = record.get("proposed_state")
    if proposed_state is None:
        return None
    target_id = record.get("target_hypothesis_id")
    if target_id is None:
        target_desc = "(no target hypothesis recorded)"
    elif hypothesis_note is not None:
        target_desc = f"{target_id!r} (unresolved -- {hypothesis_note})"
    else:
        hypothesis = hypothesis_lookup.get(target_id)
        if hypothesis is None:
            target_desc = f"{target_id!r} -- matches no hypothesis in the current corpus"
        else:
            target_desc = f"{target_id!r} ({hypothesis.statement!r})"
    return (
        f">   PROPOSED (not applied): hypothesis {target_desc} -> {proposed_state!r} -- "
        "the hypothesis set changes ONLY when you change it yourself, on the authoring page"
    )


def _format_suggestion_block(
    record: dict[str, Any],
    *,
    elapsed: str,
    fact_lookup: "Optional[dict[str, Any]]" = None,
    fact_note: "Optional[str]" = None,
    hypothesis_lookup: "Optional[dict[str, Any]]" = None,
    hypothesis_note: "Optional[str]" = None,
) -> "list[str]":
    """Every suggestion the coach ever produced -- shown, ranked down, or
    dropped -- with its move type, band, five per-rule flags and rationale
    (COACH-04, extended by 06-02/06-03 to the fifth flag). A candidate the
    critique call never classified at all (critique itself failed) has every
    flag/band/rationale field `None`; that is shown too, explicitly marked
    "not vetted", rather than being silently omitted -- the audit trail must
    show what was PROPOSED even when nothing was vetted.

    06-03-PLAN.md §H4: also renders which past-interview facts grounded this
    suggestion (if any) and, separately, any hypothesis-state proposal it
    carries (if any) -- both independent of `shown`/`band`, since a candidate
    can self-report grounding or a proposal even when it was never vetted."""
    shown = record.get("shown")
    band = record.get("band")
    status = "SHOWN" if shown else ("DROPPED" if band is not None else "DROPPED, not vetted -- critique failed")
    move = record.get("move")
    band_label = band if band is not None else "no verdict"
    lines = [f'> [{elapsed}] suggestion ({status}) [{move}/{band_label}]: "{record["text"]}"']
    if band is not None:
        flags = ", ".join(f"{field}={record.get(field)}" for field in _SUGGESTION_FLAG_FIELDS)
        lines.append(f">   flags: {flags}")
        rationale = record.get("rationale")
        if rationale:
            lines.append(f">   rationale: {rationale}")
    lines.extend(_format_grounding_lines(record, fact_lookup=fact_lookup or {}, fact_note=fact_note))
    proposal_line = _format_proposal_line(
        record, hypothesis_lookup=hypothesis_lookup or {}, hypothesis_note=hypothesis_note
    )
    if proposal_line is not None:
        lines.append(proposal_line)
    return lines


def _render_timeline(
    ordered: list[dict[str, Any]],
    session_time: "Callable[[dict[str, Any]], float]",
    *,
    metadata: dict[str, Any],
    fact_lookup: "dict[str, Any]",
    fact_note: "Optional[str]",
    hypothesis_lookup: "dict[str, Any]",
    hypothesis_note: "Optional[str]",
) -> list[str]:
    survey_text_by_id = _survey_text_by_id(metadata)
    segment_text_by_seq = _segment_text_by_seq(tuple(ordered))

    lines: list[str] = []
    current_speaker: Optional[str] = None
    turn_open = False

    for record in ordered:
        kind = record.get("kind")
        if kind == "segment":
            speaker = record["speaker"]
            if speaker != current_speaker or not turn_open:
                if turn_open:
                    lines.append("")
                lines.append(f"### {_speaker_label(speaker)} [{_format_elapsed(session_time(record))}]")
                current_speaker = speaker
                turn_open = True
            # Each utterance is reproduced on its OWN line, byte-for-byte --
            # never joined into one string with another utterance's text, and
            # never reordered or reflowed (phase_specification U2, the whole
            # point of this module).
            lines.append(record["text"])
            continue

        # A suggestion or an event breaks the current speaker turn -- it must
        # appear inline at its true position in the timeline, never deferred
        # to a footnote, so a transcript gap (or a coach decision) has a
        # visible cause beside it.
        if turn_open:
            lines.append("")
            turn_open = False
            current_speaker = None

        elapsed = _format_elapsed(session_time(record))
        if kind == "suggestion":
            lines.extend(
                _format_suggestion_block(
                    record, elapsed=elapsed, fact_lookup=fact_lookup, fact_note=fact_note,
                    hypothesis_lookup=hypothesis_lookup, hypothesis_note=hypothesis_note,
                )
            )
        elif kind == "event" and record.get("event") in (EVENT_COVERAGE_ASKED, EVENT_COVERAGE_CONSIDERED):
            lines.append(
                _format_coverage_event_line(
                    record, elapsed=elapsed, survey_text_by_id=survey_text_by_id,
                    segment_text_by_seq=segment_text_by_seq,
                )
            )
        else:
            event_speaker = record.get("speaker")
            label = f" ({event_speaker})" if event_speaker else ""
            lines.append(
                f"> [{elapsed}] event{label}: {record['event']} {record.get('detail') or ''}".rstrip()
            )
    lines.append("")
    return lines


def _render_coaching_summary_section(result: SessionReadResult) -> "list[str]":
    """Cycles completed, suggestions shown, candidates dropped broken down by
    which rule fired, coverage transitions, and the coaching latency figures
    (04-03-PLAN.md §E6) -- computed from the SAME durable records the
    timeline above reads, and reporting all-zero without raising for a
    session where the coach never ran (the normal shape of every session
    recorded before this phase existed)."""
    suggestions = result.records_of_kind("suggestion")
    lines = ["## Coaching summary", ""]

    # `coach_enabled` is written into session metadata at `start` (04-01-PLAN.md
    # §C6) -- absent entirely for every session recorded before Phase 4 ever
    # existed, and `False` for any session run without a reasoning key. Either
    # way, that is the authoritative "did the coach run at all" signal, not an
    # inference from whether any suggestion happened to be produced.
    if not result.metadata.get("coach_enabled", False):
        lines.append("The coach did not run in this session.")
        return lines

    # Cycles completed: one cycle produces one group of candidates sharing a
    # trigger_seq -- counting distinct trigger_seq values is exact, never
    # estimated, and matches what `SuggestionPipeline` itself calls a cycle.
    cycle_trigger_seqs = {r["trigger_seq"] for r in suggestions}
    shown_count = sum(1 for r in suggestions if r.get("shown"))
    dropped = [r for r in suggestions if not r.get("shown")]
    dropped_by_rule: "dict[str, int]" = {field: 0 for field in _SUGGESTION_FLAG_FIELDS}
    for record in dropped:
        for field in _SUGGESTION_FLAG_FIELDS:
            if record.get(field):
                dropped_by_rule[field] += 1

    events = result.records_of_kind("event")
    asked_count = sum(1 for r in events if r.get("event") == EVENT_COVERAGE_ASKED)
    considered_count = sum(1 for r in events if r.get("event") == EVENT_COVERAGE_CONSIDERED)

    lines.append(f"- Cycles completed: {len(cycle_trigger_seqs)}")
    lines.append(f"- Suggestions shown: {shown_count}")
    dropped_breakdown = ", ".join(f"{field}={count}" for field, count in dropped_by_rule.items())
    lines.append(f"- Candidates dropped: {len(dropped)} ({dropped_breakdown})")
    lines.append(f"- Coverage: {asked_count} asked, {considered_count} considered with no match")

    # 06-03-PLAN.md §H4 (GC-02, GC-05): computed from the durably-recorded
    # `grounding_ids`/`proposed_state` fields directly -- never from the
    # (possibly absent or malformed) corpus lookup, so these two counts are
    # exact and unaffected by whether resolution ran at all.
    grounded_suggestions = sum(1 for r in suggestions if r.get("grounding_ids"))
    distinct_facts_cited = len({gid for r in suggestions for gid in (r.get("grounding_ids") or [])})
    lines.append(
        f"- Suggestions grounded in a past-interview fact: {grounded_suggestions} "
        f"({distinct_facts_cited} distinct fact(s) cited)"
    )
    proposed_count = sum(1 for r in suggestions if r.get("proposed_state") is not None)
    lines.append(
        f"- Hypothesis-state changes proposed: {proposed_count} (never applied -- accepted only "
        "when you change them yourself, on the authoring page)"
    )

    latency = result.metadata.get("coaching_latency_summary") or {"first_cycle_s": None, "stats": {"count": 0}}
    first_cycle_s = latency.get("first_cycle_s")
    stats = latency.get("stats") or {"count": 0}
    if first_cycle_s is not None:
        lines.append(f"- Coaching latency, first cycle: {first_cycle_s:.2f}s")
    if stats.get("count", 0) >= 2:
        lines.append(
            f"- Coaching latency, remaining cycles: count={stats['count']}, "
            f"median={stats['median_s']:.2f}s, p95={stats['p95_s']:.2f}s "
            f"(min={stats['min_s']:.2f}s, max={stats['max_s']:.2f}s)"
        )
    elif stats.get("count", 0) == 0 and first_cycle_s is None:
        lines.append("- No coaching latency figures were recorded.")

    # 06-01-PLAN.md §F4: retrieval's own session-start cost, printed beside
    # the per-cycle figures above -- `.get(...)`, like every other field in
    # this section, so a session recorded before this phase existed (whose
    # `coaching_latency_summary` carries no such key) renders unchanged,
    # never raising and never inventing a number.
    session_start_grounding_s = latency.get("session_start_grounding_s")
    if session_start_grounding_s is not None:
        lines.append(f"- Coaching latency, session-start grounding: {session_start_grounding_s:.2f}s")

    return lines


def write_rendered_transcript(session_dir: Path, *, context_dir: "Optional[Path]") -> Path:
    """Renders the session and writes it into the session directory,
    returning the path written -- the artifact `python -m app render`
    reports.

    06-03-PLAN.md §H4: `context_dir` is REQUIRED, explicitly nullable,
    threaded straight through to `render_session` -- see that function's own
    docstring."""
    text = render_session(session_dir, context_dir=context_dir)
    doc_path = session_dir / TRANSCRIPT_DOC_FILENAME
    doc_path.write_text(text, encoding="utf-8")
    return doc_path
