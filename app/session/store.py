"""The phase_specification S6 append-only session record store: one ordered,
line-delimited file, so there is exactly one durability mechanism to get
right and so that stream events interleave with segments in true order for
post-mortem. Every write is flushed and `os.fsync`'d before it returns --
the difference between "written" and "survives the process being killed"
(SESS-01, and the precondition SESS-02 in plan 02-03 depends on).

The fsync-per-record contract (phase_specification U1, plan 02-03 Task 1) is
made observable, not merely described: `fsync_count` counts every fsync this
store has performed on its own transcript file descriptor, so a test can
assert "one fsync per append" by count rather than by reasoning about the
implementation.

Reopening an existing session directory (rather than creating a fresh one)
resumes rather than truncates: the highest sequence number already on disk is
recovered once at open time via `app.session.reader` -- the same tolerant
parser every other consumer uses -- and a `session-resumed` event is appended
so a restarted interview is visible in the record itself, never merely
inferred from a timestamp gap (SESS-02, plan 02-03 Task 1).

Never logs a record's text -- only its length (S7, T-02-05). No STT vendor is
ever named in this module's executable code (STT-03).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"
METADATA_FILENAME = "session.json"


@dataclass(frozen=True)
class SegmentRecord:
    kind: str
    seq: int
    speaker: str
    text: str
    # ⚠️ FABRICATED, and the only field here that is. `app/stt/live.py` sets
    # it to the SAME speaker's previous `audio_end_s`, because the engine
    # contract (`benchmark/engines/base.Segment`) has no start time to record
    # -- so this absorbs however long that speaker was silent, and
    # `audio_end_s - audio_start_s` is that silence, not the speech.
    # Gate 35 (`test_phase_gates.py`) fails the build on any read of it.
    audio_start_s: float
    audio_end_s: float
    ready_wall_s: float
    latency_s: float
    detected_language: Optional[str]
    engine: str
    written_at: str


@dataclass(frozen=True)
class EventRecord:
    kind: str
    seq: int
    speaker: Optional[str]
    event: str
    detail: dict
    at_s: float
    written_at: str


@dataclass(frozen=True)
class SuggestionRecord:
    """One candidate the coach's generate stage ever produced -- shown,
    ranked down, or dropped (plan 04-02, COACH-04). The durable store gets
    the FULL candidate text and rationale by design (SESS-03 already
    persists full segment text); the logger and the wire never do (§D8's
    two-tier rule). `band`/`is_leading`/`is_hypothetical_or_future_tense`/
    `is_pitch`/`is_opinion_seeking`/`is_confirmation_only`/`is_generic`/
    `rationale` are
    all `None` together only when the candidate was never vetted at all (the
    critique call itself failed) -- the candidate is still recorded, with
    `shown=False`, so the audit trail shows what was proposed even when
    nothing was vetted.

    06-03-PLAN.md §H1 grows this record by four fields, in the same commit
    that updates this record's own pinned-equality test
    (`test_coach_audit.py::EXPECTED_SUGGESTION_KEYS`):

    - `is_confirmation_only` -- the fifth per-rule critic flag (06-02-PLAN.md
      §G4), `None` together with the other four exactly as described above.
    - `is_generic` -- the sixth per-rule critic flag (2026-08-15): the
      candidate could have been asked of anyone, in any interview, without
      hearing this one. Same `None`-together rule as every flag above. A
      record written before this flag existed reads back without the key,
      which the renderer's `record.get(field)` already handles -- that is why
      the real recorded artifacts on disk are left alone rather than migrated.
    - `is_off_topic` -- the seventh per-rule critic flag (2026-08-19): the
      candidate serves neither a survey item nor a listed hypothesis, so it
      would spend the interviewer's next question outside what this interview
      is for. Same `None`-together rule as every flag above, and the same
      read-back-without-the-key tolerance for records written before it. It
      is the only flag whose value the pipeline can act on deterministically
      -- `critic.enforce_off_topic` bands a true one `clear_violation`, so a
      record carrying `is_off_topic=True` and `shown=True` is a contradiction
      and worth investigating rather than a state the code can produce.
    - `grounding_ids` -- which retrieved-fact ids (`Candidate.grounding_ids`,
      06-02-PLAN.md §G5) the model says backed this specific candidate.
      Default `()` is an absence, not an assertion that nothing grounded it.
    - `target_hypothesis_id` -- the hypothesis a `test_hypothesis` candidate
      targeted (`Candidate.target_hypothesis_id`).
    - `proposed_state` -- the hypothesis state this candidate's question
      would PROPOSE (`Candidate.proposed_state`). Never applied by anything
      on the live path (D-01) -- the interviewer reads it here and
      separately, explicitly, accepts or rejects it on the authoring page.

    All four are durable-only, by design (§D8's two-tier rule, unchanged):
    gate 15 keeps every one of them off the wire by construction (a
    allowlist projection that excludes anything not explicitly named), so a
    future reader never has to reason out why a phone-facing field is
    missing here.

    The field set is intentionally pinned by an equality test
    (`test_coach_audit.py`), not a contains-check, so a field nobody thought
    to add also fails -- this record's shape becomes costly to change once a
    real session has written it and 04-03's/06-03's renderer reads it."""

    kind: str
    seq: int
    trigger_seq: int
    text: str
    move: str
    band: Optional[str]
    is_leading: Optional[bool]
    is_hypothetical_or_future_tense: Optional[bool]
    is_pitch: Optional[bool]
    is_opinion_seeking: Optional[bool]
    is_confirmation_only: Optional[bool]
    is_generic: Optional[bool]
    is_off_topic: Optional[bool]
    rationale: Optional[str]
    shown: bool
    written_at: str
    grounding_ids: "tuple[str, ...]" = ()
    target_hypothesis_id: Optional[str] = None
    proposed_state: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def create_session_dir(sessions_root: Path) -> Path:
    """A fresh, not-yet-existing session directory under `sessions_root`,
    named by start timestamp. Defensively disambiguates a same-second
    collision rather than crashing a session that is about to hold real
    interview audio."""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = sessions_root / timestamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = sessions_root / f"{timestamp}-{suffix}"
    return candidate


def _truncate_incomplete_final_line(transcript_path: Path) -> int:
    """Drops a trailing partial line (one not terminated by a newline) and
    returns how many bytes were discarded.

    Called on the resume path only. A record whose `append()` returned always
    ends in a newline written under the same lock and fsync'd, so an
    unterminated tail is by construction a write the process died inside --
    never a durable record. Returns 0 for a cleanly-terminated or empty file,
    which is the normal case.
    """
    raw = transcript_path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return 0

    keep = raw.rfind(b"\n") + 1  # 0 when the file is one partial line
    dropped = len(raw) - keep
    with open(transcript_path, "r+b") as handle:
        handle.truncate(keep)
        handle.flush()
        os.fsync(handle.fileno())
    return dropped


class SessionStore:
    """One store per session directory. Thread-safe: two stream workers
    append concurrently from plan 02-02 onward, so sequence assignment and
    the write itself are both taken under one lock (S6).

    Opening a directory that already holds a transcript file is a RESUME, not
    a fresh session: the existing file is never truncated or rewritten, the
    sequence counter continues from the highest sequence number already on
    disk, and a `session-resumed` event is appended so the resume is visible
    in the record itself (SESS-02)."""

    def __init__(self, session_dir: Path) -> None:
        self._dir = session_dir
        self._transcript_path = self._dir / TRANSCRIPT_FILENAME
        self._metadata_path = self._dir / METADATA_FILENAME
        is_resume = self._transcript_path.is_file()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # A crash leaves a partial final line BY DEFINITION -- that is exactly
        # what the reader is built to tolerate. Appending straight onto it
        # would weld the next record onto the partial one, producing a
        # malformed line that becomes an INTERIOR line as soon as anything
        # further is written -- and the reader refuses interior corruption, so
        # the whole session would read as corrupt forever, including the
        # transcript the interview actually captured.
        #
        # Truncating that partial line does NOT violate "never truncates":
        # a partial line is a record whose `append()` never returned, so it
        # was never durable and no contract covers it. Padding with a newline
        # instead would be worse -- it converts the tail into precisely the
        # malformed interior line the reader refuses.
        dropped_partial_bytes = 0
        if is_resume:
            dropped_partial_bytes = _truncate_incomplete_final_line(self._transcript_path)
        # Opened in append mode: this NEVER seeks or rewrites a byte belonging
        # to a record whose append() returned.
        self._file = open(self._transcript_path, "a", encoding="utf-8")
        self._latencies: dict[str, list[float]] = {}
        self._opened_monotonic = time.monotonic()
        self._fsync_count = 0
        self._resumed = is_resume
        self._seq = 0

        if is_resume:
            # Deferred import: app.session.reader imports this module's
            # filename constants, so importing it at module scope here would
            # be circular. By the time __init__ runs both modules are fully
            # loaded.
            from app.session.reader import read_session

            existing = read_session(self._dir)
            highest_seq = max((r.get("seq", -1) for r in existing.records), default=-1)
            self._seq = highest_seq + 1
            self.append_event(
                event="session-resumed",
                detail={
                    "resumed_from_seq": self._seq,
                    "prior_record_count": len(existing.records),
                    # Recorded so the drop is VISIBLE in the transcript rather
                    # than silent. Non-zero means a record was mid-write when
                    # the process died; it was never durable, but the reader
                    # of this session deserves to know bytes were discarded.
                    "dropped_partial_tail_bytes": dropped_partial_bytes,
                },
            )

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path

    @property
    def fsync_count(self) -> int:
        """The number of times this store has called `os.fsync` on the
        transcript file descriptor -- one per `append_segment`/`append_event`
        call plus one on `close()`. Makes the fsync-per-record contract an
        assertion by count, not a claim in a comment (phase_specification
        U1, plan 02-03 Task 1)."""
        return self._fsync_count

    @property
    def resumed(self) -> bool:
        return self._resumed

    def segment_count(self, speaker: str) -> int:
        """How many `segment` records have been appended for `speaker` so
        far -- one entry is recorded per segment as a side effect of
        `append_segment`'s own latency tracking, so this is always exact,
        never estimated. Used by `app/main.py` to print per-speaker segment
        counts on stop (phase_specification Task 3: "a user who cannot see
        what the system is doing cannot tell a working system from a dead
        one")."""
        return len(self._latencies.get(speaker, []))

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Session provenance (S6): engine name/model, device name(s), native
        sample rate(s), start time -- whatever the caller has at open time.
        The latency summary is added at `close()`."""
        payload = dict(metadata)
        payload.setdefault("started_at", _now_iso())
        self._metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def update_metadata(self, patch: dict[str, Any]) -> None:
        """Merges `patch` into the existing `session.json`, preserving
        everything already written (mirrors the read-merge-rewrite shape
        `close()` already uses for `closed_at`/`latency_summary`). Used by
        `app/coach/coverage.py` (plan 04-01) to record a coverage-replay
        mismatch count without inventing a second metadata-write path."""
        existing: dict[str, Any] = {}
        if self._metadata_path.is_file():
            existing = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        existing.update(patch)
        self._metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def append_segment(
        self,
        *,
        speaker: str,
        text: str,
        audio_start_s: float,
        audio_end_s: float,
        ready_wall_s: float,
        detected_language: Optional[str],
        engine: str,
    ) -> SegmentRecord:
        latency_s = max(0.0, ready_wall_s - audio_end_s)
        with self._lock:
            seq = self._next_seq()
            record = SegmentRecord(
                kind="segment",
                seq=seq,
                speaker=speaker,
                text=text,
                audio_start_s=audio_start_s,
                audio_end_s=audio_end_s,
                ready_wall_s=ready_wall_s,
                latency_s=latency_s,
                detected_language=detected_language,
                engine=engine,
                written_at=_now_iso(),
            )
            self._latencies.setdefault(speaker, []).append(latency_s)
            self._write(record)
        logger.debug(
            "segment appended seq=%d speaker=%s text_len=%d audio_start_s=%.2f "
            "audio_end_s=%.2f latency_s=%.2f language=%s engine=%s",
            seq, speaker, len(text), audio_start_s, audio_end_s, latency_s,
            detected_language, engine,
        )
        return record

    def append_suggestion(
        self,
        *,
        trigger_seq: int,
        text: str,
        move: str,
        band: Optional[str],
        is_leading: Optional[bool],
        is_hypothetical_or_future_tense: Optional[bool],
        is_pitch: Optional[bool],
        is_opinion_seeking: Optional[bool],
        is_confirmation_only: Optional[bool],
        is_generic: Optional[bool],
        is_off_topic: Optional[bool],
        rationale: Optional[str],
        shown: bool,
        grounding_ids: "tuple[str, ...]" = (),
        target_hypothesis_id: Optional[str] = None,
        proposed_state: Optional[str] = None,
    ) -> SuggestionRecord:
        """Same lock-then-write-then-fsync shape as `append_segment`/
        `append_event` (plan 04-02, COACH-04) -- one fsync per record, no new
        persistence mechanism. Never logs `text` -- only its length, matching
        `append_segment`'s own discipline (T-04-27).

        06-03-PLAN.md §H1: `is_confirmation_only` is REQUIRED, and so are
        `is_generic` (2026-08-15) and `is_off_topic` (2026-08-19), mirroring the
        existing flags exactly (a candidate the critique never
        classified passes `None`, explicitly, from the caller -- never a
        silent default). `grounding_ids`/`target_hypothesis_id`/
        `proposed_state` default to their empty/absent shape because they are
        candidate-level self-reports that exist independently of whether the
        critique stage ever ran."""
        with self._lock:
            seq = self._next_seq()
            record = SuggestionRecord(
                kind="suggestion",
                seq=seq,
                trigger_seq=trigger_seq,
                text=text,
                move=move,
                band=band,
                is_leading=is_leading,
                is_hypothetical_or_future_tense=is_hypothetical_or_future_tense,
                is_pitch=is_pitch,
                is_opinion_seeking=is_opinion_seeking,
                is_confirmation_only=is_confirmation_only,
                is_generic=is_generic,
                is_off_topic=is_off_topic,
                rationale=rationale,
                shown=shown,
                written_at=_now_iso(),
                grounding_ids=tuple(grounding_ids),
                target_hypothesis_id=target_hypothesis_id,
                proposed_state=proposed_state,
            )
            self._write(record)
        logger.debug(
            "suggestion appended seq=%d trigger_seq=%d move=%s band=%s shown=%s text_len=%d",
            seq, trigger_seq, move, band, shown, len(text),
        )
        return record

    def append_event(
        self, *, event: str, speaker: Optional[str] = None, detail: Optional[dict] = None
    ) -> EventRecord:
        with self._lock:
            seq = self._next_seq()
            record = EventRecord(
                kind="event",
                seq=seq,
                speaker=speaker,
                event=event,
                detail=detail or {},
                at_s=time.monotonic() - self._opened_monotonic,
                written_at=_now_iso(),
            )
            self._write(record)
        logger.debug("event appended seq=%d event=%s speaker=%s", seq, event, speaker)
        return record

    def close(self) -> dict:
        """Flush, fsync, and close the transcript file; write the latency
        summary into session.json. Returns the summary."""
        summary = self._latency_summary()
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._fsync_count += 1
            self._file.close()
        existing: dict[str, Any] = {}
        if self._metadata_path.is_file():
            existing = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        existing["closed_at"] = _now_iso()
        existing["latency_summary"] = summary
        self._metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return summary

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _write(self, record: Any) -> None:
        # flush() then os.fsync() on the underlying descriptor, in that
        # order, INSIDE the same lock that serializes concurrent stream
        # workers (S6) -- this is the entire difference between "written"
        # and "survives the process being killed without warning" (SESS-02).
        line = json.dumps(asdict(record), ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._fsync_count += 1

    def _latency_summary(self) -> dict:
        """Per-speaker (phase_specification U2 -- the rendered document ends
        with a per-speaker summary, and the terminal prints the same figure
        the moment a session closes). Shared with `app.session.render` via
        `latency_stats_for_speaker` so a session re-read from disk after the
        process exited computes the identical numbers this live summary
        would have."""
        return {
            speaker: latency_stats_with_modes(values)
            for speaker, values in self._latencies.items()
        }


def latency_stats(values: "list[float]") -> dict:
    """count/median/p95/min/max over a list of measured `latency_s` values,
    or `{"count": 0}` when there are too few samples for those figures to
    mean anything (phase_specification U2). Shared by `SessionStore` (the
    live, in-process summary) and `app.session.render` (recomputed from the
    persisted `latency_s` values on a later, possibly separate, read) so the
    two are always the same number by construction, never two
    implementations that can drift apart."""
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    p95_index = max(0, int(len(sorted_values) * 0.95) - 1)
    return {
        "count": len(sorted_values),
        "median_s": statistics.median(sorted_values),
        "p95_s": sorted_values[p95_index],
        "min_s": sorted_values[0],
        "max_s": sorted_values[-1],
    }


# The boundary between the two ways this engine finalizes a token, in seconds.
#
# **Measured, not chosen.** Per-segment `latency_s` across three real sessions
# is BIMODAL: a fast cluster at ~0.65-1.0s and a slow one at ~4.7-5.6s, with
# almost nothing between them -- one 1.31s and one 2.65s sample in the whole
# corpus. This constant sits in that empty band, which is why its exact value
# does not matter much; anywhere from ~1.5s to ~3.0s separates the same two
# populations.
#
# The mechanism, as far as the data shows it: a pause lets the engine's
# endpoint detection flush everything it is holding, and that segment is ready
# almost immediately. During CONTINUOUS speech there is no endpoint, so tokens
# become final only as they age out of a rolling window -- a constant offset
# trailing the live audio head, not a backlog. The tell is that the two modes
# alternate within a single minute of one session, and that the slow mode
# recovers instantly (4.71s to 0.70s between consecutive segments) rather than
# draining the way a queue would.
FINALIZATION_BOUNDARY_S = 2.0


def latency_stats_with_modes(values: "list[float]") -> dict:
    """`latency_stats`, plus the two finalization modes reported separately.

    **Why this exists, and it is a reporting fix rather than a performance
    one.** A session's median latency is largely a function of how much the
    speaker paused, not of how healthy the pipeline is. Read a recording of
    continuous playback and the median lands in the slow mode; read a
    turn-taking interview and it lands in the fast one. On 2026-08-12 that
    produced a full investigation into a "2.6x latency regression" against a
    ~1.83s baseline -- and there was no regression. The baseline had been
    measured on a 25-second benchmark clip with natural pauses, the session had
    not, and sessions recorded LATER on the same code came back at 0.89s and
    1.77s. One number had been asked to describe two populations.

    So both are reported, and the single median is kept beside them rather than
    replaced: it is still the right figure for "what did this session feel
    like", and it is still what every existing reader expects.

    **A SEPARATE function, deliberately not folded into `latency_stats`.** That
    one is shared with `app.coach.pipeline.pipeline_latency_summary`, whose
    cycle latency is a model round-trip and is not bimodal in this way at all;
    splitting it there would print a meaningless boundary over coach numbers.
    And it is used by BOTH `SessionStore._latency_summary` and
    `app.session.render`, for the same reason `latency_stats` is -- a live
    summary and one recomputed from disk after the process exited must be the
    same numbers by construction, never two implementations that can drift.
    """
    stats = latency_stats(values)
    if not values:
        return stats
    fast = [value for value in values if value < FINALIZATION_BOUNDARY_S]
    slow = [value for value in values if value >= FINALIZATION_BOUNDARY_S]
    stats["endpoint_final_count"] = len(fast)
    stats["window_final_count"] = len(slow)
    # Each median only when that mode HAS samples. A median over an empty list
    # raises, and a zero would read as "instant" -- the most flattering possible
    # lie about a mode that did not occur.
    if fast:
        stats["endpoint_final_median_s"] = statistics.median(fast)
    if slow:
        stats["window_final_median_s"] = statistics.median(slow)
    # Only true when BOTH modes are actually represented. A session that is
    # entirely one mode has an honest single median and must not be labelled
    # bimodal on the strength of one stray sample.
    stats["is_bimodal"] = bool(fast) and bool(slow)
    return stats
