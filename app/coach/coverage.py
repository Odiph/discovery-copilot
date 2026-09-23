"""The coverage detector: watches the INTERVIEWER's own finalized transcript
segments and decides whether what they just said asked one of the survey's
still-unasked questions (04-CONTEXT.md D-03, 04-01-PLAN.md §C2/§C3/§C5).

**D-03, structural, not a convention:** coverage is written from the
interviewer's own stream and from NOTHING else. `submit()` silently ignores
any record whose `speaker` is not the interviewer -- an interviewee segment
can never move coverage state, even when the matcher would have said yes.

**D-04, structural, not a convention:** `submit(record)` is the ONLY public
entry point this class exposes, and it takes a `SegmentRecord` -- the same
type every finalized STT segment already is. There is no method here a
generated or proposed suggestion could call; nothing "resembling a
suggestion" has a write path into coverage state.

**Never on a capture or STT thread (04-RESEARCH.md §4, non-negotiable #7).**
`submit()` enqueues onto a BOUNDED queue and returns -- no network, no
blocking, no raising, ever. A dedicated worker thread drains the queue and
makes the (much slower) vendor call. This is the direct analogue of
`app/stt/live.py::StreamWorker._persist`'s thread->asyncio bridge, except the
coverage call is an order of magnitude slower than a broadcast, so this
plan gives it its own thread rather than reusing that bridge (§C3).

The queue is bounded and drops the OLDEST entry when full, keeping the
newest and recording a named event (T-04-07) -- a stale coverage match is
worth less than a fresh one.

Two persisted states only, `unasked`/`asked` (`CoverageState`, §C5) -- no
third `answered` state; "asked but not answered" is 04-02's live, per-call
judgement, never persisted here. Every match decision is recorded, including
non-matches (`coverage-considered`), so a false positive is visible in the
post-call read even though nothing un-asks a question once asked (Pitfall 1).

On construction, state is rebuilt by replaying every `coverage-asked` event
already in the session's durable record (SESS-02 extended) -- no second
persistence file.

This module must NEVER log or record segment/candidate TEXT -- only ids,
sequence numbers, timestamps and exception type names (S7, T-04-11). It must
NEVER import `anthropic` -- the vendor SDK is named nowhere in this file; a
client is received by injection and driven only through
`app/coach/reasoning_client.with_timeout()`.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from app.coach import reasoning_client
from app.coach.context import SurveyItem
from app.session.reader import read_session
from app.session.store import SegmentRecord, SessionStore

logger = logging.getLogger(__name__)

SPEAKER_INTERVIEWER = "interviewer"

EVENT_COVERAGE_ASKED = "coverage-asked"
EVENT_COVERAGE_CONSIDERED = "coverage-considered"
EVENT_COVERAGE_QUEUE_DROP = "coverage-queue-drop"
EVENT_COVERAGE_FAILED = "coverage-failed"
EVENT_REASONING_FAILURE = "reasoning-timeout"

_COVERAGE_TOOL_NAME = "report_coverage_match"

_MATCH_SYSTEM_PROMPT = (
    "You track which items of a pre-written interview survey the interviewer has "
    "actually asked, from the interviewer's own words. Prefer reporting no match over "
    "guessing: a missed match only delays the next suggestion by one turn, but a wrong "
    "match silently removes that survey item from attention for the rest of the "
    "interview. Only report a match when the interviewer's utterance plausibly asked "
    "the substance of that specific survey question, even if paraphrased."
)


def _coverage_tool_schema() -> dict:
    return {
        "name": _COVERAGE_TOOL_NAME,
        "description": (
            "Report which unasked survey item (if any) the interviewer's utterance "
            "most plausibly asked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question_id": {
                    "type": ["string", "null"],
                    "description": "The id of the matched survey item, or null if none was asked.",
                },
            },
            "required": ["question_id"],
        },
    }


def _build_match_prompt(segment_text: str, unasked_items: Sequence[SurveyItem]) -> str:
    numbered = "\n".join(f"- id={item.id!r}: {item.text}" for item in unasked_items)
    return (
        "Unasked survey items:\n"
        f"{numbered}\n\n"
        "The interviewer just said:\n"
        f"{segment_text}\n\n"
        "Call report_coverage_match with the id of the survey item this utterance "
        "asked, or null if it did not clearly ask any of them."
    )


def match_coverage_question(
    client: Any,
    *,
    segment_text: str,
    unasked_items: Sequence[SurveyItem],
    model: str,
    timeout_s: float,
) -> Optional[str]:
    """Drives ONE coverage-match call through the shared client seam. Takes
    `client` by injection (real or `app.coach.stub.StubAnthropicClient`) and
    is the single function both the product code and 04-01 Task 3's probe
    call -- the probe is required to go through this function rather than a
    hand-written copy, so it can never drift into testing something the
    product doesn't do."""
    bound = reasoning_client.with_timeout(client, timeout_s)
    response = bound.messages.create(
        model=model,
        max_tokens=200,
        system=_MATCH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_match_prompt(segment_text, unasked_items)}],
        tools=[_coverage_tool_schema()],
        tool_choice={"type": "tool", "name": _COVERAGE_TOOL_NAME},
    )
    valid_ids = {item.id for item in unasked_items}
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == _COVERAGE_TOOL_NAME:
            question_id = block.input.get("question_id")
            return question_id if question_id in valid_ids else None
    return None


@dataclass(frozen=True)
class CoverageState:
    """Immutable. Two states only (§C5): a question id is either in `asked`
    or it isn't -- there is no third `answered` state persisted here."""

    asked: "frozenset[str]" = frozenset()

    def is_asked(self, question_id: str) -> bool:
        return question_id in self.asked

    def unasked_items(self, survey: Sequence[SurveyItem]) -> "tuple[SurveyItem, ...]":
        return tuple(item for item in survey if item.id not in self.asked)

    def with_asked(self, question_id: str) -> "CoverageState":
        return CoverageState(asked=self.asked | {question_id})


class CoverageDetector:
    """Owns the bounded queue and the dedicated worker thread (§C3). Public
    surface is deliberately exactly `submit`, `start`, `stop` and `state` --
    D-04 is enforced by there being no OTHER public method a proposed
    suggestion could ever call into."""

    def __init__(
        self,
        *,
        client: Any,
        store: SessionStore,
        survey: Sequence[SurveyItem],
        model: str = reasoning_client.DEFAULT_MODEL,
        timeout_s: float = 3.0,
        queue_maxsize: int = 16,
        on_change: Optional[Callable[[CoverageState], None]] = None,
    ) -> None:
        self._client = client
        self._store = store
        self._survey = tuple(survey)
        self._model = model
        self._timeout_s = timeout_s
        self._on_change = on_change
        self._queue: "queue.Queue[SegmentRecord]" = queue.Queue(maxsize=queue_maxsize)
        self._state_lock = threading.Lock()
        self._state = self._replay_state()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def state(self) -> CoverageState:
        with self._state_lock:
            return self._state

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="coach-coverage", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None

    def submit(self, record: SegmentRecord) -> None:
        """The ONLY entry point (D-04). Never blocks, never raises, never
        touches the network -- non-interviewer segments are dropped
        immediately (D-03); a full queue drops its OLDEST entry and records
        a named event rather than blocking the caller."""
        if record.speaker != SPEAKER_INTERVIEWER:
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._store.append_event(
                    event=EVENT_COVERAGE_QUEUE_DROP,
                    speaker=SPEAKER_INTERVIEWER,
                    detail={"reason": "queue-full"},
                )
            try:
                self._queue.put_nowait(record)
            except queue.Full:  # pragma: no cover -- only under a concurrent racing drain
                pass

    def _replay_state(self) -> CoverageState:
        """Rebuilds coverage state from the durable record alone (SESS-02
        extended, §C5) -- no second persistence file. A transition naming a
        question id the current survey no longer contains is ignored with a
        warning, and the mismatch count is recorded into session metadata so
        a post-call read can see a survey edit happened between runs."""
        result = read_session(self._store.session_dir)
        valid_ids = {item.id for item in self._survey}
        asked: "set[str]" = set()
        mismatches = 0
        for record in result.records_of_kind("event"):
            if record.get("event") != EVENT_COVERAGE_ASKED:
                continue
            detail = record.get("detail") or {}
            question_id = detail.get("question_id")
            if question_id in valid_ids:
                asked.add(question_id)
            else:
                mismatches += 1
                logger.warning(
                    "coverage replay: question_id %r is no longer in the survey -- ignored",
                    question_id,
                )
        if mismatches:
            self._store.update_metadata({"coverage_replay_mismatches": mismatches})
        return CoverageState(asked=frozenset(asked))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_guarded(record)
        # Drain whatever is left, without blocking, so a stop() during a
        # burst never silently discards an already-queued interviewer
        # segment. Deliberately NEVER makes the vendor call here (independent
        # review finding D): `stop()` joins with a bounded timeout, but this
        # drain used to process every queued record through the SAME
        # full-round-trip `_process_guarded` as the live loop -- up to
        # `queue_maxsize x timeout_s` (e.g. 16 x 3.0s = 48s) under a vendor
        # outage, which is exactly the condition that fills this queue in
        # the first place. That let `stop()` return with the thread still
        # running, racing `app/main.py`'s next step (`store.close()`)
        # against this thread's own `append_event` calls. Recording each
        # remaining record as considered-but-not-matched, with reason
        # "shutdown", keeps the drain O(microseconds) per record regardless
        # of backlog depth -- bounded, and still never a silent discard.
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            self._record_shutdown_drain(record)

    def _record_shutdown_drain(self, record: SegmentRecord) -> None:
        """Same degrade-and-continue discipline as `_process_guarded` -- the
        store may already be mid-close during a raced shutdown, so a failure
        here must never propagate and block the drain loop."""
        try:
            self._store.append_event(
                event=EVENT_COVERAGE_CONSIDERED,
                speaker=SPEAKER_INTERVIEWER,
                detail={"seq": record.seq, "matched_question_id": None, "reason": "shutdown"},
            )
        except Exception as exc:  # noqa: BLE001 -- the store may be gone during shutdown
            logger.warning("coverage shutdown drain failed to record error_type=%s", type(exc).__name__)

    def _process_guarded(self, record: SegmentRecord) -> None:
        """Degrade and continue, never die.

        Only the vendor call inside `_process` is wrapped; everything else --
        disk I/O, `append_event` against a store closed by a shutdown race,
        timestamp parsing, dataclass construction from a resumed session's
        records -- can raise and unwind this worker's loop. The thread would
        then exit for the rest of the interview while the detector's own state
        still looks fine, so every later cycle keeps proposing questions the
        interviewer has already asked, with no signal anywhere that coverage
        stopped moving.

        This is the same degrade-and-continue discipline `app/session/
        audio_writer.py` established: record the failure as a named event
        carrying the error TYPE only (never transcript text), and keep going.
        """
        try:
            self._process(record)
        except Exception as exc:  # noqa: BLE001 -- a dead detector is worse than any single failure
            logger.warning("coverage processing failed error_type=%s", type(exc).__name__)
            try:
                self._store.append_event(
                    event=EVENT_COVERAGE_FAILED,
                    speaker=SPEAKER_INTERVIEWER,
                    detail={"error_type": type(exc).__name__},
                )
            except Exception:  # noqa: BLE001 -- the store itself may be gone during shutdown
                pass

    def _process(self, record: SegmentRecord) -> None:
        with self._state_lock:
            unasked = self._state.unasked_items(self._survey)
        if not unasked:
            return
        try:
            question_id = match_coverage_question(
                self._client,
                segment_text=record.text,
                unasked_items=unasked,
                model=self._model,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 -- one coverage-match failure must never
            # disable the detector or touch capture/STT/the phone surface (SURF-06).
            self._store.append_event(
                event=EVENT_REASONING_FAILURE,
                speaker=SPEAKER_INTERVIEWER,
                detail={"stage": "coverage", "error_type": type(exc).__name__},
            )
            return

        if question_id is None:
            self._store.append_event(
                event=EVENT_COVERAGE_CONSIDERED,
                speaker=SPEAKER_INTERVIEWER,
                detail={"seq": record.seq, "matched_question_id": None},
            )
            return

        with self._state_lock:
            self._state = self._state.with_asked(question_id)
            changed_state = self._state
        self._store.append_event(
            event=EVENT_COVERAGE_ASKED,
            speaker=SPEAKER_INTERVIEWER,
            detail={
                "question_id": question_id,
                "matched_seq": record.seq,
                "matched_audio_end_s": record.audio_end_s,
            },
        )
        self._notify_change(changed_state)

    def _notify_change(self, state: CoverageState) -> None:
        """Phase C: tells the caller a question just moved to `asked`, so the
        overlay's next-question panel can be re-broadcast.

        Called AFTER the durable `coverage-asked` event, and inside its own
        try/except, for two separate reasons. Ordering: the durable record is
        what `_replay_state` rebuilds from, so it must never be skipped
        because a display concern failed first. Isolation: this runs inside
        `_process_guarded`, which would otherwise record a *coverage* failure
        for what is really a broadcast failure -- and worse, a raise here
        would make a genuine, already-persisted coverage transition look like
        a processing error in the post-call read.

        The callback itself must not block (it schedules onto the event loop
        and returns -- see `app/main.py`'s `_survey_broadcast_callback`); this
        is the coverage worker thread, and the same never-block-a-worker rule
        the whole module is built on applies to it."""
        if self._on_change is None:
            return
        try:
            self._on_change(state)
        except Exception as exc:  # noqa: BLE001 -- a display concern must never fail coverage
            logger.warning("coverage on_change failed error_type=%s", type(exc).__name__)
