"""The connection manager for the phone (and optional second-monitor) surface
-- shaped like `app/capture/supervisor.py::StreamSupervisor`: it owns state,
and every connection-set transition is a decision, never a silent drop
(phase_specification §P3).

Also home to `wire_payload()`, the allowlist projection that is the whole
point of non-negotiable #9 (03-01-PLAN.md §P4): `SegmentRecord` is never
serialized wholesale. `StreamWorker._persist` fabricates each segment's
`audio_start_s` as the end of the SAME speaker's previous segment, so it
silently absorbs however long that speaker was quiet -- `app/session/render.py`'s
docstring documents this as unsafe to sort on (the bug fixed in commit
74cbd6f). The fabricated value is excluded BY CONSTRUCTION here, not by a
caller remembering to strip it: no phone, no second screen, and no future
consumer can sort on a value that was never on the wire in the first place.

Never logs message content -- connection counts and error types only (S7
extended to this package, T-03-09).
"""

from __future__ import annotations

import logging
import threading
from typing import AbstractSet, Any, Optional, Sequence

from fastapi import WebSocket

from app import config as config_mod
from app.coach.context import SurveyItem
from app.coach.generate import Candidate
from app.session.store import SegmentRecord
from app.web.status import ServerStatus, status_message

logger = logging.getLogger(__name__)

# The exact, and only, key set a `segment` wire message may carry
# (03-01-PLAN.md §P4). `audio_start_s` is deliberately absent -- see module
# docstring. Tested as set equality, not a negative check, so a future field
# nobody thought to forbid also fails the assertion.
WIRE_SEGMENT_KEYS = frozenset({"type", "seq", "speaker", "text", "audio_end_s", "detected_language"})

# 04-03-PLAN.md §E2: the exact, and only, key set a `suggestion` wire message
# may carry -- the primary suggestion's text and up to two alternative texts,
# and NOTHING else. No per-rule flags, no band, no rationale, no move type,
# no dropped candidates, no coverage state -- those live only in the durable
# `SuggestionRecord` (app/coach/pipeline.py) and are read after the call
# (04-02-PLAN.md §D8). Excluded here BY CONSTRUCTION, the same discipline
# `wire_payload()` above already applies to `audio_start_s`.
WIRE_SUGGESTION_KEYS = frozenset({"type", "primary", "alternatives"})

# Phase C: the exact, and only, key set a `survey` wire message may carry --
# the next question the interviewer has not yet asked, plus how far through
# the survey they are. Pinned by set equality exactly like the two above.
#
# Why this message exists at all: every other thing the overlay can show is
# the output of a model call, so when the model is slow or fails the surface
# is empty. This one is computed from `survey.json` plus the coverage state
# `CoverageDetector` already maintains off the interviewer's own microphone
# stream -- NO vendor call, no timeout, no failure mode. That is what makes
# Phase 4 criterion 6 ("degrades to transcript-only, never goes blank")
# STRUCTURALLY true rather than incidentally true.
#
# Deliberately NOT carried: the rest of the survey. The surface is for
# glancing (COACH-06) -- a full checklist invites reading. `asked_count`/
# `total_count` give position without giving a list to scan.
WIRE_SURVEY_KEYS = frozenset(
    {"type", "next_question_id", "next_question_text", "next_question_order", "asked_count", "total_count"}
)


def wire_payload(record: SegmentRecord) -> dict[str, Any]:
    """Projects a `SegmentRecord` onto the wire allowlist. This is the ONLY
    function that may turn a `SegmentRecord` into something sent over the
    socket -- never `dataclasses.asdict(record)`, which would leak
    `audio_start_s` (and `ready_wall_s`/`latency_s`/`engine`/`written_at`,
    none of which any consumer needs or should see)."""
    return {
        "type": "segment",
        "seq": record.seq,
        "speaker": record.speaker,
        "text": record.text,
        "audio_end_s": record.audio_end_s,
        "detected_language": record.detected_language,
    }


def suggestion_payload(
    ranked: "Sequence[Candidate]", *, max_shown: int = config_mod.DEFAULT_COACH_MAX_SHOWN
) -> dict[str, Any]:
    """Projects a ranked suggestion set onto the wire allowlist (04-03-PLAN.md
    §E2) -- the ONLY function that may turn a ranked candidate list into a
    `suggestion` wire message. `ranked` must be non-empty and already ordered
    primary-first (`critic.rank_candidates()`'s own output order) -- callers
    decide whether to broadcast at all (an empty ranked set means "nothing
    worth showing this cycle" and must never blank a healthy pipeline's
    region, per D-05/COACH-06); this function only ever projects a non-empty
    one onto the wire.

    `max_shown` (independent review finding E) is the SAME knob
    (`app/config.py`'s `DEFAULT_COACH_MAX_SHOWN`/`APP_COACH_MAX_SHOWN`) that
    already caps `ranked` itself, upstream, via `critic.rank_candidates
    (max_shown=...)` -- this parameter exists only so `APP_COACH_MAX_SHOWN`
    raised past 3 is not silently re-truncated back down to "one primary
    plus two alternatives" on the wire by a second, independent hardcoded
    cap here."""
    texts = [candidate.text for candidate in ranked]
    return {
        "type": "suggestion",
        "primary": texts[0],
        "alternatives": texts[1:max_shown],
    }


def survey_payload(
    survey: "Sequence[SurveyItem]", *, asked: "AbstractSet[str]"
) -> dict[str, Any]:
    """Projects the survey and the current coverage state onto the wire
    allowlist -- the ONLY function that may turn them into a `survey` wire
    message.

    Unlike `suggestion_payload()`, this one is ALWAYS sendable: an exhausted
    survey (every question asked) and an empty survey are both normal states,
    reported as `next_question_*: None` rather than by declining to broadcast.
    A caller that skipped the send in those cases would leave the panel
    showing a question the interviewer has already asked, which is worse than
    saying nothing -- so the "callers decide whether to broadcast" rule the
    two projectors above follow deliberately does NOT apply here.

    Ordering is re-derived from `order` HERE rather than trusted from the
    caller. `load_survey()` already returns items sorted, but "the next
    question" is the whole meaning of this message, and a future caller
    assembling a survey by any other route (a merged multi-project read, say)
    must not be able to change which question the interviewer is shown by
    changing a list's order."""
    ordered = sorted(survey, key=lambda item: item.order)
    next_item: "Optional[SurveyItem]" = next(
        (item for item in ordered if item.id not in asked), None
    )
    # Counted against the survey's OWN ids, never `len(asked)`: coverage state
    # is replayed from the durable record and can name a question a later
    # survey edit removed (`CoverageDetector._replay_state` tolerates exactly
    # that, with a warning). Counting the raw set would then report more
    # questions asked than the survey has.
    asked_count = sum(1 for item in ordered if item.id in asked)
    return {
        "type": "survey",
        "next_question_id": next_item.id if next_item is not None else None,
        "next_question_text": next_item.text if next_item is not None else None,
        "next_question_order": next_item.order if next_item is not None else None,
        "asked_count": asked_count,
        "total_count": len(ordered),
    }


def hello_payload(
    *,
    session_id: str,
    server_started_at: str,
    heartbeat_interval_ms: int,
    heartbeat_timeout_ms: int,
    reconnect_backoff_base_ms: int,
    reconnect_backoff_cap_ms: int,
    project: "Optional[str]",
    grounding_sources: "Sequence[str]",
    coach_enabled: bool,
) -> dict[str, Any]:
    """The one-time `hello` message sent on every fresh connect, so the client
    can tell a fresh page load from a reconnect (03-01-PLAN.md §P4). Carries
    no resume position of any kind -- D-01 is an absence, not a field to
    remember to omit.

    03-02-PLAN.md Task 1: also the one source of truth for the client's
    heartbeat/backoff timing knobs (`app/config.py`'s `DEFAULT_HEARTBEAT_*`/
    `DEFAULT_RECONNECT_BACKOFF_*`) -- never hardcoded a second time in
    `app.js`.

    Phase A: `project` and `grounding_sources` say which knowledge is in play.
    Carried on `hello` rather than as a new message type because it is fixed
    for the whole run -- it is decided by the CLI arguments before the first
    audio frame and can never change while the session is open, so a
    broadcast channel for it would be a channel that never fires twice. A
    reconnect re-sends `hello`, so the panel survives a sleeping phone.

    Both have NO default. `hello` is the one place a client learns what it is
    looking at, and a defaulted "no sources" would render an identical, and
    wrong, panel for a run that had several -- indistinguishable from a run
    that had none.

    `coach_enabled` has no default for a stronger version of the same reason,
    and it is here because of a defect the 2026-08-12 audit found: the overlay
    decided whether to show its coaching panels from HOW THE PROJECT IS FILED
    -- a string it knew before the child process even started -- and never
    from whether a coach came up. A run with no `ANTHROPIC_API_KEY` therefore
    showed "Next question" and "Coach" panels, resting and empty, for a whole
    interview. `UNCOACHED_TEXT`'s own comment says a panel that promises
    something for a whole interview teaches the operator that the surface
    lies, and that is exactly what shipped.

    It rides on `hello` rather than on a channel of its own for the reason
    stated above: it is decided before the first audio frame and cannot change
    while the session is open, so a broadcast for it would never fire twice.
    A reconnect re-sends `hello`, so a phone that slept through the start
    still learns it. This is a fact about the RUN, not about the operator's
    intent -- `--no-coach` and a missing key both arrive here as `False`,
    because from a client's point of view they are the same situation, and
    the client that needs to word them differently already knows which one it
    asked for.

    **Node paths only, and note what that means on a phone.** These are labels
    the OPERATOR typed, never customer speech, so gate 12's no-text rule is
    untouched. But they are rendered on a screen that may be sitting in a
    cafe, and an operator who names a node after a customer has put that name
    on it. That is a deliberate trade the user asked for -- the operator must
    always be able to see which knowledge is in play -- and it is the reason
    this carries the path and nothing else: no brief, no survey, no fact."""
    return {
        "type": "hello",
        "session_id": session_id,
        "server_started_at": server_started_at,
        "heartbeat_interval_ms": heartbeat_interval_ms,
        "heartbeat_timeout_ms": heartbeat_timeout_ms,
        "reconnect_backoff_base_ms": reconnect_backoff_base_ms,
        "reconnect_backoff_cap_ms": reconnect_backoff_cap_ms,
        "project": project,
        "grounding_sources": list(grounding_sources),
        "coach_enabled": coach_enabled,
    }


class ConnectionHub:
    """One instance per run. Owns the active-connection set; every method
    that touches it takes the lock, keeping the critical section small
    (matches `app/session/store.py`'s locking convention). Guarded because it
    is touched from both the WebSocket accept path and the broadcast path
    scheduled onto the loop via `run_coroutine_threadsafe`."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = threading.Lock()

    @property
    def connection_count(self) -> int:
        with self._lock:
            return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts the socket and adds it to the active set. Callers must
        have already validated the join token BEFORE calling this -- a
        rejected token must never reach this method (D-05: a wrong token is
        rejected without ever entering the connection set)."""
        await websocket.accept()
        with self._lock:
            self._connections.add(websocket)
        logger.debug("connection added total=%d", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            self._connections.discard(websocket)
        logger.debug("connection removed total=%d", len(self._connections))

    async def broadcast(self, record: SegmentRecord) -> None:
        """Fans a finalized segment out to every active connection."""
        await self._broadcast_payload(wire_payload(record))

    async def broadcast_status(self, status: ServerStatus) -> None:
        """Fans a `server_status` change out to every active connection --
        the same per-connection isolation as `broadcast()` (03-02-PLAN.md
        Task 2 §Q6/§Q7). Callers decide WHETHER to call this (only on an
        actual change -- see `app/web/status.py::StatusBoard.observe`); this
        method itself always sends unconditionally."""
        await self._broadcast_payload(status_message(status))

    async def broadcast_suggestion(
        self, ranked: "Sequence[Candidate]", *, max_shown: int = config_mod.DEFAULT_COACH_MAX_SHOWN
    ) -> None:
        """Fans a ranked suggestion set out to every active connection -- the
        same per-connection isolation as `broadcast()`/`broadcast_status()`
        above (04-03-PLAN.md §E2). Callers decide WHETHER to call this (never
        with an empty `ranked` -- see `suggestion_payload()`'s own docstring);
        this method itself always sends unconditionally, mirroring
        `broadcast_status()`'s own contract. `max_shown` is threaded straight
        through to `suggestion_payload()` -- see its docstring (finding E)."""
        await self._broadcast_payload(suggestion_payload(ranked, max_shown=max_shown))

    async def broadcast_survey(
        self, survey: "Sequence[SurveyItem]", *, asked: "AbstractSet[str]"
    ) -> None:
        """Fans the next-question state out to every active connection -- the
        same per-connection isolation as every other broadcast here. Sent
        unconditionally, including when the survey is exhausted or empty: see
        `survey_payload()`'s docstring for why this one has no
        callers-decide-whether rule."""
        await self._broadcast_payload(survey_payload(survey, asked=asked))

    async def _broadcast_payload(self, payload: dict) -> None:
        """Fans one JSON-able payload out to every active connection. Each
        per-connection `send_json` is wrapped in its OWN try/except so one
        dead connection (a sleeping phone, say) can never prevent a second
        connection (a second monitor) from receiving the same payload --
        the one-dead-connection-must-not-starve-another rule this whole
        class exists to enforce."""
        with self._lock:
            targets = list(self._connections)

        dead: list[WebSocket] = []
        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception as exc:  # noqa: BLE001 -- one dead connection must not break the loop
                logger.warning("broadcast send failed error_type=%s", type(exc).__name__)
                dead.append(websocket)

        if dead:
            with self._lock:
                for websocket in dead:
                    self._connections.discard(websocket)
