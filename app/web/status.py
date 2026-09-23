"""Server-side health status, pushed to every connected client alongside
`hello` on every fresh connect (03-02-PLAN.md Task 2 §Q6/§Q7).

Shaped like `app/capture/supervisor.py`'s config/state pair: a small frozen
value (`ServerStatus`) plus the transition logic (`StatusBoard`), kept
completely separate from `app/web/hub.py` so both are unit-testable without a
socket.

The mechanism that keeps the two failure classes in `03-RESEARCH.md` §6 from
ever looking alike: a laptop-side failure (an STT worker died, or capture
stopped) is a DISTINCT message type (`server_status`) from a `segment`, and
the CURRENT status is re-sent on every fresh connect -- so it survives a
phone reconnect, even though transcript content deliberately does NOT
(D-01). This does not violate D-01: D-01 forbids replaying missed
*transcript* and says the client renders whatever arrives next; a fresh
status push on connect is exactly "whatever arrives next", and it is the one
mechanism this phase has for telling a laptop-side failure apart from a
phone-side connectivity problem (the client-local, unpersisted gap marker in
`app/static/live-state.mjs`, which explicitly does NOT survive a reconnect --
it is local state that disappears with the socket that detected it).

Never carries transcript text or a vendor name -- only a failed side and a
reason class (T-03-16, extending 02-01 §S7 and the STT-03 no-vendor-name
posture to this new status path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

STATUS_HEALTHY = "healthy"
STATUS_FAILED = "failed"
# A stream can be useless without being dead. `failed` means transcription has
# STOPPED; this means it is still running and something is wrong with what is
# reaching it -- a muted call app, an unplugged headset, a backed-up queue.
# A distinct value rather than a reuse of `failed` because "your recording
# stopped" is FALSE here, and saying it would be exactly the false alarm the
# failure vocabulary was built to avoid.
STATUS_DEGRADED = "degraded"

# 04-03-PLAN.md §E5: the coach's failed side and reason class, alongside the
# existing stream-failure vocabulary. `FAILED_SIDE_COACH` is not a speaker
# label (it names neither "interviewee" nor "interviewer") -- it is what the
# client branches on to pick the calmer "suggestions unavailable" sentence
# instead of the (false, in this case) "transcription stopped" one.
FAILED_SIDE_COACH = "coach"
REASON_STREAM_FAILURE = "stream-failure"
REASON_COACH_FAILURE = "coach-failure"
# Kept apart rather than folded into one "degraded" reason: no audio arriving
# and audio arriving late have different remedies (check the call app / check
# the machine), and collapsing them would rebuild a smaller version of the
# defect this tier exists to fix.
REASON_NO_AUDIO = "no-audio"
REASON_LAGGING = "lagging"
# Distinct from REASON_NO_AUDIO on the same reasoning, and the distinction is
# the point: `no-audio` means the device stopped handing over frames, and this
# means frames are still arriving but nothing is coming back transcribed while
# the OTHER side is fine. Their remedies differ completely -- one is "the
# device died", the other is "you are recording the wrong endpoint" -- and the
# second is the one that went undetected for a whole interview on 2026-08-17.
REASON_NO_TRANSCRIPT = "no-transcript"

# The three vocabularies above, as sets, so they can be pinned by EQUALITY the
# way `SERVER_MESSAGE_TYPES` already is in `app/web/server.py`.
#
# Their absence was the gap the 2026-08-12 audit walked through. `type` had a
# collection and two set-equality assertions; `status`, `reason` and
# `failed_side` had eight bare literals and no collection at all, so nothing
# could be checked against them -- and `app/static/app.js` compared each one as
# a hand-typed string. Renaming REASON_NO_AUDIO left 1092 tests green and the
# phone permanently in the wrong branch.
#
# `failed_side` carries a SPEAKER label as well as `coach`, which is why the
# set is assembled from `app/main.py`'s `ALL_SPEAKERS` rather than restating
# the two words here. Imported inside the tuple below rather than at module
# scope on purpose: `app/main.py` imports this module, and a top-level import
# back would be a cycle.
ALL_STATUSES = (STATUS_HEALTHY, STATUS_DEGRADED, STATUS_FAILED)
ALL_REASONS = (
    REASON_STREAM_FAILURE,
    REASON_COACH_FAILURE,
    REASON_NO_AUDIO,
    REASON_LAGGING,
    REASON_NO_TRANSCRIPT,
)


def all_failed_sides() -> "tuple[str, ...]":
    """Every value `failed_side` can carry: the two speakers, plus `coach`.

    A function rather than a constant because the speaker labels live in
    `app/main.py`, which imports this module -- naming them again here is the
    duplication this whole block exists to remove."""
    from app.main import ALL_SPEAKERS

    return (*ALL_SPEAKERS, FAILED_SIDE_COACH)


@dataclass(frozen=True)
class ServerStatus:
    """The entire server-status wire shape. `failed_side`/`reason` are only
    ever a speaker label and a short reason CLASS (e.g. "stream-failure") --
    never a message, a vendor name, or any transcript text."""

    value: str
    failed_side: Optional[str] = None
    reason: Optional[str] = None


HEALTHY_STATUS = ServerStatus(value=STATUS_HEALTHY)


def status_message(status: ServerStatus) -> dict[str, Any]:
    """Projects a `ServerStatus` onto the wire -- the ONLY function that may
    turn one into a `server_status` message, matching `wire_payload()`'s
    allowlist-projection precedent in `app/web/hub.py`."""
    return {
        "type": "server_status",
        "status": status.value,
        "failed_side": status.failed_side,
        "reason": status.reason,
    }


@dataclass(frozen=True)
class StreamHealth:
    """The slice of one `StreamPair`'s publicly-exposed state the status
    poller needs -- a plain value, not a `StreamPair` import, so this module
    stays fully unit-testable without constructing a real capture/STT stack
    (mirrors `app/capture/supervisor.py::SupervisedCapture`'s Protocol-not-
    concrete-dependency shape)."""

    speaker: str
    worker_failed: bool
    capture_dead: bool
    # Both DEFAULT so every caller that predates the degraded tier keeps
    # working unchanged -- and both are already computed, debounced, by
    # `StreamSupervisor` (`is_idle`/`is_lagging`), which wrote them to the
    # store as events while the status poller ran a separate loop that never
    # asked. The information existed; nothing said it.
    idle: bool = False
    lagging: bool = False
    # Frames ARE arriving; nothing is coming back transcribed while the other
    # side is producing. Computed by `app/session/feed_watch.py` and folded in
    # here rather than broadcast separately, so this module stays the one place
    # that decides what the banner says.
    not_transcribing: bool = False


def derive_status(healths: Sequence[StreamHealth], *, coach_unhealthy: bool = False) -> ServerStatus:
    """The healthy value if nothing has failed; otherwise a failed status
    naming which side and a reason class. Reads state each stream pair
    ALREADY exposes publicly (`StreamWorker.failed`, `CaptureStream.is_dead`)
    -- adds no new callback into `app/capture/supervisor.py` or
    `app/stt/live.py` beyond what 03-01 already added.

    04-03-PLAN.md §E5: `coach_unhealthy` folds `CoachHealth.is_unhealthy()`
    (04-02-PLAN.md §D7) into this SAME derivation function rather than a
    second status mechanism. A dead capture/STT stream is checked FIRST and
    wins if both have failed -- a dead capture is the more urgent fact, and
    telling the user their recording stopped when it is the coach that died
    would itself be the false alarm this ordering prevents. Defaults to
    `False` so every pre-Phase-4 caller (and every existing test) is
    unaffected."""
    for health in healths:
        if health.worker_failed or health.capture_dead:
            return ServerStatus(value=STATUS_FAILED, failed_side=health.speaker, reason=REASON_STREAM_FAILURE)
    if coach_unhealthy:
        return ServerStatus(value=STATUS_FAILED, failed_side=FAILED_SIDE_COACH, reason=REASON_COACH_FAILURE)
    # Only once nothing has FAILED. The whole sequence is scanned for failures
    # before any degraded stream is considered, so iteration order can never
    # decide severity -- a degraded interviewee listed first must not mask a
    # dead interviewer, which would be a worse bug than the one this tier
    # fixes. `failed_side` carries the side here too: the key is named for the
    # failure case that came first, but a second key meaning the same thing
    # would widen the wire allowlist for no gain.
    for health in healths:
        if health.idle:
            return ServerStatus(
                value=STATUS_DEGRADED, failed_side=health.speaker, reason=REASON_NO_AUDIO
            )
    # After `idle` and before `lagging`. If frames stopped arriving at all,
    # `no-audio` is the more precise diagnosis and should win; but a side that
    # is transcribing nothing is a worse fault than a side running late, and
    # ordering it after `lagging` would let a slow interviewer stream mask a
    # dead interviewee one.
    for health in healths:
        if health.not_transcribing:
            return ServerStatus(
                value=STATUS_DEGRADED, failed_side=health.speaker, reason=REASON_NO_TRANSCRIPT
            )
    for health in healths:
        if health.lagging:
            return ServerStatus(
                value=STATUS_DEGRADED, failed_side=health.speaker, reason=REASON_LAGGING
            )
    return HEALTHY_STATUS


class StatusBoard:
    """Owns 'the last broadcast status' so a poller can decide whether a
    newly observed state differs from what was already pushed -- a status
    that has not changed is not re-broadcast every poll (§Q7). Also what a
    fresh connect reads from directly, so a client that connects AFTER a
    failure still receives it (§Q6)."""

    def __init__(self) -> None:
        self._current: ServerStatus = HEALTHY_STATUS

    @property
    def current(self) -> ServerStatus:
        return self._current

    def observe(self, status: ServerStatus) -> bool:
        """Records `status` as current. Returns `True` if it differs from
        what was already current (callers should broadcast), `False` if
        nothing changed."""
        changed = status != self._current
        self._current = status
        return changed
