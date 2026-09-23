"""A stream can be useless without being dead, and the banner must say so.

`StreamHealth` carried two booleans -- `worker_failed` and `capture_dead` --
so `derive_status` returned HEALTHY for every condition short of death:

  - the interviewee's call app is muted, or their headset was unplugged, and
    the capture is producing synthesized silence (`idle`);
  - the STT queue is backing up and audio is arriving late (`lagging`).

`StreamSupervisor` already detects both, debounced, on a 0.5s cadence, and
writes them to the store as `idle` / `lagging` / `recovered` events. It kept
them in private attributes, and the status poller ran its own loop that never
asked. The gap between "the system knows" and "the system says" was the whole
defect: a muted interviewee produced a confident green banner, live, during
the one event that cannot be re-run. You find out afterwards, from the
transcript.

Severity ordering is the part to get right. Degraded must sit BELOW both
failure kinds, so nothing that used to report `failed` now reports something
softer -- a real stream failure outranks "no audio for a while", and the
existing ordering (stream failure beats coach failure) is untouched.
"""

from __future__ import annotations

import pytest

from app.web import status as status_mod
from app.web.status import (
    FAILED_SIDE_COACH,
    HEALTHY_STATUS,
    REASON_COACH_FAILURE,
    REASON_STREAM_FAILURE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    StreamHealth,
    derive_status,
)


def _health(speaker="interviewee", **kwargs) -> StreamHealth:
    base = {"worker_failed": False, "capture_dead": False}
    base.update(kwargs)
    return StreamHealth(speaker=speaker, **base)


def test_a_silent_interviewee_is_not_reported_as_healthy():
    # The defect, in one line. Their call app is muted; capture is alive and
    # producing synthesized frames; the operator is looking at a green banner
    # wondering why the transcript stopped moving.
    assert derive_status([_health(idle=True)]) != HEALTHY_STATUS


def test_an_idle_stream_names_the_side_and_a_reason_class():
    result = derive_status([_health(speaker="interviewee", idle=True)])
    assert result.value == STATUS_DEGRADED
    assert result.failed_side == "interviewee"
    assert result.reason == status_mod.REASON_NO_AUDIO


def test_a_lagging_stream_is_degraded_with_its_own_reason():
    # Distinct from idle on purpose: "no audio is arriving" and "audio is
    # arriving late" have different remedies, and collapsing them would
    # rebuild a smaller version of the bug this fixes.
    result = derive_status([_health(speaker="interviewer", lagging=True)])
    assert result.value == STATUS_DEGRADED
    assert result.reason == status_mod.REASON_LAGGING
    assert result.reason != status_mod.REASON_NO_AUDIO


def test_nothing_wrong_is_still_exactly_the_healthy_singleton():
    # The over-correction guard. If a healthy poll started returning a NEW
    # equal-but-distinct object, `StatusBoard` would see a change every cycle
    # and rebroadcast forever.
    assert derive_status([_health(), _health(speaker="interviewer")]) is HEALTHY_STATUS


def test_a_dead_stream_still_outranks_a_degraded_one():
    # Severity ordering. A degraded tier that could mask a failure would be a
    # worse bug than the one being fixed -- the operator would be told
    # "audio is late" while transcription has actually stopped.
    result = derive_status([_health(speaker="interviewee", idle=True, capture_dead=True)])
    assert result.value == STATUS_FAILED
    assert result.reason == REASON_STREAM_FAILURE


def test_a_failure_on_the_OTHER_side_still_outranks_a_degraded_stream():
    # The sneaky version of the ordering test: the degraded stream is checked
    # FIRST in the sequence, so an implementation that returns on the first
    # interesting health it sees reports "degraded" while the other side is
    # dead. Iteration order must not decide severity.
    result = derive_status([
        _health(speaker="interviewee", idle=True),
        _health(speaker="interviewer", worker_failed=True),
    ])
    assert result.value == STATUS_FAILED
    assert result.failed_side == "interviewer"


def test_a_coach_failure_also_outranks_a_degraded_stream():
    result = derive_status([_health(idle=True)], coach_unhealthy=True)
    assert result.value == STATUS_FAILED
    assert result.failed_side == FAILED_SIDE_COACH
    assert result.reason == REASON_COACH_FAILURE


def test_the_new_fields_default_to_false_so_every_existing_caller_is_unaffected():
    # `StreamHealth` is constructed in `app/main.py` and in a number of tests
    # that predate this change. Defaults are what keep those honest rather
    # than needing a sweep.
    assert derive_status([StreamHealth(speaker="interviewee", worker_failed=False, capture_dead=False)]) is HEALTHY_STATUS


def test_the_supervisor_exposes_what_the_poller_needs_to_read():
    # The seam. The supervisor computed `_idle`/`_lagging` privately and the
    # poller had no way to ask, which is why the information existed and was
    # never said. Without public readers this whole file tests a function
    # nothing can feed.
    from app.capture.supervisor import StreamSupervisor

    assert hasattr(StreamSupervisor, "is_idle")
    assert hasattr(StreamSupervisor, "is_lagging")


@pytest.mark.parametrize("value", [STATUS_DEGRADED])
def test_degraded_is_a_distinct_wire_value(value):
    # Not a third boolean and not a reuse of "failed": the phone and the
    # overlay have to be able to word it differently, because "your recording
    # stopped" is false here and would be the false alarm the failure
    # vocabulary was careful to avoid in the first place.
    assert value not in (status_mod.STATUS_HEALTHY, status_mod.STATUS_FAILED)
    message = status_mod.status_message(
        derive_status([_health(idle=True)])
    )
    assert message["status"] == STATUS_DEGRADED
    # The wire KEY SET is unchanged -- same message type, new value -- so the
    # server-message allowlist and its gate are untouched.
    assert set(message) == {"type", "status", "failed_side", "reason"}
