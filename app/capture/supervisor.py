"""Owns the health-to-event translation for ONE stream (speaker side):
idle onset/recovery, lag onset, drops with a running count, device change,
degraded/dead, all routed through the SessionStore at the position where
they happened (phase_specification T1/T2, S5). This is the only thing
standing between a dead or degraded stream and a user who keeps trusting it
(OPS-05) -- silence is never an acceptable outcome here.

State transitions are debounced (edge-triggered): an idle minute produces
exactly one `idle` event, not one per poll cycle. `poll_once()` is public and
side-effect-complete on its own so a test can drive it deterministically,
with no dependency on the background thread's timing; `start()`/`stop()`
just call it on an interval.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from app.capture.devices import DeviceInfo, DeviceOpenError
from app.session.store import SessionStore

logger = logging.getLogger(__name__)

EVENT_IDLE = "idle"
EVENT_RECOVERED = "recovered"
EVENT_LAGGING = "lagging"
EVENT_DROPPED = "dropped"
EVENT_DEVICE_CHANGED = "device-changed"
EVENT_REOPENED = "reopened"
EVENT_DEAD = "dead"


class SupervisedCapture(Protocol):
    """The slice of `CaptureStream` the supervisor depends on -- deliberately
    a Protocol, not a concrete import-time dependency, so a deterministic
    test can drive a stub satisfying just this shape without any hardware."""

    @property
    def health(self): ...

    @property
    def device(self) -> DeviceInfo: ...

    @property
    def is_dead(self) -> bool: ...

    @property
    def queue(self): ...  # exposes `.maxsize`

    def reopen(self, new_device: DeviceInfo) -> None: ...

    def mark_dead(self) -> None: ...


@dataclass(frozen=True)
class SupervisorConfig:
    poll_interval_s: float = 0.5
    idle_threshold_s: float = 5.0
    lag_queue_fraction: float = 0.5
    device_poll_interval_s: float = 2.0


class StreamSupervisor:
    """One per stream. Polls `capture.health` on an interval, debounces
    state transitions, and appends named events to the store. Also owns the
    device-change watch and the single re-open attempt (T-02-13, phase
    research's second trap)."""

    def __init__(
        self,
        *,
        speaker: str,
        capture: SupervisedCapture,
        store: SessionStore,
        device_resolver: Optional[Callable[[], DeviceInfo]] = None,
        config: Optional[SupervisorConfig] = None,
    ) -> None:
        self._speaker = speaker
        self._capture = capture
        self._store = store
        self._device_resolver = device_resolver
        self._config = config or SupervisorConfig()

        self._idle = False
        self._lagging = False
        self._last_dropped = 0
        self._last_device_check = 0.0
        self._last_failed_identity: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_idle(self) -> bool:
        """Whether this side has stopped producing real audio.

        Public because the status poller runs its own loop and had no way to
        ask: this was computed here, debounced, written to the store as an
        `idle` event, and never reached the banner. A muted interviewee
        therefore produced a green status, live, during the one event that
        cannot be re-run.

        Already debounced by `_check_idle`'s threshold, so a reader gets the
        settled answer rather than a per-frame flicker."""
        return self._idle

    @property
    def is_lagging(self) -> bool:
        """Whether the capture queue is backing up. Same reasoning as
        `is_idle`: detected here, recorded as an event, previously unsayable."""
        return self._lagging

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"supervisor-{self._speaker}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self._config.poll_interval_s)

    def poll_once(self) -> None:
        """One complete supervision cycle: idle/recovery, lag, drops, and
        (interval-gated) the device-change watch. Public so tests can drive
        it deterministically without depending on thread timing."""
        if self._capture.is_dead:
            return
        self._check_idle()
        self._check_lag()
        self._check_drops()
        self._check_device_change()

    def _check_idle(self) -> None:
        health = self._capture.health
        now = time.monotonic()
        is_idle = (
            health.last_real_audio_monotonic is None
            or (now - health.last_real_audio_monotonic) >= self._config.idle_threshold_s
        )
        if is_idle and not self._idle:
            self._idle = True
            self._store.append_event(
                event=EVENT_IDLE, speaker=self._speaker,
                detail={"consecutive_synthesized_frames": health.consecutive_synthesized_frames},
            )
        elif not is_idle and self._idle:
            self._idle = False
            self._store.append_event(event=EVENT_RECOVERED, speaker=self._speaker, detail={})

    def _check_lag(self) -> None:
        health = self._capture.health
        maxsize = self._capture.queue.maxsize or 1
        is_lagging = (health.queue_depth / maxsize) >= self._config.lag_queue_fraction
        if is_lagging and not self._lagging:
            self._lagging = True
            self._store.append_event(
                event=EVENT_LAGGING, speaker=self._speaker,
                detail={"queue_depth": health.queue_depth, "queue_max": maxsize},
            )
        elif not is_lagging and self._lagging:
            self._lagging = False

    def _check_drops(self) -> None:
        health = self._capture.health
        if health.dropped_frame_count > self._last_dropped:
            self._last_dropped = health.dropped_frame_count
            self._store.append_event(
                event=EVENT_DROPPED, speaker=self._speaker,
                detail={"total_dropped": self._last_dropped},
            )

    def _check_device_change(self) -> None:
        if self._device_resolver is None or self._capture.is_dead:
            return
        now = time.monotonic()
        if now - self._last_device_check < self._config.device_poll_interval_s:
            return
        self._last_device_check = now

        try:
            new_device = self._device_resolver()
        except DeviceOpenError:
            return  # transient resolution failure -- not a change, try again next interval

        current = self._capture.device
        if new_device.identity == current.identity:
            return
        if new_device.identity == self._last_failed_identity:
            return  # exactly ONE attempt against a device that already failed -- no retry loop

        self._store.append_event(
            event=EVENT_DEVICE_CHANGED, speaker=self._speaker,
            detail={"old_device": current.name, "new_device": new_device.name},
        )
        try:
            self._capture.reopen(new_device)
        except DeviceOpenError as exc:
            self._last_failed_identity = new_device.identity
            self._capture.mark_dead()
            self._store.append_event(
                event=EVENT_DEAD, speaker=self._speaker,
                detail={"device": new_device.name, "reason": str(exc)},
            )
        else:
            self._last_failed_identity = None
            self._store.append_event(
                event=EVENT_REOPENED, speaker=self._speaker, detail={"device": new_device.name},
            )
