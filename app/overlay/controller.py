"""The loop: window events in, session events out, redraw.

The only place tkinter and `InterviewSession` meet. Kept separate from both so
the window stays a dumb renderer and the session stays a dumb supervisor --
and so the whole state machine below can be driven in a test by calling its
methods, with a fake window and a fake session.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable, Optional, Sequence

from app import project_meta
from app.overlay.model import STATE_RECORDING, STATE_STOPPING, OverlayState
from app.overlay.session import SessionEvent

logger = logging.getLogger(__name__)

# How often the queue is drained and the clock advanced. 250ms is under the
# threshold where a suggestion feels delayed, and far above the rate at which
# redrawing costs anything -- the window is a dozen labels.
POLL_INTERVAL_MS = 250


class OverlayController:
    """Owns the `OverlayState` and is the only thing allowed to change it."""

    def __init__(
        self,
        *,
        window: Any,
        list_projects: "Callable[[], Sequence[str]]",
        start_session: "Callable[..., Any]",
        clock: "Callable[[], float]" = time.monotonic,
    ) -> None:
        self._window = window
        self._list_projects = list_projects
        self._start_session = start_session
        self._clock = clock
        self._events: "queue.Queue[SessionEvent]" = queue.Queue()
        self._session: "Optional[Any]" = None
        self._started_at: "Optional[float]" = None
        self.state = OverlayState()

    # --- window callbacks -------------------------------------------------

    def on_record_pressed(self) -> None:
        """Reads the project list HERE rather than at construction, so a
        project authored while the overlay sat idle appears without a
        restart. The overlay is meant to live on the desktop all day."""
        try:
            projects = list(self._list_projects())
        except Exception as exc:  # noqa: BLE001 -- an unreadable tree must not kill the overlay
            logger.warning("listing projects failed error_type=%s", type(exc).__name__)
            self._render(self.state.failed("could not read the project list"))
            return
        self._window.set_projects(projects)
        self._render(self.state.picking())

    def on_project_chosen(self, project: str, *, loopback_only: bool = False) -> None:
        # The picker reports a cancel as an empty choice, so both paths land
        # here and there is exactly one place that leaves the picker.
        if not project:
            self._render(self.state.cancelled())
            return
        # Decided ONCE, here, and passed to both the child and the state: the
        # unfiled bucket has no `context/`, so there is no brief and no survey
        # to ground a coach in. Two independent derivations of this -- one for
        # the command line, one for what gets drawn -- is exactly how a
        # panel ends up promising a coach that was never started.
        coached = not project_meta.is_unfiled(project)
        try:
            # `loopback_only` is threaded to BOTH the child and the state for
            # the same reason `coached` is: two independent derivations of one
            # fact is how a panel ends up describing a recording that is not
            # the one running.
            self._session = self._start_session(
                project, self._events, coached=coached, loopback_only=loopback_only
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("starting the interview failed error_type=%s", type(exc).__name__)
            self._render(self.state.failed(f"could not start: {type(exc).__name__}"))
            return
        self._started_at = self._clock()
        self._render(
            self.state.recording(project, coached=coached, loopback_only=loopback_only)
        )

    def on_mic_toggled(self) -> None:
        """Releases or re-opens the microphone mid-interview.

        The state is set from the value we ASKED for, not from a confirmation,
        and that is deliberate: the child acknowledges nothing today, and a
        control that waited for a reply it will never get would sit there
        looking broken. `set_microphone` swallows its own failures and logs
        them, so the panel can disagree with reality only if the pipe is dead
        -- at which point the interview is over anyway.

        Ignored unless recording, like `on_stop_pressed`: there is no
        microphone to release from the picker."""
        if self.state.phase != STATE_RECORDING:
            return
        enabled = not self.state.mic_on
        session = self._session
        if session is not None:
            session.set_microphone(enabled)
        self._render(self.state.with_microphone(enabled))

    def on_stop_pressed(self) -> None:
        """Renders `stopping` BEFORE asking the session to stop.

        `stop()` blocks for as long as the clean shutdown takes -- up to
        twenty seconds while two capture threads, two STT workers and a
        server come down. Without this the button would simply stop
        responding, and the operator would press it again."""
        if self.state.phase != STATE_RECORDING:
            return
        self._render(self.state.stopping())
        session, self._session = self._session, None
        if session is not None:
            session.stop()

    # --- the tick ---------------------------------------------------------

    def poll(self) -> None:
        """Drains every queued event, then advances the clock. Called on a
        timer from the tkinter main loop."""
        drained = False
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            drained = True
            self._apply(event)

        if self.state.phase == STATE_RECORDING and self._started_at is not None:
            elapsed = int(self._clock() - self._started_at)
            if elapsed != self.state.elapsed_s:
                self.state = self.state.ticked(elapsed)
                drained = True

        if drained:
            self._window.render(self.state)

    def _apply(self, event: SessionEvent) -> None:
        if event.kind == "wire" and event.payload is not None:
            self.state = self.state.applied(event.payload)
            return
        if event.kind == "microphone" and event.payload is not None:
            # AUTHORITATIVE, and it overrides the optimistic value set when the
            # control was clicked. That optimism is deliberate -- a control
            # that waits for a round trip before moving reads as broken -- but
            # it must be correctable, or a device that failed to re-open leaves
            # the panel reading "MIC ON" over a closed microphone.
            self.state = self.state.with_microphone(bool(event.payload["enabled"]))
            return
        if event.kind == "started":
            return
        if event.kind == "failed":
            # A failed WIRE connection is not a failed interview -- the
            # recording is still running and still writing the transcript.
            # Only leave the recording state if we were never in it.
            if self.state.phase in (STATE_RECORDING, STATE_STOPPING):
                logger.warning("overlay lost its display: %s", event.message)
                return
            self.state = self.state.failed(event.message or "the interview failed")
            return
        if event.kind == "exited":
            self._started_at = None
            self._session = None
            self.state = OverlayState(error=event.message)

    def _render(self, state: OverlayState) -> None:
        self.state = state
        self._window.render(state)

    # --- driving the tkinter loop ----------------------------------------

    def attach(self, root: Any) -> None:
        """Schedules `poll` on tkinter's own timer. Uses `after` rather than a
        thread because every tkinter call must happen on the thread that
        created the window."""

        def _tick() -> None:
            try:
                self.poll()
            except Exception as exc:  # noqa: BLE001 -- one bad frame must not stop the clock
                logger.warning("overlay poll failed error_type=%s", type(exc).__name__)
            root.after(POLL_INTERVAL_MS, _tick)

        root.after(POLL_INTERVAL_MS, _tick)
