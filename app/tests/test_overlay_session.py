"""The overlay's supervisor and controller, driven against fakes.

No subprocess, no socket, no audio device and no window. Every dependency of
`InterviewSession` and `OverlayController` is injected precisely so this file
can exercise the whole lifecycle -- including the stop path, which is the one
with real consequences for a recorded interview.
"""

from __future__ import annotations

import os
import queue
import signal

import pytest

from app.overlay.controller import OverlayController
from app.overlay.model import STATE_IDLE, STATE_PICKING, STATE_RECORDING, STATE_STOPPING
from app.overlay.session import InterviewSession, SessionEvent, parse_join_token


# --- reading the token off the child's own output ----------------------------------------


def test_the_token_is_read_from_the_line_the_interview_actually_prints():
    # Pinned against the real format from `app/web/join.py`, because this is
    # the one brittle coupling in the overlay: it parses a line printed for a
    # HUMAN. If that line changes, this test is what says so.
    line = "Join URL: http://192.168.68.100:8765/s/91WuFQzqbft8laiFJ0bI0A"
    assert parse_join_token(line) == "91WuFQzqbft8laiFJ0bI0A"


@pytest.mark.parametrize(
    "line", ["Session path: C:\\x\\y", "Coach enabled: brief has 6 item(s)", "", "no url here"]
)
def test_an_ordinary_line_yields_no_token(line):
    assert parse_join_token(line) is None


def test_a_token_with_url_safe_punctuation_survives():
    assert parse_join_token("Join URL: http://127.0.0.1:8765/s/a-b_c123") == "a-b_c123"


# --- the command line --------------------------------------------------------------------


def _session(project="acme", **kwargs):
    return InterviewSession(project, events=queue.Queue(), **kwargs)


def test_the_command_is_the_one_a_person_would_type():
    argv = _session("acme/onboarding", python_executable="py.exe").command()
    assert argv[1:] == ["-u", "-m", "app", "start", "--project", "acme/onboarding"]


def test_unbuffered_output_is_not_optional():
    # The join token is read off stdout. A block-buffered pipe holds it until
    # the buffer fills, which for a process that prints a few lines and then
    # goes quiet for an hour is forever.
    assert "-u" in _session().command()


def test_every_extra_source_is_passed_through():
    argv = _session("acme", read_from=("acme/billing", "acme/support")).command()
    assert argv.count("--read-from") == 2
    assert "acme/billing" in argv and "acme/support" in argv


# --- stopping is a Ctrl+C, not a kill ----------------------------------------------------


class _FakeProcess:
    def __init__(self, *, exits_after: int = 1, lines=(), exit_code: int = 0):
        self.signals: "list[int]" = []
        self.killed = False
        self._polls = 0
        self._exits_after = exits_after
        self._exit_code = exit_code
        self.stdout = iter(lines)

    def send_signal(self, sig):
        self.signals.append(sig)

    def poll(self):
        self._polls += 1
        return self._exit_code if self._polls >= self._exits_after else None

    def kill(self):
        self.killed = True


def test_stopping_interrupts_the_interview_rather_than_terminating_it():
    # SESS-01: a keyboard interrupt CLOSES the session -- the ordered-shutdown
    # block flushes the store and stamps `closed_at`. `terminate()` on Windows
    # is TerminateProcess: the child dies where it stands, session.json never
    # gets closed_at, and the transcript keeps whatever partial tail was
    # mid-write. That is exactly what a force-kill produced when this was
    # tested by hand.
    events: "queue.Queue[SessionEvent]" = queue.Queue()
    session = InterviewSession("acme", events=events)
    process = _FakeProcess()
    session._process = process

    session.stop()

    expected = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    assert process.signals == [expected]
    assert process.killed is False, "a session that closed cleanly must never also be killed"
    assert events.get_nowait().kind == "exited"


def test_a_child_torn_down_by_the_break_is_never_reported_as_closed():
    # The sneaky failure, and the one that actually happened: `stop()` used to
    # treat ANY exit as a clean one. A CTRL_BREAK kill exits FAST, so the
    # overlay reported "closed" for a session that never ran `store.close()` --
    # the guard returning SUCCESS on the exact failure it exists to catch.
    # 0xC000013A is what Windows gives a process torn down by a console event.
    events: "queue.Queue[SessionEvent]" = queue.Queue()
    session = InterviewSession("acme", events=events)
    process = _FakeProcess(exit_code=3221225786)
    session._process = process

    session.stop()

    event = events.get_nowait()
    assert event.kind == "exited"
    assert "closed" != (event.message or "")
    assert "tail" in (event.message or ""), "the operator must be told this transcript is suspect"


def test_an_ordinary_nonzero_exit_is_also_not_reported_as_closed():
    # Not only the console-event code: a child that crashed on its way down
    # also failed to flush. Pinning ONLY 0xC000013A would let every other
    # unclean exit keep reporting success.
    events: "queue.Queue[SessionEvent]" = queue.Queue()
    session = InterviewSession("acme", events=events)
    process = _FakeProcess(exit_code=1)
    session._process = process

    session.stop()

    assert "closed" != (events.get_nowait().message or "")


def test_a_child_that_ignores_the_interrupt_is_eventually_killed_and_says_so(monkeypatch):
    # The operator needs to know THIS recording may be missing its closed_at
    # and may have a truncated tail. A silent kill would leave them trusting a
    # transcript that deserves a second look.
    monkeypatch.setattr("app.overlay.session.GRACEFUL_STOP_TIMEOUT_S", 0.4)
    events: "queue.Queue[SessionEvent]" = queue.Queue()
    session = InterviewSession("acme", events=events)
    process = _FakeProcess(exits_after=10_000)
    session._process = process

    session.stop()

    assert process.killed is True
    event = events.get_nowait()
    assert event.kind == "exited"
    assert "force-closed" in (event.message or "")


def test_stopping_before_anything_started_is_harmless():
    session = InterviewSession("acme", events=queue.Queue())
    session.stop()  # must not raise


def test_the_child_runs_in_its_own_process_group_on_windows():
    # A console control event goes to every process in the group. Without the
    # new group, stopping the interview would stop the OVERLAY too.
    if os.name != "nt":
        pytest.skip("process groups only matter for CTRL_BREAK_EVENT on Windows")
    import subprocess

    seen: "list[int]" = []

    class _Spy(InterviewSession):
        def _spawn_child(self, argv):
            seen.append(subprocess.CREATE_NEW_PROCESS_GROUP)
            return _FakeProcess()

    _Spy("acme", events=queue.Queue())._spawn_child([])
    assert seen == [subprocess.CREATE_NEW_PROCESS_GROUP]


# --- the controller's state machine ------------------------------------------------------


class _FakeWindow:
    def __init__(self):
        self.rendered: "list[str]" = []
        self.projects: "tuple[str, ...]" = ()

    def set_projects(self, projects):
        self.projects = tuple(projects)

    def render(self, state):
        self.rendered.append(state.phase)


class _FakeSession:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _controller(projects=("acme",), start=None, clock=None):
    window = _FakeWindow()
    started: "list[str]" = []

    # `**_` swallows keywords this double does not care about (e.g.
    # `loopback_only`). The controller's own contract with the real
    # factory is pinned in test_overlay_capture_mode.py; a double that
    # had to be edited for every new keyword would just get **_ added
    # by whoever hit it first, so it may as well say why.
    def _start(project, events, *, coached=True, **_):
        started.append(project)
        return (start or _FakeSession)()

    ticks = iter(clock or [])
    controller = OverlayController(
        window=window,
        list_projects=lambda: projects,
        start_session=_start,
        clock=(lambda: next(ticks)) if clock else None or (lambda: next(ticks, 0.0)),
    )
    return controller, window, started


def test_pressing_record_shows_the_projects_that_exist_right_now():
    # Read at press time, not at construction: the overlay is meant to sit on
    # the desktop all day, and a project authored meanwhile must appear
    # without a restart.
    controller, window, _ = _controller(projects=("acme", "globex"))
    controller.on_record_pressed()
    assert window.projects == ("acme", "globex")
    assert controller.state.phase == STATE_PICKING


def test_an_unreadable_project_tree_does_not_kill_the_overlay():
    window = _FakeWindow()

    def _boom():
        raise OSError("the disk is gone")

    controller = OverlayController(
        window=window, list_projects=_boom, start_session=lambda p, e, **_kw: _FakeSession()
    )
    controller.on_record_pressed()
    assert controller.state.phase == STATE_IDLE
    assert "project list" in (controller.state.error or "")


def test_cancelling_returns_to_idle_and_starts_nothing():
    controller, _window, started = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("")
    assert controller.state.phase == STATE_IDLE
    assert started == []


def test_choosing_a_project_starts_that_interview():
    controller, _window, started = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    assert started == ["acme"]
    assert controller.state.phase == STATE_RECORDING and controller.state.project == "acme"


def test_a_failure_to_start_returns_to_idle_with_a_reason():
    class _Boom:
        def __init__(self):
            raise RuntimeError("no such interpreter")

    controller, _window, _ = _controller(start=_Boom)
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    assert controller.state.phase == STATE_IDLE
    assert "could not start" in (controller.state.error or "")


def test_stop_renders_stopping_before_it_blocks():
    # `stop()` blocks for as long as the clean shutdown takes -- up to twenty
    # seconds. Without this the button would simply stop responding and the
    # operator would press it again.
    controller, window, _ = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    controller.on_stop_pressed()
    assert window.rendered[-1] == STATE_STOPPING


def test_stop_is_ignored_when_nothing_is_recording():
    controller, window, _ = _controller()
    before = len(window.rendered)
    controller.on_stop_pressed()
    assert len(window.rendered) == before


def test_a_wire_message_reaches_the_state():
    controller, _window, _ = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    controller._events.put(SessionEvent(kind="wire", payload={
        "type": "survey", "next_question_text": "What broke?", "asked_count": 1, "total_count": 4,
    }))
    controller.poll()
    assert controller.state.question_line == "What broke?"


def test_losing_the_display_does_not_stop_the_recording():
    # The interview is still running and still writing the transcript. Losing
    # the overlay's view of it is bad; dropping out of the recording state
    # over it would tell the operator the interview had ended when it had not.
    controller, _window, _ = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    controller._events.put(SessionEvent(kind="failed", message="lost the connection"))
    controller.poll()
    assert controller.state.phase == STATE_RECORDING


def test_the_interview_exiting_returns_the_overlay_to_idle_with_the_reason():
    controller, _window, _ = _controller()
    controller.on_record_pressed()
    controller.on_project_chosen("acme")
    controller._events.put(SessionEvent(kind="exited", message="closed"))
    controller.poll()
    assert controller.state.phase == STATE_IDLE and controller.state.error == "closed"


def test_the_clock_advances_only_while_recording():
    controller, window, _ = _controller(clock=[0.0, 5.0, 9.0])
    controller.on_record_pressed()
    controller.on_project_chosen("acme")   # consumes 0.0
    controller.poll()                      # 5.0
    assert controller.state.elapsed_s == 5
    controller.poll()                      # 9.0
    assert controller.state.elapsed_s == 9


def test_a_poll_with_nothing_to_report_does_not_redraw():
    # The window rebuilds every child widget on render. Redrawing four times a
    # second with nothing changed is visible flicker for no information.
    controller, window, _ = _controller()
    controller.on_record_pressed()
    before = len(window.rendered)
    controller.poll()
    assert len(window.rendered) == before
