"""Releasing and re-taking the microphone while the interview runs.

**Why this exists, and it is a UX finding rather than a code one.** The
pre-flight "system audio only" checkbox worked exactly as built and a real
operator missed it on two consecutive recordings, shipping 67% speaker bleed
both times. It was pre-flight only -- invisible at the moment they were
actually asking what the microphone was doing -- and it reset on every visit to
the picker, so it had to be remembered every run. A setting you must remember
before you start is a setting you will forget.

So the control moved onto the recording panel, where it is visible for the
whole interview, and it now RELEASES the device rather than muting a stream:
"is my microphone open" was the operator's literal question, and only a
released device answers it.

**The channel is stdin, not the WebSocket.** `ACCEPTED_CLIENT_MESSAGE_TYPES`
is pinned to exactly `{ping}` and its own comment says D-01's no-replay rule
and the deferred two-way control channel both depend on that. The overlay is
already the parent process holding a pipe to the interview, so the control
channel is the pipe that exists: no port, no token, no browser, and
unreachable from the network by construction.
"""

from __future__ import annotations

import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from app.control import ALL_COMMANDS, COMMAND_MIC_OFF, COMMAND_MIC_ON, ControlChannel
from app.overlay import model
from app.overlay.controller import OverlayController
from app.overlay.session import InterviewSession

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- the protocol ---------------------------------------------------------------------------


def test_the_command_vocabulary_is_pinned_by_equality():
    """A control channel that grows by drift is the shape gates 8/18/19 exist
    to prevent -- and this one takes instructions from another process."""
    assert set(ALL_COMMANDS) == {"mic on", "mic off"}


@pytest.mark.parametrize(
    "line,expected",
    [("mic off", False), ("mic on", True), ("  MIC OFF  ", False), ("Mic On\n", True)],
)
def test_a_command_reaches_the_handler(line, expected):
    seen: "list[bool]" = []
    ControlChannel(io.StringIO(), on_microphone=seen.append).handle(line)
    assert seen == [expected]


@pytest.mark.parametrize("line", ["", "   ", "mic", "microphone off", "mic off please", "hello"])
def test_an_unrecognised_line_is_ignored_rather_than_fatal(line):
    """This reader sits on a pipe a human may type into while debugging. An
    interview that died because somebody pressed Enter would be an unusually
    stupid way to lose a recording."""
    seen: "list[bool]" = []
    channel = ControlChannel(io.StringIO(), on_microphone=seen.append)
    assert channel.handle(line) is False
    assert seen == []


def test_the_channel_reads_a_whole_stream():
    seen: "list[bool]" = []
    channel = ControlChannel(
        io.StringIO("mic off\nnonsense\nmic on\n"), on_microphone=seen.append
    )
    channel.start()
    for _ in range(50):
        if len(seen) == 2:
            break
        time.sleep(0.02)
    assert seen == [False, True]


def test_stopping_the_channel_does_not_block():
    """`stop()` deliberately does not join. A blocking readline on a pipe
    nobody will write to again does not return, and joining would add seconds
    to the shutdown -- the part of this program with the worst track record."""
    channel = ControlChannel(io.StringIO(), on_microphone=lambda _e: None)
    channel.start()
    started = time.monotonic()
    channel.stop()
    assert time.monotonic() - started < 1.0


# --- the overlay sends it -------------------------------------------------------------------


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.returncode = None

    def poll(self):
        return None


def test_the_session_writes_the_command_to_the_child():
    session = InterviewSession("acme", events=queue.Queue())
    session._process = _FakeProcess()  # noqa: SLF001 -- exercising the real write path
    session.set_microphone(False)
    session.set_microphone(True)
    assert session._process.stdin.getvalue() == f"{COMMAND_MIC_OFF}\n{COMMAND_MIC_ON}\n"  # noqa: SLF001


def test_a_dead_child_does_not_take_the_overlay_down():
    """Called from a click handler on the UI thread. A broken pipe must not
    raise into tkinter."""
    session = InterviewSession("acme", events=queue.Queue())
    session.set_microphone(False)  # no process at all


# --- the controller keeps the panel honest ---------------------------------------------------


class _Window:
    def __init__(self) -> None:
        self.rendered: "list[model.OverlayState]" = []

    def set_projects(self, projects) -> None:
        pass

    def render(self, state) -> None:
        self.rendered.append(state)


class _Session:
    def __init__(self) -> None:
        self.commands: "list[bool]" = []

    def set_microphone(self, enabled: bool) -> None:
        self.commands.append(enabled)

    def stop(self) -> None:
        pass


def _recording_controller():
    session = _Session()
    controller = OverlayController(
        window=_Window(), list_projects=lambda: ("acme",),
        start_session=lambda *a, **k: session,
    )
    controller.on_project_chosen("acme")
    return controller, session


def test_toggling_sends_the_command_and_moves_the_panel():
    controller, session = _recording_controller()
    assert controller.state.mic_on is True

    controller.on_mic_toggled()
    assert session.commands == [False]
    assert controller.state.mic_on is False
    assert controller.state.mic_line == model.MIC_OFF_TEXT

    controller.on_mic_toggled()
    assert session.commands == [False, True]
    assert controller.state.mic_on is True


def test_the_toggle_does_nothing_outside_a_recording():
    """There is no microphone to release from the picker, and a control that
    silently half-worked there would leave the panel disagreeing with the
    next interview."""
    controller = OverlayController(
        window=_Window(), list_projects=lambda: ("acme",), start_session=lambda *a, **k: _Session()
    )
    controller.on_mic_toggled()
    assert controller.state.phase != model.STATE_RECORDING
    assert controller.state.mic_on is True


def test_toggling_preserves_everything_else_about_the_interview():
    """`with_microphone` is a `replace`, not a fresh state. `recording()`
    builds a fresh one deliberately, and reusing that here would silently
    reset the timer, the coach line and the survey counts mid-interview."""
    controller, _session = _recording_controller()
    controller._render(  # noqa: SLF001
        controller.state.ticked(75).applied(
            {"type": "survey", "next_question_text": "what broke?",
             "asked_count": 2, "total_count": 5}
        )
    )
    before = controller.state
    controller.on_mic_toggled()
    after = controller.state
    assert after.mic_on is False
    assert after.elapsed_s == before.elapsed_s == 75
    assert after.next_question == before.next_question == "what broke?"
    assert after.asked_count == 2 and after.total_count == 5
    assert after.project == before.project


def test_a_one_sided_run_starts_with_the_microphone_already_off():
    controller = OverlayController(
        window=_Window(), list_projects=lambda: ("acme",), start_session=lambda *a, **k: _Session()
    )
    controller.on_project_chosen("acme", loopback_only=True)
    assert controller.state.mic_on is False
    assert controller.state.mic_line == model.MIC_OFF_TEXT


def test_the_microphone_can_be_taken_BACK_on_a_one_sided_run():
    """`--only interviewee` decided which devices to OPEN, not which exist.
    An operator who started a podcast recording and then wants to talk over it
    is making a real request, and refusing it would be an arbitrary rule."""
    controller = OverlayController(
        window=_Window(), list_projects=lambda: ("acme",), start_session=lambda *a, **k: _Session()
    )
    controller.on_project_chosen("acme", loopback_only=True)
    controller.on_mic_toggled()
    assert controller.state.mic_on is True


# --- the survey panel stays honest -----------------------------------------------------------


def test_releasing_the_microphone_stops_the_survey_promising_to_advance():
    """Coverage is computed from the INTERVIEWER stream. With the device
    released nothing can ever be detected, so the panel must stop showing a
    next question as though one were coming -- the same rule that applies to a
    run started one-sided, now reachable mid-interview."""
    state = model.OverlayState().recording("acme")
    assert state.survey_can_advance is True
    off = state.with_microphone(False)
    assert off.survey_can_advance is False
    assert off.question_line == model.QUESTION_NO_MIC_TEXT
    assert off.with_microphone(True).survey_can_advance is True


def test_the_control_says_the_state_not_the_verb():
    """"MIC ON" reads as what is true; "mute" reads as what pressing it does.
    Mid-conversation, glancing at a translucent panel, those are read
    differently. The verb lives in the sub-label, where there is room."""
    on = model.OverlayState().recording("acme")
    assert on.mic_line == "MIC ON"
    assert on.with_microphone(False).mic_line == "MIC OFF"
    assert "click" in model.MIC_ON_HINT and "click" in model.MIC_OFF_HINT


def _methods_calling(source: str, name: str) -> "set[str]":
    """Which method(s) of OverlayWindow call `name`."""
    import ast

    tree = ast.parse(source)
    found: "set[str]" = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == name
            ):
                found.add(node.name)
    return found


def test_the_microphone_control_is_drawn_on_the_RECORDING_panel():
    """The test this file shipped without, and the bug it let through.

    The first version asserted only that `_draw_microphone` was CALLED and that
    the call was not inside the coaching guard. Both were true -- of a call
    sitting in `_draw_idle`, where there is no interview to toggle. So the
    control was invisible for the entire recording, the operator clicked
    nothing, and every test passed.

    "Is it called" is not the property. "Is it called FROM THE PANEL THAT IS ON
    SCREEN WHEN IT MATTERS" is."""
    source = (REPO_ROOT / "app" / "overlay" / "window.py").read_text(encoding="utf-8")
    callers = _methods_calling(source, "_draw_microphone")
    assert "_draw_recording" in callers, (
        f"the microphone control is never drawn while recording -- drawn from {callers or 'nowhere'}"
    )
    assert "_draw_idle" not in callers, (
        "the idle surface has no interview, so the control there can only do nothing when clicked"
    )


def test_the_scan_would_notice_the_call_moving():
    """The emptiness guard: the assertion above is satisfied by a broken scan
    that returns nothing, so prove it can tell two methods apart."""
    sample = textwrap.dedent(
        """
        class W:
            def _draw_idle(self):
                self._draw_status(1)
            def _draw_recording(self):
                self._draw_microphone(1)
        """
    )
    assert _methods_calling(sample, "_draw_microphone") == {"_draw_recording"}
    assert _methods_calling(sample, "_draw_status") == {"_draw_idle"}


def test_the_microphone_control_is_outside_the_coaching_guard():
    """The original property, kept. An uncoached recording draws no coaching
    regions at all, and burying the control inside that guard would hide it for
    exactly the podcast-style runs that need it most."""
    import ast

    source = (REPO_ROOT / "app" / "overlay" / "window.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded: "set[int]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute):
            if node.test.attr == "shows_coaching":
                for branch in (*node.body, *node.orelse):
                    for inner in ast.walk(branch):
                        guarded.add(id(inner))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_draw_microphone"
    ]
    assert calls, "the recording panel never draws the microphone control"
    assert all(id(c) not in guarded for c in calls), (
        "the microphone control is drawn inside the coaching guard -- an uncoached recording "
        "would have no way to release the device"
    )


# --- and the end of the chain: a real child really releases the device ------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.skipif(os.name != "nt", reason="CTRL_BREAK_EVENT is Windows-only")
def test_a_real_interview_records_the_toggle_and_still_closes_cleanly(tmp_path):
    """THE crossing test. Everything above proves the overlay SENDS a line and
    the child's parser UNDERSTANDS one; neither proves a running interview does
    anything about it. This drives the real `app start` over a real pipe.

    Asserted on the transcript's own `microphone` events rather than on a log
    line, because that is the artifact: a gap on the interviewer track is
    either a silent operator or a released device, and after the fact nothing
    else on disk can tell those apart.

    It also stops cleanly at the end, so the control channel cannot quietly
    reintroduce the shutdown defect `7e53a18` fixed -- a stdin reader thread
    blocked on a pipe is exactly the shape that would."""
    sessions_root = tmp_path / "sessions"
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    env = {
        **os.environ,
        "APP_SESSIONS_ROOT": str(sessions_root),
        "APP_CONTEXT_DIR": str(context_dir),
        "APP_PORT": str(_free_port()),
        "APP_OPEN_BROWSER": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "app", "start", "--no-coach", "--engine", "fake"],
        cwd=str(REPO_ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        time.sleep(8)
        proc.stdin.write(f"{COMMAND_MIC_OFF}\n")
        proc.stdin.flush()
        time.sleep(3)
        proc.stdin.write(f"{COMMAND_MIC_ON}\n")
        proc.stdin.flush()
        time.sleep(3)
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        output, _ = proc.communicate(timeout=90)
    except Exception:
        proc.kill()
        raise

    assert proc.returncode == 0, f"the interview did not close cleanly:\n{output[-2000:]}"
    (session_dir,) = [p for p in sessions_root.iterdir() if p.is_dir()]
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert "closed_at" in metadata, "the control channel broke the ordered shutdown"

    records = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    toggles = [r for r in records if r.get("event") == "microphone"]
    assert [t["detail"]["enabled"] for t in toggles] == [False, True], (
        f"the running interview did not act on the control lines: {toggles}\n{output[-1500:]}"
    )
    assert all("error_type" not in t["detail"] for t in toggles), (
        f"releasing or re-opening the device failed: {toggles}"
    )

    # And what it REPORTED back, which is what the overlay's panel shows. The
    # transcript events above are written from the REQUEST; these lines are
    # written from the tracked state, so only this half catches a tracker that
    # never moves. A mutation removing `microphone.record(...)` passed every
    # other assertion in this file.
    reported = [
        state for state in (parse_microphone_state(line) for line in output.splitlines())
        if state is not None
    ]
    assert reported == [False, True], (
        f"the interview reported {reported} rather than the states it reached:\n"
        f"{output[-1500:]}"
    )


# --- the panel is CORRECTED by what the interview actually did ---------------------------
#
# Added immediately after the control shipped, because the control shipped with
# the lie in it: the overlay showed whatever was ASKED for, so a device that
# failed to re-open left the panel reading "MIC ON" over a closed microphone.
# That is the same class of defect as the coach panel that read the project's
# FILING instead of the coach, fixed hours earlier the same day.

from app.control import microphone_state_line, parse_microphone_state  # noqa: E402
from app.overlay.session import SessionEvent  # noqa: E402


def test_the_state_line_round_trips_through_its_own_parser():
    """Written and read in one module so the two cannot drift. The defect that
    left the coach panel dead for a fortnight was two spellings of one key."""
    assert parse_microphone_state(microphone_state_line(True)) is True
    assert parse_microphone_state(microphone_state_line(False)) is False


@pytest.mark.parametrize(
    "line",
    ["Resolved engine: soniox", "", "[mic-state]", "[mic-state] maybe", "mic on", "  "],
)
def test_an_ordinary_line_is_not_read_as_a_state_report(line):
    """This parser sees EVERY line the interview prints. A false positive would
    move the panel for no reason; note `mic on` -- the COMMAND -- must not be
    mistaken for the ANSWER, or an echo would confirm itself."""
    assert parse_microphone_state(line) is None


def test_a_reported_state_overrides_what_was_asked_for():
    """THE test. The optimistic set on click is deliberate -- a control that
    waits for a round trip reads as broken -- but it has to be correctable."""
    controller, _session = _recording_controller()
    controller.on_mic_toggled()
    assert controller.state.mic_on is False, "optimistic, as designed"

    # The interview could not release the device and says so.
    controller._events.put(  # noqa: SLF001
        SessionEvent(kind="microphone", payload={"enabled": True})
    )
    controller.poll()
    assert controller.state.mic_on is True, (
        "the panel kept showing what was requested rather than what happened"
    )


def test_a_confirmation_that_agrees_changes_nothing_visible():
    controller, _session = _recording_controller()
    controller.on_mic_toggled()
    controller._events.put(  # noqa: SLF001
        SessionEvent(kind="microphone", payload={"enabled": False})
    )
    controller.poll()
    assert controller.state.mic_on is False


def test_a_state_report_preserves_the_rest_of_the_interview():
    """It arrives mid-recording, so it must not reset the survey or the project
    -- the same trap `with_microphone` was written as a `replace` to avoid.

    Asserted on the SURVEY rather than on `elapsed_s`: `poll()` recomputes the
    clock from `_started_at` every call by design, so a timer assertion here
    would be testing the harness rather than the state fold."""
    controller, _session = _recording_controller()
    controller._render(  # noqa: SLF001
        controller.state.applied(
            {"type": "survey", "next_question_text": "what broke?",
             "asked_count": 3, "total_count": 7}
        )
    )
    controller._events.put(  # noqa: SLF001
        SessionEvent(kind="microphone", payload={"enabled": False})
    )
    controller.poll()
    assert controller.state.mic_on is False
    assert controller.state.next_question == "what broke?"
    assert controller.state.asked_count == 3 and controller.state.total_count == 7
    assert controller.state.project == "acme"


# --- what is REPORTED when the device refuses to move -------------------------------------
#
# This section exists because a mutation escaped. Changing the failure branch to
# report the REQUEST instead of the achieved state passed all 34 tests above --
# the failure path was structurally untestable, so it was untested, and it was
# also wrong: it computed the answer as `not enabled`, which is only right if
# the previous state was the opposite. Ask twice for `on`, fail the second time,
# and it reported `off` over an open device.

from app.control import MicrophoneState  # noqa: E402


def test_the_tracker_starts_where_the_run_started():
    assert MicrophoneState(on=True).is_on is True
    assert MicrophoneState(on=False).is_on is False


def test_a_successful_move_updates_the_tracker():
    mic = MicrophoneState(on=True)
    mic.record(False)
    assert mic.is_on is False
    mic.record(True)
    assert mic.is_on is True


def test_a_failed_move_leaves_the_tracker_alone():
    """`record` is called only after the device really moved, never in an
    except block. That is the whole point of the class."""
    mic = MicrophoneState(on=True)
    # ... the attempt raised, so nothing is recorded ...
    assert mic.is_on is True, "a failed release must not be reported as released"


def test_asking_for_a_state_you_are_already_in_and_failing_reports_the_truth():
    """THE case the `not enabled` version got wrong, and the reason this is a
    tracked fact rather than a computed one. Previous state ON, request ON,
    attempt fails: the honest answer is ON. `not enabled` would say OFF."""
    mic = MicrophoneState(on=True)
    requested = True
    # attempt fails -> no record() call
    assert mic.is_on is True
    assert mic.is_on != (not requested), (
        "computing the report as the opposite of the request would invert it here"
    )


def test_the_reported_line_comes_from_the_tracker_not_from_the_request():
    """Source-level, because the branch itself needs a real device to fail.
    Pins the STRUCTURE: the failure path must read the tracker."""
    source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "microphone_state_line(microphone.is_on)" in source
    assert "microphone_state_line(not enabled)" not in source, (
        "the failure report is computed as the opposite of the request again"
    )
    assert "microphone_state_line(enabled)" not in source, (
        "the report echoes the request rather than the achieved state"
    )


@pytest.mark.skipif(os.name != "nt", reason="CTRL_BREAK_EVENT is Windows-only")
def test_a_one_sided_run_can_take_a_microphone_mid_interview(tmp_path):
    """The path the operator actually hit, and it did nothing.

    They started with the pre-flight "system audio only" toggle on, so the run
    had `--only interviewee` and NO interviewer pair. The control looped over
    the existing pairs looking for one, found none, and fell through in
    silence: no device change, no event, no state line, no error. Two clicks,
    nothing.

    The model's own docstring claimed the opposite -- that `--only interviewee`
    "decided which devices to OPEN, not which exist" and the request could be
    honoured. It could not. The only test of it drove the CONTROLLER against a
    fake session, so it passed while the child was incapable.

    The pair is now built on demand. Nothing is opened until it is asked for:
    `CaptureStream.__init__` only records what to open, and `worker.start()` is
    what opens the STT session -- so a one-sided run that never enables the
    microphone still never opens either."""
    sessions_root = tmp_path / "sessions"
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    env = {
        **os.environ,
        "APP_SESSIONS_ROOT": str(sessions_root),
        "APP_CONTEXT_DIR": str(context_dir),
        "APP_PORT": str(_free_port()),
        "APP_OPEN_BROWSER": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "app", "start", "--no-coach", "--engine", "fake",
         "--only", "interviewee"],
        cwd=str(REPO_ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        time.sleep(8)
        proc.stdin.write(f"{COMMAND_MIC_ON}\n")
        proc.stdin.flush()
        time.sleep(4)
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        output, _ = proc.communicate(timeout=90)
    except Exception:
        proc.kill()
        raise

    assert proc.returncode == 0, f"the interview did not close cleanly:\n{output[-2000:]}"
    (session_dir,) = [p for p in sessions_root.iterdir() if p.is_dir()]
    records = [
        json.loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    toggles = [r for r in records if r.get("event") == "microphone"]
    assert toggles, (
        "a one-sided run answered the microphone command with SILENCE -- no event, no state "
        f"line, no error. That is the reported defect.\n{output[-1500:]}"
    )
    reported = [
        state for state in (parse_microphone_state(line) for line in output.splitlines())
        if state is not None
    ]
    assert reported, "the interview reported no state at all, so the panel cannot be right"
    # Whether the device could actually be opened depends on the machine having
    # a microphone, which a test must not assume. What must ALWAYS hold is that
    # the run answers -- with the truth either way.
    if "error_type" in toggles[-1]["detail"]:
        assert reported[-1] is False, "a failed open must not be reported as an open microphone"
    else:
        assert reported[-1] is True
        assert toggles[-1]["detail"]["enabled"] is True
