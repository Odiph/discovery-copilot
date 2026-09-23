"""Supervising one interview from the overlay: spawn it, listen to it, stop it
cleanly.

Three threads and no asyncio, deliberately. The overlay's main thread belongs
to tkinter, so everything here hands work back through a `queue.Queue` that
the window drains on a timer -- the same thread->UI bridge
`app/main.py::_suggestion_broadcast_callback` uses in the other direction.

**Stopping must be a Ctrl+C, not a kill.** SESS-01 is that a keyboard
interrupt CLOSES the session rather than abandoning it: the ordered-shutdown
block in `_async_main`'s `finally` stops the workers, terminates PyAudio and
flushes the store. `Popen.terminate()` on Windows is `TerminateProcess` --
the child dies where it stands, `session.json` never gets its `closed_at`,
and the transcript keeps whatever partial tail was mid-write. That is exactly
what a force-kill produced when this was tested by hand.

So the child is created in its own process group and stopped with
`CTRL_BREAK_EVENT`. The new group is what keeps that signal off the OVERLAY's
own process -- a console control event goes to every process in the group, and
without the new group the overlay would kill itself along with the interview.

**`CTRL_BREAK_EVENT` is NOT turned into a `KeyboardInterrupt` on its own, and
this docstring used to claim it was.** Python's console handler converts only
`CTRL_C_EVENT`; `SIGBREAK` sits at `SIG_DFL`, whose default action terminates.
So the signal chosen precisely because it was not a kill behaved as one, and
every overlay recording made before 2026-08-12 lost `store.close()`. The child
now installs the handler itself -- see `app/interrupt.py` -- and this module
checks the exit CODE rather than inferring a clean close from the process
merely being gone.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.control import COMMAND_MIC_OFF, COMMAND_MIC_ON, parse_microphone_state

logger = logging.getLogger(__name__)

# The child prints `Join URL: http://<lan-ip>:<port>/s/<token>`. Only the
# token is taken from it: the overlay connects over LOOPBACK regardless of
# what the LAN IP resolved to, because it is on the same machine and has no
# reason to leave it.
_JOIN_URL_PATTERN = re.compile(r"/s/([A-Za-z0-9_\-]+)")

# How long a Ctrl+C is given to close the session before the process is killed
# outright. Generous on purpose -- the shutdown path stops two capture
# threads, two STT workers, the coach's threads and a uvicorn server, and a
# transcript flushed late is better than one truncated early.
GRACEFUL_STOP_TIMEOUT_S = 20.0

# How long to wait for the child to print its join line before giving up.
# A start that has not reached the join URL by now has failed at devices or
# at the brief, and both print their own error first.
TOKEN_WAIT_TIMEOUT_S = 45.0


@dataclass(frozen=True)
class SessionEvent:
    """One thing the window needs to know. `kind` is `wire` (a broadcast
    message from the interview), `microphone` (the device state the interview
    ACTUALLY reached, read off its stdout), `started`, `failed`, or `exited`.

    `microphone` is deliberately its own kind rather than a `wire` payload: it
    did not come off the WebSocket, and the phone never sees it. Folding it in
    would make the one authoritative signal about the microphone
    indistinguishable from a broadcast every client receives."""

    kind: str
    payload: "Optional[dict[str, Any]]" = None
    message: "Optional[str]" = None


# What Windows reports for a process torn down by a console control event.
# Named so a reader can tell it apart from an ordinary failure -- and because
# it is the code every overlay recording produced before `app/interrupt.py`
# taught the child to catch the break instead of dying on it.
STATUS_CONTROL_C_EXIT = 3221225786


def _exit_message(code: int) -> str:
    """What the operator is told about a stopped interview.

    `"closed"` is reserved for a genuinely clean shutdown. Every other exit
    means `store.close()` did not finish, so `closed_at` is missing and the
    transcript's tail is whatever was mid-write -- the operator has to know
    that about THIS recording, because nothing downstream will tell them."""
    if code == 0:
        return "closed"
    if code == STATUS_CONTROL_C_EXIT:
        return "the interview was cut short before it flushed -- check the transcript's tail"
    return f"the interview exited with code {code} before it flushed -- check the transcript's tail"


def parse_join_token(line: str) -> "Optional[str]":
    """The token out of the child's join line, or `None`.

    Its own function so the parsing can be tested without spawning anything
    -- and because this is the one brittle coupling in the whole overlay: it
    reads a line the interview prints for a HUMAN. If that line ever changes,
    this is what breaks, and `test_overlay_session.py` pins the exact format
    against the real string `app/web/join.py` emits."""
    match = _JOIN_URL_PATTERN.search(line)
    return match.group(1) if match else None


class InterviewSession:
    """One `python -m app start` child, its stdout reader, and its WebSocket
    client.

    Every dependency is injected -- the interpreter, the repo root, the port,
    and the two functions that actually spawn and connect. That is what lets
    `test_overlay_session.py` drive the whole lifecycle against fakes, with no
    audio device, no port bound and no subprocess."""

    def __init__(
        self,
        project: str,
        *,
        events: "queue.Queue[SessionEvent]",
        read_from: "tuple[str, ...]" = (),
        coached: bool = True,
        loopback_only: bool = False,
        python_executable: str = sys.executable,
        repo_root: "Optional[Path]" = None,
        port: int = 8765,
        spawn: "Optional[Any]" = None,
        connect: "Optional[Any]" = None,
    ) -> None:
        self.project = project
        self.read_from = tuple(read_from)
        self.coached = coached
        self.loopback_only = loopback_only
        self._events = events
        self._python = python_executable
        self._repo_root = repo_root or Path(__file__).resolve().parents[2]
        self._port = port
        self._spawn = spawn or self._spawn_child
        self._connect = connect
        self._process: "Optional[Any]" = None
        self._token: "Optional[str]" = None
        self._stopping = threading.Event()
        self._token_found = threading.Event()
        self._threads: "list[threading.Thread]" = []

    # --- command line -----------------------------------------------------

    def command(self) -> "list[str]":
        """The exact command a hand-typed run would use.

        `-u` because the join token is read off stdout and a block-buffered
        pipe would hold it until the buffer filled -- which, for a process
        that prints a handful of lines and then goes quiet for an hour, is
        forever.

        `--project` is ALWAYS passed, `--no-coach` or not. An unfiled run
        omitting it would write `"project": null` into `session.json`, which
        is indistinguishable from a legacy pre-project recording -- and filing
        it later would stop being a rename.

        `--only interviewee` is passed for a LOOPBACK-ONLY recording, and the
        reason it exists on this surface at all was measured on 2026-08-12:
        recording a podcast played through SPEAKERS put the same speech on
        both tracks -- 15 of 21 microphone segments were near-duplicates of a
        loopback segment about 0.35s earlier, so 71% of the "interviewer"
        track was the podcast. For a one-sided recording the microphone has
        nothing to contribute and everything to pollute, and not opening it is
        the only defence that cannot be got wrong."""
        argv = [self._python, "-u", "-m", "app", "start", "--project", self.project]
        if self.loopback_only:
            argv += ["--only", "interviewee"]
        for source in self.read_from:
            argv += ["--read-from", source]
        if not self.coached:
            # An EXPLICIT flag, never "let the missing context/ do it". `start`
            # is required to fail loudly and create no session directory when a
            # brief or survey cannot load, and that rule is worth more than the
            # convenience of reaching this state by accident.
            argv.append("--no-coach")
        return argv

    def _spawn_child(self, argv: "list[str]") -> Any:
        creation_flags = 0
        if os.name == "nt":
            # See the module docstring: the new group is what keeps
            # CTRL_BREAK_EVENT off this process.
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen(
            argv,
            cwd=str(self._repo_root),
            # The control channel. The child reads one line per command off
            # this pipe (`app/control.py`); it is how the microphone is
            # released mid-interview without a port, a token or a browser, and
            # it is unreachable from the network because only this process can
            # write to it.
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creation_flags,
        )

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._process = self._spawn(self.command())
        self._start_thread(self._pump_stdout, "overlay-stdout")
        self._start_thread(self._await_token_then_connect, "overlay-ws")

    def _start_thread(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _pump_stdout(self) -> None:
        """Drains the child's stdout forever, and picks the token out on the
        way past.

        Draining is not optional and not merely tidy: a child whose stdout
        pipe fills blocks on its next `print`, and the interview prints
        warnings throughout. Stopping reading after the token was found would
        deadlock the interview partway through, which would look exactly like
        the capture hanging."""
        assert self._process is not None
        stream = getattr(self._process, "stdout", None)
        if stream is None:
            return
        for raw_line in stream:
            line = raw_line.rstrip()
            reported = parse_microphone_state(line)
            if reported is not None:
                # The state the interview ACTUALLY reached, which is not
                # necessarily the one that was asked for. Emitted as its own
                # event kind rather than folded into `wire`: it did not come
                # off the WebSocket, and pretending it did would make the one
                # honest signal about the microphone indistinguishable from a
                # broadcast the phone also receives.
                self._events.put(
                    SessionEvent(kind="microphone", payload={"enabled": reported})
                )
                continue
            logger.info("interview: %s", line)
            if self._token is None:
                token = parse_join_token(line)
                if token is not None:
                    self._token = token
                    self._token_found.set()
        # The stream closed, so the child is finishing. `exited` is emitted
        # here rather than from a waiter thread because this is the moment the
        # overlay can be sure there is no more output to show.
        if not self._stopping.is_set():
            self._events.put(SessionEvent(kind="exited", message="the interview ended on its own"))

    def _await_token_then_connect(self) -> None:
        if not self._token_found.wait(timeout=TOKEN_WAIT_TIMEOUT_S):
            self._events.put(
                SessionEvent(
                    kind="failed",
                    message="the interview never reported a join URL -- check the terminal",
                )
            )
            return
        self._events.put(SessionEvent(kind="started"))
        try:
            self._consume_wire()
        except Exception as exc:  # noqa: BLE001 -- a dead socket must not kill the recording
            # The interview is still running and still writing the transcript.
            # Losing the display is bad; stopping the capture over it is
            # worse, so this degrades exactly the way the phone surface does.
            logger.warning("overlay wire connection failed error_type=%s", type(exc).__name__)
            self._events.put(
                SessionEvent(kind="failed", message="lost the connection to the interview")
            )

    def _consume_wire(self) -> None:
        import json

        connect = self._connect
        if connect is None:
            from websockets.sync.client import connect as connect  # noqa: PLC0414

        url = f"ws://127.0.0.1:{self._port}/s/{self._token}/ws"
        with connect(url, open_timeout=10) as socket:
            while not self._stopping.is_set():
                try:
                    raw = socket.recv(timeout=1.0)
                except TimeoutError:
                    continue
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict):
                    self._events.put(SessionEvent(kind="wire", payload=payload))

    def set_microphone(self, enabled: bool) -> None:
        """Asks the running interview to release or re-open the microphone.

        Failures are logged and swallowed, never raised: this is called from a
        click handler on the overlay's UI thread, and a broken pipe -- a child
        that has already exited, say -- must not take the window down with it.
        The overlay's own state is set by the CALLER regardless, so the panel
        keeps reflecting what the operator asked for even if the message did
        not land; the alternative is a control that appears to do nothing."""
        process = self._process
        if process is None or process.stdin is None:
            logger.warning("no interview to send the microphone command to")
            return
        command = COMMAND_MIC_ON if enabled else COMMAND_MIC_OFF
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sending the microphone command failed error_type=%s", type(exc).__name__
            )

    def stop(self) -> None:
        """Ctrl+C, then wait, then kill. See the module docstring for why the
        first step is not `terminate()`."""
        self._stopping.set()
        process = self._process
        if process is None:
            return

        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except Exception as exc:  # noqa: BLE001 -- fall through to the kill below
            logger.warning("interrupting the interview failed error_type=%s", type(exc).__name__)

        deadline = time.monotonic() + GRACEFUL_STOP_TIMEOUT_S
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                # The exit CODE, not merely the fact of exiting. A child torn
                # down by the console event exits FAST, so "it is gone" arrives
                # sooner on the failure path than on the success one -- which
                # is how this reported "closed" for every unflushed session
                # before `app/interrupt.py` existed. Anything non-zero means
                # the ordered shutdown did not finish.
                self._events.put(SessionEvent(kind="exited", message=_exit_message(code)))
                return
            time.sleep(0.2)

        # Only now, and say so -- a session closed this way may be missing its
        # `closed_at` and may have a truncated transcript tail, and the
        # operator should know that about this specific recording.
        logger.warning("the interview did not close within %.0fs -- killing it", GRACEFUL_STOP_TIMEOUT_S)
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass
        self._events.put(
            SessionEvent(
                kind="exited",
                message="force-closed after it did not shut down -- check the transcript's tail",
            )
        )
