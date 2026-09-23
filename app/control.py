"""A one-line control channel on the interview's own stdin.

**Why stdin and not the WebSocket.** `app/web/server.py` pins
`ACCEPTED_CLIENT_MESSAGE_TYPES` to exactly `{ping}`, and says so in its own
words: D-01's no-replay rule and the deferred two-way control channel both
depend on that set staying exactly the heartbeat. Adding a control message
there would be an architectural decision about the PHONE protocol, taken in
order to solve a problem the phone does not have -- the operator with the
microphone is at the laptop, and the overlay that needs to command it is
already the parent process holding a pipe to this one.

So the control channel is the pipe that already exists. It needs no port, no
token, no browser, and it is unreachable from the network by construction:
only the process that spawned this one can write to its stdin.

**The protocol is one line per command, and deliberately tiny.** Two verbs,
no arguments, no JSON. A control channel that can express more than the
overlay can ask for is a surface to get wrong; when a third command is needed,
adding it here is three lines and a decision.

**Unknown lines are IGNORED, never fatal.** This reader sits on the same pipe
a human may type into while debugging, and an interview that died because
somebody pressed Enter would be an unusually stupid way to lose a recording.
Every line is logged at debug either way, so a typo is findable.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, TextIO

logger = logging.getLogger(__name__)

# The whole vocabulary. Pinned as a set so a test can assert it by equality --
# a control channel that grows by drift is the shape gates 8/18/19 exist to
# prevent, and this one takes instructions from another process.
COMMAND_MIC_ON = "mic on"
COMMAND_MIC_OFF = "mic off"
ALL_COMMANDS = (COMMAND_MIC_ON, COMMAND_MIC_OFF)

# What the interview prints on stdout to report the state it ACTUALLY reached.
#
# The commands above are a request; this is the answer, and the two are not the
# same thing. Without it the overlay showed whatever the operator ASKED for --
# so a device that failed to re-open left the panel reading "MIC ON" over a
# closed microphone, which is precisely the class of lie `UNCOACHED_TEXT` and
# `coach_confirmed` already exist to prevent, in a control shipped hours after
# both.
#
# stdout, because the overlay already drains it and must keep doing so (a full
# pipe blocks the interview). A prefix rather than bare text so a line of
# ordinary output can never be mistaken for a state report.
MIC_STATE_PREFIX = "[mic-state]"


def microphone_state_line(enabled: bool) -> str:
    """The exact line the interview prints, built in ONE place so the writer
    and the parser cannot drift -- the defect that left the coach panel dead
    for a fortnight was two spellings of one wire key."""
    return f"{MIC_STATE_PREFIX} {'on' if enabled else 'off'}"


def parse_microphone_state(line: str) -> "Optional[bool]":
    """The reported state, or `None` for any line that is not one.

    Returns `None` rather than raising: this reads EVERY line the interview
    prints, most of which are ordinary output."""
    text = line.strip()
    if not text.startswith(MIC_STATE_PREFIX):
        return None
    value = text[len(MIC_STATE_PREFIX):].strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    return None


class MicrophoneState:
    """What the microphone is ACTUALLY doing, as distinct from what was asked.

    Exists because the first cut computed the failure report as `not enabled`
    -- the opposite of the request -- which is only right if the previous state
    was the opposite, and that is a guess. Ask twice for `on`, fail the second
    time, and it would report `off` over an open device.

    So the state is tracked and only moves when an attempt SUCCEEDED. The
    reported value is then a fact rather than an inference, and the failure
    path stops depending on the caller having been in the state it assumed."""

    def __init__(self, *, on: bool) -> None:
        self._on = on

    @property
    def is_on(self) -> bool:
        return self._on

    def record(self, enabled: bool) -> None:
        """Call ONLY after the device really moved. Never in an except block:
        that is the whole point of the class."""
        self._on = enabled


class ControlChannel:
    """Reads commands off a text stream on a daemon thread.

    Takes the stream and the handler as parameters rather than reaching for
    `sys.stdin` itself, for the same reason every loader in this project takes
    its directory: it makes the whole thing testable with a `StringIO` and no
    subprocess."""

    def __init__(
        self,
        stream: "TextIO",
        *,
        on_microphone: "Callable[[bool], None]",
    ) -> None:
        self._stream = stream
        self._on_microphone = on_microphone
        self._thread: "Optional[threading.Thread]" = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="control-channel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Sets the flag and returns. It does NOT join the thread.

        A blocking `readline()` on a pipe nobody will write to again does not
        return, so joining here would add up to five seconds to every
        shutdown -- and the shutdown is the part of this program with the worst
        track record. The thread is a daemon reading a pipe that is about to
        close; leaving it is correct."""
        self._stop.set()

    def _run(self) -> None:
        try:
            for raw_line in self._stream:
                if self._stop.is_set():
                    return
                self.handle(raw_line)
        except Exception as exc:  # noqa: BLE001 -- a broken pipe must never end the interview
            logger.debug("control channel closed error_type=%s", type(exc).__name__)

    def handle(self, raw_line: str) -> bool:
        """Applies one line. Returns whether it was a command we know.

        Public and synchronous so the whole protocol can be tested without a
        thread: the threading above is delivery, and this is the behaviour."""
        line = raw_line.strip().lower()
        if not line:
            return False
        if line == COMMAND_MIC_ON:
            logger.info("control: microphone on")
            self._on_microphone(True)
            return True
        if line == COMMAND_MIC_OFF:
            logger.info("control: microphone off")
            self._on_microphone(False)
            return True
        logger.debug("control: ignoring unrecognised line %r", line[:40])
        return False
