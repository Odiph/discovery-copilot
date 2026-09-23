"""Making a console CTRL_BREAK arrive as a `KeyboardInterrupt`.

Its own module, and deliberately tiny, so the child processes that need this
can import it without dragging in `app/main.py`'s import list (pyaudio, the
capture stack, uvicorn). `app/tests/test_overlay_stop_signal.py` spawns real
children against it; a heavyweight import would make that test slow and would
couple a signal question to an audio library.

**Why this exists at all.** The overlay stops an interview by sending
`CTRL_BREAK_EVENT` to the child's process group -- never `terminate()`, which
on Windows is `TerminateProcess` and would skip the ordered shutdown that
flushes the store. But Python installs a console control handler that converts
only `CTRL_C_EVENT` into `KeyboardInterrupt`. `SIGBREAK` is left at `SIG_DFL`,
and the default action for a break event is to terminate the process. So the
signal chosen precisely BECAUSE it was not a kill was behaving as one: measured
2026-08-12, an unpatched child dies with exit code 3221225786 (`0xC000013A`)
without ever entering its `except KeyboardInterrupt` block.

The consequence was silent and total: no `closed_at`, no `latency_summary`, and
a transcript with whatever partial tail was mid-write -- on every overlay
recording ever made, while the overlay reported "closed".
"""

from __future__ import annotations

import signal


def install_stop_handler() -> bool:
    """Routes `SIGBREAK` into the same handler `SIGINT` already uses, so a
    console break raises `KeyboardInterrupt` instead of killing the process.

    Returns whether a handler was installed -- `False` off Windows, where
    `SIGBREAK` does not exist and `SIGINT` already does the right thing, and
    `False` when called off the main thread, where `signal.signal` raises
    `ValueError` by design. The return value is reported rather than logged so
    a caller can decide; this module deliberately holds no logger and no policy.

    **The off-thread case is returned, never swallowed in silence.** A caller
    that ignores it gets exactly the defect this module exists to fix -- a stop
    that looks like an interrupt and behaves like a kill -- so `run_start` says
    so out loud. Two tests (`test_precheck.py`, `test_tracer_capture_to_disk.py`)
    legitimately drive `run_start` from a worker thread, which is how this path
    was found.

    Safe to call more than once: installing the same handler twice is a no-op.
    """
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is None:
        return False
    try:
        signal.signal(sigbreak, signal.default_int_handler)
    except ValueError:
        # Not the main thread. Nothing can be installed from here.
        return False
    return True


def ignore_stop_signals() -> bool:
    """Makes further stop signals do nothing, for the duration of a shutdown
    that is already underway.

    **Why this is needed on top of `install_stop_handler`, measured 2026-08-12
    by reproducing a real stop with a real child.** `uvicorn.Server.serve()`
    runs inside `capture_signals()`, which installs uvicorn's OWN
    SIGINT/SIGBREAK handler for as long as the server lives -- so the handler
    installed above is not the one in force during an interview. Uvicorn
    records the signal, shuts down gracefully, and then, on leaving that
    context manager, calls `signal.raise_signal(captured_signal)`
    (`uvicorn/server.py:340`) to "trigger the expected behaviour now".

    That re-raise is delivered to the main thread at the next bytecode
    boundary, which is somewhere inside the ordered shutdown -- NOT reliably at
    the `await` that surfaced it, which is why catching it at one call site did
    not work. It killed the process before `store.close()`, so every overlay
    recording ever made lacked `closed_at` and `latency_summary`.

    **The rule this encodes: once the flush has begun, a stop signal can no
    longer do anything useful.** The interview is ending either way. The only
    question left is whether the transcript reaches disk, and an interrupt can
    only answer "no". So they are ignored from the top of the shutdown onward
    -- narrowly, deliberately, and never during the interview itself, where an
    interrupt is exactly how an operator ends a recording.

    Returns whether anything was installed: `False` off the main thread, where
    `signal.signal` raises by design.
    """
    signals = [signal.SIGINT, getattr(signal, "SIGBREAK", None)]
    installed = False
    for sig in signals:
        if sig is None:
            continue
        try:
            signal.signal(sig, signal.SIG_IGN)
        except ValueError:
            # Not the main thread. Nothing can be installed from here.
            return False
        installed = True
    return installed
