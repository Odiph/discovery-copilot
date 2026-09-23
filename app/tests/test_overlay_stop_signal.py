"""The STOP path, tested across the seam a double cannot reach.

`test_overlay_session.py` already asserts that `InterviewSession.stop()` calls
`send_signal(CTRL_BREAK_EVENT)` rather than `terminate()`. That test uses a
FAKE process, so it proves the overlay's INTENT and nothing about whether a
real child survives the signal long enough to close its session.

It does not. Measured 2026-08-12, twice: a Python child spawned with
`CREATE_NEW_PROCESS_GROUP` and sent `CTRL_BREAK_EVENT` with NO `SIGBREAK`
handler installed dies with exit code 3221225786 (`0xC000013A`,
`STATUS_CONTROL_C_EXIT`) and its `except KeyboardInterrupt` block never runs.
Python's console handler converts only `CTRL_C_EVENT` into `KeyboardInterrupt`;
`SIGBREAK` is left at `SIG_DFL`, whose default action is to terminate. Every
overlay recording ever made therefore lost `store.close()` -- no `closed_at`,
no `latency_summary`, and an unflushed transcript tail.

The second test below is the mutation, kept permanently as a test: it asserts
the UNPATCHED child really is killed. Without it, the first test could pass for
a reason that has nothing to do with the handler (a child that exits on its own,
a signal that never arrives) and nobody would know.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="CTRL_BREAK_EVENT and SIGBREAK are Windows console concepts"
)

# 0xC000013A -- the exit code Windows gives a process torn down by a console
# control event. Named rather than inlined so a future reader can tell this
# apart from an ordinary non-zero exit.
STATUS_CONTROL_C_EXIT = 3221225786

_CHILD_WITH_HANDLER = """
import sys, time
from app.interrupt import install_stop_handler
install_stop_handler()
print("ready", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    open(sys.argv[1], "w", encoding="utf-8").write("closed")
    sys.exit(0)
"""

_CHILD_WITHOUT_HANDLER = """
import sys, time
print("ready", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    open(sys.argv[1], "w", encoding="utf-8").write("closed")
    sys.exit(0)
"""


def _run_child_and_break(source: str, marker: Path) -> "tuple[int, bool]":
    """Spawns a real child the way `InterviewSession` does, waits for it to be
    genuinely inside its loop, sends the real signal, and reports what
    happened. Returns `(returncode, marker_written)`."""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", source, str(marker)],
        cwd=str(REPO_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for "ready" rather than sleeping: a signal delivered before the
        # child reaches its try block would prove nothing either way.
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            pytest.fail("the child neither closed nor died within 15s")
    finally:
        if process.poll() is None:
            process.kill()
    return process.returncode, marker.is_file()


def test_a_child_that_installs_the_handler_closes_cleanly_on_ctrl_break(tmp_path):
    # The property the whole STOP path depends on: the interrupt must arrive as
    # a KeyboardInterrupt so `_async_main`'s `finally` runs and the store is
    # flushed. Exit code 0 alone is not enough -- a process can exit 0 without
    # ever entering its handler -- so the marker is what is actually asserted.
    marker = tmp_path / "closed.txt"
    returncode, wrote_marker = _run_child_and_break(_CHILD_WITH_HANDLER, marker)
    assert wrote_marker, "the child's KeyboardInterrupt handler never ran"
    assert returncode == 0


def test_a_child_without_the_handler_is_killed_outright(tmp_path):
    # The mutation, kept as a test. If this ever starts passing with a clean
    # exit, then CTRL_BREAK_EVENT has begun behaving like CTRL_C_EVENT on this
    # platform and the test above no longer proves the handler is what saved
    # it -- at which point the handler could be deleted silently.
    marker = tmp_path / "closed.txt"
    returncode, wrote_marker = _run_child_and_break(_CHILD_WITHOUT_HANDLER, marker)
    assert not wrote_marker, "an unpatched child should never reach its handler"
    assert returncode == STATUS_CONTROL_C_EXIT


def test_run_start_installs_the_handler_before_the_interview_begins():
    # Behavioural, not textual: the installer must actually be CALLED on the
    # `start` path. A textual scan would pass on an import that is never used,
    # which is precisely how a forcing function rots.
    from app import main as main_mod

    calls: "list[int]" = []
    original_install = main_mod.install_stop_handler
    original_run = main_mod.asyncio.run
    try:
        main_mod.install_stop_handler = lambda: calls.append(1) or True  # type: ignore[assignment]
        main_mod.asyncio.run = lambda coro: (coro.close(), 0)[1]  # type: ignore[assignment]
        result = main_mod.run_start(object())
    finally:
        main_mod.install_stop_handler = original_install  # type: ignore[assignment]
        main_mod.asyncio.run = original_run  # type: ignore[assignment]

    assert calls == [1], "run_start must install the SIGBREAK handler"
    assert result == 0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# --- the real child, with the real server, which is what actually broke ------------------


@pytest.mark.skipif(os.name != "nt", reason="CTRL_BREAK_EVENT is Windows-only")
def test_a_real_interview_with_the_server_running_still_writes_closed_at(tmp_path):
    """THE test this file was missing, and the reason `closed_at` was still
    absent from every overlay recording after `7fcfe05`.

    Every other test here drives a SYNTHETIC child -- a few lines that install
    the handler and write a file. That proves the signal mechanism and nothing
    about `app start`, and the thing that actually broke was neither the signal
    nor the handler: it was UVICORN.

    `uvicorn.Server.serve()` runs inside `capture_signals()`, which installs
    uvicorn's own SIGINT/SIGBREAK handler, records the signal, shuts down
    gracefully -- and then, on leaving that context manager, calls
    `signal.raise_signal(captured_signal)` (`uvicorn/server.py:340`) to
    "trigger the expected behaviour now". Delivered at an arbitrary bytecode
    boundary inside the ordered shutdown, that killed the process before
    `store.close()`.

    So the missing coverage was exactly the seam: a real child AND a real
    server. Measured 2026-08-12 -- before the fix this test produces exit code
    3221225786 and a session with no `closed_at`; after it, exit 0 and both
    markers present.
    """
    repo_root = Path(__file__).resolve().parents[2]
    sessions_root = tmp_path / "sessions"
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    env = {
        **os.environ,
        "APP_SESSIONS_ROOT": str(sessions_root),
        "APP_CONTEXT_DIR": str(context_dir),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        # An interview, not a test double: the server binds, the socket
        # accepts, uvicorn installs its handlers. That is the whole point.
        "APP_OPEN_BROWSER": "0",
        # An EPHEMERAL port, chosen here and handed over. The default 8765 is
        # fixed and would collide with a real interview running on this
        # machine -- and a test that kills a real recording to prove a point
        # about clean shutdown would be an unusually bad joke.
        "APP_PORT": str(_free_port()),
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "app", "start", "--no-coach", "--engine", "fake"],
        cwd=str(repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        time.sleep(8)  # let the server bind and the capture threads settle
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        output, _ = proc.communicate(timeout=90)
    except Exception:
        proc.kill()
        raise

    assert proc.returncode == 0, (
        f"the interview did not exit cleanly (code {proc.returncode}; "
        f"3221225786 is STATUS_CONTROL_C_EXIT, i.e. killed by the break):\n{output[-2000:]}"
    )
    sessions = [p for p in sessions_root.iterdir() if p.is_dir()]
    assert len(sessions) == 1, f"expected one session dir, got {sessions}"
    metadata = json.loads((sessions[0] / "session.json").read_text(encoding="utf-8"))
    assert "closed_at" in metadata, (
        "the ordered shutdown did not reach store.close() -- this is the defect that made every "
        f"overlay recording end with an unflushed tail:\n{output[-2000:]}"
    )
    assert "latency_summary" in metadata
