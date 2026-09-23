"""Task 1 done criteria for 03-02-PLAN.md: the application-level heartbeat,
the pinned client/server message-type sets (D-01/D-02's standing gate), the
JS unit suite wired into this repo's test run, and the connection-hub
behaviours a real reconnect exercises (03-01-PLAN.md §P9: a real uvicorn
server, stdlib HTTP, the synchronous `websockets` client -- never
`starlette.testclient.TestClient`, which is not a dependency this project
has)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from websockets.sync.client import connect as ws_connect

from app.web import server as web_server
from app.web.hub import ConnectionHub

# Reuse the exact real-server harness 03-01 already built and tested --
# never a second, parallel implementation of the same thing.
from app.tests.test_web_serving import STATIC_DIR, _running_server, _store, _events

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_TEST_DIR = REPO_ROOT / "app" / "tests" / "js"


# --- The protocol surface is pinned by set equality, permanently (D-01/D-02/T-03-14/T-03-15) --


def test_accepted_and_server_message_type_sets_are_exact():
    assert set(web_server.ACCEPTED_CLIENT_MESSAGE_TYPES) == {"ping"}
    # 04-03-PLAN.md §E1: deliberately updated in the same commit that adds
    # MESSAGE_TYPE_SUGGESTION to app/web/server.py -- see gate 8's own
    # comment in app/tests/test_phase_gates.py for the full reasoning.
    assert set(web_server.SERVER_MESSAGE_TYPES) == {
        "hello", "pong", "segment", "server_status", "suggestion", "survey",
    }


def test_pong_message_type_literal_comes_from_one_place_not_two():
    """The Python constant and the JS module's own literal must be the exact
    same string -- proven by actually importing the real ESM module via node
    and reading its exported value back, never by re-typing the literal a
    second time in this test and hoping it stays in sync."""
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node is not on PATH -- cannot cross-check live-state.mjs's own literal")
    live_state_path = (REPO_ROOT / "app" / "static" / "live-state.mjs").as_posix()
    script = (
        f"import('file://{live_state_path}')"
        f".then(m => process.stdout.write(m.MESSAGE_TYPE_PONG))"
    )
    result = subprocess.run(
        [node_path, "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, f"node failed importing live-state.mjs: {result.stderr}"
    assert result.stdout == web_server.MESSAGE_TYPE_PONG


def test_a_real_client_sending_ping_receives_the_pong_the_client_module_consumes(tmp_path):
    token = "heartbeat-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)  # hello
            client.recv(timeout=5)  # server_status, pushed alongside hello
            client.send(json.dumps({"type": "ping"}))
            pong = json.loads(client.recv(timeout=5))
            assert pong == {"type": web_server.MESSAGE_TYPE_PONG}


def test_hello_carries_the_heartbeat_and_backoff_config_from_one_source(tmp_path):
    from app import config as config_mod

    token = "hello-config-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
        heartbeat_interval_ms=12345, heartbeat_timeout_ms=6789,
        reconnect_backoff_base_ms=111, reconnect_backoff_cap_ms=9999,
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            hello = json.loads(client.recv(timeout=5))
            assert hello["heartbeat_interval_ms"] == 12345
            assert hello["heartbeat_timeout_ms"] == 6789
            assert hello["reconnect_backoff_base_ms"] == 111
            assert hello["reconnect_backoff_cap_ms"] == 9999


def test_unknown_or_malformed_client_message_is_ignored_without_raising_or_closing(tmp_path):
    token = "malformed-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)  # hello
            client.recv(timeout=5)  # server_status, pushed alongside hello
            client.send("not even json")
            client.send(json.dumps({"type": "resume-please"}))
            client.send(json.dumps({"no_type_field": True}))
            client.send(json.dumps({"type": "ping"}))
            pong = json.loads(client.recv(timeout=5))
            assert pong == {"type": "pong"}, "the socket must stay alive and responsive after garbage input"


# --- Reconnection: no replay, connection-set correctness, isolation (D-01) --------------------


def test_reconnecting_client_receives_no_past_segments(tmp_path):
    token = "no-replay-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"

        with ws_connect(ws_url, open_timeout=5) as first_client:
            first_client.recv(timeout=5)  # hello
            first_client.recv(timeout=5)  # server_status, pushed alongside hello

            from app.session.store import SegmentRecord
            import asyncio

            for seq in range(3):
                record = SegmentRecord(
                    kind="segment", seq=seq, speaker="interviewee", text=f"segment {seq}",
                    audio_start_s=0.0, audio_end_s=float(seq), ready_wall_s=1.0, latency_s=0.1,
                    detected_language="en", engine="fake", written_at="2026-08-05T00:00:00.000+00:00",
                )
                future = asyncio.run_coroutine_threadsafe(hub.broadcast(record), loop)
                future.result(timeout=5)
                first_client.recv(timeout=5)  # each segment reaches the first client

        # A FRESH client connects after 3 segments were already broadcast.
        with ws_connect(ws_url, open_timeout=5) as second_client:
            hello = json.loads(second_client.recv(timeout=5))
            assert hello["type"] == "hello"
            status = json.loads(second_client.recv(timeout=5))
            assert status["type"] == "server_status"

            # Broadcast one MORE segment now -- only this one should ever
            # reach the second client.
            from app.session.store import SegmentRecord as _SR
            import asyncio as _asyncio

            new_record = _SR(
                kind="segment", seq=99, speaker="interviewee", text="only this one",
                audio_start_s=0.0, audio_end_s=99.0, ready_wall_s=1.0, latency_s=0.1,
                detected_language="en", engine="fake", written_at="2026-08-05T00:00:00.000+00:00",
            )
            fut = _asyncio.run_coroutine_threadsafe(hub.broadcast(new_record), loop)
            fut.result(timeout=5)
            message = json.loads(second_client.recv(timeout=5))
            assert message["text"] == "only this one", "the second client must never receive a past segment"


def test_abrupt_disconnect_removes_connection_and_does_not_stop_a_second_client(tmp_path):
    token = "abrupt-disconnect-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        first = ws_connect(ws_url, open_timeout=5)
        first.recv(timeout=5)
        first.recv(timeout=5)
        with ws_connect(ws_url, open_timeout=5) as second:
            second.recv(timeout=5)
            second.recv(timeout=5)
            assert hub.connection_count == 2

            first.close()  # abrupt -- no cooperative shutdown message

            deadline = time.monotonic() + 3.0
            while hub.connection_count != 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert hub.connection_count == 1, "an abruptly-closed connection must be removed from the active set"

            from app.session.store import SegmentRecord
            import asyncio

            record = SegmentRecord(
                kind="segment", seq=0, speaker="interviewee", text="still reaches the survivor",
                audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.0, latency_s=0.1,
                detected_language="en", engine="fake", written_at="2026-08-05T00:00:00.000+00:00",
            )
            future = asyncio.run_coroutine_threadsafe(hub.broadcast(record), loop)
            future.result(timeout=5)
            message = json.loads(second.recv(timeout=5))
            assert message["text"] == "still reaches the survivor"


def test_three_reconnects_leave_the_connection_count_correct_each_time(tmp_path):
    token = "three-reconnects-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        for attempt in range(3):
            with ws_connect(ws_url, open_timeout=5) as client:
                client.recv(timeout=5)
                client.recv(timeout=5)
                assert hub.connection_count == 1, f"attempt {attempt}: expected exactly one active connection"
            deadline = time.monotonic() + 3.0
            while hub.connection_count != 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert hub.connection_count == 0, f"attempt {attempt}: connection must be removed after close"


def test_disconnect_is_recorded_as_a_named_event_with_no_content(tmp_path):
    token = "disconnect-event-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)
            client.recv(timeout=5)

    deadline = time.monotonic() + 3.0
    disconnect_events = []
    while time.monotonic() < deadline:
        disconnect_events = [e for e in _events(store) if e.get("event") == "ws-disconnected"]
        if disconnect_events:
            break
        time.sleep(0.02)

    assert len(disconnect_events) == 1
    detail = disconnect_events[0]["detail"]
    assert set(detail) == {"reason", "connection_count"}
    assert isinstance(detail["reason"], str)
    assert isinstance(detail["connection_count"], int)


# --- live-state.mjs is pure: no DOM, no timer creation, no socket reference -------------------


def _strip_js_comments_and_strings(source: str) -> str:
    """A deliberately simple (not a full JS parser) but robust-enough
    stripper: removes `//` line comments, `/* */` block comments, and the
    contents of string/template literals, so a docstring/comment mentioning
    "document" or "setTimeout" in explanatory prose can never trip this
    gate -- exactly the same rationale as `test_windows_traps.py`'s AST-based
    S13 gate over a fragile regex."""
    result = []
    i = 0
    n = len(source)
    while i < n:
        two = source[i:i + 2]
        ch = source[i]
        if two == "//":
            while i < n and source[i] != "\n":
                i += 1
            continue
        if two == "/*":
            end = source.find("*/", i + 2)
            i = (end + 2) if end != -1 else n
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def test_live_state_module_is_pure_no_dom_no_timers_no_socket():
    source = (REPO_ROOT / "app" / "static" / "live-state.mjs").read_text(encoding="utf-8")
    stripped = _strip_js_comments_and_strings(source)
    forbidden = ("document", "window.", "setTimeout(", "setInterval(", "new WebSocket", "XMLHttpRequest", "fetch(")
    violations = [name for name in forbidden if name in stripped]
    assert not violations, f"live-state.mjs must stay pure (no DOM/timer/socket): found {violations}"


# --- The JS unit suite runs as part of THIS suite, never silently skipped --------------------


def test_js_live_state_suite_runs_via_node():
    node_path = shutil.which("node")
    if node_path is None:
        pytest.skip("node is not on PATH -- live-state.mjs's JS unit tests could not run at all")

    test_files = sorted(str(p) for p in JS_TEST_DIR.glob("*.test.mjs"))
    assert test_files, f"no *.test.mjs files found under {JS_TEST_DIR}"

    result = subprocess.run(
        [node_path, "--test", *test_files],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, (
        f"node --test over {test_files} failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
    match = re.search(r"pass (\d+)", result.stdout)
    assert match is not None, "could not find node's own pass count in its summary output"
    assert int(match.group(1)) > 0, "node reported zero passing tests -- that is not a real green run"
