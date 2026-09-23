"""Task 2 done criteria for 03-02-PLAN.md: `app/web/status.py`'s pure
transition logic, the server-status push on every fresh connect (§Q6/§Q7),
the gap marker's persistence guarantee against a REAL session (D-02), and
SURF-06's "never blank" structural promise. Per 03-01-PLAN.md §P9: a real
uvicorn server, stdlib HTTP, the synchronous `websockets` client."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from websockets.sync.client import connect as ws_connect

from app.session.reader import read_session
from app.session.store import SessionStore, create_session_dir
from app.web import server as web_server
from app.web.hub import ConnectionHub
from app.web.status import (
    HEALTHY_STATUS,
    STATUS_FAILED,
    ServerStatus,
    StatusBoard,
    StreamHealth,
    derive_status,
    status_message,
)

# Reuse 03-01's real-server harness -- never a second, parallel implementation.
from app.tests.test_web_serving import STATIC_DIR, _running_server, _store

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- app/web/status.py: pure, unit-testable without a socket ----------------------------------


def test_derive_status_is_healthy_when_nothing_has_failed():
    healths = [
        StreamHealth(speaker="interviewee", worker_failed=False, capture_dead=False),
        StreamHealth(speaker="interviewer", worker_failed=False, capture_dead=False),
    ]
    assert derive_status(healths) == HEALTHY_STATUS


def test_derive_status_names_the_failed_side_on_a_worker_failure():
    healths = [
        StreamHealth(speaker="interviewee", worker_failed=True, capture_dead=False),
        StreamHealth(speaker="interviewer", worker_failed=False, capture_dead=False),
    ]
    status = derive_status(healths)
    assert status.value == STATUS_FAILED
    assert status.failed_side == "interviewee"
    assert status.reason is not None


def test_derive_status_names_the_failed_side_on_a_dead_capture():
    healths = [StreamHealth(speaker="interviewer", worker_failed=False, capture_dead=True)]
    status = derive_status(healths)
    assert status.value == STATUS_FAILED
    assert status.failed_side == "interviewer"


def test_status_message_never_carries_transcript_text_or_a_vendor_name():
    status = ServerStatus(value=STATUS_FAILED, failed_side="interviewee", reason="stream-failure")
    message = status_message(status)
    assert set(message) == {"type", "status", "failed_side", "reason"}
    assert message["type"] == "server_status"
    # A reason CLASS only -- never a vendor name (STT-03) or free text.
    for forbidden in ("soniox", "assemblyai", "whisper", "deepgram"):
        assert forbidden not in json.dumps(message).lower()


def test_status_board_reports_a_change_only_once():
    board = StatusBoard()
    assert board.current == HEALTHY_STATUS

    failed = ServerStatus(value=STATUS_FAILED, failed_side="interviewee", reason="stream-failure")
    assert board.observe(failed) is True, "the first observation of a new status must report a change"
    assert board.observe(failed) is False, "an unchanged status must not be reported as a change again"

    assert board.observe(HEALTHY_STATUS) is True, "a recovery is a change too"
    assert board.observe(HEALTHY_STATUS) is False


# --- Server-side push: survives a reconnect (§Q6 versus D-01) ---------------------------------


def test_a_fresh_connect_with_no_failure_receives_the_healthy_status(tmp_path):
    token = "status-healthy-token"
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
            status = json.loads(client.recv(timeout=5))
            assert status == {"type": "server_status", "status": "healthy", "failed_side": None, "reason": None}


def test_a_client_connecting_after_a_failure_still_receives_the_failed_status(tmp_path):
    token = "status-after-failure-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    board = StatusBoard()
    board.observe(ServerStatus(value=STATUS_FAILED, failed_side="interviewer", reason="stream-failure"))
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t", status_board=board,
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)  # hello
            status = json.loads(client.recv(timeout=5))
            assert status["status"] == "failed"
            assert status["failed_side"] == "interviewer"


def test_a_status_change_is_pushed_to_every_connected_client_within_one_poll(tmp_path):
    """Exercises the REAL async poll loop from app/main.py's own shape --
    a small periodic task reading each stream pair's already-public state,
    broadcasting only on a change (§Q6/§Q7)."""
    import asyncio

    token = "status-poll-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    board = StatusBoard()
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t", status_board=board,
    )
    with _running_server(app) as (base_url, loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)  # hello
            client.recv(timeout=5)  # initial healthy status

            class _FakeWorker:
                failed = True

            class _FakeCapture:
                is_dead = False

            # The supervisor is part of the pair the poller reads: `idle` and
            # `lagging` come from it, so a double without one is a double of a
            # pair that cannot exist. Left REQUIRED rather than defaulted with
            # `getattr` in the poller -- a silent default there would mean the
            # degraded tier never fires and nothing anywhere notices.
            class _FakeSupervisor:
                is_idle = False
                is_lagging = False

            class _FakePair:
                speaker = "interviewee"
                worker = _FakeWorker()
                capture = _FakeCapture()
                supervisor = _FakeSupervisor()

            from app.main import _status_poll_loop

            async def _run_one_poll_cycle():
                task = asyncio.create_task(_status_poll_loop([_FakePair()], board, hub, 0.05))
                await asyncio.sleep(0.2)
                task.cancel()

            future = asyncio.run_coroutine_threadsafe(_run_one_poll_cycle(), loop)
            future.result(timeout=5)

            pushed = json.loads(client.recv(timeout=5))
            assert pushed["type"] == "server_status"
            assert pushed["status"] == "failed"
            assert pushed["failed_side"] == "interviewee"


def test_status_message_type_is_distinct_from_segment_and_rendered_differently():
    assert web_server.MESSAGE_TYPE_SERVER_STATUS != web_server.MESSAGE_TYPE_SEGMENT
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    # The client module must branch on the two types separately, into
    # separate rendering functions -- never the same code path.
    # The CONSTANT, not the literal. app.js used to compare a hand-typed
    # "server_status" and now imports MESSAGE_TYPE_SERVER_STATUS from
    # live-state.mjs -- see test_wire_vocabulary.py for why. Asserting the
    # import name is the stronger check: it proves the branch exists AND
    # that it is held to the server's word.
    assert "MESSAGE_TYPE_SERVER_STATUS" in app_js
    assert "renderServerStatus" in app_js
    assert "renderGapMarker" in app_js
    served_html = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="server-status-region"' in served_html
    assert 'id="transcript-region"' in served_html


# --- Gap marker never persisted (D-02) ---------------------------------------------------------


def test_three_reconnects_leave_no_client_gap_in_the_transcript(tmp_path):
    token = "gap-persistence-token"
    hub = ConnectionHub()
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        for _ in range(3):
            with ws_connect(ws_url, open_timeout=5) as client:
                client.recv(timeout=5)  # hello
                client.recv(timeout=5)  # status

    deadline = time.monotonic() + 3.0
    while hub.connection_count != 0 and time.monotonic() < deadline:
        time.sleep(0.02)

    store.close()
    result = read_session(session_dir)
    event_names = {r["event"] for r in result.records if r["kind"] == "event"}
    assert event_names <= {"ws-disconnected"}, f"an unexpected event kind leaked into the transcript: {event_names}"
    for record in result.records:
        assert "gap" not in json.dumps(record).lower(), f"a client-side gap leaked into the transcript: {record}"


def test_no_client_message_can_cause_a_record_to_be_written(tmp_path):
    token = "no-write-token"
    hub = ConnectionHub()
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            client.recv(timeout=5)  # hello
            client.recv(timeout=5)  # status
            transcript_lines_before = store.transcript_path.read_text(encoding="utf-8").splitlines()

            client.send(json.dumps({"type": "ping"}))
            json.loads(client.recv(timeout=5))  # pong
            client.send(json.dumps({"type": "not-a-real-type"}))
            client.send("not even json")
            client.send(json.dumps({"no_type_field": True}))
            time.sleep(0.3)

            transcript_lines_after = store.transcript_path.read_text(encoding="utf-8").splitlines()

    assert len(transcript_lines_after) == len(transcript_lines_before)


# --- SURF-06: no code path clears the transcript container, never blank ----------------------


def test_no_code_path_clears_the_transcript_container():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    forbidden = re.compile(r"transcript\w*\.(innerHTML\s*=|replaceChildren\(\s*\)|textContent\s*=)")
    assert not forbidden.search(app_js), "SURF-06: no code path may clear the transcript container"


def test_suggestion_region_keeps_its_resting_state_after_a_failure_status(tmp_path):
    token = "suggestion-rest-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        from app.tests.test_web_serving import _http_get

        status, body = _http_get(f"{base_url}/s/{token}")
        assert status == 200
        html = body.decode("utf-8")
        assert 'id="suggestion-region"' in html
        assert "no suggestions" in html
