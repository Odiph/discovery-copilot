"""Task 2 done criteria for routing, the connection hub, the thread->asyncio
bridge, and the served page (03-01-PLAN.md §P2-§P5, §P9). Per §P9: a real
uvicorn server on an ephemeral loopback port in a background thread, stdlib
HTTP (`urllib.request`), and the synchronous client from the `websockets`
library this project already pins for the vendor adapter -- never
`starlette.testclient.TestClient`/httpx, which is not a dependency this
project actually has (see 03-01-PLAN.md §P9)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
import uvicorn
from websockets.sync.client import connect as ws_connect

from app import config as config_mod
from app.capture.stream import CaptureFrame
from app.session.store import SegmentRecord, SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.stt.live import StreamWorker
from app.web import server as web_server
from app.web.hub import ConnectionHub, wire_payload
from benchmark.engines.base import Segment

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


# --- shared test infrastructure --------------------------------------------------------


@contextlib.contextmanager
def _running_server(app, *, host: str = "127.0.0.1", port: int = 0):
    """Runs `app` under a REAL uvicorn server, in its own background thread,
    driving its OWN explicitly-created event loop -- not `uvicorn.Server.run()`
    (which hides the loop inside its own `asyncio.run()` call). Yields
    `(base_url, loop)` so a test can hand that same loop to a real
    `StreamWorker`, reproducing the exact thread -> asyncio bridge production
    code uses (§P3)."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn server never reported started"
    actual_port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://{host}:{actual_port}", loop
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _http_get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    return store


def _events(store: SessionStore) -> list[dict]:
    return [
        json.loads(line)
        for line in store.transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_record(**overrides) -> SegmentRecord:
    fields = dict(
        kind="segment", seq=1, speaker="interviewee", text="hello world",
        audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.2, latency_s=0.2,
        detected_language="en", engine="fake", written_at="2026-08-05T00:00:00.000+00:00",
    )
    fields.update(overrides)
    return SegmentRecord(**fields)


def _expected_hello(
    *,
    session_id: str,
    server_started_at: str,
    project: "str | None" = None,
    grounding_sources: "list[str] | None" = None,
    coach_enabled: bool = False,
) -> dict:
    """The `hello` shape every test in THIS file expects when `create_app` is
    called with no heartbeat/backoff overrides -- i.e. `app/config.py`'s own
    `DEFAULT_*` values, served to the client as the single source of truth
    (03-02-PLAN.md Task 1).

    Phase A adds `project` and `grounding_sources`. The defaults here mirror
    `create_app`'s -- "no project, no extra sources" -- which is what an
    unprojected run genuinely sends. Note this helper is compared by whole-dict
    EQUALITY at its call sites, which is why adding the fields to the payload
    without adding them here failed two tests immediately. Keep that equality;
    a subset check would let a future field reach the phone unnoticed."""
    return {
        "type": "hello",
        "session_id": session_id,
        "server_started_at": server_started_at,
        "heartbeat_interval_ms": config_mod.DEFAULT_HEARTBEAT_INTERVAL_MS,
        "heartbeat_timeout_ms": config_mod.DEFAULT_HEARTBEAT_TIMEOUT_MS,
        "reconnect_backoff_base_ms": config_mod.DEFAULT_RECONNECT_BACKOFF_BASE_MS,
        "reconnect_backoff_cap_ms": config_mod.DEFAULT_RECONNECT_BACKOFF_CAP_MS,
        "project": project,
        "grounding_sources": grounding_sources if grounding_sources is not None else [],
        # Mirrors `create_app`'s own default, which is False -- a server built
        # without a coach says so, rather than leaving a client to assume one.
        "coach_enabled": coach_enabled,
    }


def _run_loop_in_background_thread() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _stop_background_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)
    loop.close()


# --- wire_payload: exact key-set allowlist ---------------------------------------------


def test_wire_payload_is_exact_key_set_allowlist():
    record = _sample_record()
    payload = wire_payload(record)
    assert set(payload) == {"type", "seq", "speaker", "text", "audio_end_s", "detected_language"}
    assert payload["type"] == "segment"
    assert payload["audio_end_s"] == record.audio_end_s
    # The fabricated field -- excluded BY CONSTRUCTION, not by a negative check.
    assert "audio_start_s" not in payload


# --- ConnectionHub: per-connection failure isolation, driven with no network at all ----


class _FakeConnection:
    def __init__(self, *, raise_on_send: bool = False) -> None:
        self.raise_on_send = raise_on_send
        self.received: list[dict] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        if self.raise_on_send:
            raise RuntimeError("simulated dead connection")
        self.received.append(payload)


def test_broadcast_isolates_one_dead_connection_from_another():
    async def scenario() -> None:
        hub = ConnectionHub()
        good = _FakeConnection()
        bad = _FakeConnection(raise_on_send=True)
        await hub.connect(good)
        await hub.connect(bad)
        assert hub.connection_count == 2

        record = _sample_record()
        await hub.broadcast(record)

        assert good.received == [wire_payload(record)]
        assert hub.connection_count == 1, "the raising connection must be removed from the active set"

    asyncio.run(scenario())


# --- The thread -> asyncio bridge: real thread, real loop, real connected client -------


def test_finalized_segment_crosses_thread_boundary_to_a_real_connected_client(tmp_path):
    token = "delivery-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="delivery-session", server_started_at="t0",
    )

    with _running_server(app) as (base_url, loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            hello = json.loads(client.recv(timeout=5))
            assert hello == _expected_hello(session_id="delivery-session", server_started_at="t0")
            status = json.loads(client.recv(timeout=5))
            assert status["type"] == "server_status"

            frame_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
            worker = StreamWorker(
                engine=FakeStreamingEngine(text="bridge delivery probe", cadence_bytes=20),
                frame_queue=frame_queue, store=store, speaker="interviewee",
                sample_rate=16000, loop=loop, broadcast=hub.broadcast,
            )
            worker.start()
            try:
                frame_queue.put(
                    CaptureFrame(seq=0, pcm16=b"\x00" * 20, kind="synthesized", captured_monotonic=0.0)
                )
                message = json.loads(client.recv(timeout=5))
            finally:
                worker.stop()

    assert message["type"] == "segment"
    assert message["speaker"] == "interviewee"
    assert message["text"] == "bridge delivery probe"
    assert set(message) == {"type", "seq", "speaker", "text", "audio_end_s", "detected_language"}


def test_persist_returns_fast_even_when_broadcast_is_slow(tmp_path):
    loop, thread = _run_loop_in_background_thread()
    try:
        store = _store(tmp_path)

        async def slow_broadcast(record: SegmentRecord) -> None:
            await asyncio.sleep(2.0)

        worker = StreamWorker(
            engine=FakeStreamingEngine(text="slow broadcast probe"),
            frame_queue=queue.Queue(), store=store, speaker="interviewee",
            sample_rate=16000, loop=loop, broadcast=slow_broadcast,
        )
        segment = Segment(text="slow broadcast probe", audio_end_s=1.0, ready_wall_s=1.1, detected_language="en")

        result: dict = {}

        def call_persist() -> None:
            start = time.monotonic()
            worker._persist(segment)
            result["elapsed"] = time.monotonic() - start

        caller = threading.Thread(target=call_persist)
        caller.start()
        caller.join(timeout=5.0)

        assert "elapsed" in result, "the calling thread never returned -- _persist blocked"
        assert result["elapsed"] < 0.05, (
            f"_persist took {result['elapsed']:.3f}s against a 2s-slow broadcast -- "
            f"this is the .result() trap: it must schedule and return immediately"
        )
        # Let the scheduled slow_broadcast actually finish on the loop before
        # tearing it down -- purely test hygiene (avoids an "unawaited
        # coroutine" warning from stopping the loop mid-sleep); irrelevant to
        # the assertion above, which already ran the instant _persist returned.
        time.sleep(2.1)
    finally:
        _stop_background_loop(loop, thread)


def test_persist_records_named_event_when_broadcast_raises_without_blocking(tmp_path):
    loop, thread = _run_loop_in_background_thread()
    try:
        store = _store(tmp_path)

        async def raising_broadcast(record: SegmentRecord) -> None:
            raise RuntimeError("simulated broadcast failure")

        worker = StreamWorker(
            engine=FakeStreamingEngine(text="raising broadcast probe"),
            frame_queue=queue.Queue(), store=store, speaker="interviewee",
            sample_rate=16000, loop=loop, broadcast=raising_broadcast,
        )
        segment = Segment(text="raising broadcast probe", audio_end_s=1.0, ready_wall_s=1.1, detected_language="en")

        result: dict = {}

        def call_persist() -> None:
            start = time.monotonic()
            worker._persist(segment)
            result["elapsed"] = time.monotonic() - start

        caller = threading.Thread(target=call_persist)
        caller.start()
        caller.join(timeout=5.0)
        assert result["elapsed"] < 0.5, "a raising broadcast must never block the STT thread either"

        deadline = time.monotonic() + 3.0
        failures: list[dict] = []
        while time.monotonic() < deadline:
            failures = [e for e in _events(store) if e.get("event") == "broadcast-failure"]
            if failures:
                break
            time.sleep(0.02)

        assert len(failures) == 1, "a broadcast failure must be recorded as exactly one named event"
        assert failures[0]["detail"]["error_type"] == "RuntimeError"
        assert "message" not in failures[0]["detail"], "never record the raw exception text -- error_type only (S7)"
    finally:
        _stop_background_loop(loop, thread)


# --- build_uvicorn_server: the websocket ping/pong bound is explicit, never left ------
# --- to ride on `websockets`' own undocumented default (independent review finding B) --


def test_build_uvicorn_server_carries_the_default_ws_ping_knobs_from_config(tmp_path):
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token="ws-ping-default-token", hub=hub, static_dir=STATIC_DIR,
        store=store, session_id="s", server_started_at="t",
    )
    server = web_server.build_uvicorn_server(app, host="127.0.0.1", port=0)
    assert server.config.ws_ping_interval == config_mod.DEFAULT_WS_PING_INTERVAL_S
    assert server.config.ws_ping_timeout == config_mod.DEFAULT_WS_PING_TIMEOUT_S
    store.close()


def test_build_uvicorn_server_honours_an_explicit_ws_ping_override(tmp_path):
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token="ws-ping-override-token", hub=hub, static_dir=STATIC_DIR,
        store=store, session_id="s", server_started_at="t",
    )
    server = web_server.build_uvicorn_server(
        app, host="127.0.0.1", port=0, ws_ping_interval_s=7.0, ws_ping_timeout_s=9.0,
    )
    assert server.config.ws_ping_interval == 7.0
    assert server.config.ws_ping_timeout == 9.0
    store.close()


# --- Routing: token-gated, generic 404, nothing echoed ---------------------------------


def test_correct_token_serves_page_wrong_token_and_root_get_generic_404(tmp_path):
    token = "route-token-correct"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        status, body = _http_get(f"{base_url}/s/{token}")
        assert status == 200
        assert b"transcript-region" in body

        wrong_token = "definitely-not-the-token"
        status, body = _http_get(f"{base_url}/s/{wrong_token}")
        assert status == 404
        assert wrong_token.encode() not in body

        status, body = _http_get(f"{base_url}/s/")
        assert status == 404

        status, body = _http_get(f"{base_url}/")
        assert status == 404

        status, _ = _http_get(f"{base_url}/s/{token}/app.js")
        assert status == 200
        status, body = _http_get(f"{base_url}/s/{wrong_token}/app.js")
        assert status == 404
        assert wrong_token.encode() not in body

        status, _ = _http_get(f"{base_url}/s/{token}/style.css")
        assert status == 200
        status, body = _http_get(f"{base_url}/s/{wrong_token}/style.css")
        assert status == 404
        assert wrong_token.encode() not in body


def test_websocket_correct_token_is_accepted_and_sends_one_hello(tmp_path):
    token = "ws-token-correct"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="ws-session", server_started_at="ws-t0",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            hello = json.loads(client.recv(timeout=5))
            assert hello == _expected_hello(session_id="ws-session", server_started_at="ws-t0")


def test_websocket_wrong_token_is_rejected_before_ever_entering_the_connection_set(tmp_path):
    token = "ws-token-correct-2"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + "/s/wrong-token/ws"
        with pytest.raises(Exception):
            with ws_connect(ws_url, open_timeout=5) as client:
                client.recv(timeout=2)
        assert hub.connection_count == 0, "a rejected token must never reach the hub's active set"


# --- Non-ASCII token: uniform 404/close, never a 500 or an unhandled exception (D-05) ---------
#
# `secrets.compare_digest` on two `str` requires ASCII on both sides. Starlette percent-decodes
# path params, so a non-ASCII token (Hebrew is plausible on this project -- 03-CONTEXT.md D-04)
# used to raise `TypeError` inside `_token_matches`, producing an HTTP 500 (an observable deviation
# from D-05's uniform-404 posture -- a LAN scanner could distinguish "wrong token" from "malformed
# token" by response class) and, on the socket route, an ASGI-level exception instead of the
# deliberate close-before-accept path.


def test_non_ascii_token_gets_the_same_uniform_404_as_a_wrong_ascii_token(tmp_path):
    token = "route-token-ascii-only"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        non_ascii_token = "אב"  # Hebrew, percent-decoded by Starlette from the URL
        encoded = urllib.parse.quote(non_ascii_token)

        status, body = _http_get(f"{base_url}/s/{encoded}")
        assert status == 404, f"a non-ASCII token must get the same uniform 404 as a wrong token, got {status}"
        assert non_ascii_token.encode("utf-8") not in body

        status, body = _http_get(f"{base_url}/s/{encoded}/app.js")
        assert status == 404
        assert non_ascii_token.encode("utf-8") not in body


def test_websocket_non_ascii_token_is_rejected_before_ever_entering_the_connection_set(tmp_path):
    token = "ws-token-ascii-only"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        non_ascii_token = "אב"
        encoded = urllib.parse.quote(non_ascii_token)
        ws_url = base_url.replace("http://", "ws://") + f"/s/{encoded}/ws"
        with pytest.raises(Exception):
            with ws_connect(ws_url, open_timeout=5) as client:
                client.recv(timeout=2)
        assert hub.connection_count == 0, "a rejected non-ASCII token must never reach the hub's active set"


# --- App factory: assets read from disk exactly once, never per-request ---------------


def test_static_assets_are_read_once_at_factory_time_never_per_request(tmp_path):
    static_copy = tmp_path / "static-copy"
    static_copy.mkdir()
    for name in ("index.html", "style.css", "app.js", "live-state.mjs"):
        (static_copy / name).write_bytes((STATIC_DIR / name).read_bytes())

    hub = ConnectionHub()
    store = _store(tmp_path / "session")
    app = web_server.create_app(
        token="asset-once-token", hub=hub, static_dir=static_copy, store=store,
        session_id="s", server_started_at="t",
    )

    # Deleted AFTER the factory already read them -- subsequent responses
    # must be entirely unaffected, proving no handler ever touches disk.
    for name in ("index.html", "style.css", "app.js", "live-state.mjs"):
        (static_copy / name).unlink()

    with _running_server(app) as (base_url, _loop):
        for path, expect_nonempty in (
            ("/s/asset-once-token", True),
            ("/s/asset-once-token/app.js", True),
            ("/s/asset-once-token/style.css", True),
            ("/s/asset-once-token/live-state.mjs", True),
        ):
            status, body = _http_get(f"{base_url}{path}")
            assert status == 200
            assert bool(body) == expect_nonempty


# --- Rendered page: both regions, non-collapsing suggestion region --------------------


def test_served_page_contains_both_regions_and_transcript_region_starts_empty(tmp_path):
    token = "regions-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        status, body = _http_get(f"{base_url}/s/{token}")
        assert status == 200
        html = body.decode("utf-8")

        assert 'id="suggestion-region"' in html
        assert 'id="transcript-region"' in html
        assert "no suggestions" in html

        transcript_match = re.search(r'<div id="transcript-region"[^>]*>(.*?)</div>', html, re.DOTALL)
        assert transcript_match is not None, "transcript region markup not found in served page"
        assert transcript_match.group(1).strip() == "", "the transcript region must ship empty"

        status, css_body = _http_get(f"{base_url}/s/{token}/style.css")
        assert status == 200
        assert "min-height" in css_body.decode("utf-8")


# --- Hardware smoke test: request latency while a real capture stream runs (A2) --------


@pytest.mark.hardware
def test_hardware_request_latency_stays_low_while_a_real_capture_stream_runs(tmp_path):
    import pyaudiowpatch as pyaudio

    from app.capture.devices import resolve_default_loopback_device
    from app.capture.stream import CaptureStream

    pa = pyaudio.PyAudio()
    try:
        device = resolve_default_loopback_device(pa)
        capture = CaptureStream(
            pyaudio, pa, device, target_sample_rate=16000, frame_ms=100,
            queue_max_seconds=10.0, label="interviewee",
        )
        capture.start()
        try:
            token = "hardware-latency-token"
            hub = ConnectionHub()
            store = _store(tmp_path)
            app = web_server.create_app(
                token=token, hub=hub, static_dir=STATIC_DIR, store=store,
                session_id="s", server_started_at="t",
            )
            with _running_server(app) as (base_url, _loop):
                durations: list[float] = []
                for _ in range(20):
                    start = time.monotonic()
                    status, _ = _http_get(f"{base_url}/s/{token}")
                    durations.append(time.monotonic() - start)
                    assert status == 200
                slowest = max(durations)
                assert slowest < 0.25, (
                    f"slowest of 20 requests took {slowest:.3f}s while a real capture stream ran -- "
                    f"the 'capture threads and the loop do not contend' assumption may not hold here"
                )
        finally:
            capture.stop()
    finally:
        pa.terminate()
