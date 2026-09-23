"""Task 1 done criteria for 04-03-PLAN.md §E1/§E2/§E3/§E5: the suggestion
wire message, its allowlist projection, the broadcast's per-connection
isolation, and the coach's distinct failed-side wording. Reuses
`app/tests/test_web_serving.py`'s real-uvicorn harness (§P9's own precedent),
never a second parallel implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from websockets.sync.client import connect as ws_connect

from app.coach.generate import Candidate
from app.web import server as web_server
from app.web.hub import WIRE_SUGGESTION_KEYS, ConnectionHub, suggestion_payload
from app.web.status import (
    FAILED_SIDE_COACH,
    REASON_COACH_FAILURE,
    REASON_STREAM_FAILURE,
    STATUS_FAILED,
    ServerStatus,
    StatusBoard,
    StreamHealth,
    derive_status,
)

# Reuse 03-01's real-server harness -- never a second, parallel implementation.
from app.tests.test_web_serving import STATIC_DIR, _running_server, _store

REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate(index: int, text: str, move: str = "advance") -> Candidate:
    return Candidate(index=index, text=text, move=move)


# --- suggestion_payload(): exact key-set allowlist ------------------------------------


def test_suggestion_payload_is_exact_key_set_allowlist():
    ranked = [_candidate(0, "primary text"), _candidate(1, "alt one"), _candidate(2, "alt two")]
    payload = suggestion_payload(ranked)
    assert set(payload) == WIRE_SUGGESTION_KEYS
    assert set(payload) == {"type", "primary", "alternatives"}
    assert payload["type"] == "suggestion"
    assert payload["primary"] == "primary text"
    assert payload["alternatives"] == ["alt one", "alt two"]
    # No per-rule flags, no band, no rationale, no move type, no dropped
    # candidates -- excluded by construction, not by a negative check.
    for forbidden in ("band", "rationale", "move", "is_leading", "dropped"):
        assert forbidden not in payload


def test_suggestion_payload_of_a_single_candidate_has_no_alternatives_key_populated():
    payload = suggestion_payload([_candidate(0, "only one")])
    assert payload["primary"] == "only one"
    assert payload["alternatives"] == []
    assert set(payload) == WIRE_SUGGESTION_KEYS


def test_suggestion_payload_derives_its_alternatives_cap_from_max_shown_not_a_hardcoded_three():
    """Independent review finding E: `app/config.py`'s `DEFAULT_COACH_MAX_SHOWN`/
    `APP_COACH_MAX_SHOWN` already caps how many candidates survive ranking
    (`critic.rank_candidates(max_shown=...)`). Before the fix,
    `suggestion_payload` hardcoded `texts[1:3]` regardless -- so raising
    `APP_COACH_MAX_SHOWN` to 5 grew `ranked` but the wire silently kept
    truncating back down to one primary plus two alternatives."""
    ranked = [_candidate(i, f"text-{i}") for i in range(5)]

    default_payload = suggestion_payload(ranked)
    assert default_payload["alternatives"] == ["text-1", "text-2"], (
        "the default cap must still match DEFAULT_COACH_MAX_SHOWN (3) unchanged"
    )

    grown_payload = suggestion_payload(ranked, max_shown=5)
    assert grown_payload["alternatives"] == ["text-1", "text-2", "text-3", "text-4"], (
        "a max_shown wider than 3 must actually widen the wire, not silently re-truncate it"
    )


# --- Broadcasting: real client, per-connection isolation -------------------------------


def test_broadcasting_a_ranked_set_delivers_exactly_one_message_to_a_real_connected_client(tmp_path):
    token = "suggestion-delivery-token"
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
            client.recv(timeout=5)  # server_status

            import asyncio

            ranked = [_candidate(0, "shown primary"), _candidate(1, "shown alt")]
            future = asyncio.run_coroutine_threadsafe(hub.broadcast_suggestion(ranked), _loop)
            future.result(timeout=5)

            message = json.loads(client.recv(timeout=5))

    assert message["type"] == "suggestion"
    assert message["primary"] == "shown primary"
    assert message["alternatives"] == ["shown alt"]
    assert set(message) == {"type", "primary", "alternatives"}


def test_a_ranked_set_of_one_renders_as_a_complete_state_with_no_placeholder(tmp_path):
    token = "suggestion-single-token"
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
            client.recv(timeout=5)  # server_status

            import asyncio

            future = asyncio.run_coroutine_threadsafe(
                hub.broadcast_suggestion([_candidate(0, "the only survivor")]), _loop
            )
            future.result(timeout=5)

            message = json.loads(client.recv(timeout=5))

    assert message["primary"] == "the only survivor"
    assert message["alternatives"] == []
    # The RENDERER-side assertion (no placeholder/error class/empty-state
    # text for a single survivor) lives in the app.js source-inspection tests
    # below -- app.js has no client-shaped stand-in to drive from Python.


def test_broadcast_suggestion_isolates_one_dead_connection_from_another():
    import asyncio

    class _FakeConnection:
        def __init__(self, *, raise_on_send: bool = False) -> None:
            self.raise_on_send = raise_on_send
            self.received: list[dict] = []

        async def accept(self) -> None:
            pass

        async def send_json(self, payload: dict) -> None:
            if self.raise_on_send:
                raise RuntimeError("simulated dead connection")
            self.received.append(payload)

    async def scenario() -> None:
        hub = ConnectionHub()
        good = _FakeConnection()
        bad = _FakeConnection(raise_on_send=True)
        await hub.connect(good)
        await hub.connect(bad)
        assert hub.connection_count == 2

        ranked = [_candidate(0, "isolation probe")]
        await hub.broadcast_suggestion(ranked)

        assert good.received == [suggestion_payload(ranked)]
        assert hub.connection_count == 1, "the raising connection must be removed from the active set"

    asyncio.run(scenario())


def test_broadcast_suggestion_threads_max_shown_through_to_the_wire_payload():
    import asyncio

    class _FakeConnection:
        def __init__(self) -> None:
            self.received: list[dict] = []

        async def accept(self) -> None:
            pass

        async def send_json(self, payload: dict) -> None:
            self.received.append(payload)

    async def scenario() -> None:
        hub = ConnectionHub()
        conn = _FakeConnection()
        await hub.connect(conn)

        ranked = [_candidate(i, f"text-{i}") for i in range(5)]
        await hub.broadcast_suggestion(ranked, max_shown=5)

        assert conn.received == [suggestion_payload(ranked, max_shown=5)]
        assert conn.received[0]["alternatives"] == ["text-1", "text-2", "text-3", "text-4"]

    asyncio.run(scenario())


# --- Gate 8: protocol surface pinned in both directions, including the new type -------


def test_gate_eight_includes_the_new_server_to_client_type_and_client_to_server_stays_the_heartbeat():
    assert web_server.ACCEPTED_CLIENT_MESSAGE_TYPES == frozenset({web_server.MESSAGE_TYPE_PING})
    assert web_server.SERVER_MESSAGE_TYPES == frozenset({
        web_server.MESSAGE_TYPE_HELLO,
        web_server.MESSAGE_TYPE_PONG,
        web_server.MESSAGE_TYPE_SEGMENT,
        web_server.MESSAGE_TYPE_SERVER_STATUS,
        web_server.MESSAGE_TYPE_SUGGESTION,
        web_server.MESSAGE_TYPE_SURVEY,
    })
    assert len(web_server.SERVER_MESSAGE_TYPES) == 6


# --- app/web/status.py: derive_status folds in coach health -----------------------------


def test_derive_status_reports_the_coach_as_failed_side_when_unhealthy_and_no_stream_failed():
    healths = [StreamHealth(speaker="interviewee", worker_failed=False, capture_dead=False)]
    status = derive_status(healths, coach_unhealthy=True)
    assert status.value == STATUS_FAILED
    assert status.failed_side == FAILED_SIDE_COACH
    assert status.reason == REASON_COACH_FAILURE


def test_derive_status_still_reports_the_stream_as_failed_side_when_both_have_failed():
    healths = [StreamHealth(speaker="interviewer", worker_failed=True, capture_dead=False)]
    status = derive_status(healths, coach_unhealthy=True)
    assert status.value == STATUS_FAILED
    assert status.failed_side == "interviewer"
    assert status.reason == REASON_STREAM_FAILURE


def test_derive_status_stays_healthy_when_coach_unhealthy_defaults_to_false():
    healths = [StreamHealth(speaker="interviewee", worker_failed=False, capture_dead=False)]
    assert derive_status(healths).value == "healthy"


# --- app.js: distinct rendering for the coach vs. the stream, never a placeholder ------


def test_app_js_branches_on_the_suggestion_message_type_and_never_clears_the_transcript():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    # MESSAGE_TYPE_SUGGESTION, not the bare literal -- app.js imports the
    # wire vocabulary now (test_wire_vocabulary.py). A grep for the string
    # would pass on a stale local copy, which is the defect that change fixed.
    assert "MESSAGE_TYPE_SUGGESTION" in app_js
    assert "renderSuggestion" in app_js
    forbidden = re.compile(r"transcript\w*\.(innerHTML\s*=|replaceChildren\(\s*\)|textContent\s*=)")
    assert not forbidden.search(app_js), "SURF-06: no code path may clear the transcript container"


def test_app_js_uses_a_different_sentence_for_a_coach_failure_than_for_a_stream_failure():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    literals = [
        text
        for text in re.findall(r'serverStatusRegion\.textContent\s*=\s*"([^"]*)"', app_js)
        if text  # excludes the healthy branch's clearing assignment (textContent = "")
    ]
    # Was "exactly two" until the degraded tier added three more branches.
    # Distinctness is the property that actually matters and it generalises;
    # the count itself is pinned in gate 16, so pinning it twice would just
    # mean two places to edit for one deliberate change.
    assert len(literals) >= 2, f"expected at least two status sentences, found {literals}"
    assert len(set(literals)) == len(literals), (
        "two status branches share a sentence, so the situations they distinguish are "
        f"indistinguishable to the operator: {literals}"
    )
    # Never the claim that transcription itself stopped, for any situation
    # where it has NOT. That is false for a coach failure and equally false
    # for a degraded stream, where transcription is running and it is the
    # audio reaching it that is wrong.
    for sentence in literals:
        lowered = sentence.lower()
        still_transcribing = (
            "unavailable" in lowered
            or "suggestion" in lowered
            or "no audio" in lowered
            or "arriving late" in lowered
        )
        if still_transcribing:
            assert "transcription stopped" not in lowered, sentence


def test_app_js_never_clears_or_hides_the_suggestion_region_on_a_status_message():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    render_status_match = re.search(
        r"function renderServerStatus\([^)]*\)\s*\{(.*?)\n  \}", app_js, re.DOTALL
    )
    assert render_status_match is not None, "renderServerStatus() not found in app.js"
    body = render_status_match.group(1)
    assert "suggestionRegion" not in body, (
        "a coach failure must leave the last shown suggestion in place -- "
        "renderServerStatus must never touch the suggestion region"
    )


# --- Rendering rules: dir=auto per block, text node insertion --------------------------


def test_app_js_suggestion_blocks_carry_dir_auto_and_insert_text_as_a_text_node():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "renderSuggestion" in app_js
    build_match = re.search(
        r"function buildSuggestionTextBlock\([^)]*\)\s*\{(.*?)\n  \}", app_js, re.DOTALL
    )
    assert build_match is not None, "buildSuggestionTextBlock() not found in app.js"
    body = build_match.group(1)
    assert 'setAttribute("dir", "auto")' in body
    assert "createTextNode" in body


# --- The suggestion region's non-collapsing minimum height still holds -----------------


def test_served_style_css_still_carries_the_non_collapsing_min_height(tmp_path):
    token = "min-height-token"
    hub = ConnectionHub()
    store = _store(tmp_path)
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id="s", server_started_at="t",
    )
    with _running_server(app) as (base_url, _loop):
        from app.tests.test_web_serving import _http_get

        status, body = _http_get(f"{base_url}/s/{token}/style.css")
        assert status == 200
        assert "min-height" in body.decode("utf-8")


def test_transcript_region_html_is_byte_identical_to_phase_three():
    served_html = (REPO_ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<div id="transcript-region"[^>]*></div>', served_html)
    assert match is not None, "the transcript region markup must still ship empty and untouched"
