"""The ASGI app factory and the uvicorn embedding (03-01-PLAN.md §P2, §P7;
03-CONTEXT.md D-05). The one invariant this module exists to protect: the
asyncio event loop hosting uvicorn runs on the MAIN thread -- capture and STT
stay exactly where they already are, as plain background OS threads
(03-RESEARCH.md §1). Since uvicorn 0.13.0, `Server.install_signal_handlers()`
is a no-op off the main thread (Python itself only allows signal handlers on
the main thread of the main interpreter) -- so if this module's server were
ever run from a background thread, Ctrl-C would silently stop being handled
by it. `run_start` in `app/main.py` is the only caller that constructs and
drives the server returned here, and it does so via its own top-level
`asyncio.run()`, never a nested one.

Every route lives beneath `/s/<token>` (D-05) -- there is no unauthenticated
static mount. A wrong or absent token gets a generic 404 that echoes nothing,
compared in constant time via `secrets.compare_digest` (never `==`, which
leaks a timing side-channel for free). The four static assets are read from
disk exactly ONCE, at factory time, into memory -- a synchronous file read
inside an `async def` handler would stall the entire event loop and therefore
freeze every connected client at once (03-RESEARCH.md §1 Pitfall 1).

03-02-PLAN.md Task 1: `ACCEPTED_CLIENT_MESSAGE_TYPES` and `SERVER_MESSAGE_TYPES`
are the whole protocol surface, pinned as module-level constants and asserted
equal by set equality in `app/tests/test_web_reconnect.py` -- this is what
permanently forbids a resume/backfill request (D-01), a client-reported gap
(D-02), and a two-way control channel (deferred per 03-CONTEXT.md) from ever
being silently reintroduced. The heartbeat (`ping`/`pong`) is the ONLY
client->server message this protocol has, or is ever meant to have.
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response

from app import config as config_mod
from app.session.store import SessionStore
from app.web.hub import ConnectionHub, hello_payload
from app.web.status import StatusBoard, status_message

logger = logging.getLogger(__name__)

_NOT_FOUND = PlainTextResponse("Not Found", status_code=404)

# WebSocket close code used to reject a bad token BEFORE ever accepting the
# connection -- so a rejected socket never enters the hub's active set
# (D-05). 4404 is in the app-defined range (4000-4999) per RFC 6455 §7.4.2.
_WS_CLOSE_BAD_TOKEN = 4404

# The message type literals, defined ONCE here and imported by
# `app/static/live-state.mjs`'s test (`app/tests/test_web_reconnect.py`
# imports both and asserts the literal that both sides consume comes from
# one place, never typed twice and hoped about).
MESSAGE_TYPE_HELLO = "hello"
MESSAGE_TYPE_PING = "ping"
MESSAGE_TYPE_PONG = "pong"
MESSAGE_TYPE_SEGMENT = "segment"
MESSAGE_TYPE_SERVER_STATUS = "server_status"
# 04-03-PLAN.md §E1: the protocol grows by exactly one message, deliberately.
# A ranked suggestion set (app/coach/pipeline.py's on_ranked callback, wired
# in app/main.py) reaches the phone through this ONE new type. Adding it here
# without also updating SERVER_MESSAGE_TYPES below AND gate 8's own assertion
# in the SAME commit is the intended friction (§E1) -- the protocol surface
# grows by decision, never by drift.
MESSAGE_TYPE_SUGGESTION = "suggestion"
# Phase C: the protocol grows by exactly one more message, deliberately, and
# under the same friction as the line above -- adding it here without also
# updating SERVER_MESSAGE_TYPES, gate 8's assertion, and the two other
# set-equality assertions that pin this surface (test_web_reconnect.py,
# test_web_suggestion.py) in the SAME commit fails the suite.
#
# This is the one server->client message that carries no model output. It is
# computed from the authored survey plus coverage state, so it is the only
# thing the surface can show that cannot be delayed or emptied by a slow or
# failing vendor call.
MESSAGE_TYPE_SURVEY = "survey"

# The ENTIRE client->server protocol surface (D-01/D-02/03-CONTEXT.md's
# deferred two-way control channel). Asserted as SET EQUALITY, not a
# "contains" check and not a negative grep for forbidden vocabulary --
# equality is what catches a field nobody thought to forbid, and cannot be
# invalidated by a comment the way a negative grep can (T-03-14, T-03-15).
#
# 04-03-PLAN.md §E1: deliberately UNTOUCHED by the suggestion message above.
# D-01's no-replay rule and the deferred two-way control channel (dismiss/pin
# a suggestion from the phone) both depend on this set staying exactly the
# heartbeat -- adding a suggestion type is a server->client change ONLY.
ACCEPTED_CLIENT_MESSAGE_TYPES = frozenset({MESSAGE_TYPE_PING})

# The ENTIRE server->client protocol surface.
SERVER_MESSAGE_TYPES = frozenset(
    {
        MESSAGE_TYPE_HELLO,
        MESSAGE_TYPE_PONG,
        MESSAGE_TYPE_SEGMENT,
        MESSAGE_TYPE_SERVER_STATUS,
        MESSAGE_TYPE_SUGGESTION,
        MESSAGE_TYPE_SURVEY,
    }
)


def _parse_client_message(raw: str) -> Optional[dict[str, Any]]:
    """Returns the parsed message dict if -- and only if -- `raw` is valid
    JSON, is a JSON object, and its `type` is one of
    `ACCEPTED_CLIENT_MESSAGE_TYPES`. Returns `None` for anything else:
    malformed JSON, a non-object payload, or an unrecognized `type` -- all
    ignored without raising and without ever logging the content (S7,
    T-03-14)."""
    try:
        message = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(message, dict):
        return None
    if message.get("type") not in ACCEPTED_CLIENT_MESSAGE_TYPES:
        return None
    return message


class PortInUseError(Exception):
    """Raised when the pinned port is already bound by another process. The
    message names the port and how to change it -- never a silent fallback to
    a different port, which would desynchronize the printed/QR'd join URL
    from what is actually listening (03-RESEARCH.md §4, OPS-05)."""


def check_port_available(port: int, *, host: str = "0.0.0.0") -> None:
    """Raises `PortInUseError` if `port` cannot be bound on `host` right now.
    Called before `uvicorn.Config`/`uvicorn.Server` are constructed, so a
    taken port is a typed, actionable error rather than something the caller
    has to parse out of an uvicorn log line."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise PortInUseError(
            f"port {port} is already in use -- is another session already running? "
            f"Set APP_PORT to use a different one."
        ) from exc
    finally:
        probe.close()


@dataclass(frozen=True)
class _StaticAssets:
    index_html: bytes
    style_css: bytes
    app_js: bytes
    live_state_mjs: bytes


def _load_static_assets(static_dir: Path) -> _StaticAssets:
    """Reads the four static assets from disk exactly once. No request
    handler below ever touches the filesystem -- every response below is
    served from these in-memory bytes."""
    return _StaticAssets(
        index_html=(static_dir / "index.html").read_bytes(),
        style_css=(static_dir / "style.css").read_bytes(),
        app_js=(static_dir / "app.js").read_bytes(),
        live_state_mjs=(static_dir / "live-state.mjs").read_bytes(),
    )


def create_app(
    *,
    token: str,
    hub: ConnectionHub,
    static_dir: Path,
    store: SessionStore,
    session_id: str,
    server_started_at: str,
    status_board: Optional["StatusBoard"] = None,
    survey_provider: Optional["Callable[[], Optional[dict[str, Any]]]"] = None,
    project: Optional[str] = None,
    grounding_sources: "Sequence[str]" = (),
    coach_enabled: bool = False,
    heartbeat_interval_ms: int = config_mod.DEFAULT_HEARTBEAT_INTERVAL_MS,
    heartbeat_timeout_ms: int = config_mod.DEFAULT_HEARTBEAT_TIMEOUT_MS,
    reconnect_backoff_base_ms: int = config_mod.DEFAULT_RECONNECT_BACKOFF_BASE_MS,
    reconnect_backoff_cap_ms: int = config_mod.DEFAULT_RECONNECT_BACKOFF_CAP_MS,
) -> FastAPI:
    """Builds the ASGI app. Registers exactly five routes, all beneath the
    per-run token: the page, the three assets, and the socket. No other route
    ever serves anything -- a catch-all falls through to a generic 404 that
    echoes nothing (D-05).

    `status_board` defaults to a fresh, healthy `StatusBoard` when omitted
    (e.g. in tests that don't exercise 03-02's server-status path);
    `app/main.py`'s real call site passes the ONE board instance its status
    poller also writes to, so a fresh connect and a poll both read the same
    "last known status" (§Q6/§Q7).

    `coach_enabled` says whether a coach ACTUALLY came up on this run -- not
    whether one was asked for. It defaults to `False`, which is the honest
    default for a server built without one, and it is the value a client uses
    to decide whether to promise suggestions at all. The live caller cannot
    forget it: gate 31 pins `app/main.py`'s call keyword set by equality
    against this signature, which is why a default here is safe where the
    same default on `hub.hello_payload` would not be -- that function is the
    wire projection itself, and a wrong value there renders a wrong panel with
    nothing downstream to catch it.

    `survey_provider` (Phase C) is called on every fresh connect and returns
    an already-projected `survey` wire payload, or `None` when there is no
    survey to report (`--no-serve` never reaches here; a key-less run has no
    coach and therefore no coverage state). It returns the PAYLOAD rather
    than the survey and coverage state themselves so this module keeps
    importing nothing from `app.coach` -- `app/web/hub.py` stays the only
    place that projects anything onto the wire, and `app/main.py` stays the
    single injection point. Pushed alongside `hello` for the same reason the
    server status is: a browser that joins mid-interview, or reconnects after
    a sleep, must be correct immediately rather than at the next coverage
    change (which may be minutes away, or never)."""
    assets = _load_static_assets(static_dir)
    board = status_board if status_board is not None else StatusBoard()
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    def _token_matches(candidate: str) -> bool:
        # `secrets.compare_digest` on two `str` requires ASCII on both
        # sides -- Starlette percent-decodes path params, so a non-ASCII
        # candidate (Hebrew is plausible on this project, D-04) would
        # otherwise raise `TypeError` here, producing an HTTP 500 (a
        # observable deviation from D-05's uniform-404 posture) instead of
        # the same generic rejection a wrong-but-ASCII token gets. A
        # non-ASCII candidate can never equal the token (tokens are
        # generated from an ASCII alphabet, D-05), so short-circuiting to
        # "no match" here is correct, not just safe.
        if not candidate.isascii():
            return False
        return secrets.compare_digest(candidate, token)

    @app.get("/s/{candidate_token}")
    async def get_page(candidate_token: str) -> Response:
        if not _token_matches(candidate_token):
            return _NOT_FOUND
        return Response(
            content=assets.index_html,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/s/{candidate_token}/app.js")
    async def get_app_js(candidate_token: str) -> Response:
        if not _token_matches(candidate_token):
            return _NOT_FOUND
        return Response(
            content=assets.app_js,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/s/{candidate_token}/style.css")
    async def get_style_css(candidate_token: str) -> Response:
        if not _token_matches(candidate_token):
            return _NOT_FOUND
        return Response(
            content=assets.style_css,
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/s/{candidate_token}/live-state.mjs")
    async def get_live_state_mjs(candidate_token: str) -> Response:
        if not _token_matches(candidate_token):
            return _NOT_FOUND
        return Response(
            content=assets.live_state_mjs,
            media_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/s/{candidate_token}/ws")
    async def websocket_endpoint(websocket: WebSocket, candidate_token: str) -> None:
        if not _token_matches(candidate_token):
            # Never accepted -- rejected before ever entering the hub's active
            # set (D-05). Sending "websocket.close" before "websocket.accept"
            # refuses the handshake at the ASGI level.
            await websocket.close(code=_WS_CLOSE_BAD_TOKEN)
            return
        await hub.connect(websocket)
        disconnect_reason = "unknown"
        try:
            await websocket.send_json(
                hello_payload(
                    session_id=session_id,
                    server_started_at=server_started_at,
                    heartbeat_interval_ms=heartbeat_interval_ms,
                    heartbeat_timeout_ms=heartbeat_timeout_ms,
                    reconnect_backoff_base_ms=reconnect_backoff_base_ms,
                    reconnect_backoff_cap_ms=reconnect_backoff_cap_ms,
                    project=project,
                    grounding_sources=grounding_sources,
                    coach_enabled=coach_enabled,
                )
            )
            # Pushed on EVERY fresh connect, alongside hello -- a laptop-side
            # failure survives a reconnect even though transcript content
            # deliberately does not (§Q6 versus D-01: this is "whatever
            # arrives next", not a replay of missed transcript).
            await websocket.send_json(status_message(board.current))
            # Phase C: pushed on every fresh connect too, for the same reason
            # the status above is -- but guarded, because a provider that
            # raises must not take the whole connection down with it. The
            # survey panel going momentarily unpopulated is a far smaller
            # failure than a browser that cannot connect at all, and this
            # message exists precisely to be the thing that still works when
            # other things do not.
            if survey_provider is not None:
                try:
                    survey = survey_provider()
                except Exception as exc:  # noqa: BLE001 -- never fail a connect over the survey panel
                    logger.warning("survey provider failed error_type=%s", type(exc).__name__)
                    survey = None
                if survey is not None:
                    await websocket.send_json(survey)
            while True:
                raw = await websocket.receive_text()
                message = _parse_client_message(raw)
                if message is None:
                    # Unaccepted type or malformed payload: ignored without
                    # raising, without closing the socket, and without ever
                    # logging its content (S7, T-03-14).
                    continue
                if message.get("type") == MESSAGE_TYPE_PING:
                    await websocket.send_json({"type": MESSAGE_TYPE_PONG})
        except WebSocketDisconnect as exc:
            # 03-RESEARCH Open Question 2: recorded so the server's own view
            # of a disconnect can be compared against the client's own
            # heartbeat-timeout log during Task 3's real-phone rehearsal.
            disconnect_reason = f"websocket-close-code-{exc.code}"
        finally:
            await hub.disconnect(websocket)
            store.append_event(
                event="ws-disconnected",
                detail={"reason": disconnect_reason, "connection_count": hub.connection_count},
            )

    @app.get("/{_catch_all:path}")
    async def catch_all(_catch_all: str) -> Response:
        # Covers "/", "/s/" (empty token segment), and anything else not
        # matched above -- always a generic 404, nothing ever echoed back.
        return _NOT_FOUND

    return app


def build_uvicorn_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    ws_ping_interval_s: float = config_mod.DEFAULT_WS_PING_INTERVAL_S,
    ws_ping_timeout_s: float = config_mod.DEFAULT_WS_PING_TIMEOUT_S,
) -> "object":
    """Constructs `uvicorn.Config` + `uvicorn.Server` -- never `uvicorn.run()`,
    which calls `asyncio.run()` internally and cannot be nested inside this
    process's own top-level `asyncio.run()` (03-RESEARCH.md §1). Deferred
    import: `uvicorn` is only needed here, at the one call site that embeds
    it, keeping every other module in this package importable without it.

    `ws_ping_interval_s`/`ws_ping_timeout_s` (independent review finding B)
    are passed EXPLICITLY, sourced from `app/config.py`'s own
    `DEFAULT_WS_PING_*` knobs -- never left to ride on `websockets`'
    undocumented default. This is the only bound on
    `app/web/hub.py::ConnectionHub._broadcast_payload`'s sequential fan-out:
    a half-open phone whose write buffer fills parks `send_json()` on
    backpressure rather than raising, and without an explicit ping/pong
    timeout a future dependency bump could silently make that starvation
    indefinite instead of bounded."""
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        ws_ping_interval=ws_ping_interval_s,
        ws_ping_timeout=ws_ping_timeout_s,
    )
    return uvicorn.Server(config)
