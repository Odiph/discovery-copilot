"""JSON-RPC 2.0 over stdio, hand-rolled against the standard library.

**Why hand-rolled.** `requirements.txt` pins exactly five packages beyond
Phase 1's, and gate 3 asserts that count by equality so a sixth cannot arrive
by drift. MCP's stdio transport does not need one: it is newline-delimited
JSON-RPC 2.0, which is `json` plus a loop.

**The framing rules, stated once here because getting them subtly wrong is
the classic failure of a server whose handlers are all correct:**

- One JSON object per line on stdout, UTF-8, terminated by `\\n`.
- A message may NEVER contain an embedded newline -- `json.dumps` escapes
  them inside strings by default, which is why the encoder below must not be
  "improved" with `indent=`.
- A request carries an `id` and gets exactly one response. A NOTIFICATION
  carries no `id` and gets NO response at all -- answering one desynchronizes
  a strict client, which is why `notifications/initialized` is consumed
  silently.
- A line that cannot be parsed is answered with `id: null` and continues the
  loop. A server that exits on a bad line takes the operator's whole agent
  session down with it.

**stdout belongs to the protocol, absolutely.** Nothing in this package's
import graph may `print`. Diagnostics go to stderr, which is why
`__main__.py` configures logging there and never here.

**Read-only.** See `app/mcp/__init__.py` for why that is a decision (gate 19)
rather than a first cut, and for the privacy boundary this server enforces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO, Optional

from app.mcp import tools as tools_mod

SERVER_NAME = "discovery-copilot"
SERVER_VERSION = "0.1.0"

# The protocol revision this server implements. A client that asks for a
# different one is answered with THIS value rather than an error -- the spec's
# own negotiation: the server states what it supports and the client decides
# whether it can live with it.
PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _response(request_id: Any, result: "dict[str, Any]") -> "dict[str, Any]":
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> "dict[str, Any]":
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _initialize(params: "dict[str, Any]") -> "dict[str, Any]":
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "instructions": (
            "Read-only access to this machine's interview discovery corpus. Start with "
            "list_projects. No verbatim customer speech is exposed by any tool, by design: "
            "facts carry an abstracted claim, never the quote behind it, and no transcript "
            "text or audio is reachable here."
        ),
    }


def _tools_call(params: "dict[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    name = params.get("name")
    if not isinstance(name, str):
        raise tools_mod.ToolArgumentError(
            f"'name' is required and must be a string, got {type(name).__name__}"
        )
    payload = tools_mod.call_tool(name, params.get("arguments"), repo_root=repo_root)
    # MCP's tool result envelope. The document is JSON inside a text content
    # block -- the shape every client renders -- rather than a bare object.
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": False,
    }


def handle_message(message: Any, *, repo_root: "Optional[Path]") -> "Optional[dict[str, Any]]":
    """One decoded message in, at most one response out. `None` means "this
    was a notification, say nothing" -- never "something went wrong"."""
    if not isinstance(message, dict):
        return _error(None, INVALID_REQUEST, "a JSON-RPC message must be an object")

    request_id = message.get("id")
    is_notification = "id" not in message
    method = message.get("method")
    params = message.get("params")
    if not isinstance(params, dict):
        params = {}

    if not isinstance(method, str):
        return None if is_notification else _error(
            request_id, INVALID_REQUEST, "'method' is required and must be a string"
        )

    try:
        if method == "initialize":
            result = _initialize(params)
        elif method == "tools/list":
            result = {"tools": list(tools_mod.TOOL_DEFINITIONS)}
        elif method == "tools/call":
            result = _tools_call(params, repo_root=repo_root)
        elif method == "ping":
            result = {}
        elif method.startswith("notifications/"):
            # Consumed silently. `notifications/initialized` is the one every
            # client sends; anything else in that namespace is equally
            # answer-free by definition.
            return None
        else:
            return None if is_notification else _error(
                request_id, METHOD_NOT_FOUND, f"unknown method {method!r}"
            )
    except tools_mod.ToolArgumentError as exc:
        return None if is_notification else _error(request_id, INVALID_PARAMS, str(exc))
    except Exception as exc:  # noqa: BLE001 -- an unexpected failure is an ANSWER, not an exit
        # Only the exception TYPE, for the reason `tools.py` states: a
        # message may carry the offending record, and the offending record
        # may carry a customer's words.
        return None if is_notification else _error(
            request_id, INTERNAL_ERROR, f"{method} failed ({type(exc).__name__})"
        )

    return None if is_notification else _response(request_id, result)


def serve(stdin: BinaryIO, stdout: BinaryIO, *, repo_root: "Optional[Path]" = None) -> None:
    """Reads newline-delimited JSON-RPC messages until EOF.

    Binary streams on both sides, deliberately: the framing is a byte-level
    contract (UTF-8, one `\\n` per message) and a text wrapper would hand the
    platform's own newline translation a say in it -- on Windows a text
    stdout turns every `\\n` into `\\r\\n`, which is tolerable for this
    transport but is not a thing to leave to chance on a protocol stream."""
    for raw_line in stdin:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            # `id` is null: the id is unknowable when the line did not parse.
            # The message names the failure and never echoes the line, which
            # could be anything at all.
            response = _error(None, PARSE_ERROR, f"line is not valid JSON ({type(exc).__name__})")
        else:
            response = handle_message(message, repo_root=repo_root)
        if response is None:
            continue
        stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        stdout.flush()
