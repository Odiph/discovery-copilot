"""Recording WHO was interviewed, after the interview -- and the ONLY module
in this codebase that puts bytes into a finished session's `session.json`.

**Why this is its own file rather than a few lines added to the hub.**
`app/web/review.py`'s docstring states, and its structure enforces, that the
hub itself stays read-only: the accept/reject card lives in
`app/web/accept.py` and the hypothesis editor in `app/web/author.py`, both
mounted onto the hub's app rather than implemented inside it. The reason there
is gate 19 (`app/tests/test_phase_gates.py`), which pins by set equality the
modules that may write the hypothesis artifact; the reason HERE is the same
shape of argument without a gate behind it yet. A session directory is the one
artifact in this product that cannot be regenerated -- the interview is over
and the audio is usually gone -- so the set of modules that can write into one
should stay small enough to read in a sitting, and small enough that a future
gate can pin it.

**And this module deliberately does not NAME that other artifact.** Gate 19
matches two regexes PER FILE, not per statement: any module that both spells
the hypothesis file's name and contains a write primitive joins the pinned set.
This module contains a write primitive and has nothing whatever to do with
hypotheses, so spelling the filename in this very paragraph -- in prose, in a
comparison, for a reader's benefit -- was enough to trip the gate on the first
run. That is the gate working, not the gate being wrong: a textual check is
what makes it impossible to sneak past, and the cost of that strength is that
prose has to route around it. Hence "the hypothesis artifact" above.

**It does NOT go through `SessionStore`, and that is the whole trap.**
`SessionStore(session_dir)` on a directory that already holds a transcript is a
RESUME by construction: it recovers the sequence counter and APPENDS a
`session-resumed` event to `transcript.jsonl`. Naming the person who was in the
room is not a resumption of the interview, and a label typed on the hub three
days later must not put a record into the transcript claiming the session
started again. `SessionStore.update_metadata` is exactly the method whose name
suggests it is the right tool here, which is why the trap is worth naming: the
event is written by the CONSTRUCTOR, before `update_metadata` is ever called,
so a reviewer checking `update_metadata` itself would find nothing wrong.

**Read strictly, then write** -- `app/web/accept.py`'s rule, applied to a
different artifact for the same reason. The merge reads the existing
`session.json` with `json.loads` and REFUSES on anything it cannot parse. The
tempting tolerant read (treat unparseable as `{}`) would take a metadata file
holding the engine, the device names, the sample rates, the stored survey and
the latency summary, and replace all of it with a single `participant` key. A
file we cannot read is a file we refuse to replace.

**Loopback-only, like every other write route here** -- and the POST checks the
Host header ITSELF rather than trusting whichever app it is mounted on to have
installed the hub's middleware, exactly as `app/web/accept.py::post_decision`
does and for the same reason: a write must not depend on a caller having
remembered.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import projects as projects_mod
from app.session.participant import PARTICIPANT_KEY, normalize_participant, read_participant
from app.session.reader import list_session_ids
from app.session.store import METADATA_FILENAME
from app.web.author import _FORBIDDEN, _atomic_write_json, _host_header_is_loopback

logger = logging.getLogger(__name__)

PARTICIPANT_ROUTE = "/api/participant"


def save_participant(
    sessions_root: Path, session_id: "Optional[str]", label: Any
) -> "tuple[Optional[str], Optional[str]]":
    """Records (or clears) one session's participant label. Returns
    `(stored_label, None)` or `(None, error)`; on any error NOTHING is written,
    because every check completes before the atomic write is reached -- the
    "validated, then written, never the other order" discipline `author.py` and
    `accept.py` already apply to the context artifacts.

    `stored_label` is `None` on a successful CLEAR as well as on an error, so
    callers must branch on the error, never on the label. That asymmetry is
    deliberate and is the same one `normalize_participant` documents: "nobody
    is named" is a legitimate stored state, not a failure.

    **The session id is checked by MEMBERSHIP of the real listing, never by
    pattern.** This value arrives from a browser -- the hub's picker puts it
    there -- and `sessions_root / requested` with a `..` in it resolves outside
    the project entirely. An allowlist built from what is actually on disk
    cannot be talked round by traversal, by a symlink or by a case difference,
    and it needs no separate argument about which characters are dangerous.
    This route re-does that check rather than trusting the hub to have done it,
    for the same reason the loopback guard is re-done below.

    Every other key in `session.json` survives, byte for byte in value if not
    in formatting: the engine, both device names, both native sample rates, the
    stored `context_survey` the coverage report replays against, and the
    latency summary. A write that dropped any of them would destroy provenance
    that cannot be recomputed, in exchange for a name.
    """
    root = Path(sessions_root)
    if session_id is None:
        # Not an error the operator can act on by retrying -- there is simply
        # no interview on screen to attach a name to. Said plainly rather than
        # dressed as a validation failure.
        return None, "no interview is selected, so there is nothing to record a participant for"
    if session_id not in list_session_ids(root):
        # Deliberately does not echo the requested value back into the message.
        # It is attacker-controlled text heading for a page that renders it,
        # and the operator does not need it repeated to understand what
        # happened. `app/web/review.py::resolve_session` words it identically.
        return None, "that session is not one of this project's interviews"

    normalized, error = normalize_participant(label)
    if error is not None:
        return None, error

    metadata_path = root / session_id / METADATA_FILENAME
    existing: "dict[str, Any]" = {}
    if metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning(
                "refusing to replace an unreadable session.json session=%s error_type=%s",
                session_id, type(exc).__name__,
            )
            return None, (
                "this session's metadata could not be read, so it will not be replaced -- "
                "recording a name must never cost an interview its provenance"
            )
        if not isinstance(existing, dict):
            return None, (
                "this session's metadata is not a JSON object, so it will not be replaced"
            )

    # A COPY, then a merge. Mutating `existing` in place would be harmless here
    # (it is a local parse result) and is avoided anyway, because the next
    # reader of this function should not have to work that out to be sure the
    # caller's data was not altered underneath them.
    merged = dict(existing)
    if normalized is None:
        # CLEARED, and the key is removed rather than set to `null`. See
        # `app/session/participant.py`: a session recorded before this field
        # existed and a session someone deliberately blanked are the same
        # situation -- nobody knows who it was -- and a stored `null` would
        # assert a distinction no byte on disk supports.
        merged.pop(PARTICIPANT_KEY, None)
    else:
        merged[PARTICIPANT_KEY] = normalized
    _atomic_write_json(metadata_path, merged)
    return normalized, None


def register_participant_routes(
    app: FastAPI,
    *,
    project: str,
    session_id: "Optional[str]",
    repo_root: "Optional[Path]" = None,
) -> None:
    """Mounts the participant GET+POST onto an existing app -- the hub's, so
    the name and the evidence it belongs to are one sitting on one port. A
    second server would force a cross-origin fetch and a CORS surface for the
    sake of one short string, which is a worse trade than sharing the page the
    reader is already looking at.

    The GET exists even though `/api/review` already carries the label, and the
    duplication is deliberate: `/api/review` reads the corpus, the hypotheses,
    the coverage and the transcript, any of which can be slow or can degrade,
    and a caller that only wants to know who was interviewed should not have to
    survive all of that. It is the same argument `/api/sessions` already makes
    for being served separately from `/api/review`.
    """

    def _selected(session: "Optional[str]") -> "Optional[str]":
        return session_id if session is None else session

    @app.get(PARTICIPANT_ROUTE)
    async def get_participant(session: "Optional[str]" = None) -> JSONResponse:
        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        chosen = _selected(session)
        if chosen is not None and chosen not in list_session_ids(paths.sessions_root):
            return JSONResponse(
                {"error": "that session is not one of this project's interviews"},
                status_code=400,
            )
        # `participant: null` on the wire is the COMMON answer, and it is
        # returned as a successful read rather than as a 404 or an error: no
        # recording made before this field existed carries a label, so
        # "nobody is named" must be an ordinary, unalarming reply.
        return JSONResponse({
            "session_id": chosen,
            "participant": (
                None if chosen is None else read_participant(paths.sessions_root / chosen)
            ),
        })

    @app.post(PARTICIPANT_ROUTE)
    async def post_participant(request: Request):
        # This route's OWN guard, not the mounting app's. The hub installs the
        # same check in middleware, but a write must not depend on a caller
        # having remembered to.
        if not _host_header_is_loopback(request.headers.get("host", "")):
            await request.body()
            return _FORBIDDEN
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 -- malformed JSON is a 400, never a 500
            return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": "the participant body must be a JSON object"}, status_code=400
            )
        try:
            paths = projects_mod.resolve(project, repo_root=repo_root)
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        chosen = _selected(body.get("session"))
        # `body.get(PARTICIPANT_KEY)` with NO default, so an absent key and an
        # explicit `null` both arrive as `None` and both mean "clear it". A
        # client that omits the field is not silently treated as having sent
        # something else.
        stored, error = save_participant(
            paths.sessions_root, chosen, body.get(PARTICIPANT_KEY)
        )
        if error is not None:
            return JSONResponse({"error": error}, status_code=400)
        # Never logs the label itself. It is the name of a real person who sat
        # for an interview, and this repo already keeps transcript text out of
        # every log line for the same reason (S7, T-02-05); a name is not a
        # softer case of that rule.
        logger.info(
            "participant label saved session=%s cleared=%s", chosen, stored is None
        )
        return JSONResponse({"saved": True, "session_id": chosen, "participant": stored})
