"""The loopback-only authoring surface for the brief and survey
(04-03-PLAN.md §E4; 04-CONTEXT.md D-07).

A genuinely SEPARATE FastAPI app instance from `app/web/server.py`'s phone
surface -- reusing the PATTERN (FastAPI, static assets read once from disk
at factory time, documentation endpoints disabled), never the INSTANCE.
`04-RESEARCH.md`'s Pitfall 3 names the failure this avoids: adding authoring
routes to `create_app()` would inherit that app's `0.0.0.0` binding and put a
named customer's brief on the LAN behind nothing.

Deliberate divergences from the phone surface, stated here because §E4 calls
them out as the parts that must not be re-derived:

- **No join token.** A network-layer token adds no real protection for
  content already readable by anything running as the same OS user, and
  reusing the phone surface's token model here would blur the ONE invariant
  gate 9 protects (every route on the LAN-REACHABLE app lives beneath the
  token). This app is never LAN-reachable at all -- see the loopback-only
  bind in `build_uvicorn_server()` below -- so no exception is carved into
  gate 9; its routes are simply never on the phone app's route table at all
  (`app/tests/test_web_author.py` asserts this explicitly, alongside gate 9
  passing unmodified).
- **Loopback binding ONLY**, hardcoded in `build_uvicorn_server()` below,
  never a parameter a caller could widen to the wildcard address the phone
  surface deliberately uses. Gate 14 (`app/tests/test_phase_gates.py`)
  asserts this on the CONSTRUCTED server object's own configuration, not on
  a constant that could silently drift from what actually gets bound.
- **Runs its server directly** -- `app/main.py`'s `run_author` calls
  `asyncio.run(server.serve())` with nothing else sharing that loop. This
  command shares its process with no capture or STT threads, so it needs
  none of `start`'s careful main-thread event-loop arrangement; see
  `app/web/server.py`'s own module docstring for why `start` needs that
  dance and why this command does not.

The three static assets (`index.html`/`author.js`/`author.css`) are read
from disk once, at factory time, into memory -- the same discipline
`app/web/server.py::_load_static_assets` uses, for the same reason (no
request handler may block the event loop on a synchronous file read). The
save endpoints are the one deliberate exception to "no route performs file
I/O on the request path": writing the brief/survey IS their entire job, and
they do so atomically (temp file in the same directory, flush, fsync,
`os.replace`) so a partially written file is never a failure state a user
one minute from a call has to debug.

Save endpoints validate against `app/coach/context.py`'s OWN `ContextItem`/
`SurveyItem` schemas -- imported, never redefined, so the two halves can
never drift. A malformed item is rejected by naming its id, with NOTHING
written to disk. A `SurveyItem`'s `order` is always recomputed as the item's
position in the posted array, ignoring whatever value the client sent -- so
`order` can never drift from "the order the user actually arranged them in"
(Claude's discretion, recorded here per 04-CONTEXT.md).

**A third artifact, added in 05-03-PLAN.md §K13/§K14: hypotheses.** Same
page, same app instance, same guard, same atomic write -- the `/api/hypotheses`
GET+POST pair validates against `app/coach/corpus.py`'s OWN `Hypothesis`
model (imported, never redefined, for the identical reason the brief/survey
schemas are imported above). A hypothesis without a falsification condition
-- absent or whitespace-only -- is rejected by naming its id, with NOTHING
written to disk (D-04): the same "validated, then written, never the other
order" discipline this module already applies to the brief and survey.

**And this module STAMPS the criterion's pre-registration boundary.** A
`Hypothesis.criterion` claims a decision rule was fixed before the
interviewing; nothing in the text can back that up, so the hypotheses save
path records `criterion_registered_after` -- the session ids that already
existed at this node when the criterion was written -- always recomputed,
never taken from the request, exactly as `SurveyItem.order` already is. A
value the user could type would be the same unbacked claim with a credential
attached. `_stamp_pre_registration_boundaries` below is that rule and argues
when the boundary moves; `Hypothesis.criterion_registered_after` argues the
field's shape.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app.coach.context import BRIEF_FILENAME, SURVEY_FILENAME, ContextItem, SurveyItem
from app.coach.corpus import HYPOTHESES_FILENAME, Hypothesis
from app.projects import SESSIONS_DIR_NAME

# Hardcoded, not a parameter -- see module docstring and gate 14. A future
# convenience edit to "make it reachable from the tablet" is exactly how
# this gets undone, and hardcoding here (rather than accepting a `host`
# kwarg) is what makes that edit visible in a diff of THIS line specifically.
_AUTHOR_HOST = "127.0.0.1"


@dataclass(frozen=True)
class _AuthorStaticAssets:
    index_html: bytes
    author_js: bytes
    author_css: bytes


def _load_static_assets(static_dir: Path) -> _AuthorStaticAssets:
    return _AuthorStaticAssets(
        index_html=(static_dir / "index.html").read_bytes(),
        author_js=(static_dir / "author.js").read_bytes(),
        author_css=(static_dir / "author.css").read_bytes(),
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Temp file in the SAME directory, flush, fsync, `os.replace` -- atomic
    on both POSIX and Windows when source and destination share a
    filesystem (§E4). If anything fails before `os.replace` runs, `path`
    itself was never opened for writing and is therefore left exactly as it
    was; the next reader (`app/coach/context.py`'s own loaders, or the next
    save) is never handed a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _read_existing(path: Path) -> "list[dict]":
    """Used only to pre-fill the page's own form on load -- tolerant, never
    raising: a missing or malformed file reads back as an empty list rather
    than blocking the page from loading at all. `app/coach/context.py`'s
    loaders (which DO raise loudly) remain the only validating readers this
    codebase uses at `start` time; this function exists solely to show the
    user what is already on disk before they edit it."""
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return raw if isinstance(raw, list) else []


STATE_MISSING = "missing"
STATE_PRESENT = "present"
STATE_UNREADABLE = "unreadable"


def _existing_state(path: Path) -> str:
    """Whether the artifact on disk is absent, readable, or there-but-unusable.

    `_read_existing` deliberately renders all three as `[]`, which is right for
    filling a form and catastrophic for deciding whether to overwrite. This
    function is the distinction the SAVE path needs, and it is deliberately a
    separate function rather than a second return value: the read stays
    tolerant, and the write gets to be strict, without either one having to
    care about the other's rules.

    `unreadable` is NOT "renders as empty". A file containing exactly `[]` is a
    legitimate state -- an empty survey is a real thing to have -- and treating
    it as damaged would block authoring a node from scratch, which is this
    page's main job. Only a file that exists and cannot be understood as a list
    counts, which includes valid JSON of the wrong shape: real corruption
    usually stays valid JSON."""
    if not path.is_file():
        return STATE_MISSING
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return STATE_UNREADABLE
    return STATE_PRESENT if isinstance(raw, list) else STATE_UNREADABLE


def _label_for(entry: Any, index: int) -> str:
    if isinstance(entry, dict) and "id" in entry:
        return repr(entry["id"])
    return f"<item at index {index}>"


def _validate_brief_payload(body: Any) -> "tuple[Optional[list[dict]], Optional[str]]":
    if not isinstance(body, list):
        return None, "brief must be a JSON array of items"
    validated: "list[dict]" = []
    for index, entry in enumerate(body):
        try:
            item = ContextItem.model_validate(entry)
        except ValidationError as exc:
            return None, f"item {_label_for(entry, index)} is invalid: {exc}"
        validated.append(item.model_dump())
    return validated, None


def _validate_survey_payload(body: Any) -> "tuple[Optional[list[dict]], Optional[str]]":
    if not isinstance(body, list):
        return None, "survey must be a JSON array of items"
    validated: "list[dict]" = []
    seen_ids: "set[str]" = set()
    for index, entry in enumerate(body):
        payload = dict(entry) if isinstance(entry, dict) else entry
        if isinstance(payload, dict):
            # Always the item's position in the posted array (§E4) --
            # whatever `order` value the client sent (if any) is overwritten.
            payload["order"] = index
        try:
            item = SurveyItem.model_validate(payload)
        except ValidationError as exc:
            return None, f"item {_label_for(entry, index)} is invalid: {exc}"
        if item.id in seen_ids:
            return None, f"item {item.id!r} is a duplicate id"
        seen_ids.add(item.id)
        validated.append(item.model_dump())
    return validated, None


def _sessions_root_for(context_dir: Path) -> Path:
    """The interviews filed at the SAME node as `context_dir`.

    Derived from `context_dir` rather than accepted as a parameter, and that is
    the deliberate part. `app/projects.py::resolve` builds `context_dir` and
    `sessions_root` from ONE `node_dir` as siblings, and `app/config.py`'s two
    defaults (`context`, `sessions`) are siblings as well -- so under both
    layouts this product actually has, the relationship is exact rather than
    guessed. `SESSIONS_DIR_NAME` is IMPORTED from the module that owns the
    convention, never spelled again here, on this file's own stated precedent
    (":45-47", imported never redefined, so the two halves cannot drift).

    A `sessions_root` PARAMETER was the alternative and is worse here: this
    factory's only caller is `app/main.py::run_author`, so a parameter nothing
    passed would leave every hypothesis in the product stamped `()` -- and `()`
    is an assertion ("no interview existed"), not a blank. A seam that defaults
    to a false claim is worse than no seam.

    The one layout this cannot see is a user who overrides `APP_SESSIONS_ROOT`
    alone, to somewhere that is not `APP_CONTEXT_DIR`'s sibling. That stamps an
    empty boundary where interviews exist, which is the wrong direction, and it
    is stated here rather than papered over: the fix, if that layout ever
    ships, is to thread the resolved root through from `main.py`, which already
    holds it -- never to widen the guess."""
    return Path(context_dir).parent / SESSIONS_DIR_NAME


def _session_ids_at(sessions_root: Path) -> "tuple[str, ...]":
    """Every interview that exists at this node, in id order.

    DIRECTORIES only, matching `app/web/console.py::_past_interviews`' own
    listing -- a stray file beside the sessions is not an interview, and every
    id recorded that never was one permanently weakens a real pre-registration
    claim. An absent `sessions/` is the ordinary first case (D-01), not an
    error: a node authored before any call has none."""
    if not sessions_root.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in sessions_root.iterdir() if entry.is_dir()))


def _stored_boundary(entry: Any) -> "Optional[list[str]]":
    """A previously stored `criterion_registered_after`, or `None` if there
    is not one that can be carried forward. Garbage on disk reads as `None`
    -- carrying it forward would 400 every future save (the value is injected
    from disk, so no client edit could ever clear it), and the fallback is to
    re-stamp with what exists NOW, which weakens the claim rather than
    inventing one."""
    if not isinstance(entry, dict):
        return None
    value = entry.get("criterion_registered_after")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def _stamp_pre_registration_boundaries(body: Any, *, context_dir: Path) -> Any:
    """Records, on every hypothesis being saved, WHICH INTERVIEWS ALREADY
    EXISTED when its criterion text was written.

    Always recomputed, never taken from the request -- `SurveyItem.order`'s own
    stated precedent (":48-51"), and for a much sharper reason. A
    user-supplied `criterion_registered_after` would be evidence of nothing at
    all: it is the same unbacked claim about order that the bare `criterion`
    already was, only now wearing a credential. The page POSTs the whole array
    on every save, so a stale payload claiming `[]` while two interviews sit on
    disk is not even an attack -- it is the ordinary failure this must stop.

    **The boundary moves on a CRITERION EDIT, and on nothing else.** Both
    halves of that rule are load-bearing:

    - Moving on an edit is the decision that looks wrong and is not. An edited
      criterion is a DIFFERENT RULE, written later, and the interviews now on
      disk came first. Freezing the boundary at the original write would let a
      rule be rewritten to fit the data while still displaying the original
      pre-registration -- shooting the arrow and painting the target around it,
      with the paint certified. Moving it always WEAKENS the claim, the only
      safe direction for something nobody can re-verify afterwards. Empty text
      becoming real text is a criterion edit like any other, so a criterion
      written for the first time after a call records that call honestly.
    - NOT moving otherwise is what keeps the page usable. Every save re-sends
      every criterion, so if editing a statement, adding a hypothesis or
      reordering the list moved the boundary, a genuinely pre-registered rule
      would lose its standing to an unrelated edit and the field would decay
      into "when did you last press Save" -- precisely the timestamp it was
      shaped as a session-id tuple to avoid being.

    The previous text is read through `_read_existing` (tolerant): a file that
    cannot be read yields no stored criteria, so every entry counts as changed
    and gets stamped with what exists now. Nothing is written on that path
    anyway -- `_save_items` refuses an unreadable file with a 409 before the
    write is reached.

    A non-list body, or a non-dict entry, is passed through untouched: it is
    `_validate_hypotheses_payload`'s job to reject those, and by its own name."""
    if not isinstance(body, list):
        return body
    stored: "dict[str, dict]" = {}
    for entry in _read_existing(Path(context_dir) / HYPOTHESES_FILENAME):
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            stored[entry["id"]] = entry
    existing_sessions = list(_session_ids_at(_sessions_root_for(context_dir)))

    stamped: "list[Any]" = []
    for entry in body:
        if not isinstance(entry, dict):
            stamped.append(entry)
            continue
        previous = stored.get(entry.get("id")) if isinstance(entry.get("id"), str) else None
        carried = _stored_boundary(previous)
        unchanged = (
            previous is not None
            and carried is not None
            and previous.get("criterion", "") == entry.get("criterion", "")
        )
        stamped.append(
            {**entry, "criterion_registered_after": carried if unchanged else existing_sessions}
        )
    return stamped


def _validate_hypotheses_payload(body: Any) -> "tuple[Optional[list[dict]], Optional[str]]":
    """§K14: validates against `Hypothesis` -- IMPORTED from
    `app/coach/corpus.py`, never redefined here, so the falsifier rule lives
    in exactly one place. `Hypothesis` has no `order` field (unlike
    `SurveyItem`), so there is nothing to recompute here beyond the
    duplicate-id check every other save endpoint in this module already
    applies. A missing or whitespace-only `falsifier` is rejected by
    `Hypothesis`'s own field validator, naming the offending id via
    `_label_for` below exactly like every other rejection in this module."""
    if not isinstance(body, list):
        return None, "hypotheses must be a JSON array of items"
    validated: "list[dict]" = []
    seen_ids: "set[str]" = set()
    for index, entry in enumerate(body):
        try:
            item = Hypothesis.model_validate(entry)
        except ValidationError as exc:
            return None, f"item {_label_for(entry, index)} is invalid: {exc}"
        if item.id in seen_ids:
            return None, f"item {item.id!r} is a duplicate id"
        seen_ids.add(item.id)
        validated.append(item.model_dump())
    return validated, None


async def _save_items(
    request: Request,
    path: Path,
    validator: "Callable[[Any], tuple[Optional[list[dict]], Optional[str]]]",
) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 -- malformed JSON is a 400, never a 500
        return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
    validated, error = validator(body)
    if error is not None:
        # Nothing written -- validation happens entirely before the atomic
        # write below is ever called (§E4: "no file is written" on a bad item).
        return JSONResponse({"error": error}, status_code=400)
    # BEFORE the write, and after validation so a bad payload still reports its
    # own fault first. Overwriting a file whose contents were never understood
    # destroys work that was recoverable by hand a moment earlier -- and the
    # operator reaches this exact state by FOLLOWING the tool's own advice,
    # because `start` tells them to open the authoring page when a file will
    # not load, and the page then renders a malformed file as an empty form.
    # A 409 rather than a 400: the request is fine, the state on disk is not.
    if _existing_state(path) == STATE_UNREADABLE:
        return JSONResponse(
            {
                "error": (
                    f"{path.name} already exists on disk but could not be read as a list, so "
                    "your edit was NOT saved -- saving would have replaced a file whose "
                    "contents are still recoverable by hand. Fix or move that file, then "
                    "reload this page."
                ),
                "state": STATE_UNREADABLE,
            },
            status_code=409,
        )
    _atomic_write_json(path, validated)
    return JSONResponse({"saved": len(validated)})


_FORBIDDEN = Response(status_code=403)


def _host_header_is_loopback(host_header: str) -> bool:
    """True only for `127.0.0.1[:port]` or `localhost[:port]` (case
    insensitive). This app deliberately carries no token (module docstring)
    -- the Host header is the only signal left, once a request has already
    reached this process, that it did not originate from DNS rebinding (an
    attacker-controlled domain resolved to 127.0.0.1, so CORS never even
    applies)."""
    hostname = host_header.rsplit(":", 1)[0]
    return hostname.lower() in ("127.0.0.1", "localhost")


def create_author_app(*, context_dir: Path, static_dir: Path) -> FastAPI:
    """Builds the authoring ASGI app. No token, no docs endpoints, seven
    routes total -- the page, its two static assets, and the brief/survey/
    hypotheses GET+POST pairs (05-03-PLAN.md §K13). Never registered on
    `app/web/server.py::create_app()`'s own app -- gate 9 (every phone-surface
    route lives beneath the token prefix) is asserted unmodified against THAT
    app, never this one.

    Guarded by one `http` middleware, checked before any route runs (found
    in an independent review): the loopback bind stops the LAN but not the
    user's OWN browser. Starlette's `request.json()` parses the body
    regardless of `Content-Type`, and this app has no CORS middleware, no
    Host check, and no token -- so without these two guard clauses, ANY
    webpage open in the user's browser could fire a no-preflight
    cross-origin POST (`Content-Type: text/plain`, a valid-JSON body) at
    `/api/survey`/`/api/brief`/`/api/hypotheses` and silently overwrite the
    brief/survey/hypotheses that ground the generate prompt and the coach's
    own hypothesis-tracking (persistent prompt injection into the coach), and
    DNS rebinding could READ them past the loopback bind entirely. Both
    guards return a bare 403 with an empty body -- never a detail that would
    help an attacker iterate."""
    assets = _load_static_assets(static_dir)
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def _guard_cross_origin_requests(request: Request, call_next):
        # A rejected POST's body is drained (never inspected) before
        # returning -- if bytes are left unread on the socket when uvicorn
        # closes the response, the OS can send an RST instead of a clean
        # FIN, which surfaces to the client as an intermittent
        # ConnectionAbortedError rather than the 403 actually sent. Reading
        # it here is safe: Starlette caches the body, so a route that later
        # calls `request.json()` (the happy path never reaches this branch)
        # would read the cached bytes, never a second time off the wire.
        if not _host_header_is_loopback(request.headers.get("host", "")):
            if request.method == "POST":
                await request.body()
            return _FORBIDDEN
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                await request.body()
                return _FORBIDDEN
        return await call_next(request)

    @app.get("/")
    async def get_page() -> Response:
        return Response(
            content=assets.index_html, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/author.js")
    async def get_author_js() -> Response:
        return Response(
            content=assets.author_js,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/author.css")
    async def get_author_css() -> Response:
        return Response(
            content=assets.author_css, media_type="text/css", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        """Which artifacts are absent, readable, or present-but-unusable.

        Additive on purpose -- the three GETs above keep returning bare arrays,
        so nothing that already reads them has to change. This exists because
        refusing at save time is only a safety net: without it the operator
        still types a whole form before learning the file on disk was never
        readable, and an empty form for a DAMAGED file looks exactly like an
        empty form for a node nobody has authored yet."""
        return JSONResponse(
            {
                "brief": _existing_state(context_dir / BRIEF_FILENAME),
                "survey": _existing_state(context_dir / SURVEY_FILENAME),
                "hypotheses": _existing_state(context_dir / HYPOTHESES_FILENAME),
            }
        )

    @app.get("/api/brief")
    async def get_brief() -> JSONResponse:
        return JSONResponse(_read_existing(context_dir / BRIEF_FILENAME))

    @app.get("/api/survey")
    async def get_survey() -> JSONResponse:
        return JSONResponse(_read_existing(context_dir / SURVEY_FILENAME))

    @app.post("/api/brief")
    async def post_brief(request: Request) -> JSONResponse:
        return await _save_items(request, context_dir / BRIEF_FILENAME, _validate_brief_payload)

    @app.post("/api/survey")
    async def post_survey(request: Request) -> JSONResponse:
        return await _save_items(request, context_dir / SURVEY_FILENAME, _validate_survey_payload)

    register_hypothesis_routes(app, context_dir=context_dir)

    return app


def register_hypothesis_routes(app: FastAPI, *, context_dir: Path) -> None:
    """Mounts the hypotheses GET and POST onto an app.

    **Extracted so the hub can build hypotheses WITHOUT a second writer.** Gate
    19 pins the modules permitted to put bytes into `context/hypotheses.json`
    to `{author.py, accept.py}`, TEXTUALLY and per FILE. The hub needed to
    create a hypothesis and fill in a missing criterion; implementing that in
    `app/web/review.py` would have meant either admitting that module to the
    gate's set -- retiring the gate's reach over everything else the hub does
    and will do -- or writing a second save path with a second copy of the
    stamping rule. This is the third option: the same route, mounted twice.

    The stamping rule in particular is the part most dangerous to reimplement.
    It is the difference between a pre-registered decision rule and one typed
    once the answers were in, and a plausible-looking second copy would be
    wrong in a way nothing on the page could show."""

    @app.get("/api/hypotheses")
    async def get_hypotheses() -> JSONResponse:
        return JSONResponse(_read_existing(context_dir / HYPOTHESES_FILENAME))

    @app.post("/api/hypotheses")
    async def post_hypotheses(request: Request) -> JSONResponse:
        # Stamped, THEN validated -- and the stamp is wrapped around the
        # validator here rather than folded into it, deliberately.
        # `app/web/accept.py` imports `_validate_hypotheses_payload` verbatim
        # (its own docstring states why it must not own a second validator),
        # and it has no sessions_root and must never acquire one: an accept
        # happens AFTER the interview it was proposed by, so a re-stamp there
        # would push every accepted hypothesis's boundary past the very
        # evidence the accept was judged on. Keeping the stamp on this route
        # only is what makes an accept a pure state change.
        def _validate(body: Any) -> "tuple[Optional[list[dict]], Optional[str]]":
            return _validate_hypotheses_payload(
                _stamp_pre_registration_boundaries(body, context_dir=context_dir)
            )

        return await _save_items(request, context_dir / HYPOTHESES_FILENAME, _validate)


def build_uvicorn_server(app: FastAPI, *, port: int) -> "object":
    """Constructs the authoring server, ALWAYS bound to `_AUTHOR_HOST`
    (loopback) -- never the wildcard address `app/web/server.py`'s phone
    surface deliberately uses, and never a parameter this function accepts
    from its caller (see module docstring). Deferred `uvicorn` import,
    mirroring `app/web/server.py::build_uvicorn_server`'s own precedent."""
    import uvicorn

    config = uvicorn.Config(app, host=_AUTHOR_HOST, port=port, log_level="warning")
    return uvicorn.Server(config)
