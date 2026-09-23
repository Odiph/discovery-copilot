"""Phase B: the local console -- pick a project, see what is in it, start the
interview.

**One command, one process, no second launcher.** `python -m app console`
serves a loopback-only page; when the operator presses Start, this app records
the selection and stops its own server, and the SAME process then runs
`start`. A web page never spawns a subprocess. That keeps the README's
existing principle intact and avoids the failure mode where a launcher
survives the thing it launched.

**A sibling of the authoring app, not a new security surface.** It reuses
`app/web/author.py`'s two guards verbatim -- the loopback bind stops the LAN
but not the operator's OWN browser, and Starlette parses a JSON body whatever
the `Content-Type` says, so any page open in that browser could otherwise fire
a no-preflight cross-origin POST at this app. Here that would mean choosing
which customer's knowledge grounds the next interview, which is a worse
outcome than the authoring hole those guards were added for, not a lesser one.

**This module reads. It never writes.** Not one route mutates anything on
disk; the only state it holds is the operator's selection, in memory, until
the process moves on to `start`. Two standing gates depend on that and on
what this module names:

- **Gate 19** pins the modules that WRITE `context/hypotheses.json` to exactly
  `{app/web/author.py}` and detects them TEXTUALLY. This module therefore
  never names that file: it calls `corpus.load_hypotheses`, which names it.
- **Gate 20** pins the modules allowed to read `Fact.quoted_text`. This module
  reports a fact COUNT and reads no field of a fact at all.

**Project paths arriving here are untrusted input.** They come from a request
body, and `app.projects.resolve` is what confines them to the tree -- the same
function the CLI uses, never a second path-handling implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app import project_meta
from app import projects as projects_mod
from app.coach import context as coach_context
from app.coach import corpus as coach_corpus
from app.web.author import _FORBIDDEN, _host_header_is_loopback

logger = logging.getLogger(__name__)

# Hardcoded for the same reason `app/web/author.py`'s is, and it must stay a
# literal rather than a parameter: a future "make it reachable from the
# tablet" convenience edit has to be visible in a diff of THIS line.
_CONSOLE_HOST = "127.0.0.1"

RENDERED_TRANSCRIPT_NAME = "transcript.md"

# How many past interviews the page lists. Found by RUNNING the console
# against the real `_eval` node, which carries 273 sessions: the full list
# rendered every one of them and pushed the Start button several screens
# down, which makes the console unusable for the exact project most likely to
# be opened first. The count is always reported in full beside the list, so
# the cap is visible rather than a silent truncation.
MAX_LISTED_INTERVIEWS = 10


@dataclass(frozen=True)
class ConsoleSelection:
    """What the operator chose. Returned by `serve_console` to its caller,
    which then runs the interview -- this is the whole interface between the
    page and the rest of the process."""

    project: str
    read_from: "tuple[str, ...]"


@dataclass(frozen=True)
class _ConsoleStaticAssets:
    index_html: bytes
    console_js: bytes
    console_css: bytes


def _load_static_assets(static_dir: Path) -> _ConsoleStaticAssets:
    return _ConsoleStaticAssets(
        index_html=(static_dir / "index.html").read_bytes(),
        console_js=(static_dir / "console.js").read_bytes(),
        console_css=(static_dir / "console.css").read_bytes(),
    )


def _node_detail(project_path: str, *, repo_root: "Optional[Path]" = None) -> dict[str, Any]:
    """Everything the page shows for ONE node.

    Every read here degrades rather than raising. A console that refuses to
    open because one project among twenty has a malformed survey is a console
    that cannot be used to reach the other nineteen -- and the operator is
    most likely opening it precisely BECAUSE something is wrong. Each section
    therefore carries its own `error` string instead, so the page can show
    "this node's survey is broken" beside the nodes that are fine.

    That is the opposite of `start`'s discipline, deliberately: `start` fails
    loudly on a malformed brief because it is about to ground a live coach in
    it. Reading a directory listing is not that."""
    paths = projects_mod.resolve(project_path, repo_root=repo_root)
    detail: dict[str, Any] = {
        "project": paths.path,
        "label": paths.label,
        "root": paths.root,
        "ancestors": list(paths.ancestors),
    }

    try:
        survey = coach_context.load_survey(paths.context_dir)
        detail["survey"] = [
            {"id": item.id, "order": item.order, "text": item.text} for item in survey
        ]
        detail["survey_error"] = None
    except Exception as exc:  # noqa: BLE001 -- see docstring: one bad node must not close the console
        detail["survey"] = []
        detail["survey_error"] = f"{type(exc).__name__}: {exc}"

    try:
        hypotheses = coach_corpus.load_hypotheses(paths.context_dir, missing_ok=True)
        detail["hypotheses"] = [
            {"id": h.id, "statement": h.statement, "falsifier": h.falsifier, "state": h.state}
            for h in hypotheses
        ]
        detail["hypotheses_error"] = None
    except Exception as exc:  # noqa: BLE001
        detail["hypotheses"] = []
        detail["hypotheses_error"] = f"{type(exc).__name__}: {exc}"

    # A COUNT, never the facts. A fact carries a previous customer's verbatim
    # speech in `quoted_text`, and the console is a browser page -- the count
    # is what tells the operator whether this node has accumulated anything,
    # which is the only question the picker needs answered.
    try:
        if (paths.context_dir / coach_corpus.FACTS_FILENAME).is_file():
            detail["fact_count"] = len(coach_corpus.load_facts(paths.context_dir, sessions_root=None))
        else:
            detail["fact_count"] = 0
        detail["facts_error"] = None
    except Exception as exc:  # noqa: BLE001
        detail["fact_count"] = None
        detail["facts_error"] = f"{type(exc).__name__}: {exc}"

    all_interviews = _past_interviews(paths.sessions_root)
    detail["sessions"] = all_interviews[:MAX_LISTED_INTERVIEWS]
    detail["session_total"] = len(all_interviews)
    return detail


def _past_interviews(sessions_root: Path) -> "list[dict[str, Any]]":
    """Past interviews at this node, newest first.

    Session directory names are `%Y%m%dT%H%M%S`, so a reverse lexicographic
    sort IS newest-first without parsing a date -- and it stays correct for a
    same-second collision, which `create_session_dir` disambiguates with a
    `-1` suffix that sorts after the bare name.

    `rendered` says whether `python -m app render` has been run, because a
    session with no rendered transcript has nothing for the page to link to
    yet. Nothing here opens a transcript: the console lists interviews, it
    does not display customer speech."""
    if not sessions_root.is_dir():
        return []
    entries: "list[dict[str, Any]]" = []
    for entry in sorted(sessions_root.iterdir(), key=lambda item: item.name, reverse=True):
        if not entry.is_dir():
            continue
        entries.append(
            {"id": entry.name, "rendered": (entry / RENDERED_TRANSCRIPT_NAME).is_file()}
        )
    return entries


def _validate_start_request(body: Any) -> "tuple[Optional[ConsoleSelection], Optional[str]]":
    """Validates a Start request into a `ConsoleSelection`, or an error
    message.

    Re-applies the SAME same-root rule the CLI's `--read-from` enforces. The
    page also filters its picker to in-branch nodes, and that is not a reason
    to trust what arrives -- the picker is a convenience for the operator; the
    check is the rule. Anything reaching this process could have skipped the
    page entirely."""
    if not isinstance(body, dict):
        return None, "expected a JSON object"

    project = body.get("project")
    if not isinstance(project, str):
        return None, "'project' must be a string naming the node to file this interview under"

    raw_sources = body.get("read_from", [])
    if not isinstance(raw_sources, list) or not all(isinstance(s, str) for s in raw_sources):
        return None, "'read_from' must be a list of project paths"

    try:
        filing = projects_mod.resolve(project)
    except projects_mod.ProjectPathError as exc:
        return None, str(exc)

    seen: "set[str]" = set()
    sources: "list[str]" = []
    for candidate in raw_sources:
        try:
            resolved = projects_mod.resolve(candidate)
        except projects_mod.ProjectPathError as exc:
            return None, str(exc)
        if resolved.root != filing.root:
            return None, (
                f"{candidate!r} is under a different root ({resolved.root!r}) than "
                f"{project!r} ({filing.root!r}) -- knowledge may be shared only within "
                f"one hierarchy"
            )
        if resolved.path == filing.path:
            return None, f"{candidate!r} is the filing node itself, which is always read"
        if resolved.path in seen:
            return None, f"{candidate!r} was selected twice"
        seen.add(resolved.path)
        sources.append(resolved.path)

    return ConsoleSelection(project=filing.path, read_from=tuple(sources)), None


def create_console_app(
    *,
    static_dir: Path,
    on_start: "Any",
    repo_root: "Optional[Path]" = None,
) -> FastAPI:
    """Builds the console ASGI app.

    `on_start` is called with the validated `ConsoleSelection` when the
    operator presses Start, and is what stops the server so the process can
    move on. Passed in rather than reached for, so the whole app is testable
    without running uvicorn -- and so this module has no opinion about how
    the server is shut down."""
    assets = _load_static_assets(static_dir)
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def _guard_cross_origin_requests(request: Request, call_next):
        # Byte-for-byte the authoring app's guard, including the body drain:
        # bytes left unread on the socket when uvicorn closes the response can
        # make the OS send an RST rather than a clean FIN, which reaches the
        # client as an intermittent ConnectionAbortedError instead of the 403
        # actually sent.
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

    @app.get("/console.js")
    async def get_console_js() -> Response:
        return Response(
            content=assets.console_js,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/console.css")
    async def get_console_css() -> Response:
        return Response(
            content=assets.console_css, media_type="text/css", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/projects")
    async def get_projects() -> JSONResponse:
        # The PICKER's view: archived nodes (and their subtrees) dropped, the
        # unfiled bucket last. `/api/project/{path}` below is deliberately NOT
        # filtered -- archiving hides a node from the list, it never closes
        # it, and an operator who addresses one by path still gets it.
        return JSONResponse(
            [
                {
                    "path": node.path,
                    "label": node.label,
                    "depth": node.depth,
                    "has_context": node.has_context,
                    "has_sessions": node.has_sessions,
                }
                for node in project_meta.for_picker(
                    projects_mod.list_projects(repo_root=repo_root), repo_root=repo_root
                )
            ]
        )

    @app.get("/api/project/{project_path:path}")
    async def get_project(project_path: str) -> JSONResponse:
        try:
            return JSONResponse(_node_detail(project_path, repo_root=repo_root))
        except projects_mod.ProjectPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/start")
    async def post_start(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 -- a malformed body is a 400, never a 500
            return JSONResponse({"error": "body is not valid JSON"}, status_code=400)

        selection, error = _validate_start_request(body)
        if selection is None:
            return JSONResponse({"error": error}, status_code=400)

        # Counts and node paths only -- never a survey question, a hypothesis
        # or a fact (S7).
        logger.info(
            "console start project=%s extra_sources=%d",
            selection.project,
            len(selection.read_from),
        )
        on_start(selection)
        return JSONResponse(
            {"project": selection.project, "read_from": list(selection.read_from)}
        )

    return app


def build_console_server(app: FastAPI, *, port: int) -> uvicorn.Server:
    """The console server, ALWAYS bound to `_CONSOLE_HOST`. Mirrors
    `app/web/author.py`'s own builder so the loopback bind is established in
    exactly one visible place per app."""
    config = uvicorn.Config(app, host=_CONSOLE_HOST, port=port, log_level="warning")
    return uvicorn.Server(config)
