"""The nine standing promises this project makes structural (phase_specification
U4, 03-03-PLAN.md §R5), turned into tests a future phase cannot break quietly.
Each one is cheap, fast and permanent -- every one of them is a property this
project has already lost once by not checking it.

Gates 1-4 are Phase 2's (02-03-PLAN.md §U4); gates 5-9 are Phase 3's
(03-03-PLAN.md §R5), added alongside rather than in a new module.

<!-- planner-discipline-allow: soniox, assemblyai, whisper, deepgram,
getUserMedia, MediaRecorder, MediaDevices, RTCPeerConnection -- these
literals are named ONLY in this plan's specification (02-03-PLAN.md §U4) and
in this gate module itself, never in any other task's action prose, so they
cannot leak into a source comment elsewhere that would then trip their own
gate. -->
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from websockets.sync.client import connect as ws_connect

from app.coach import context as coach_context_mod
from app.coach import corpus as coach_corpus_mod
from app.coach import corpus_store as coach_corpus_store
from app.coach import critic as coach_critic_mod
from app.coach import extract as coach_extract_mod
from app.coach import generate as coach_generate_mod
from app.coach import prompts as coach_prompts_mod
from app.coach import reasoning_client
from app.coach.context import PROVENANCE_VALUES, SurveyItem
from app.coach.coverage import CoverageDetector, CoverageState
from app.coach.critic import Judgment
from app.coach.extract import EXTRACT_TOOL_NAME
from app.coach.generate import Candidate
from app.coach.pipeline import SuggestionPipeline
from app.coach.stub import StubAnthropicClient, tool_use_response
from app.mcp import server as mcp_server_mod
from app.mcp import tools as mcp_tools_mod
from app.session import render as session_render_mod
from app.session.reader import read_session
from app.session.store import SegmentRecord, SessionStore, SuggestionRecord, create_session_dir
from app.web import author as web_author_mod
from app.web import join as join_mod
from app.web import server as web_server_mod
from app.web.hub import ConnectionHub

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
STATIC_DIR = REPO_ROOT / "app" / "static"

# Directories that are never part of the shipped application surface and
# must not be scanned by any gate below.
# `projects/` is excluded DELIBERATELY, decided rather than discovered.
#
# The gates built on this walk (1, 2, 3) constrain THIS REPOSITORY'S SOURCE:
# no STT vendor named in capture/session/web code, no browser media-capture
# identifier, no deployment descriptor. `projects/` holds customer material
# and never holds source, so scanning it can only produce a false positive --
# a user's notes file mentioning one of the forbidden identifiers would fail
# a gate about code.
#
# The sharper reason is gate 2's failure message: it echoes the MATCHING LINE
# verbatim. A customer's own sentence would be printed into test output, which
# is the one place in this project nobody thought to check for customer
# speech. `sessions/` and `context/` are not excluded here only because their
# contents (.jsonl, .json, .wav) fall outside every suffix filter in use; that
# is luck, not design, and it stops being true the moment someone drops a .md
# in one. Do not treat their absence from this set as a precedent.
#
# This does not weaken the privacy gates: 17 and 29 compare `git status`
# snapshots and never use this walk.
_EXCLUDED_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", ".planning", "node_modules", ".claude", "projects",
}


def _iter_repo_files(*, suffixes: "tuple[str, ...] | None" = None):
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        yield path


# --- Gate 1 (extended by 03-03-PLAN.md §R5 item 5): no STT vendor named in capture, ------
# --- session, OR web executable code -----------------------------------------------


def test_gate_one_no_vendor_identifier_in_capture_or_session_executable_code():
    # Take the vendor list from phase_specification §U4, not from this line.
    # `app/web` added in 03-03-PLAN.md §R5 item 5: the surface broadcasts a
    # record carrying an engine name as a VALUE, but must never import or
    # mention a vendor -- this keeps STT-03 true across the new layer.
    vendor_pattern = re.compile(r"^[^#]*\b(soniox|assemblyai|whisper|deepgram)\b", re.IGNORECASE)
    violations = []
    for package in ("app/capture", "app/session", "app/web"):
        for py_file in sorted((REPO_ROOT / package).rglob("*.py")):
            for lineno, raw_line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), start=1):
                if vendor_pattern.search(raw_line):
                    violations.append(f"{py_file}:{lineno}: {raw_line}")
    assert not violations, "a later change reintroduced a vendor name into capture, storage, or the web layer:\n" + "\n".join(violations)


# --- Gate 2: no browser media-capture API anywhere in the repository ---------------


def test_gate_two_no_browser_media_capture_identifier_anywhere():
    # Take the identifier list from phase_specification §U4. These four
    # literals are deliberately named ONLY here and in this plan's own
    # specification -- never in a source comment anywhere else, or that
    # comment would trip this very gate.
    forbidden_identifiers = ("getUserMedia", "MediaRecorder", "MediaDevices", "RTCPeerConnection")
    forbidden_pattern = re.compile("|".join(re.escape(name) for name in forbidden_identifiers))

    violations = []
    for path in _iter_repo_files(suffixes=(".py", ".js", ".mjs", ".ts", ".jsx", ".tsx", ".html", ".md")):
        if path == THIS_FILE:
            continue
        for lineno, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if forbidden_pattern.search(raw_line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {raw_line}")
    assert not violations, (
        "the phone is receive-only, and that is what keeps a plain non-TLS LAN "
        "connection legal in Phase 3 -- a browser media-capture identifier was "
        "found:\n" + "\n".join(violations)
    )


# --- Gate 3: zero deployed services, exactly one new dependency --------------------


_DEPLOYMENT_DESCRIPTOR_NAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Procfile",
    "fly.toml",
    "vercel.json",
    "render.yaml",
    "railway.json",
}


def test_gate_three_no_deployment_descriptor_anywhere():
    violations = [
        str(path.relative_to(REPO_ROOT))
        for path in _iter_repo_files()
        if path.name in _DEPLOYMENT_DESCRIPTOR_NAMES
    ]
    assert not violations, (
        "one process, zero deployed services -- a deployment descriptor was "
        "found: " + ", ".join(violations)
    )


def _parse_pinned_packages(requirements_path: Path) -> set[str]:
    """Resolves `-r other-file.txt` references so the returned set matches
    what `pip install -r requirements_path` would actually install -- exactly
    what "the pinned dependency set has grown by exactly one package" means."""
    pins: set[str] = set()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r"):
            referenced = line[2:].strip()
            pins |= _parse_pinned_packages(requirements_path.parent / referenced)
            continue
        pins.add(line)
    return pins


def test_gate_three_exactly_five_new_pinned_dependencies_versus_phase_one():
    # Deliberately raised from 4 to 5 in 04-01-PLAN.md §C7, per CONTEXT.md's
    # instruction that this be a conscious, justified change, never a silent one
    # (the gate's own 1 -> 4 bump in 03-01-PLAN.md Task 2 §P8 set this precedent).
    # Phase 2 contributed the original `1`: PyAudioWPatch. Phase 3 (03-RESEARCH.md
    # Standard Stack) added three more:
    #   - fastapi==0.141.1  -- the ASGI web framework serving the phone surface (D-05).
    #   - uvicorn==0.52.1   -- the ASGI server embedding that framework on the main
    #     thread alongside the existing capture/STT threads (03-RESEARCH.md §1).
    #   - qrcode[pil]==8.2  -- QR rendering (terminal ANSI + PNG) for the join flow
    #     (03-RESEARCH.md §4); Pillow arrives transitively via the `[pil]` extra and is
    #     NOT a separate pinned line, so it does not count against this literal-line
    #     assertion.
    # Phase 4 (04-01-PLAN.md §C4/§C7) adds exactly one more:
    #   - anthropic==0.120.2 -- the official Python SDK for the reasoning vendor's
    #     Messages API, constructed at exactly one site (app/coach/reasoning_client.py).
    #     PyPI-resident, MIT, github.com/anthropics/anthropic-sdk-python -- 04-RESEARCH.md's
    #     Package Legitimacy Audit returns OK/Approved. `anthropic` declares `pydantic`
    #     as its own dependency (used for schema validation by app/coach/context.py
    #     too); that arrival is a declared dependency of a package THIS plan pins
    #     deliberately, so it is NOT pinned separately and does NOT count here -- a
    #     different, legitimate relationship from the one 03-01-PLAN.md §P9 rejected
    #     (an UNRELATED vendor SDK's transitive dependency tree).
    # Transitive arrivals (Pillow, colorama, pydantic) never count --
    # `_parse_pinned_packages` parses literal requirements-file lines, not a resolved
    # dependency tree.
    phase1_pins = _parse_pinned_packages(REPO_ROOT / "benchmark" / "requirements.txt")
    root_pins = _parse_pinned_packages(REPO_ROOT / "requirements.txt")

    missing = phase1_pins - root_pins
    assert not missing, f"Phase 1's pinned dependencies must never be removed: {missing}"

    added = root_pins - phase1_pins
    assert len(added) == 5, (
        f"expected exactly five new pinned dependencies versus Phase 1 "
        f"(PyAudioWPatch + fastapi + uvicorn + qrcode[pil] + anthropic), got {added}"
    )


# --- Gate 4: nothing private is committable -----------------------------------------


def test_gate_four_session_artifacts_are_git_ignored():
    for relative_path in ("sessions/", "sessions/some-session/transcript.jsonl",
                           "sessions/some-session/session.json", "sessions/some-session/interviewee.wav"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative_path],
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"{relative_path} is NOT reported as git-ignored"


def test_gate_four_a_completed_run_leaves_nothing_committable_under_sessions(tmp_path):
    # Exercises the real SessionStore against a REAL path under the repo's
    # own sessions/ root (not tmp_path) so `git status --porcelain` is a
    # meaningful assertion about this repository's actual ignore rules,
    # exactly like the plan's own <verify> command.
    sessions_root = REPO_ROOT / "sessions"
    session_dir = create_session_dir(sessions_root)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    store.append_segment(
        speaker="interviewee", text="gate four probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "sessions"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert result.stdout.strip() == "", (
        f"a completed run left something committable under sessions/: {result.stdout!r}"
    )


# --- 03-03-PLAN.md §R5: shared test infrastructure for gates 7 and 9 ----------------


@contextlib.contextmanager
def _running_gate_server(app):
    """Runs `app` under a REAL uvicorn server on an ephemeral loopback port,
    in its own background thread driving its own event loop -- the same shape
    as `app/tests/test_web_serving.py`'s own `_running_server` fixture (§P9:
    no third-party TestClient/httpx, which is not a dependency this project
    actually has)."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
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
        yield actual_port
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _gate_app(*, token: str, store: SessionStore, session_id: str):
    return web_server_mod.create_app(
        token=token, hub=ConnectionHub(), static_dir=STATIC_DIR, store=store,
        session_id=session_id, server_started_at="2026-01-01T00:00:00.000+00:00",
    )


# --- Gate 6: no third-party origin, no build step in the static assets -------------


_URL_PATTERN = re.compile(r"[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_FORBIDDEN_JS_MANIFEST_NAMES = {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}


def test_gate_six_no_third_party_origin_or_build_step_in_static_assets():
    # First half: a security control -- a page that loads a remote resource
    # can carry the join token off-device in a referrer header, and there is
    # no third party in this architecture that has any business being there.
    violations = []
    for path in sorted(STATIC_DIR.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _URL_PATTERN.finditer(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {match.group(0)}")
    assert not violations, (
        "a static asset references an absolute external URL of some scheme -- a remote "
        "include is how the join token could leave the device in a referrer header:\n"
        + "\n".join(violations)
    )

    # Second half: keeps "one process, zero deployed services, no build" honest
    # now that the repository contains JavaScript for the first time.
    manifests = [
        str(path.relative_to(REPO_ROOT))
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and path.name in _FORBIDDEN_JS_MANIFEST_NAMES
        and not any(part in _EXCLUDED_DIR_NAMES for part in path.parts)
    ]
    node_modules_dirs = [
        str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob("node_modules") if path.is_dir()
    ]
    assert not manifests, f"a JavaScript package manifest/lockfile was found: {manifests}"
    assert not node_modules_dirs, f"an installed node_modules directory was found: {node_modules_dirs}"


# --- Gate 7: the join token never reaches a log line --------------------------------


def test_gate_seven_the_join_token_never_reaches_a_log_line(tmp_path, caplog, capsys):
    # Same shape as 02-01's transcript-marker test: run with a real token,
    # capture log records at debug level, and assert the token appears in the
    # printed join output and in NO log record. Printing is not logging
    # (T-03-23) -- the join URL is deliberately printed so a human can read
    # or scan it.
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "gate7-probe"})

    caplog.set_level(logging.DEBUG)
    join_info = join_mod.create_join(port=0, session_dir=session_dir, store=store, emit_qr=False)
    token = join_info.token
    printed = capsys.readouterr().out
    assert token in printed, "the join token must still be PRINTED -- the user has to read/scan it"

    app = _gate_app(token=token, store=store, session_id=session_dir.name)
    with _running_gate_server(app) as actual_port:
        with ws_connect(f"ws://127.0.0.1:{actual_port}/s/{token}/ws", open_timeout=5) as ws:
            ws.recv()  # hello
            ws.recv()  # server_status
            ws.send(json.dumps({"type": "ping"}))
            ws.recv()  # pong

    # Excludes the TEST's own simulated client library (`websockets.client`,
    # standing in for a real phone here): a real phone's browser never emits
    # a Python log record at all, so its own outgoing request line logged by
    # this test's stand-in client is test-harness noise, not a production
    # risk. Every record from THIS application's own loggers (`app.*`,
    # `uvicorn*`) is in scope.
    app_records = [r for r in caplog.records if not r.name.startswith("websockets")]
    assert app_records, "no application log records were captured -- the assertion below would be vacuous"
    for record in app_records:
        formatted = record.getMessage()
        assert token not in formatted, f"the join token leaked into a log record: {formatted!r}"


# --- Gate 8: the protocol surface is pinned in both directions ---------------------


def test_gate_eight_the_protocol_surface_is_pinned_in_both_directions():
    # Moving this out of a plan's verify command and into the gate module is
    # what makes D-01's "no resume position" and the deferred two-way control
    # channel permanent rather than true-on-the-day (03-03-PLAN.md §R5 item 8).
    #
    # 04-03-PLAN.md §E1: deliberately, consciously updated here in the SAME
    # commit that adds MESSAGE_TYPE_SUGGESTION to app/web/server.py -- the
    # protocol surface grows by decision, never by drift. The client->server
    # set is UNTOUCHED: D-01's no-replay rule and the deferred two-way control
    # channel (dismiss/pin a suggestion from the phone) both depend on it
    # staying exactly the heartbeat.
    assert web_server_mod.ACCEPTED_CLIENT_MESSAGE_TYPES == frozenset({web_server_mod.MESSAGE_TYPE_PING}), (
        "D-01/D-02: the client->server protocol surface must remain exactly {'ping'} -- no "
        "resume request, no client-reported gap, no two-way control channel"
    )
    assert web_server_mod.SERVER_MESSAGE_TYPES == frozenset({
        web_server_mod.MESSAGE_TYPE_HELLO,
        web_server_mod.MESSAGE_TYPE_PONG,
        web_server_mod.MESSAGE_TYPE_SEGMENT,
        web_server_mod.MESSAGE_TYPE_SERVER_STATUS,
        web_server_mod.MESSAGE_TYPE_SUGGESTION,
        # Phase C, added here in the SAME commit that adds MESSAGE_TYPE_SURVEY to
        # app/web/server.py -- the friction is the point.
        web_server_mod.MESSAGE_TYPE_SURVEY,
    }), "the server->client protocol surface must remain exactly these six message types"


# --- Gate 9: every registered route lives beneath the token prefix -----------------


def test_gate_nine_every_registered_route_lives_beneath_the_token_prefix(tmp_path):
    # Iterating the app's own route table is worth more than any number of
    # individual 404 tests -- it catches the route nobody wrote a test for,
    # which is exactly how D-05 would get quietly undone (03-03-PLAN.md §R5
    # item 9).
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "gate9-probe"})
    app = _gate_app(token="gate9-any-token-value", store=store, session_id=session_dir.name)

    all_paths = [getattr(route, "path", None) for route in app.routes]
    non_token_routes = [
        path for path in all_paths
        if path is not None and path != "/{_catch_all:path}" and not path.startswith("/s/{candidate_token}")
    ]
    assert not non_token_routes, (
        "a route was registered outside the token-scoped prefix -- this is exactly how D-05 "
        f"would get quietly undone: {non_token_routes}"
    )
    # The catch-all fallback must exist too -- otherwise an unmatched path
    # would 500 rather than 404.
    assert "/{_catch_all:path}" in all_paths, "no generic catch-all route is registered"


# --- Phase 4 (04-01-PLAN.md §C7/§C8/§C9): four new standing gates ------------------
#
# Gates 10-11 land with Task 1 (§C8 "the same git-ignore shape as gate 4"; §C9 point 3
# "the gate that keeps the seam a seam"). Gates 12-13 land with Task 2 (§gates). The
# plan's own Task 2 <done> text says "eleven standing gates... including the two new
# ones" -- that arithmetic assumed Task 1 contributed zero NEW pytest gates (its two
# properties were meant to be proven only by the plan's own inline <verify> shell/
# python one-liners). This module instead codifies all four as permanent gates,
# exactly as §C8/§C9 each explicitly instruct ("Add a standing gate..."), which is why
# this repository now carries thirteen, not eleven -- see 04-01-SUMMARY.md.


# --- Gate 10: context/brief.json and context/survey.json are git-ignored -----------


def test_gate_ten_context_artifacts_are_git_ignored():
    # Extended in 05-01-PLAN.md §K1/§K9 (Task 2): the two Phase 5 artifacts
    # (context/facts.json, context/hypotheses.json) land on the ignored side,
    # and their format samples on the must-NOT-be-ignored side -- same rule,
    # same commit discipline as the brief/survey pair already here.
    # `context/declined_proposals.json` joins the list in the commit that added
    # it (`app/web/accept.py`), on the same discipline: it records that a named
    # human declined a state change for a named hypothesis after a named
    # interview, which is a judgement about a real customer conversation and
    # belongs in git no more than the corpus it sits beside does.
    for relative_path in (
        "context/brief.json", "context/survey.json", "context/facts.json",
        "context/hypotheses.json", "context/declined_proposals.json",
    ):
        result = subprocess.run(["git", "check-ignore", "-q", relative_path], cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"{relative_path} is NOT reported as git-ignored"
    # Credential files, on the same side and for a sharper reason. `.gitignore`
    # named `.env` EXACTLY until 2026-08-14, which ignored the one file everybody
    # thinks of and left every neighbour committable. Not hypothetical: an
    # editing script produced `.env.env.bak` holding three live API keys, and it
    # appeared in `git status` as untracked -- one `git add -A` from the history.
    # Asserted name by name rather than by re-reading the glob, so this fails if
    # the pattern is ever narrowed back.
    for relative_path in (
        ".env", ".env.bak", ".env.local", ".env.backup", ".env.env.bak", ".env.production",
    ):
        result = subprocess.run(["git", "check-ignore", "-q", relative_path], cwd=str(REPO_ROOT))
        assert result.returncode == 0, (
            f"{relative_path} is NOT git-ignored -- a credential file is one `git add -A` "
            "away from the history"
        )
    for relative_path in (
        "context/brief.example.json", "context/facts.example.json", "context/hypotheses.example.json",
        # A shipped format sample, and the reason the `.env*` widening above
        # carries a `!.env.example` re-admission line.
        ".env.example",
    ):
        result = subprocess.run(["git", "check-ignore", "-q", relative_path], cwd=str(REPO_ROOT))
        assert result.returncode != 0, (
            f"{relative_path} must NOT be git-ignored -- it is a shipped format sample"
        )


# --- Gate 11: exactly one module under app/coach/ imports the vendor SDK -----------


def test_gate_eleven_exactly_one_coach_module_imports_the_vendor_sdk():
    # AST-based, not a text scan (T-04-12) -- this project's own S13 precedent moved
    # from regex to AST after a regex false-positived on a module's own explanatory
    # docstring prose (STATE.md decision, 02-02).
    hits: list[str] = []
    for py_file in sorted((REPO_ROOT / "app" / "coach").glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "anthropic" for alias in node.names
            ):
                hits.append(py_file.name)
            elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "anthropic":
                hits.append(py_file.name)
    assert sorted(set(hits)) == ["reasoning_client.py"], (
        f"exactly one module under app/coach/ may import the vendor SDK -- got {sorted(set(hits))}"
    )


# --- Gate 12: no coach module puts transcript text into a log line or event detail --


def test_gate_twelve_no_coach_module_puts_transcript_text_into_a_log_line_or_event_detail(tmp_path, caplog):
    # Same shape as 02-01's transcript-marker gate, extended to app/coach/ (T-04-11):
    # a distinctive marker travels through the real coverage-match call path (against
    # the offline stub) and must land ONLY in the transcript file (SESS-03 -- full
    # segment text belongs there by design), never in an event detail or a log record.
    marker = "gate-twelve-distinctive-marker-6f1a2b9c"
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "gate12-probe"})
    survey = (SurveyItem(id="s1", order=0, text="A survey question"),)
    stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )

    caplog.set_level(logging.DEBUG)
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text=marker, audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        deadline = time.monotonic() + 5.0
        matched = False
        while time.monotonic() < deadline:
            polled = read_session(session_dir)
            if any(r.get("event") == "coverage-asked" for r in polled.records_of_kind("event")):
                matched = True
                break
            time.sleep(0.02)
        assert matched, "the stub-primed match never landed as a coverage-asked event"
    finally:
        detector.stop()
    store.close()

    result = read_session(session_dir)
    assert any(marker in r.get("text", "") for r in result.records_of_kind("segment")), (
        "the marker must appear in the transcript file -- full segment text belongs there (SESS-03)"
    )
    for event_record in result.records_of_kind("event"):
        assert marker not in json.dumps(event_record.get("detail") or {}), (
            f"segment text leaked into an event detail: {event_record}"
        )
    for log_record in caplog.records:
        assert marker not in log_record.getMessage(), (
            f"segment text leaked into a log record: {log_record.getMessage()!r}"
        )


# --- Gate 13: the reasoning key never reaches a log line ---------------------------


def test_gate_thirteen_the_reasoning_key_never_reaches_a_log_line(caplog, monkeypatch):
    # Same shape as gate 7's join-token gate: run with a known dummy key value,
    # capture debug-level records, assert the value appears nowhere (T-04-03).
    dummy_key = "sk-ant-gate13-dummy-000111222333"
    monkeypatch.setenv("ANTHROPIC_API_KEY", dummy_key)
    caplog.set_level(logging.DEBUG)
    client = reasoning_client.build_client()
    assert client.api_key == dummy_key, "sanity: the key must actually reach the client"
    assert client.max_retries == 0
    for log_record in caplog.records:
        assert dummy_key not in log_record.getMessage(), (
            f"the reasoning key leaked into a log record: {log_record.getMessage()!r}"
        )


# --- 04-03-PLAN.md §E7: three more standing gates -----------------------------------
#
# The plan's own §E7 prose calls these "gates 12, 13 and 14" -- written before this
# executor discovered the module already carries FOURTEEN test functions (not eleven
# or thirteen; see the 04-01-SUMMARY.md precedent for the same kind of arithmetic
# mismatch). To avoid colliding with the already-existing gates 12/13 above (the
# no-transcript-text-in-coach-logs gate and the reasoning-key-never-logged gate), these
# three are numbered 14, 15 and 16 here -- the properties they prove are exactly the
# ones §E7 describes, just renumbered to match what is actually on disk.


# --- Gate 14: the authoring server binds the loopback address only, never the -------
# --- wildcard address the phone surface deliberately uses --------------------------


def test_gate_fourteen_the_authoring_server_binds_the_loopback_address_only(tmp_path):
    context_dir = tmp_path / "context"
    static_dir = REPO_ROOT / "app" / "static" / "author"
    app = web_author_mod.create_author_app(context_dir=context_dir, static_dir=static_dir)
    server = web_author_mod.build_uvicorn_server(app, port=0)
    assert server.config.host == "127.0.0.1", (
        "the authoring server must bind the loopback address only -- a one-character "
        f"convenience edit is all it takes to undo this; got host={server.config.host!r}"
    )
    assert server.config.host != "0.0.0.0", (
        "the authoring server must never bind the wildcard address the phone surface "
        "deliberately uses"
    )


# --- Gate 15: the suggestion wire payload's key set is exactly the shown text ------


def test_gate_fifteen_the_suggestion_wire_payload_key_set_is_exactly_the_shown_text():
    from app.web.hub import WIRE_SUGGESTION_KEYS, suggestion_payload

    ranked = [Candidate(index=0, text="primary", move="advance"), Candidate(index=1, text="alt", move="deepen")]
    payload = suggestion_payload(ranked)
    assert set(payload) == WIRE_SUGGESTION_KEYS == {"type", "primary", "alternatives"}, (
        "the suggestion payload must carry ONLY the shown text -- no flags, no band, "
        "no rationale, no move type, no dropped candidates -- ever"
    )


# --- Gate 16: the coach's failure wording is distinct from the stream-failure's ----


def test_gate_sixteen_the_coach_failure_wording_is_distinct_from_the_stream_failure_wording():
    app_js = (REPO_ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    literals = [
        text
        for text in re.findall(r'serverStatusRegion\.textContent\s*=\s*"([^"]*)"', app_js)
        if text  # excludes the healthy branch's own clearing assignment (textContent = "")
    ]
    # Widened from "exactly two" when the DEGRADED tier landed, and from five to
    # SEVEN when `no-transcript` landed: the region now words no audio from the
    # call, no audio from the mic, nothing transcribed from the call, nothing
    # transcribed from the mic, audio arriving late, a coach failure, and a
    # stream failure. The two `no-transcript` sentences are deliberately a PAIR
    # rather than one shared sentence -- the whole reason the reason code exists
    # is that the two sides have different remedies ("you are capturing the wrong
    # output" vs "the wrong mic is selected"), so a single sentence covering both
    # would defeat it. The count is still pinned by equality so an eighth branch
    # has to be a deliberate edit here, exactly as gates 8/18/19 require of their
    # own sets.
    assert len(literals) == 7, (
        f"the number of serverStatusRegion.textContent sentences changed -- add the new "
        f"branch here with a reason, never by drift. Found {literals}"
    )
    # The property, and the reason this gate exists: every branch must say
    # something DIFFERENT. Collapsing any two -- most dangerously a degraded
    # stream and a stream failure -- tells the operator their recording died
    # when it has not, which is the false alarm the whole status vocabulary
    # was built to avoid. Distinctness catches that directly, where the old
    # pairwise `!=` could only see the two branches that existed then.
    assert len(set(literals)) == len(literals), (
        f"two status branches share a sentence, so the situations they distinguish are "
        f"indistinguishable to the operator: {literals}"
    )


# --- 05-02-PLAN.md §K9: two more standing gates -------------------------------------
#
# Gate 17 lands with Task 1's own real-extraction-into-the-repo's-own-trees property;
# Gate 18 lands with Task 2's schema/validator pin. Nineteen standing gates total.


# --- Gate 17: a real extraction into the repository's OWN trees leaves nothing -----
# --- committable -- restores any corpus it found, exactly like gate 6's own shape ---


def test_gate_seventeen_a_real_extraction_leaves_nothing_committable():
    # Deliberately writes under the repository's OWN sessions/ and context/
    # trees, not tmp_path -- gate 6's own comment explains why this choice is
    # the entire point of the gate (a future "fix" to a temp path would
    # quietly disarm it). Unlike gate 6, the corpus lives at a FIXED path a
    # real user's data occupies, so this gate saves and restores whatever was
    # there before it ran, and asserts that it did.
    #
    # The status check below is deliberately UNSCOPED, and that is the whole
    # point. An earlier version of this gate ran
    # `git status --porcelain -- sessions context`, which could only ever see
    # the two directories it already expected the code to use -- so it passed
    # cheerfully while a mutated corpus path wrote `leaked_facts.json` into the
    # repository root. That is the identical blind spot `git check-ignore` has
    # and that this gate exists to close: the interesting failure is code
    # writing somewhere NOBODY named, and a pathspec can only ask about paths
    # you name. Compare a full before/after snapshot instead, which sees a new
    # file anywhere in the tree while still tolerating unrelated work in
    # progress. Never reintroduce a pathspec here.
    context_dir = REPO_ROOT / "context"
    facts_path = context_dir / "facts.json"
    backup = facts_path.read_bytes() if facts_path.is_file() else None

    def _tree_status() -> "set[str]":
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        return {line for line in completed.stdout.splitlines() if line.strip()}

    before = _tree_status()
    try:
        sessions_root = REPO_ROOT / "sessions"
        session_dir = create_session_dir(sessions_root)
        store = SessionStore(session_dir)
        store.write_metadata({"purpose": "gate17-probe"})
        store.append_segment(
            speaker="interviewee", text="gate seventeen probe claim text",
            audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.1,
            detected_language="en", engine="fake",
        )
        store.close()

        read_result = read_session(session_dir)
        segments = tuple(SegmentRecord(**r) for r in read_result.records_of_kind("segment"))
        seq = segments[0].seq

        stub = StubAnthropicClient(
            responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [{
                "claim": "gate seventeen probe claim", "seqs": [seq], "provenance": "evidence",
                "quoted_text": "gate seventeen probe claim text", "bearings": [],
            }]})]
        )
        result = coach_extract_mod.extract_facts(
            stub, session_id=session_dir.name, transcript=segments, hypotheses=(),
            model="m", max_tokens=100, timeout_s=1.0,
        )
        assert result.accepted, "sanity: the probe fact must actually have been accepted"
        coach_corpus_store.replace_session_facts(context_dir, session_dir.name, result.accepted)

        appeared = _tree_status() - before
        assert appeared == set(), (
            "a real extraction left something committable ANYWHERE in this repository "
            "(gate 5/gate 6 answer a question about a path you NAME; this answers about "
            "the path the code actually wrote, wherever that turned out to be): "
            f"{sorted(appeared)!r}"
        )
    finally:
        if backup is not None:
            facts_path.write_bytes(backup)
        elif facts_path.is_file():
            facts_path.unlink()
        assert (facts_path.read_bytes() if facts_path.is_file() else None) == backup, (
            "gate 17 must restore whatever corpus was there before it ran -- a gate that can "
            "destroy a real corpus is not one anybody will keep running"
        )


# --- Gate 18: the model-facing provenance/required-field contract cannot drift from --
# --- the code-side validator ---------------------------------------------------------


def test_gate_eighteen_the_extraction_tool_schemas_provenance_enum_equals_the_imported_tuple():
    schema = coach_extract_mod._extract_tool_schema()
    fact_item_schema = schema["input_schema"]["properties"]["facts"]["items"]
    assert fact_item_schema["properties"]["provenance"]["enum"] == sorted(PROVENANCE_VALUES), (
        "a prompt that offers the model a provenance value the validator will reject (or that "
        "silently drops one the validator still accepts) must fail here -- equality, not a "
        "superset check, so a THIRD value slipping into the schema alone is also caught"
    )


def test_gate_eighteen_every_extraction_field_the_model_judges_carries_a_description():
    """`provenance` shipped with an `enum` and NO description while every
    other field had one -- and it is the field the model gets wrong (A2: 32
    of 34 real facts labelled `evidence`). The rubric explaining it lived in
    the system block, several thousand tokens from the point of decision.

    Whether restating it on the field fixes A2 is unmeasured. That it was
    ABSENT is not a thing to rediscover, so the presence of a description on
    every judged field is pinned here."""
    schema = coach_extract_mod._extract_tool_schema()
    properties = schema["input_schema"]["properties"]["facts"]["items"]["properties"]
    undocumented = sorted(
        name for name, spec in properties.items()
        # `claim` is the model's own free text, not a judgment call between
        # options -- it is described by the tool description itself.
        if name != "claim" and not spec.get("description", "").strip()
    )
    assert undocumented == [], (
        f"these extraction fields ask the model for a judgment with no guidance "
        f"at the point of decision: {undocumented}"
    )


def test_gate_eighteen_the_extraction_tool_schemas_required_field_set_equals_the_pinned_set():
    # STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: "seq" -> "seqs", updated
    # in the SAME commit as the schema edit that made this change (properties
    # and required are separately hand-typed in `_extract_tool_schema` -- this
    # is exactly the drift this gate exists to catch if the two ever disagree).
    schema = coach_extract_mod._extract_tool_schema()
    fact_item_schema = schema["input_schema"]["properties"]["facts"]["items"]
    assert set(fact_item_schema["required"]) == {
        "claim", "seqs", "provenance", "quoted_text", "bearings",
    }, (
        "a schema edit that stops requiring quoted_text would disarm the evidence backstop "
        "while every other test in the suite still passed -- this is precisely the kind of "
        "silent regression a standing gate exists for"
    )
    assert set(fact_item_schema["properties"].keys()) == set(fact_item_schema["required"]), (
        "properties and required are separately hand-typed -- pinned equal by set equality so "
        "the two can never silently drift apart (STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1)"
    )


# --- 05-03-PLAN.md §K15: gate 19, the phase's twentieth (and last) standing gate ----
#
# Aimed at a phase that has not been written yet (Phase 6's coach-side proposal
# mechanism). D-03: the user authors hypotheses; the coach only ever PROPOSES an
# update, and the set changes on an explicit user accept. The realistic way D-03
# dies is not a decision to ignore it -- it is a future module writing an accepted
# proposal straight to context/hypotheses.json because that is the shortest path,
# with "accept" happening somewhere upstream and nothing enforcing the link.


# A "write primitive" is one of this codebase's own established ways of putting
# bytes on disk -- `app/web/author.py::_atomic_write_json` and
# `app/coach/corpus_store.py::_atomic_write_json` (both built from a bare
# `open(path, "w"...)` + `os.replace`), `Path.write_text`/`write_bytes`
# (`app/session/store.py`, `app/session/render.py`), `os.rename`, and a bare
# `json.dump`. Deliberately NOT `json.loads`/`.read_text`/`open(path)` with no
# write mode, which are read-only and must never trip this gate.
_HYPOTHESIS_WRITE_INDICATOR = re.compile(
    r'_atomic_write_json\(|os\.replace\(|os\.rename\(|\.write_text\(|\.write_bytes\(|json\.dump\('
    r'|open\([^)]*["\']a?w'
)
_HYPOTHESIS_ARTIFACT_REFERENCE = re.compile(r"HYPOTHESES_FILENAME|hypotheses\.json")


def _modules_that_reference_and_write_the_hypothesis_artifact() -> "set[str]":
    hits: "set[str]" = set()
    for py_file in sorted((REPO_ROOT / "app").rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        if _HYPOTHESIS_ARTIFACT_REFERENCE.search(text) and _HYPOTHESIS_WRITE_INDICATOR.search(text):
            hits.add(str(py_file.relative_to(REPO_ROOT)).replace("\\", "/"))
    return hits


def test_gate_nineteen_the_set_of_modules_that_write_the_hypothesis_artifact_is_pinned_by_equality():
    # Equality, not a superset check (gate 8/gate 18's own precedent) -- a SECOND
    # writer must fail here first, before D-03's own runtime guarantee is ever
    # tested against it. `app/coach/corpus.py` (which defines HYPOTHESES_FILENAME
    # and reads it via `load_hypotheses`) is deliberately NOT in this set -- its
    # own module docstring states "this module never writes", and it contains no
    # write primitive at all.
    #
    # WIDENED TO TWO, on purpose, in the commit that added the accept/reject
    # card. `app/web/accept.py` is a NEW module whose only job is that one
    # write. The reason it is a new module rather than the write being added
    # to `review.py` (the hub that shows the card) or to `author.py` (which
    # already writes this file) is this gate's own mechanism: it matches its
    # two regexes PER FILE, not per statement, so membership is
    # all-or-nothing per module. Admitting the hub -- a ~250-line module that
    # renders transcripts, resolves projects and will keep growing -- would
    # retire this gate's discriminating power over everything else that file
    # ever does. A dedicated single-purpose file keeps the blast radius to a
    # module whose entire content is that write, and leaves the gate at full
    # strength over every other module in the tree.
    assert _modules_that_reference_and_write_the_hypothesis_artifact() == {
        "app/web/author.py",
        "app/web/accept.py",
    }, (
        "the set of modules that write context/hypotheses.json changed -- a new writer must "
        "either route through the authoring save path, or be added here with a stated reason "
        "(D-03/T-05-22), never added by drift"
    )


# --- 06-01-PLAN.md §F5/§F6: two more standing gates ---------------------------------
#
# GC-08: a past interview's verbatim customer speech (`Fact.quoted_text`) must never
# reach a DIFFERENT customer's live prompt or the phone wire. Gate 20 is the structural
# guarantee (if the mapping never READS the field, the property holds by construction);
# gate 21 is the behavioural backstop (a sentinel value never appears in a serialized
# request body, catching a leak through a field gate 20's structural argument does not
# cover -- e.g. `ContextItem.citation` built from a fragment of the quote).


_QUOTED_TEXT_FIELD_NAME = "quoted_text"


def _iter_non_test_app_modules() -> "list[Path]":
    """Every `.py` file under `app/`, excluding anything under a `tests`
    directory -- the WHOLE tree, not scoped to `app/coach/`. Exposed as its
    own function (not inlined into the gate below) so
    `test_gate_twenty_the_scan_itself_reaches_modules_outside_the_coach_package`
    can assert the walk's own reach directly, independent of what it happens
    to find today -- the gate-17 shape (`:618-628`) applied to the scan
    rather than to the result."""
    app_dir = REPO_ROOT / "app"
    return [py_file for py_file in sorted(app_dir.rglob("*.py")) if "tests" not in py_file.parts]


def _modules_that_read_quoted_text() -> "set[str]":
    """AST-based, not regex -- this project's own S13 precedent (STATE.md:
    "a regex false-positived on the module's own explanatory docstring
    prose"). A regex version here would match `app/coach/prompts.py`'s
    extraction rubric PROSE (which talks ABOUT `quoted_text` without ever
    reading the field) and `app/coach/corpus.py`'s own docstring, forcing the
    pinned set below to be widened to include two modules that never
    actually read the value -- after which a REAL read inside either of them
    would pass this gate silently. That is worse than no gate at all.
    Collects `ast.Attribute` reads of the field name, plus string-literal
    subscript (`d["quoted_text"]`) and `.get("quoted_text")` access of the
    same name."""
    hits: "set[str]" = set()
    for py_file in _iter_non_test_app_modules():
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        relative = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == _QUOTED_TEXT_FIELD_NAME:
                hits.add(relative)
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and (
                node.slice.value == _QUOTED_TEXT_FIELD_NAME
            ):
                hits.add(relative)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == _QUOTED_TEXT_FIELD_NAME
            ):
                hits.add(relative)
    return hits


# --- Gate 20: the module set that may read a Fact's verbatim quote is pinned -------
# --- by equality, from an AST walk over the WHOLE of app/ --------------------------


def test_gate_twenty_the_set_of_modules_that_read_a_facts_verbatim_quote_is_pinned_by_equality():
    # T-06-01/T-06-02/GC-08. Equality, not a superset check (gate 8/18/19's
    # own precedent) -- a FOURTH reader must fail here first, before the
    # live-prompt/wire consequence is ever tested downstream.
    #
    # As of THIS commit, exactly three modules read the field: the offline
    # extraction path (`extract.py`, which populates it from the model),
    # `app/main.py:909` (the CLI's post-extraction review printout), and --
    # widened HERE, in the SAME commit that adds the read (06-03-PLAN.md §H4,
    # mirroring gate 8/18/19's "the protocol surface grows by decision, never
    # by drift" discipline; 06-01-SUMMARY.md's own deviation note named this
    # obligation explicitly) -- `app/session/render.py`, the offline,
    # single-operator post-call audit path 06-RESEARCH.md §6 identifies as
    # the one correct home for the value. Neither `app/coach/context.py`
    # (the live grounding seam -- §F3's mapping reads only
    # `.id`/`.claim`/`.session_id`/`.seqs`/`.provenance`, never
    # `.quoted_text`) nor `app/coach/prompts.py` (the live prompt assembler)
    # is in this set, which is the property GC-08 exists to hold.
    #
    # `app/web/review.py` joined in the commit that added it, and the
    # reasoning is worth stating rather than waving through, because this is
    # the first BROWSER-RENDERED reader of the field. GC-08's actual
    # prohibition is a previous customer's verbatim speech reaching a
    # DIFFERENT customer's live prompt, or the phone wire. The hub is
    # neither: it is loopback-bound, opened after an interview has ended, and
    # read by the operator alone -- the same offline single-operator audit
    # category `render.py` already occupies, and for the same reason (a quote
    # is how a reader checks whether a claim means what it says; the research
    # into comparable tools found evidence-buried-one-click-deep to be the
    # loudest complaint about screens like this).
    #
    # What keeps GC-08 true is unchanged and must stay so: neither
    # `app/coach/context.py` nor `app/coach/prompts.py` is in this set, so
    # nothing on the LIVE path can read the field at all, and gate 21's
    # sentinel check over the serialized request bodies is untouched.
    #
    # Note what this does NOT license: the hub renders quotes from every
    # session in a project, so it can show one customer's words while
    # reviewing another's interview. That is the operator reading their own
    # corpus at a desk, which is the point of a corpus -- but it is the
    # reason this page is loopback-only and never the phone.
    assert _modules_that_read_quoted_text() == {
        "app/coach/extract.py", "app/main.py", "app/session/render.py", "app/web/review.py",
    }, (
        "the set of modules that may read Fact.quoted_text changed -- a new reader must be "
        "either the offline audit path (added deliberately, in the same commit, with a "
        "stated reason) or is exactly the live-path leak GC-08/T-06-01 exists to catch"
    )


def test_quoted_text_scan_reaches_modules_outside_the_coach_package():
    # NOT itself a numbered gate (deliberately outside the `^def test_gate`
    # count the plan's own arithmetic tracks) -- this proves gate 20's own
    # SCAN HELPER's reach, independent of the pinned result set above. The
    # gate-17 shape, applied to the scan rather than to its result: a scan
    # narrowed to app/coach/ would pass gate 20 cheerfully even after a real
    # leak into, say, app/web/hub.py, since the scan would never visit that
    # file at all. This proves the walk's own reach is the whole of app/,
    # independent of what it happens to find in the pinned set above.
    scanned = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in _iter_non_test_app_modules()}
    assert any(p.startswith("app/web/") for p in scanned), "the scan must reach app/web/"
    assert any(p.startswith("app/session/") for p in scanned), "the scan must reach app/session/"


# --- Gate 21: a fact's verbatim quote reaches no serialized request body -----------


def test_gate_twenty_one_a_facts_verbatim_quote_reaches_no_serialized_request_body():
    # T-06-01/T-06-02/GC-08's behavioural backstop. Builds a `Fact` whose
    # `quoted_text` is a sentinel this test itself generates, maps it through
    # the REAL `context._fact_to_context_item`, then serializes the WHOLE
    # request dict both `generate.assemble_generate_request` and
    # `critic.assemble_critique_request` return -- not a chosen sub-part.
    # Asserting only `ContextItem.text == Fact.claim` would pass while the
    # quote leaked through `.source` or `.citation` (e.g. a citation built as
    # the session id plus a fragment of the quote) -- the mutation-test below
    # proves exactly that gap.
    sentinel = "gate-twenty-one-sentinel-quote-9f3c7a2e"
    fact = coach_corpus_mod.Fact(
        id="f1", session_id="sess-1", seqs=(1,),
        claim="an abstracted claim, never the verbatim quote",
        provenance="evidence", quoted_text=sentinel, bearings=(),
    )
    item = coach_context_mod._fact_to_context_item(fact)  # noqa: SLF001 -- exercising the real mapping directly
    assert item.text == fact.claim, (
        "the cheap invariant, kept ALONGSIDE the whole-request check below, never instead of it"
    )

    static = coach_prompts_mod.static_block(brief=(item,), hypotheses=())
    assert sentinel not in static, "the sentinel leaked into the assembled static prompt block"

    generate_request = coach_generate_mod.assemble_generate_request(
        brief=(item,), survey=(), asked={}, transcript=[],
        trigger=SimpleNamespace(text="the interviewee just said something unrelated"),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100,
    )
    assert sentinel not in json.dumps(generate_request), (
        "the sentinel leaked into the serialized generate request"
    )

    critique_request = coach_critic_mod.assemble_critique_request(
        candidates=[Candidate(index=0, text="an unrelated candidate question", move="advance")],
        hypotheses=(), survey=(), trigger_text="an unrelated thing the interviewee said",
        model="m", max_tokens=100,
    )
    assert sentinel not in json.dumps(critique_request), (
        "the sentinel leaked into the serialized critique request"
    )


# --- Gate 22: the coaching latency record cannot omit the session-start -----------
# --- grounding cost (06-01-PLAN.md §F4/§F6, GC-06) ----------------------------------


def test_gate_twenty_two_the_coaching_latency_record_cannot_omit_the_session_start_grounding_cost():
    import inspect

    from app.coach.pipeline import SuggestionPipeline, pipeline_latency_summary

    # Half one: the summary's key set, by EQUALITY (gate 8/18/19's own
    # precedent) -- a fourth key added anywhere else must fail here first.
    summary = pipeline_latency_summary([], session_start_grounding_s=0.0)
    assert set(summary) == {"first_cycle_s", "stats", "session_start_grounding_s"}, (
        "the coaching latency summary's key set changed -- criterion 6 is judged from exactly "
        "these three numbers together, never a subset that silently drops the grounding cost"
    )

    # Half two, and per §F6 "the reason the no-default assertion is half the
    # gate": a presence-only check (`"session_start_grounding_s" in summary`)
    # would keep passing even if the constructor later grew a default of
    # `0.0` "so the tests are easier to write" -- after which every pipeline
    # built without measuring reports a grounding cost of zero, and criterion
    # 6 would be judged from a number that structurally cannot see retrieval.
    # Asserting the SIGNATURE's default is empty is what the presence check
    # above cannot catch -- see the mutation-test in 06-01-SUMMARY.md.
    parameter = inspect.signature(SuggestionPipeline.__init__).parameters["session_start_grounding_s"]
    assert parameter.default is inspect.Parameter.empty, (
        "SuggestionPipeline.__init__'s session_start_grounding_s parameter must have NO DEFAULT -- "
        "a default here is this phase's signature failure mode (§F6): every pipeline built without "
        "it would silently report zero retrieval cost, and ROADMAP criterion 6 would be judged from "
        "a number that structurally excludes what this phase added"
    )


# --- Gate 23: the move set is pinned AND the model is actually told about it -------
# --- (06-02-PLAN.md §G6, GC-03) ------------------------------------------------------


def test_gate_twenty_three_the_move_set_is_pinned_and_the_prompt_actually_names_every_move():
    # Half one: schema<->ALL_MOVES equality, gate 8/18/19's own precedent --
    # a fifth value anywhere must fail here first.
    assert coach_generate_mod.ALL_MOVES == {
        coach_generate_mod.MOVE_ADVANCE,
        coach_generate_mod.MOVE_DEEPEN,
        coach_generate_mod.MOVE_FILL_GAP,
        coach_generate_mod.MOVE_TEST_HYPOTHESIS,
        coach_generate_mod.MOVE_NO_QUESTION,
    }, "the move set changed -- ROADMAP criterion 3 is pinned against exactly these five values"
    schema = coach_generate_mod._generate_tool_schema()  # noqa: SLF001 -- exercising the real schema builder
    schema_moves = set(
        schema["input_schema"]["properties"]["candidates"]["items"]["properties"]["move"]["enum"]
    )
    assert schema_moves == coach_generate_mod.ALL_MOVES, "the schema enum is DERIVED and must equal ALL_MOVES exactly"

    # Half two -- the one that actually matters (§G6). The schema enum above
    # is DERIVED from ALL_MOVES and can never drift by itself; the hand-typed
    # prose in `_SYSTEM_PREAMBLE` and `decision_instruction` is what the
    # model actually reads. The mutation that would MISS half one alone: grow
    # ALL_MOVES, touch no prompt prose -- the new move becomes legal in the
    # schema and is never once described to the model, so it is legal but
    # never emitted, every offline test passes, and criterion 3 is claimed on
    # a move that never fires. This asserts every move name appears in the
    # prompt TEXT a real assembled request carries, not merely in the schema.
    request = coach_generate_mod.assemble_generate_request(
        brief=(), survey=(), asked={}, transcript=[],
        trigger=SimpleNamespace(text="something the interviewee said"),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100,
    )
    combined = "\n".join(block["text"] for block in request["system"]) + "\n" + request["messages"][-1]["content"]
    for move in coach_generate_mod.ALL_MOVES:
        assert move in combined, f"move {move!r} never appears in the assembled generate prompt text"


# --- Gate 24: the five critic flags agree by set equality across every -----------
# --- surface that names them (06-03-PLAN.md §H2, GC-04) ---------------------------
#
# 06-02 deliberately left this gate unwritten so it could be authored ONCE against
# a complete surface. The five surfaces, from PATTERNS §4: the critique schema's
# `properties`, the SAME schema's hand-typed `required` list, the `Judgment`
# dataclass, `SuggestionRecord`'s own fields, and render.py's `_SUGGESTION_FLAG_FIELDS`
# tuple. The mutation this gate is written against: a flag declared in `properties`
# and forgotten in `required` a few lines below -- a dataclass-only pin would miss
# it, the model omits the key, `_parse_critique_response` raises `KeyError`, and
# EVERY critique response on EVERY cycle becomes untrusted, presenting as a flaky
# vendor rather than as a bug.

_CRITIC_FLAG_NAMES = frozenset({
    "is_leading", "is_hypothetical_or_future_tense", "is_pitch",
    "is_opinion_seeking", "is_confirmation_only", "is_generic",
    "is_off_topic",
})


def test_gate_twenty_four_the_critic_flags_agree_by_equality_across_every_surface():
    # Equality, not a superset check (gate 8/18/19/23's own precedent) -- a flag
    # added to one surface and forgotten in another must fail HERE first, before
    # the runtime consequence (a whole critique response going untrusted, or a
    # flag silently vanishing from the post-call audit) is ever observed.
    schema = coach_critic_mod._critic_tool_schema()  # noqa: SLF001 -- exercising the real schema builder
    item_schema = schema["input_schema"]["properties"]["judgments"]["items"]
    schema_properties_flags = {name for name in item_schema["properties"] if name.startswith("is_")}
    schema_required_flags = {name for name in item_schema["required"] if name.startswith("is_")}
    judgment_flags = {f.name for f in dataclasses.fields(Judgment) if f.name.startswith("is_")}
    record_flags = {f.name for f in dataclasses.fields(SuggestionRecord) if f.name.startswith("is_")}
    renderer_flags = set(session_render_mod._SUGGESTION_FLAG_FIELDS)  # noqa: SLF001

    assert schema_properties_flags == _CRITIC_FLAG_NAMES, f"schema properties: {sorted(schema_properties_flags)}"
    assert schema_required_flags == _CRITIC_FLAG_NAMES, (
        f"schema required list: {sorted(schema_required_flags)} -- the sibling mistake this gate exists "
        "to catch is a flag declared in properties and forgotten here"
    )
    assert judgment_flags == _CRITIC_FLAG_NAMES, f"Judgment dataclass fields: {sorted(judgment_flags)}"
    assert record_flags == _CRITIC_FLAG_NAMES, f"SuggestionRecord dataclass fields: {sorted(record_flags)}"
    assert renderer_flags == _CRITIC_FLAG_NAMES, f"render.py's _SUGGESTION_FLAG_FIELDS: {sorted(renderer_flags)}"

    # Second half, mirroring gate 23's own second half: every flag name must
    # actually reach the critique prompt TEXT the model reads, not merely live
    # in a schema -- a flag pinned everywhere above but never described to the
    # model would be legal but never meaningfully judged.
    request = coach_critic_mod.assemble_critique_request(
        candidates=[Candidate(index=0, text="a candidate", move="advance")],
        hypotheses=(), survey=(), trigger_text="something the interviewee just said",
        model="m", max_tokens=100,
    )
    # `system` is a list of content blocks (it carries a cache breakpoint --
    # see `assemble_critique_request`), so the gate reads the block TEXT. The
    # thing being pinned is unchanged: every flag must reach the prose the
    # model actually reads, not merely live in the tool schema.
    system_text = "\n".join(block["text"] for block in request["system"])
    for flag in _CRITIC_FLAG_NAMES:
        assert flag in system_text, f"{flag!r} never appears in the assembled critique prompt text"


# --- Gate 25: a proposal is a proposal -- the hypothesis artifact and the --------
# --- whole repository are untouched by a cycle that genuinely proposed a --------
# --- state change (06-03-PLAN.md §H3, GC-05, D-01) --------------------------------


def test_gate_twenty_five_a_genuine_proposal_leaves_the_hypothesis_artifact_and_the_whole_repo_untouched():
    # Mirrors gate 17's own shape exactly (this file's own comment at
    # test_gate_seventeen's definition): real repository trees, not tmp_path, and
    # an UNSCOPED git-status diff -- a context-dir-scoped check could only ever see
    # the path it already expected to be touched, precisely the blind spot gate 17
    # was written to close after a mutated path leaked a file to the repo root.
    context_dir = REPO_ROOT / "context"
    hypotheses_path = context_dir / "hypotheses.json"
    backup = hypotheses_path.read_bytes() if hypotheses_path.is_file() else None
    context_dir.mkdir(parents=True, exist_ok=True)

    def _tree_status() -> "set[str]":
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        return {line for line in completed.stdout.splitlines() if line.strip()}

    fixture_bytes = json.dumps([{
        "id": "gate25-h1", "statement": "onboarding delay drives churn",
        "falsifier": "churned customers cite a different top reason more often", "state": "untested",
    }]).encode("utf-8")
    hypotheses_path.write_bytes(fixture_bytes)

    before = _tree_status()
    try:
        hypothesis = coach_corpus_mod.load_hypotheses(context_dir, missing_ok=False)[0]

        sessions_root = REPO_ROOT / "sessions"
        session_dir = create_session_dir(sessions_root)
        store = SessionStore(session_dir)
        store.write_metadata({"purpose": "gate25-probe"})
        trigger = store.append_segment(
            speaker="interviewee", text="gate twenty-five probe utterance",
            audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.1,
            detected_language="en", engine="fake",
        )

        stub = StubAnthropicClient(responses=[
            tool_use_response(coach_generate_mod.GENERATE_TOOL_NAME, {"candidates": [{
                "text": "does the churn reason still hold today?", "move": "test_hypothesis",
                "target_hypothesis_id": hypothesis.id, "proposed_state": "supported",
            }]}),
            tool_use_response(coach_critic_mod.CRITIC_TOOL_NAME, {"judgments": [{
                "index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
                "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False,
                "is_generic": False,
                "band": coach_critic_mod.BAND_CLEAN, "rationale": "fine",
            }]}),
        ])
        pipeline = SuggestionPipeline(
            client=stub, store=store, brief=(), survey=(), hypotheses=(hypothesis,),
            session_start_grounding_s=0.0,
            coverage_state_provider=lambda: CoverageState(asked=frozenset()),
            model="m", generate_timeout_s=1.0, critique_timeout_s=1.0,
            generate_max_tokens=200, critique_max_tokens=200,
            max_shown=3, failure_threshold=3,
            language_window_s=90.0, language_flip_margin=0.65, language_min_chars=40,
        )
        pipeline._process(trigger)  # noqa: SLF001 -- driving one cycle directly, matching test_coach_audit.py's own pattern
        store.close()

        result = read_session(session_dir)
        suggestions = result.records_of_kind("suggestion")
        assert len(suggestions) == 1, "sanity: the probe cycle must have produced exactly one candidate"
        assert suggestions[0]["proposed_state"] == "supported", (
            "step 1: the cycle actually produced a proposal -- a gate that never exercised one is vacuous"
        )
        assert suggestions[0]["target_hypothesis_id"] == hypothesis.id

        after_bytes = hypotheses_path.read_bytes()
        assert after_bytes == fixture_bytes, (
            "step 2: context/hypotheses.json must be byte-identical after a cycle that genuinely "
            "proposed a state change -- nothing on the live path may ever write it (D-01)"
        )

        appeared = _tree_status() - before
        assert appeared == set(), (
            "step 3: a proposal cycle left something committable ANYWHERE in this repository -- "
            f"{sorted(appeared)!r}"
        )
    finally:
        if backup is not None:
            hypotheses_path.write_bytes(backup)
        elif hypotheses_path.is_file():
            hypotheses_path.unlink()
        assert (hypotheses_path.read_bytes() if hypotheses_path.is_file() else None) == backup, (
            "gate 25 must restore whatever hypotheses corpus was there before it ran -- a gate that can "
            "destroy a real corpus is not one anybody will keep running"
        )


# --- Gate 26: the live coach is constructed with EVERY input the pipeline accepts ----
#
# Found 2026-08-10, by reading `run_start` rather than by any test: `run_start`
# loaded `coach_hypotheses`, printed its count to the console so the wiring
# LOOKED done, and never passed it to `SuggestionPipeline`. The pipeline
# defaulted `hypotheses` to `()`, so the live coach ran every cycle -- generate
# AND critique -- against an empty hypothesis set. Phase 6's entire premise (a
# coach that proposes the question which would validate or KILL an open
# hypothesis) was switched off in production while the whole suite stayed green
# and the coach reported healthy.
#
# Gate 22 anticipated exactly this shape and says so in its own comment: it
# asserts `session_start_grounding_s` has NO DEFAULT specifically as a forcing
# function on callers. That mitigation was applied to one of the two new
# constructor parameters. `hypotheses` got a compatibility default instead, and
# it is the one that was missed.
#
# So this gate is deliberately NOT "main.py passes hypotheses". A gate that
# named `hypotheses` would catch only the failure that already happened and
# would miss the next parameter added with a default -- the identical bug one
# argument along. It pins the keyword set by EQUALITY (gate 8/18/19/22's own
# precedent), so ANY future parameter must be threaded through the live caller
# or fail here first.


def _assert_live_caller_passes_every_parameter(cls) -> None:
    """The body of gates 26 and 27. Reads `app/main.py` by AST, finds the ONE
    construction of `cls`, and asserts its keyword set equals the
    constructor's parameter set.

    Shared rather than duplicated because the property is one property held by
    two classes, and a copy would drift: the second copy is exactly where a
    future reader relaxes an assertion "just for this one"."""
    import ast
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}

    main_source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(main_source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == cls.__name__
    ]
    assert len(calls) == 1, (
        f"expected exactly one {cls.__name__}(...) construction in app/main.py, found {len(calls)} -- "
        "this gate reads the live caller by AST, so a second construction site means it is now only "
        "checking one of them"
    )

    # `**kwargs` would let a caller look complete while passing anything at all,
    # which is precisely the kind of "looks wired" the original defect had.
    assert all(kw.arg is not None for kw in calls[0].keywords), (
        f"app/main.py must pass {cls.__name__}'s arguments as explicit keywords, never **kwargs -- "
        "a splat makes this gate unable to see what is actually being passed"
    )
    passed = {kw.arg for kw in calls[0].keywords}

    assert passed == accepted, (
        f"app/main.py's {cls.__name__}(...) call must pass EVERY parameter the constructor accepts, "
        f"by set equality. Missing from the live caller: {sorted(accepted - passed)!r}. "
        f"Passed but not accepted: {sorted(passed - accepted)!r}. "
        "A parameter with a default that the live caller never passes is silent: the coach starts, "
        "reports healthy, and runs degraded forever. That is exactly how `hypotheses` was lost."
    )


def test_gate_twenty_six_the_live_caller_passes_every_suggestion_pipeline_parameter():
    from app.coach.pipeline import SuggestionPipeline

    _assert_live_caller_passes_every_parameter(SuggestionPipeline)


# --- Gate 27: the same, for the OTHER coach constructor the live path builds --------
#
# Gate 26 was written against `SuggestionPipeline` because that is where the
# `hypotheses` defect happened. But the defect's SHAPE -- a constructor
# parameter with a compatibility default that the one live caller never passes,
# so the feature is off in production while the suite stays green -- has nothing
# to do with which class it was. `CoverageDetector` is constructed on exactly
# the same live path, from exactly the same function, and Phase C gives it a
# new defaulted parameter (`on_change`) of exactly that shape: omit it and the
# next-question panel simply never updates, silently, while coverage itself
# keeps working perfectly and every test stays green.
#
# Writing this as a second call to the shared helper rather than a second gate
# body is deliberate. It also means the NEXT constructor added to the live path
# is one line away from being covered, which is the only way this stays true.


def test_gate_twenty_seven_the_live_caller_passes_every_coverage_detector_parameter():
    from app.coach.coverage import CoverageDetector

    _assert_live_caller_passes_every_parameter(CoverageDetector)


# --- Gate 28: the survey wire payload's key set is exactly the next question -------
#
# Gate 15's precedent, applied to the message Phase C adds. Set equality, not a
# "contains" check: the risk here is the opposite of the suggestion payload's.
# There, the danger was leaking the critic's internals to the phone. Here, the
# temptation is to send the WHOLE survey "since we have it anyway" -- which
# turns a glanceable panel into a checklist to read (COACH-06) and puts every
# remaining question of a customer-specific survey onto a screen that may be
# sitting in a cafe.


def test_gate_twenty_eight_the_survey_wire_payload_key_set_is_exactly_the_next_question():
    from app.coach.context import SurveyItem
    from app.web.hub import WIRE_SURVEY_KEYS, survey_payload

    survey = (
        SurveyItem(id="q1", order=1, text="What are you trying to get done?"),
        SurveyItem(id="q2", order=2, text="What did you try before this?"),
    )
    payload = survey_payload(survey, asked=frozenset({"q1"}))
    assert set(payload) == WIRE_SURVEY_KEYS == {
        "type",
        "next_question_id",
        "next_question_text",
        "next_question_order",
        "asked_count",
        "total_count",
    }, (
        "the survey payload must carry ONLY the NEXT question and the position counts -- never the "
        "remaining questions, never the asked ids, never the survey itself"
    )
    # Not a redundant restatement of the key set: it proves the payload carries
    # the next UNASKED question rather than merely the right-shaped keys. A
    # mutation that always reports `survey[0]` passes the equality above.
    assert payload["next_question_id"] == "q2"
    assert payload["asked_count"] == 1 and payload["total_count"] == 2


# --- Gate 29: nothing under the project tree is ever committable -------------------

# Phase A's highest-risk change, and the reason the `projects/` ignore rule had
# to land in the SAME commit that first created the directory. Verified with
# real `git check-ignore` before any code wrote there: `projects/<n>/sessions/`
# was already covered (the `sessions/` rule is unanchored and matches at any
# depth), but `projects/<n>/context/` was NOT -- `context/*` is root-anchored
# and does not follow. Briefs, surveys, hypotheses and extracted facts under a
# project were committable.
#
# Gate 17's warning applies in full and is why this gate has TWO halves.
# `check-ignore` "answers a question about a path you NAME", so the first half
# is necessary and not sufficient; the second half actually writes a file and
# asserts the whole tree stays clean.
#
# A real subtlety found while exploring, worth stating because it inverts the
# usual intuition about which check is the safety net: git collapses an
# untracked directory to a single `?? projects/` line. So if `projects/` were
# ever left un-ignored, gate 17's before/after snapshot would go QUIET after
# the first run -- the one line is already present in `before` -- while a
# genuine leak continued underneath it. Ignoring by construction is what closes
# that hole. This gate guards the construction; it does not replace it.


def test_gate_twenty_nine_no_project_artifact_is_ever_committable():
    # Nested deliberately: a project path is a hierarchy (`acme/onboarding/
    # activation-delay`), so a rule that only covered the first level would
    # leak every child. `project.json` is in here too -- it is ignored on
    # purpose, because the blanket rule is what makes the negation trap
    # described in .gitignore unnecessary.
    for relative_path in (
        "projects/acme/project.json",
        "projects/acme/context/brief.json",
        "projects/acme/context/survey.json",
        "projects/acme/context/facts.json",
        "projects/acme/context/hypotheses.json",
        "projects/acme/sessions/20260810T120000/transcript.jsonl",
        "projects/acme/onboarding/activation-delay/context/facts.json",
    ):
        result = subprocess.run(["git", "check-ignore", "-q", relative_path], cwd=str(REPO_ROOT))
        assert result.returncode == 0, (
            f"{relative_path} is NOT reported as git-ignored -- a project's extracted facts and "
            f"brief carry verbatim customer speech"
        )

    # The shipped format samples live in the ROOT context/ precisely so the
    # rule above can be blanket. If someone ever moves them under projects/,
    # this asserts they did not silently become invisible instead.
    for relative_path in (
        "context/brief.example.json", "context/facts.example.json", "context/hypotheses.example.json",
    ):
        result = subprocess.run(["git", "check-ignore", "-q", relative_path], cwd=str(REPO_ROOT))
        assert result.returncode != 0, (
            f"{relative_path} must NOT be git-ignored -- it is a shipped format sample"
        )


def test_gate_twenty_nine_writing_a_real_project_artifact_leaves_the_tree_clean():
    # The half `check-ignore` cannot give: write actual bytes at an actual
    # path and compare a FULL, UNSCOPED tree snapshot, exactly as gate 17
    # does and for the identical reason. Never narrow this to a pathspec.
    def _tree_status() -> "set[str]":
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
        )
        return {line for line in completed.stdout.splitlines() if line.strip()}

    before = _tree_status()
    leaf = REPO_ROOT / "projects" / "_gate29" / "onboarding" / "context"
    created_root = REPO_ROOT / "projects" / "_gate29"
    assert not created_root.exists(), "gate 29 left a directory behind on a previous run"
    try:
        leaf.mkdir(parents=True)
        (leaf / "facts.json").write_text(
            '[{"claim": "they abandoned the tool after week two", "kind": "evidence"}]',
            encoding="utf-8",
        )
        (leaf.parent / "project.json").write_text('{"label": "onboarding"}', encoding="utf-8")
        after = _tree_status()
    finally:
        shutil.rmtree(created_root, ignore_errors=True)

    assert after == before, (
        "writing a project artifact changed what git reports as committable -- "
        f"appeared: {sorted(after - before)}"
    )


# --- Gate 30: every command that reads a root routes through the project seam ------

# Phase A's analogue of gates 26 and 27, and invented for the same reason they
# were: `--project` reaches a command through `_apply_project(load_config())`,
# and forgetting that wrapper on ONE command is silent. The command keeps
# working -- against the ROOT context/ and sessions/ -- so the suite stays
# green, the CLI accepts the flag, and the operator is told nothing. That is
# precisely the shape of the defect where `run_start` loaded
# `coach_hypotheses`, printed its count, and never passed it to the pipeline.
#
# Mutation-tested while writing it, and the result is why this gate exists at
# all: unwiring `run_render` failed only two CLI tests, and both failed
# because an INVALID path was no longer rejected there. A VALID
# `--project acme` on an unwired command changed nothing observable -- it just
# quietly read the wrong customer's brief. No behavioural test was going to
# catch that, so the check has to be structural.
#
# Written as "every load_config caller must route through the seam, unless
# named here" rather than "these four functions call _apply_project". The
# second form is satisfied forever by four correct call sites and says
# nothing about a FIFTH command added later -- which is the case that will
# actually happen.


_PROJECT_SEAM_EXEMPT = {
    # `precheck` touches neither root: it resolves devices and pings the
    # engine. It deliberately does not accept `--project` at all, and a test
    # in test_project_cli.py asserts argparse rejects the flag there. If this
    # ever gains a root read, delete it from this set rather than widening it.
    "run_precheck",
    # `console` has no --project: choosing one is what the page is FOR, and
    # the selection arrives from a request body rather than argparse. It does
    # not read a root itself either -- it hands the chosen paths to
    # `run_start`, which is inside the seam. Exempting the console rather
    # than widening the seam keeps the rule "argparse --project must be
    # honoured" exactly true.
    "run_console",
}


def _functions_that_load_config_without_the_project_seam() -> "set[str]":
    """AST, not a text scan -- the S13 precedent this project set after a
    regex false-positived on a docstring.

    A function is compliant when every `load_config(...)` call it contains is
    an ARGUMENT to `_apply_project(...)`. Checking containment that way,
    rather than "does the function mention `_apply_project` anywhere", is
    what makes a second unwrapped `load_config()` added beside a wrapped one
    still fail."""
    main_py = REPO_ROOT / "app" / "main.py"
    tree = ast.parse(main_py.read_text(encoding="utf-8"))

    def _is_load_config(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_config"
        )

    def _is_seam(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_apply_project"
        )

    offenders: "set[str]" = set()
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        wrapped = {
            id(argument)
            for node in ast.walk(function)
            if _is_seam(node)
            for argument in node.args
        }
        for node in ast.walk(function):
            if _is_load_config(node) and id(node) not in wrapped:
                offenders.add(function.name)
    return offenders


def test_gate_thirty_every_command_that_reads_a_root_routes_through_the_project_seam():
    assert _functions_that_load_config_without_the_project_seam() == _PROJECT_SEAM_EXEMPT, (
        "a command reads the config's roots without passing them through `_apply_project` -- "
        "so `--project` would be silently ignored there and the command would run against the "
        "ROOT context/ and sessions/ while reporting nothing. Wire it, or add it to "
        "_PROJECT_SEAM_EXEMPT with a stated reason"
    )


def test_gate_thirty_the_scan_finds_the_seam_when_it_is_removed():
    # The gate above passes when the scan finds nothing, which is also what a
    # broken scan returns. This asserts the scan can actually SEE an
    # unwrapped call, independent of what main.py happens to contain today --
    # gate 17's and gate 20's shape, applied to the walk rather than the
    # result.
    tree = ast.parse(
        "def run_thing(args):\n"
        "    cfg = config_mod.load_config()\n"
        "def run_wired(args):\n"
        "    cfg = _apply_project(config_mod.load_config(), args)\n"
    )
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert set(functions) == {"run_thing", "run_wired"}

    def _unwrapped(function: ast.FunctionDef) -> bool:
        wrapped = {
            id(argument)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_apply_project"
            for argument in node.args
        }
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load_config"
            and id(node) not in wrapped
            for node in ast.walk(function)
        )

    assert _unwrapped(functions["run_thing"]) is True
    assert _unwrapped(functions["run_wired"]) is False


# --- Gate 31: the live server call passes every create_app parameter --------------

# The third instance of the same forcing function (gates 26, 27), and added
# for the same reason: `create_app` grew `project` and `grounding_sources`
# WITH defaults, because it has many legitimate test callers that do not care
# about them. That is exactly the shape gate 26 exists to police -- a
# defaulted parameter the live caller can silently fail to pass, leaving the
# overlay reporting "no project" for a run that had one, while every test
# stays green.
#
# Deliberately NOT "does run_start pass project" -- that would miss the next
# defaulted parameter, which is the whole lesson of gate 26.


def test_gate_thirty_one_the_live_caller_passes_every_create_app_parameter():
    import inspect

    from app.web import server as web_server

    main_py = REPO_ROOT / "app" / "main.py"
    tree = ast.parse(main_py.read_text(encoding="utf-8"))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "create_app"
    ]
    assert len(calls) == 1, (
        f"expected exactly one `create_app` call in app/main.py, found {len(calls)} -- "
        "this gate's equality check is meaningless against two call sites"
    )

    passed = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    parameters = {
        name
        for name, parameter in inspect.signature(web_server.create_app).parameters.items()
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert passed == parameters, (
        "app/main.py's create_app call does not name every parameter -- a defaulted one it "
        f"fails to pass is silently wrong on the live path. missing: {sorted(parameters - passed)}; "
        f"unknown: {sorted(passed - parameters)}"
    )


# --- Gate 32: NO module under app/mcp/ reads a fact's verbatim quote ---------------
# --- (the MCP server is the widest egress surface in the project) ------------------


def _mcp_modules() -> "list[Path]":
    """Every non-test `.py` under `app/mcp/`. Its own walk rather than a
    filter over gate 20's result, so this gate can prove its own reach: a
    scan that found nothing because the package moved would otherwise pass
    an EMPTY-set assertion cheerfully forever, which is the one failure mode
    an emptiness gate has that an equality-to-a-nonempty-set gate does not."""
    mcp_dir = REPO_ROOT / "app" / "mcp"
    return [py for py in sorted(mcp_dir.rglob("*.py")) if "tests" not in py.parts]


def test_gate_thirty_two_no_mcp_module_reads_a_facts_verbatim_quote():
    # WHY THIS GATE EXISTS SEPARATELY FROM GATE 20. Gate 20 pins a set of
    # FOUR legitimate readers -- the extraction path, the CLI printout, the
    # offline renderer, and the loopback review hub. Every one of them is a
    # single operator reading their own corpus on their own machine. The MCP
    # server is categorically different: it is the widest egress surface in
    # this project, because it hands the corpus to a process the operator did
    # not write and did not audit, BY DESIGN. That is the whole point of the
    # server, and it is the reason the quote must never reach it.
    #
    # Folded into gate 20 this property would be invisible -- `app/mcp/` would
    # simply be absent from a set of four, indistinguishable from a module
    # that has not been written yet. Asserted here by EMPTINESS, scoped to one
    # package, a leak fails on its own name and its own reason.
    #
    # AST, never regex (gate 20's own S13 precedent): `app/mcp/__init__.py`
    # and `app/mcp/tools.py` both discuss `quoted_text` at length in prose,
    # explaining exactly why they never read it. A regex would trip on that
    # prose, forcing this set to be widened to include the two modules whose
    # docstrings promise the property -- after which a REAL read in either
    # would pass silently.
    scanned = {str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in _mcp_modules()}
    assert scanned, "the app/mcp/ scan found no modules at all -- an empty scan proves nothing"
    assert "app/mcp/tools.py" in scanned, (
        "the scan must reach the module that maps facts onto the wire -- that is the one "
        "file where a quote could plausibly be read"
    )

    hits = {path for path in _modules_that_read_quoted_text() if path.startswith("app/mcp/")}
    assert hits == set(), (
        f"a module under app/mcp/ reads Fact.quoted_text: {sorted(hits)}. This is not a set to "
        "widen. The MCP server leaves this machine's trust boundary by design, so a verbatim "
        "customer quote reaching it leaves the machine -- expose the abstracted `claim` instead"
    )


def test_gate_thirty_two_no_mcp_module_dumps_a_pydantic_model_wholesale():
    # The leak the AST gate above CANNOT see, and therefore the half of this
    # property that has to be asserted separately. `Fact` is a pydantic model
    # whose sixth field is `quoted_text`; `fact.model_dump()` puts the
    # verbatim quote on the wire while naming the field NOWHERE -- so it
    # passes gate 20, passes the gate above, and passes any reviewer grepping
    # for the string. Every projection under app/mcp/ is therefore written out
    # field by field, and this pins that.
    dumpers: "set[str]" = set()
    for py_file in _mcp_modules():
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        relative = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("model_dump", "model_dump_json")
            ):
                dumpers.add(relative)
    assert dumpers == set(), (
        f"a module under app/mcp/ serializes a pydantic model wholesale: {sorted(dumpers)}. "
        "A wholesale dump of `Fact` carries `quoted_text` onto the wire without ever naming "
        "it -- project the fields explicitly instead"
    )


# The constructs that bulk-copy a pydantic v2 model WITHOUT calling a method on
# it. Verified by experiment on 2026-08-12 against the real `Fact`, not assumed:
# `dict(f)`, `list(f)`, `tuple(f)`, `set(f)`, `sorted(f)`, `vars(f)`,
# `f.__dict__` and `for k, v in f:` every one of them yields the quote, because
# `BaseModel.__iter__` yields (field, value) pairs.
#
# `{**f}` was on the audit's list and does NOT leak -- it raises `TypeError:
# 'Fact' object is not a mapping`, because pydantic v2 defines `__iter__` but no
# `keys()`. Recorded because an untrue item on a threat list is worse than a
# missing one: it invites a guard against something that cannot happen and
# lends false confidence to the rest of the list.
_BULK_COPY_BUILTINS_WITH_NO_LEGITIMATE_USE_HERE = ("dict", "vars")


def test_gate_thirty_two_no_mcp_module_bulk_copies_an_object_through_iteration():
    # The `model_dump` check above is the leak everyone thinks of. This is the
    # one the 2026-08-12 audit found by attacking DIFFERENTLY, and it defeated
    # gate 20 as well: `return dict(fact)` puts `quoted_text` on the wire while
    # naming neither the attribute nor `model_dump`. Proven at the time by
    # appending a leak function to a module deliberately outside gate 20's
    # pinned set -- 13 tests passed.
    #
    # WHAT THIS GATE CAN AND CANNOT SEE, stated plainly because overstating a
    # guard's reach is how this project has been burned before. `list`, `tuple`,
    # `set` and `sorted` leak identically, and they are NOT checked here: this
    # package uses all four legitimately over ordinary collections
    # (`tuple(refs)`, `list(ALL_SCOPES)`, `sorted(_HANDLERS)`), and an AST scan
    # cannot tell a model from a tuple without type inference. Banning them
    # would force an allowlist of ten expressions that a reader would learn to
    # extend without thinking, which is worse than no gate.
    #
    # So this half covers only what has NO legitimate use in this package at
    # all, where the cost of the rule is exactly zero: `dict(x)` and `vars(x)`
    # over a bare object, and any read of `__dict__`. The other five constructs
    # are covered by the gate BELOW, which watches the bytes rather than the
    # syntax and therefore does not care which one was used.
    offenders: "set[str]" = set()
    for py_file in _mcp_modules():
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        relative = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                offenders.add(f"{relative}:{node.lineno} __dict__")
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _BULK_COPY_BUILTINS_WITH_NO_LEGITIMATE_USE_HERE
                and len(node.args) == 1
            ):
                continue
            # A literal, comprehension or generator argument constructs a NEW
            # value from parts the author wrote out, which is the safe form and
            # the one this package already uses. A bare name or attribute is an
            # object of unknown type being copied whole.
            if isinstance(node.args[0], (ast.Name, ast.Attribute)):
                offenders.add(f"{relative}:{node.lineno} {node.func.id}()")

    assert offenders == set(), (
        f"a module under app/mcp/ bulk-copies an object through iteration: {sorted(offenders)}. "
        "`dict(fact)` yields every field of a pydantic model including `quoted_text`, while "
        "naming neither the attribute nor `model_dump` -- so it passes gate 20, passes the "
        "dump check, and passes a reviewer grepping for the string. Project the fields "
        "explicitly, as every other mapping in this package does"
    )


# --- Gate 33: no tool this server serves puts a verbatim quote on the wire ---------
# --- (behavioural, over the real protocol bytes, every tool in the registry) -------


def test_gate_thirty_three_no_registered_mcp_tool_emits_a_verbatim_quote(tmp_path):
    # WHY A BEHAVIOURAL GATE, WHEN THREE STRUCTURAL ONES ALREADY GUARD THIS.
    # Gates 20 and 32 are SYNTAX gates: they answer "does any module name this
    # field, dump a model, or copy one whole". Every one of them was defeated
    # on 2026-08-12 by an attack nobody had written down, and the pattern was
    # uniform -- each guard is strong against the leak it can SEE and blind to
    # the same leak one language feature across. `dict(fact)` was that feature;
    # `getattr(fact, name)` in a loop is the next one, and it would pass all
    # three.
    #
    # This gate does not care which construct was used. It builds a corpus whose
    # quote is a sentinel, drives EVERY TOOL IN THE REGISTRY through the real
    # stdio server, and reads the raw bytes that came back. A leak fails here
    # whatever wrote it.
    #
    # It is scoped to app/mcp/ on purpose and is NOT extended to the hub. The
    # hub renders quotes deliberately -- gate 20 lists `app/web/review.py` as a
    # sanctioned reader, because a quote is how the operator checks that a claim
    # means what it says, and that page is loopback-only and read by them alone.
    # The MCP server is the opposite case by design: it hands the corpus to a
    # process the operator did not write.
    sentinel = "gate-thirty-three-sentinel-quote-6b1d90fe"
    # A SECOND sentinel, on a field that is supposed to travel. Without it this
    # whole test passes vacuously the day the harness stops driving the server
    # -- an empty response contains no sentinel either, and "no leak found" and
    # "nothing was examined" are the same green tick. This project has shipped
    # that exact vacuous pass twice.
    claim_sentinel = "gate-thirty-three-sentinel-claim-2e77c418"
    session_id = "20260812-101500-gate33"

    projects_dir = tmp_path / "projects"
    context_dir = projects_dir / "acme" / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "hypotheses.json").write_text(
        json.dumps([{"id": "h1", "statement": "a belief under test",
                     "falsifier": "three interviewees breeze through"}]),
        encoding="utf-8",
    )
    (context_dir / "survey.json").write_text(
        json.dumps([{"id": "q1", "order": 1, "text": "tell me about onboarding"}]),
        encoding="utf-8",
    )
    (context_dir / "facts.json").write_text(
        json.dumps([{
            "id": "f1", "session_id": session_id, "seqs": [1],
            "claim": claim_sentinel, "provenance": "evidence",
            "quoted_text": sentinel,
            "bearings": [{"hypothesis_id": "h1", "stance": "supports"}],
        }]),
        encoding="utf-8",
    )
    session_dir = projects_dir / "acme" / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps({"started_at": "2026-08-12T10:15:00+00:00"}), encoding="utf-8"
    )

    requests = []
    driven: "set[str]" = set()
    for index, definition in enumerate(mcp_tools_mod.TOOL_DEFINITIONS):
        name = definition["name"]
        properties = definition["inputSchema"].get("properties", {})
        arguments: "dict[str, object]" = {}
        if "project" in properties:
            arguments["project"] = "acme"
        if "scope" in properties:
            # The SUBTREE path specifically. A merge is where a projection is
            # most likely to be replaced by a bulk copy, because the merged
            # shape is the one someone writes a shortcut for.
            arguments["scope"] = "subtree"
        if "session_id" in properties:
            arguments["session_id"] = session_id
        if "hypothesis_id" in properties:
            arguments["hypothesis_id"] = "h1"
        requests.append({
            "jsonrpc": "2.0", "id": index + 1, "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        driven.add(name)

    # Pinned by EQUALITY against the registry, so a tool added later is driven
    # here or fails here. A gate over "the tools someone remembered" is the
    # scoped-pathspec mistake of gate 17, in a new place.
    assert driven == {d["name"] for d in mcp_tools_mod.TOOL_DEFINITIONS}
    assert driven == set(mcp_tools_mod._HANDLERS), (  # noqa: SLF001 -- the dispatch table is the property
        "a handler exists that no tool definition advertises, so it is reachable and unchecked"
    )

    payload = "".join(json.dumps(request) + "\n" for request in requests)
    stdout = io.BytesIO()
    mcp_server_mod.serve(io.BytesIO(payload.encode("utf-8")), stdout, repo_root=tmp_path)
    raw = stdout.getvalue().decode("utf-8")

    responses = [json.loads(line) for line in raw.split("\n") if line]
    assert len(responses) == len(requests), "the server did not answer every tool call"
    for response in responses:
        assert "error" not in response, (
            f"a tool failed, so its response proves nothing about leaking: {response}"
        )

    assert claim_sentinel in raw, (
        "the positive control never arrived -- the abstracted claim is supposed to travel, so "
        "its absence means this test examined nothing and the check below is vacuous"
    )
    assert sentinel not in raw, (
        "a verbatim customer quote crossed the MCP boundary. This is the widest egress surface "
        "in the project: it hands the corpus to a process the operator did not write. Whatever "
        "construct put it there -- an attribute read, model_dump(), dict(model), or a getattr "
        "loop -- project the abstracted `claim` instead"
    )


# --- Gate 34: the planning record is a customer-content channel too ---------------
#
# **Privacy B-1, found by audit and fixed 2026-08-13.** Every other privacy gate in
# this file guards a RUNTIME path: what reaches a log line, an event detail, a
# request body, the MCP boundary. All of them were satisfied while
# `.planning/STATE.md` -- tracked by git, pushed to a remote -- carried a verbatim
# list of the consumer brands a real customer named on a real call, plus verbatim
# Hebrew fragments lifted out of `transcript.jsonl`.
#
# It got there through the one channel nothing was watching. `.gitignore` covers
# `context/`, `sessions/`, `projects/` and `benchmark/data/` -- the places the
# CODE writes. Nothing writes `.planning/`; a person does, by hand, while writing
# up what a real call taught them. And `_EXCLUDED_DIR_NAMES` at the top of this
# file lists `.planning` explicitly, so gates 1/2/3's repository walk skips it by
# design.
#
# So: the artifact was neither ignored nor scanned. Two safety nets, one gap
# between them, and the leak lived exactly in the gap.
#
# **Why Hebrew script is the right signal, and its limits.** Every word this
# repository writes about itself is English. The interviews are Hebrew/English.
# Hebrew characters appearing in the rolling project record are therefore
# transcript-derived essentially by construction -- high precision, and it needs
# no list of customer names to check against (a list which would itself be the
# leak).
#
# It is deliberately NOT a repository-wide rule, which is why this gate names one
# file. `benchmark/fixtures/`, `benchmark/aliases.py`, `app/tests/rtl_probe.py`
# and the Phase 3 bidi research all contain Hebrew legitimately: synthetic
# fixtures, an alias table that must contain the terms it maps, and RTL rendering
# examples written before any real call existed. Widening this to the whole tree
# would fail on all of them and be switched off within a week.
#
# **What this gate does NOT do, stated so nobody mistakes it for done:** it cannot
# see English customer content -- a brand name in Latin script passes cleanly. It
# is a tripwire on the one signal that is cheap and precise, not a classifier. And
# it does nothing whatsoever about what is already in git history; the redaction
# that accompanied this gate stops the spread, it does not undo the commit.

_ROLLING_PLANNING_RECORD = REPO_ROOT / ".planning" / "STATE.md"

# U+0590-U+05FF: Hebrew, including presentation forms and cantillation.
_HEBREW_SCRIPT = re.compile(r"[\u0590-\u05FF]")


def test_gate_thirty_four_the_rolling_planning_record_carries_no_transcript_script():
    if not _ROLLING_PLANNING_RECORD.exists():
        pytest.skip("no .planning/STATE.md in this checkout")
    violations = [
        f"{_ROLLING_PLANNING_RECORD.relative_to(REPO_ROOT)}:{lineno}"
        for lineno, raw_line in enumerate(
            _ROLLING_PLANNING_RECORD.read_text(encoding="utf-8").splitlines(), start=1
        )
        if _HEBREW_SCRIPT.search(raw_line)
    ]
    # The LINE NUMBERS only, never the matching text. Gate 2's failure message
    # echoes the offending line verbatim, and this file's own header calls that
    # out as the sharper reason `projects/` is excluded from that walk: printing
    # the match would reprint customer speech into test output, which is the one
    # place nobody thought to check for it. A gate against a leak must not become
    # one.
    assert not violations, (
        "Hebrew script in the rolling planning record almost certainly came out of a real "
        "transcript, and this file is tracked by git and pushed to a remote. Describe what the "
        "speech DID -- 'a five-letter infinitive split after its second letter' -- instead of "
        "quoting it. Lines: " + ", ".join(violations)
    )


def test_gate_thirty_four_the_scan_would_catch_a_planted_fragment(tmp_path):
    """The gate above passes when the file is clean, which is also what it would
    do if the pattern were broken, the path were wrong, or the file were missing.
    Three ways to be vacuous and no way to tell them apart from success.

    So the scan is re-run here against a file that is KNOWN to contain a Hebrew
    fragment. This is the same discipline gate 30 and gate 33 already use -- a
    positive control beside the assertion, because a privacy gate that silently
    stopped scanning is worse than no gate: it reports safety it is not
    checking."""
    planted = tmp_path / "STATE.md"
    planted.write_text("a line that is fine\nand one that is \u05e7\u05dc\u05d5\u05d3 not\n", encoding="utf-8")
    hits = [
        lineno
        for lineno, raw_line in enumerate(planted.read_text(encoding="utf-8").splitlines(), start=1)
        if _HEBREW_SCRIPT.search(raw_line)
    ]
    assert hits == [2], "the Hebrew scan does not actually detect Hebrew"


# --- Gate 35: nothing READS the fabricated audio_start_s --------------------------


def test_gate_thirty_five_no_app_module_reads_the_fabricated_audio_start_s():
    """`audio_start_s` is not a measurement. `benchmark/engines/base.Segment`
    -- the contract every STT engine implements -- carries `audio_end_s` and
    no start at all, so there is no upstream start time to record.
    `app/stt/live.py` invents one from the SAME speaker's previous segment
    end, which means the value silently absorbs however long that speaker was
    quiet. Subtracting it gives a speaker's silence, not their speech, and it
    is biased toward whoever paused most -- in an interview, usually the
    interviewee, i.e. wrong in the direction that flatters the interviewer.

    Six modules already document this and route around it (`talk_share`,
    `corpus`, `trigger`, `asked`, `render`, `hub`). Not ONE of them enforces
    it, and a seventh consumer arriving tomorrow would find a plausible float
    on a dataclass with nothing to warn them. That is what this gate is: the
    scattered discipline made structural.

    AST, never a text scan -- and here that is not a stylistic preference.
    Every module named above mentions `audio_start_s` in its prose precisely
    BECAUSE it is being careful, so a text gate would fail on exactly the
    files that are doing the right thing (the same trap this project's S13
    precedent already hit, and that a `survey_hold` test hit again on its own
    docstring).
    """
    reads: "list[str]" = []
    for py_file in sorted((REPO_ROOT / "app").rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        if "audio_start_s" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            # An ATTRIBUTE access is a read: someone is using the value.
            # A keyword argument is a WRITE, pinned separately below.
            if isinstance(node, ast.Attribute) and node.attr == "audio_start_s":
                reads.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
    assert reads == [], (
        "audio_start_s is FABRICATED from the previous segment's end and means "
        "nothing on its own -- these read it: " + ", ".join(reads) + ". If you need "
        "a segment's duration, it is not recoverable from what is stored; the fix "
        "is upstream in what app/stt/live.py records, not a subtraction here."
    )


def test_gate_thirty_five_the_set_of_modules_that_write_audio_start_s_is_pinned():
    """Pinned by EQUALITY, not by subset: a new writer is as much a change to
    this contract as a new reader. `store.py` declares and persists the field;
    `live.py` is the one place the fabrication happens."""
    writers: "list[str]" = []
    for py_file in sorted((REPO_ROOT / "app").rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        if "audio_start_s" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.keyword) and node.arg == "audio_start_s":
                writers.append(py_file.name)
    assert sorted(set(writers)) == ["live.py", "store.py"], (
        f"the set of modules that write audio_start_s changed -- got {sorted(set(writers))}"
    )


def test_gate_thirty_five_detects_a_read_it_is_meant_to_catch(tmp_path):
    """The gate's own mutation check: an AST scan that never fires on a real
    read is a gate that proves nothing. Also pins the docstring blindness that
    makes the AST necessary -- prose mentioning the field must NOT trip it."""
    planted = tmp_path / "consumer.py"
    planted.write_text(
        '"""This module discusses audio_start_s at length and must not trip."""\n'
        "def duration(segment):\n"
        "    return segment.audio_end_s - segment.audio_start_s\n",
        encoding="utf-8",
    )
    tree = ast.parse(planted.read_text(encoding="utf-8"))
    reads = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "audio_start_s"
    ]
    assert reads == [3], "the scan does not actually detect a read"

    prose_only = tmp_path / "careful.py"
    prose_only.write_text(
        '"""audio_start_s is fabricated, so this module never touches it."""\n'
        "# audio_start_s: do not use\n"
        "def duration(segment):\n"
        "    return None\n",
        encoding="utf-8",
    )
    prose_tree = ast.parse(prose_only.read_text(encoding="utf-8"))
    assert [
        node for node in ast.walk(prose_tree)
        if isinstance(node, ast.Attribute) and node.attr == "audio_start_s"
    ] == [], "a careful module's own docstring would trip this gate"
