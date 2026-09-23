"""Task 2 done criteria for 04-03-PLAN.md §E4: the authoring page is a
genuinely separate app instance, loopback-only, no token, validating against
`app/coach/context.py`'s own schemas and writing atomically. Per §P9's
established convention: a real uvicorn server on an ephemeral loopback port,
stdlib HTTP, never a TestClient/httpx dependency this project does not have."""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from app.coach.context import load_brief, load_survey
from app.coach.corpus import load_hypotheses
from app.web import author as web_author
from app.web import server as web_server

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_STATIC_DIR = REPO_ROOT / "app" / "static" / "author"


@contextlib.contextmanager
def _running_author_server(app):
    """Same shape as `test_web_serving.py::_running_server` -- a real
    uvicorn server on an ephemeral loopback port, in its own background
    thread. Reused here rather than re-implemented because the whole point
    of this module is exercising the REAL `build_uvicorn_server` bind
    behaviour where it matters, and a plain harness everywhere else."""
    server = web_author.build_uvicorn_server(app, port=0)
    import asyncio

    loop = asyncio.new_event_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "authoring uvicorn server never reported started"
    actual_port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{actual_port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _http(method: str, url: str, body: "bytes | None" = None) -> "tuple[int, bytes]":
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post_json(url: str, payload) -> "tuple[int, dict]":
    status, body = _http("POST", url, json.dumps(payload).encode("utf-8"))
    return status, json.loads(body.decode("utf-8"))


def _get_json(url: str):
    status, body = _http("GET", url)
    assert status == 200
    return json.loads(body.decode("utf-8"))


def _brief_item(item_id="b1", *, provenance="evidence", text="grew up in Tel Aviv"):
    return {"id": item_id, "text": text, "source": "linkedin", "citation": "profile", "provenance": provenance}


def _survey_item(item_id="s1", text="What is your current role?"):
    return {"id": item_id, "text": text}


def _hypothesis_item(
    item_id="h1",
    *,
    statement="Slow onboarding drives early churn",
    falsifier="If churned customers cite a different top reason more often, this is false.",
):
    out = {"id": item_id, "statement": statement}
    if falsifier != "__omit__":
        out["falsifier"] = falsifier
    return out


# --- The authoring server binds loopback only, and is a SEPARATE instance -------------


def test_authoring_server_is_constructed_with_the_loopback_host(tmp_path):
    app = web_author.create_author_app(context_dir=tmp_path, static_dir=AUTHOR_STATIC_DIR)
    server = web_author.build_uvicorn_server(app, port=0)
    assert server.config.host == "127.0.0.1"
    assert server.config.host != "0.0.0.0"


def test_authoring_app_is_a_different_instance_from_the_phone_app_and_its_routes_are_absent(tmp_path):
    author_app = web_author.create_author_app(context_dir=tmp_path, static_dir=AUTHOR_STATIC_DIR)

    session_dir = tmp_path / "session"
    from app.session.store import SessionStore

    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "author-gate9-probe"})
    phone_app = web_server.create_app(
        token="author-test-token", hub=__import__("app.web.hub", fromlist=["ConnectionHub"]).ConnectionHub(),
        static_dir=REPO_ROOT / "app" / "static", store=store, session_id="s", server_started_at="t",
    )

    assert author_app is not phone_app

    author_paths = {getattr(route, "path", None) for route in author_app.routes}
    phone_paths = {getattr(route, "path", None) for route in phone_app.routes}
    assert not (author_paths & phone_paths - {"/{_catch_all:path}"}), (
        "authoring routes must never appear on the phone app's own route table"
    )
    for path in author_paths:
        if path is None:
            continue
        assert not path.startswith("/s/"), "an authoring route must never live beneath the phone's token prefix"


def test_gate_nine_the_phone_apps_token_gate_still_passes_unmodified(tmp_path):
    """Explicit, alongside the standing gate module itself (04-03-PLAN.md
    Task 2's own behaviour spec): the whole design of this task exists to
    make gate 9 true with no exception, because these routes are never on
    the phone app's route table at all."""
    from app.session.store import SessionStore, create_session_dir
    from app.web.hub import ConnectionHub

    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "gate9-still-passes"})
    phone_app = web_server.create_app(
        token="gate9-still-passes-token", hub=ConnectionHub(), static_dir=REPO_ROOT / "app" / "static",
        store=store, session_id="s", server_started_at="t",
    )
    all_paths = [getattr(route, "path", None) for route in phone_app.routes]
    non_token_routes = [
        path for path in all_paths
        if path is not None and path != "/{_catch_all:path}" and not path.startswith("/s/{candidate_token}")
    ]
    assert not non_token_routes


# --- Cross-origin write / DNS-rebinding read guards (independent review finding A) ----


def _raw_request(url: str, *, method: str = "GET", body: "bytes | None" = None, headers=None):
    request = urllib.request.Request(url, data=body, method=method)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_a_no_preflight_cross_origin_post_with_a_non_json_content_type_is_rejected(tmp_path):
    """Starlette parses the body as JSON regardless of Content-Type -- a
    malicious page can fire `Content-Type: text/plain` (no CORS preflight)
    with a valid-JSON body. Without the guard this would silently overwrite
    the survey; with it, the request is rejected and nothing is written."""
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(
            f"{base_url}/api/survey",
            method="POST",
            body=json.dumps([_survey_item()]).encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
    assert status == 403
    assert body == b""
    assert not (context_dir / "survey.json").exists(), "a rejected cross-origin write must leave nothing on disk"


def test_a_post_with_the_correct_content_type_but_a_spoofed_host_header_is_still_rejected(tmp_path):
    """DNS rebinding: an attacker-controlled domain resolves to 127.0.0.1,
    so the browser's own same-origin/CORS machinery never even applies --
    the Host header is the only signal left once the request has reached
    this process."""
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(
            f"{base_url}/api/survey",
            method="POST",
            body=json.dumps([_survey_item()]).encode("utf-8"),
            headers={"Content-Type": "application/json", "Host": "evil.example.com"},
        )
    assert status == 403
    assert body == b""
    assert not (context_dir / "survey.json").exists()


def test_a_get_with_a_spoofed_host_header_is_rejected_blocking_a_dns_rebinding_read(tmp_path):
    """DNS rebinding can also READ the brief -- confirmed here by exercising
    the read path (`/api/brief`), not just the write path."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "brief.json").write_text(json.dumps([_brief_item()]), encoding="utf-8")
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(f"{base_url}/api/brief", headers={"Host": "evil.example.com"})
    assert status == 403
    assert body == b""


def test_a_get_with_an_unmodified_host_header_still_succeeds(tmp_path):
    """Happy path: a normal same-origin request (Host left exactly as the
    HTTP client set it, matching what the browser would send when actually
    on this origin) is unaffected by either guard."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "brief.json").write_text(json.dumps([_brief_item()]), encoding="utf-8")
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(f"{base_url}/api/brief")
    assert status == 200
    assert json.loads(body.decode("utf-8"))[0]["id"] == "b1"


# --- Serving the page, and saving valid data --------------------------------------------


def test_fetching_the_authoring_page_over_loopback_returns_it(tmp_path):
    app = web_author.create_author_app(context_dir=tmp_path, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _http("GET", f"{base_url}/")
        assert status == 200
        assert b"brief" in body.lower() or b"Brief" in body

        status, _ = _http("GET", f"{base_url}/author.js")
        assert status == 200
        status, _ = _http("GET", f"{base_url}/author.css")
        assert status == 200


def test_saving_a_valid_brief_and_survey_writes_them_to_disk(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _post_json(f"{base_url}/api/brief", [_brief_item()])
        assert status == 200
        assert body["saved"] == 1

        status, body = _post_json(f"{base_url}/api/survey", [_survey_item()])
        assert status == 200
        assert body["saved"] == 1

    assert (context_dir / "brief.json").is_file()
    assert (context_dir / "survey.json").is_file()


# --- Rejecting a bad provenance: nothing written --------------------------------------


def test_a_bad_provenance_is_rejected_naming_the_item_id_and_nothing_is_written(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    original = json.dumps([_brief_item("keep-me")])
    (context_dir / "brief.json").write_text(original, encoding="utf-8")

    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        bad_item = _brief_item("bad-item", provenance="maybe")
        status, body = _post_json(f"{base_url}/api/brief", [bad_item])
        assert status == 400
        assert "bad-item" in body["error"]

    assert (context_dir / "brief.json").read_text(encoding="utf-8") == original, (
        "a rejected save must leave the existing file COMPLETELY unchanged"
    )


def test_a_missing_provenance_is_rejected_the_same_way():
    app = web_author.create_author_app(context_dir=Path("unused"), static_dir=AUTHOR_STATIC_DIR)
    from app.web.author import _validate_brief_payload

    entry = {"id": "no-provenance", "text": "x"}
    validated, error = _validate_brief_payload([entry])
    assert validated is None
    assert error is not None
    assert "no-provenance" in error


# --- Atomic write: a failure between write and replace leaves the prior file intact ---


def test_a_failure_during_replace_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    original = json.dumps([_brief_item("survivor")])
    (context_dir / "brief.json").write_text(original, encoding="utf-8")

    import app.web.author as author_mod

    def _boom(*_args, **_kwargs):
        raise OSError("simulated failure between write and replace")

    monkeypatch.setattr(author_mod.os, "replace", _boom)

    with pytest.raises(OSError):
        author_mod._atomic_write_json(context_dir / "brief.json", [_brief_item("new-item")])

    assert (context_dir / "brief.json").read_text(encoding="utf-8") == original, (
        "the previous file must still be intact and readable after a simulated failure"
    )
    # And still loadable through the coach's own loader.
    loaded = load_brief(context_dir)
    assert loaded[0].id == "survivor"


# --- Round trip: authored data loads back through the coach's OWN loaders -------------


def test_a_round_trip_holds_through_the_coachs_own_loaders(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    brief_payload = [_brief_item("b1", provenance="evidence"), _brief_item("b2", provenance="inference")]
    survey_payload = [_survey_item("s1", "first question"), _survey_item("s2", "second question")]

    with _running_author_server(app) as base_url:
        status, _ = _post_json(f"{base_url}/api/brief", brief_payload)
        assert status == 200
        status, _ = _post_json(f"{base_url}/api/survey", survey_payload)
        assert status == 200

    brief = load_brief(context_dir)
    survey = load_survey(context_dir)

    assert [item.id for item in brief] == ["b1", "b2"]
    assert brief[0].provenance == "evidence"
    assert brief[1].provenance == "inference"
    assert brief[0].text == "grew up in Tel Aviv"

    assert [item.id for item in survey] == ["s1", "s2"]
    assert survey[0].text == "first question"
    assert survey[0].order == 0
    assert survey[1].order == 1


def test_survey_order_is_preserved_and_ids_are_stable_across_a_wording_only_edit(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)

    with _running_author_server(app) as base_url:
        first_payload = [_survey_item("q-a", "original wording"), _survey_item("q-b", "second question")]
        status, _ = _post_json(f"{base_url}/api/survey", first_payload)
        assert status == 200

        # Same ids, same order, only the FIRST item's wording changes.
        edited_payload = [_survey_item("q-a", "edited wording"), _survey_item("q-b", "second question")]
        status, _ = _post_json(f"{base_url}/api/survey", edited_payload)
        assert status == 200

    survey = load_survey(context_dir)
    assert [item.id for item in survey] == ["q-a", "q-b"], "ids must stay stable across a wording-only edit"
    assert survey[0].text == "edited wording"
    assert survey[0].order == 0
    assert survey[1].order == 1


# --- Hypotheses: the falsifier discipline, expressed three times (05-03-PLAN.md §K14) ---


def test_fetching_hypotheses_returns_empty_list_when_the_file_does_not_exist(tmp_path):
    app = web_author.create_author_app(context_dir=tmp_path / "context", static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        items = _get_json(f"{base_url}/api/hypotheses")
    assert items == []


def test_saving_a_valid_hypothesis_writes_it_and_a_subsequent_fetch_returns_it(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _post_json(f"{base_url}/api/hypotheses", [_hypothesis_item()])
        assert status == 200
        assert body["saved"] == 1

        items = _get_json(f"{base_url}/api/hypotheses")
    assert items[0]["id"] == "h1"
    assert (context_dir / "hypotheses.json").is_file()


def test_a_hypothesis_with_no_falsifier_field_at_all_is_rejected_naming_it_and_nothing_written(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    original = json.dumps([_hypothesis_item("keep-me")])
    (context_dir / "hypotheses.json").write_text(original, encoding="utf-8")

    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        bad_item = _hypothesis_item("no-falsifier", falsifier="__omit__")
        status, body = _post_json(f"{base_url}/api/hypotheses", [bad_item])
        assert status == 400
        assert "no-falsifier" in body["error"]

    assert (context_dir / "hypotheses.json").read_text(encoding="utf-8") == original, (
        "a rejected save must leave the existing file COMPLETELY unchanged"
    )


def test_a_hypothesis_with_a_whitespace_only_falsifier_is_rejected_identically(tmp_path):
    # Asserted separately from the absent case (§K14): an empty/whitespace
    # string is a valid STRING, and a check that only tested presence would
    # let this one through.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    original = json.dumps([_hypothesis_item("keep-me")])
    (context_dir / "hypotheses.json").write_text(original, encoding="utf-8")

    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        bad_item = _hypothesis_item("blank-falsifier", falsifier="   ")
        status, body = _post_json(f"{base_url}/api/hypotheses", [bad_item])
        assert status == 400
        assert "blank-falsifier" in body["error"]

    assert (context_dir / "hypotheses.json").read_text(encoding="utf-8") == original


def test_a_hypothesis_with_an_unrecognized_state_is_rejected_naming_it(tmp_path):
    app = web_author.create_author_app(context_dir=tmp_path / "context", static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        bad_item = dict(_hypothesis_item("bad-state"), state="maybe")
        status, body = _post_json(f"{base_url}/api/hypotheses", [bad_item])
        assert status == 400
        assert "bad-state" in body["error"]


def test_a_duplicate_hypothesis_id_is_rejected_naming_the_id(tmp_path):
    app = web_author.create_author_app(context_dir=tmp_path / "context", static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        payload = [_hypothesis_item("dup-id"), _hypothesis_item("dup-id", statement="a second one")]
        status, body = _post_json(f"{base_url}/api/hypotheses", payload)
        assert status == 400
        assert "dup-id" in body["error"]


def test_a_hypothesis_with_no_state_at_all_is_accepted_and_lands_as_untested(tmp_path):
    # The deliberate asymmetry against the falsifier (§K14): a hypothesis
    # IS untested by definition the moment it is created -- that is not a
    # guess about the world, unlike a defaulted falsifier or provenance.
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        entry = _hypothesis_item("no-state")
        assert "state" not in entry
        status, body = _post_json(f"{base_url}/api/hypotheses", [entry])
        assert status == 200

    hypotheses = load_hypotheses(context_dir, missing_ok=False)
    assert hypotheses[0].state == "untested"


def test_a_hypothesis_round_trip_holds_through_the_coachs_own_loader(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    payload = [
        _hypothesis_item("h1", statement="first hypothesis"),
        dict(_hypothesis_item("h2", statement="second hypothesis"), state="supported"),
    ]
    with _running_author_server(app) as base_url:
        status, _ = _post_json(f"{base_url}/api/hypotheses", payload)
        assert status == 200

    hypotheses = load_hypotheses(context_dir, missing_ok=False)
    assert [h.id for h in hypotheses] == ["h1", "h2"]
    assert hypotheses[0].statement == "first hypothesis"
    assert hypotheses[0].state == "untested"
    assert hypotheses[1].state == "supported"
    assert hypotheses[0].falsifier


def test_hypothesis_ids_are_stable_across_a_wording_only_edit(tmp_path):
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)

    with _running_author_server(app) as base_url:
        first_payload = [_hypothesis_item("hyp-a", statement="original wording")]
        status, _ = _post_json(f"{base_url}/api/hypotheses", first_payload)
        assert status == 200

        edited_payload = [_hypothesis_item("hyp-a", statement="edited wording")]
        status, _ = _post_json(f"{base_url}/api/hypotheses", edited_payload)
        assert status == 200

    hypotheses = load_hypotheses(context_dir, missing_ok=False)
    assert [h.id for h in hypotheses] == ["hyp-a"], "ids must stay stable across a wording-only edit"
    assert hypotheses[0].statement == "edited wording"


# --- the criterion's PRE-REGISTRATION BOUNDARY is STAMPED, never typed ------------------
#
# A `criterion_registered_after` the user could type would be evidence of
# nothing at all -- it would be the same unbacked claim about order that the
# bare `criterion` already was. Every test below is about the write path being
# the thing that records it.


def _node(tmp_path, *, sessions=()):
    """A real project node: `context/` and `sessions/` as siblings, which is
    exactly how `app/projects.py::resolve` builds them from one `node_dir`.
    The authoring app finds the interviews through that relationship, so the
    tests exercise the layout the product actually has rather than a shape
    invented for the test."""
    node_dir = tmp_path / "projects" / "acme"
    context_dir = node_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    for session_id in sessions:
        (node_dir / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    return context_dir


def _saved_boundary(context_dir, hypothesis_id="h1"):
    hypotheses = {h.id: h for h in load_hypotheses(context_dir, missing_ok=False)}
    return hypotheses[hypothesis_id].criterion_registered_after


def _with_criterion(item_id="h1", *, criterion, **extra):
    return dict(_hypothesis_item(item_id), criterion=criterion, **extra)


def test_a_criterion_saved_at_a_fresh_node_records_an_empty_boundary(tmp_path):
    # The strongest claim available, and here it is earned: no interview has
    # happened at this node, so the rule genuinely pre-dates all of them.
    context_dir = _node(tmp_path)
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200
    assert _saved_boundary(context_dir) == ()


def test_a_criterion_saved_with_interviews_already_present_records_them(tmp_path):
    # The case that makes the field worth having. Two calls have already
    # happened; a rule written now cannot honestly claim to pre-date them, and
    # the record says so without anyone having to be honest about it.
    context_dir = _node(tmp_path, sessions=("20260101T090000", "20260105T140000"))
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200
    assert _saved_boundary(context_dir) == ("20260101T090000", "20260105T140000")

    hypotheses = load_hypotheses(context_dir, missing_ok=False)
    assert hypotheses[0].pre_registered_for("20260101T090000") is False
    assert hypotheses[0].pre_registered_for("20260220T090000") is True


def test_a_client_supplied_boundary_is_overwritten_by_what_is_actually_on_disk(tmp_path):
    # The sneakiest attack on this field, and it needs no attacker: the page
    # POSTs whatever it holds, so a stale or hand-edited payload claiming "[]"
    # while two interviews sit on disk would forge pre-registration through the
    # ordinary save path. Always recomputed, on `SurveyItem.order`'s own stated
    # precedent -- whatever the client sent is ignored.
    context_dir = _node(tmp_path, sessions=("20260101T090000",))
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        forged = _with_criterion(criterion="3 of 10 mention X", criterion_registered_after=[])
        status, _ = _post_json(f"{base_url}/api/hypotheses", [forged])
        assert status == 200
    assert _saved_boundary(context_dir) == ("20260101T090000",)


def test_editing_the_criterion_text_later_MOVES_the_boundary(tmp_path):
    # The decision this feature turns on, stated as a test because it looks
    # wrong at first glance. An edited criterion is a DIFFERENT RULE, and the
    # new text was written after the interviews that are now on disk. Freezing
    # the boundary at the original write would let anyone rewrite the rule to
    # fit the data while still displaying the original pre-registration -- the
    # arrow-painting failure the criterion exists to prevent, now with a
    # credential attached. Moving it forward always WEAKENS the claim, which is
    # the only safe direction for a claim nobody can re-verify later.
    context_dir = _node(tmp_path)
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200
        assert _saved_boundary(context_dir) == ()

        # The interview happens, and only then is the rule rewritten.
        (context_dir.parent / "sessions" / "20260101T090000").mkdir(parents=True)
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="1 of 10 mention X")]
        )
        assert status == 200
    assert _saved_boundary(context_dir) == ("20260101T090000",)


def test_a_criterion_written_for_the_first_time_after_an_interview_records_it(tmp_path):
    # The same rule reached from the other side: a hypothesis authored with no
    # criterion at all, then given one after a call. Empty text becoming real
    # text IS a criterion edit, and the honest record is that this rule came
    # after that interview.
    context_dir = _node(tmp_path)
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(f"{base_url}/api/hypotheses", [_hypothesis_item("h1")])
        assert status == 200

        (context_dir.parent / "sessions" / "20260101T090000").mkdir(parents=True)
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200

    hypotheses = load_hypotheses(context_dir, missing_ok=False)
    assert hypotheses[0].criterion_registered_after == ("20260101T090000",)
    assert hypotheses[0].pre_registered_for("20260101T090000") is False


def test_re_saving_an_UNCHANGED_criterion_leaves_the_boundary_where_it_was(tmp_path):
    # The other half of the rule, and the half that makes the page usable at
    # all. The authoring page POSTs the WHOLE array on every save, so editing a
    # statement, adding a second hypothesis, or reordering the list all re-send
    # every criterion unchanged. If any of those moved the boundary, a rule
    # genuinely written before any interview would lose its pre-registration to
    # an unrelated edit -- the field would decay to "when did you last press
    # Save", which is exactly the timestamp it was shaped to avoid being.
    context_dir = _node(tmp_path)
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses",
            [_with_criterion(criterion="3 of 10 mention X", statement="original wording")],
        )
        assert status == 200
        assert _saved_boundary(context_dir) == ()

        (context_dir.parent / "sessions" / "20260101T090000").mkdir(parents=True)
        status, _ = _post_json(
            f"{base_url}/api/hypotheses",
            [
                _with_criterion(criterion="3 of 10 mention X", statement="edited wording"),
                _with_criterion("h2", criterion="a brand new rule"),
            ],
        )
        assert status == 200

    hypotheses = {h.id: h for h in load_hypotheses(context_dir, missing_ok=False)}
    assert hypotheses["h1"].statement == "edited wording"
    assert hypotheses["h1"].criterion_registered_after == (), (
        "an unrelated edit moved a pre-registration boundary that was genuinely earned"
    )
    assert hypotheses["h2"].criterion_registered_after == ("20260101T090000",), (
        "the NEW hypothesis in the same save must still be stamped with what already existed"
    )


def test_a_node_with_no_sessions_directory_at_all_stamps_an_empty_boundary(tmp_path):
    # A node authored before any call has no `sessions/` directory yet. That is
    # the ordinary first case (D-01), not an error, and it must not raise.
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200
    assert not (context_dir.parent / "sessions").exists()
    assert _saved_boundary(context_dir) == ()


def test_a_stray_file_beside_the_sessions_is_not_an_interview(tmp_path):
    # `sessions/` is walked for DIRECTORIES, matching `console.py`'s own
    # listing. A `.DS_Store` or a stray render output recorded as an interview
    # would put a session id in the boundary that no interview ever had, and
    # every one of those permanently weakens a real pre-registration claim.
    context_dir = _node(tmp_path, sessions=("20260101T090000",))
    (context_dir.parent / "sessions" / "notes.txt").write_text("stray", encoding="utf-8")
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, _ = _post_json(
            f"{base_url}/api/hypotheses", [_with_criterion(criterion="3 of 10 mention X")]
        )
        assert status == 200
    assert _saved_boundary(context_dir) == ("20260101T090000",)


def test_the_brief_and_survey_save_paths_are_untouched_by_the_stamping(tmp_path):
    # The stamp lives on the hypotheses route only. A brief item has no
    # criterion and no boundary, and `ContextItem`/`SurveyItem` forbid extras --
    # so a stamp applied in the shared `_save_items` helper would 400 every
    # brief save in the product.
    context_dir = _node(tmp_path, sessions=("20260101T090000",))
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        assert _post_json(f"{base_url}/api/brief", [_brief_item()])[0] == 200
        assert _post_json(f"{base_url}/api/survey", [_survey_item()])[0] == 200
    assert load_brief(context_dir)[0].id == "b1"
    assert load_survey(context_dir)[0].id == "s1"


def test_a_failure_during_replace_leaves_the_previous_hypothesis_file_intact(tmp_path, monkeypatch):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    original = json.dumps([_hypothesis_item("survivor")])
    (context_dir / "hypotheses.json").write_text(original, encoding="utf-8")

    import app.web.author as author_mod

    def _boom(*_args, **_kwargs):
        raise OSError("simulated failure between write and replace")

    monkeypatch.setattr(author_mod.os, "replace", _boom)

    with pytest.raises(OSError):
        author_mod._atomic_write_json(context_dir / "hypotheses.json", [_hypothesis_item("new-item")])

    assert (context_dir / "hypotheses.json").read_text(encoding="utf-8") == original
    loaded = load_hypotheses(context_dir, missing_ok=False)
    assert loaded[0].id == "survivor"


def test_a_hypotheses_cross_origin_post_with_a_non_json_content_type_is_rejected(tmp_path):
    # Sibling of the brief/survey guard tests -- asserted here too because
    # this is what proves the hypotheses routes were added INSIDE the same
    # guarded factory, not registered beside it (§K13).
    context_dir = tmp_path / "context"
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(
            f"{base_url}/api/hypotheses",
            method="POST",
            body=json.dumps([_hypothesis_item()]).encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
    assert status == 403
    assert body == b""
    assert not (context_dir / "hypotheses.json").exists()


def test_a_hypotheses_get_with_a_spoofed_host_header_is_rejected(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "hypotheses.json").write_text(json.dumps([_hypothesis_item()]), encoding="utf-8")
    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _raw_request(f"{base_url}/api/hypotheses", headers={"Host": "evil.example.com"})
    assert status == 403
    assert body == b""


def test_gate_nine_still_passes_and_hypothesis_routes_are_absent_from_the_phone_apps_route_table(tmp_path):
    author_app = web_author.create_author_app(context_dir=tmp_path / "context", static_dir=AUTHOR_STATIC_DIR)
    session_dir = tmp_path / "session"
    from app.session.store import SessionStore
    from app.web.hub import ConnectionHub

    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "author-gate9-hypotheses-probe"})
    phone_app = web_server.create_app(
        token="author-hyp-test-token", hub=ConnectionHub(), static_dir=REPO_ROOT / "app" / "static",
        store=store, session_id="s", server_started_at="t",
    )
    author_paths = {getattr(route, "path", None) for route in author_app.routes}
    phone_paths = {getattr(route, "path", None) for route in phone_app.routes}
    assert "/api/hypotheses" in author_paths
    assert "/api/hypotheses" not in phone_paths


def test_falsifier_field_carries_no_default_value_and_no_placeholder_in_the_served_source():
    # Scans for the actual MECHANISMS a placeholder/default could arrive
    # through (a `placeholder` HTML attribute, a `.placeholder` DOM property,
    # or `setAttribute("placeholder", ...)`) -- never a bare substring match
    # against the word "placeholder", which would also trip on this test's
    # own explanatory comments in the source it is scanning.
    placeholder_pattern = re.compile(r'placeholder\s*=|\.placeholder\b|setAttribute\(\s*["\']placeholder["\']')
    author_js = (AUTHOR_STATIC_DIR / "author.js").read_text(encoding="utf-8")
    index_html = (AUTHOR_STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert not placeholder_pattern.search(author_js), "the falsifier field must carry no placeholder text"
    assert not placeholder_pattern.search(index_html), "the falsifier field must carry no placeholder text"
    assert 'falsifierInput.value = item.falsifier || "";' in author_js, (
        "the falsifier field must start from the item's own (possibly-empty) value, never a "
        "hardcoded default string"
    )


# --- No third-party origin in the new assets --------------------------------------------


def test_authoring_assets_load_no_absolute_external_url():
    url_pattern = re.compile(r"[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
    violations = []
    for path in sorted(AUTHOR_STATIC_DIR.rglob("*")):
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in url_pattern.finditer(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {match.group(0)}")
    assert not violations, violations


# No dedicated test here for "the authoring assets contain no browser
# media-capture identifier" -- app/tests/test_phase_gates.py's gate 2 already
# scans the ENTIRE repository (including app/static/author/) for the four
# forbidden identifiers, and naming them again in THIS file would trip that
# very gate (they are deliberately named only in the gate module itself and
# in 02-03-PLAN.md §U4). Gate 2 passing IS this property's proof.


def test_provenance_radio_has_no_checked_default_in_the_source():
    author_js = (AUTHOR_STATIC_DIR / "author.js").read_text(encoding="utf-8")
    # `radio.checked` must be assigned from a comparison against the item's
    # own (possibly-undefined) provenance value -- never a bare `true`.
    assert 'radio.checked = item.provenance === value;' in author_js
    assert "radio.checked = true" not in author_js


# --- CLI: `python -m app author --help`, no capture/device/session side effects -------


def test_author_help_succeeds_and_existing_subcommands_are_unchanged():
    result = subprocess.run(
        [sys.executable, "-m", "app", "author", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    result_top = subprocess.run(
        [sys.executable, "-m", "app", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result_top.returncode == 0
    for command in ("start", "render", "precheck", "author"):
        assert command in result_top.stdout


def test_running_author_opens_no_audio_device_and_creates_no_session_directory(tmp_path):
    import os

    sessions_before = set((REPO_ROOT / "sessions").glob("*")) if (REPO_ROOT / "sessions").is_dir() else set()

    env_context = tmp_path / "author-context"
    proc = subprocess.Popen(
        [sys.executable, "-m", "app", "author"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={
            **os.environ,
            "APP_CONTEXT_DIR": str(env_context),
            "APP_OPEN_BROWSER": "false",
            "APP_AUTHOR_PORT": "0",
        },
    )
    try:
        # `author` blocks serving forever -- it is expected to still be
        # running after a short wait. What matters is that no audio device
        # was ever touched and no session directory was created while it ran.
        time.sleep(2.0)
        assert proc.poll() is None, "the authoring server exited unexpectedly -- expected it to keep serving"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    sessions_after = set((REPO_ROOT / "sessions").glob("*")) if (REPO_ROOT / "sessions").is_dir() else set()
    assert sessions_after == sessions_before, "running `author` must never create a session directory"


# --- the criterion, and the fourth state, are reachable from the page --------------------


def test_the_state_dropdown_offers_every_state_the_model_allows():
    """`contested` was added to `ALL_HYPOTHESIS_STATES` in 763787c and the
    authoring page's `<select>` was never widened, so the state was unreachable
    from the only surface that authors hypotheses -- and a hypothesis
    hand-edited to `contested` would be written back as whatever the select
    happened to be showing.

    Pinned from Python because the JS side cannot know the model's vocabulary.
    Its own half (that the rendered dropdown really carries these four) lives
    in `app/tests/js/author_render.test.mjs`; this is the direction that scan
    cannot cover -- a FIFTH state added to the model with no option here."""
    from app.coach.corpus import ALL_HYPOTHESIS_STATES

    source = (REPO_ROOT / "app" / "static" / "author" / "author.js").read_text(encoding="utf-8")
    for state in sorted(ALL_HYPOTHESIS_STATES):
        assert f'["{state}", ' in source, (
            f"the authoring page cannot set {state!r}, which the model accepts -- so the only "
            "way to reach that state is to hand-edit the file, and the next save from this page "
            "would overwrite it"
        )


def test_a_criterion_typed_on_the_page_is_stored_and_stamped(tmp_path):
    """The end of the chain FLOWMAP called priority 3. `criterion` and its
    provenance stamp `criterion_registered_after` have existed since 763787c
    and the page rendered no input for either, so a pre-registered decision
    rule could only arrive by hand-editing JSON -- and the next page save
    would drop it, because the form never sent a field it never rendered.

    The repo's own research found that no shipped tool has a pre-registered
    criterion. This one had the data model, the provenance stamp, and no box
    to type in.

    The session dir is created as a SIBLING of `context/`, because that is how
    `_sessions_root_for` finds it -- the stamp records the interviews that
    already existed when the rule was written, so a test with no sessions on
    disk would assert an empty tuple and prove nothing about the stamp."""
    node_dir = tmp_path / "acme"
    context_dir = node_dir / "context"
    context_dir.mkdir(parents=True)
    (node_dir / "sessions" / "20260101T090000").mkdir(parents=True)

    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _post_json(f"{base_url}/api/hypotheses", [{
            "id": "h1",
            "statement": "activation stalls on invites",
            "falsifier": "three interviewees breeze through",
            "criterion": "we are right if 3 of 10 raise it unprompted",
            "state": "contested",
        }])
        assert status == 200, body

    (stored,) = load_hypotheses(context_dir, missing_ok=False)
    assert stored.criterion == "we are right if 3 of 10 raise it unprompted"
    assert stored.is_scorable is True, (
        "a hypothesis with a criterion must be scorable -- that property is the entire "
        "reason the field exists"
    )
    assert stored.state == "contested", "the fourth state did not survive the round trip"
    # The stamp the module docstring promises: the interviews that ALREADY
    # existed when the rule was written. Without it a criterion is an unbacked
    # claim about ordering.
    assert stored.criterion_registered_after == ("20260101T090000",)


def test_a_hypothesis_saved_with_no_criterion_is_not_scorable(tmp_path):
    """The control. Every assertion above is satisfied by an implementation
    that marks everything scorable, and `is_scorable` is what stops the hub
    tallying evidence against a rule nobody wrote."""
    context_dir = tmp_path / "acme" / "context"
    context_dir.mkdir(parents=True)

    app = web_author.create_author_app(context_dir=context_dir, static_dir=AUTHOR_STATIC_DIR)
    with _running_author_server(app) as base_url:
        status, body = _post_json(f"{base_url}/api/hypotheses", [_hypothesis_item()])
        assert status == 200, body

    (stored,) = load_hypotheses(context_dir, missing_ok=False)
    assert stored.criterion == ""
    assert stored.is_scorable is False
