"""Phase B: the local console.

Two things this file is really protecting. First, that the console **reads and
never writes** -- it is a picker, and the moment it can mutate a project it
becomes a second write path into the artifacts that ground the coach. Second,
that it re-applies the same-hierarchy rule on the SERVER, because the page's
own filtered picker is a convenience for the operator and anything reaching
this process could have skipped the page entirely.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import STATIC_CONSOLE_DIR
from app.web.console import (
    ConsoleSelection,
    _node_detail,
    _past_interviews,
    _validate_start_request,
    create_console_app,
)


def _node(tmp_path, path, *, survey=None, hypotheses=None, facts=None, sessions=()):
    context_dir = tmp_path / "projects" / path / "context"
    context_dir.mkdir(parents=True)
    if survey is not None:
        (context_dir / "survey.json").write_text(json.dumps(survey), encoding="utf-8")
    if hypotheses is not None:
        (context_dir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    if facts is not None:
        (context_dir / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    for session_id in sessions:
        (tmp_path / "projects" / path / "sessions" / session_id).mkdir(parents=True)
    return context_dir


def _client(tmp_path, seen=None):
    app = create_console_app(
        static_dir=STATIC_CONSOLE_DIR,
        on_start=(seen.append if seen is not None else (lambda _selection: None)),
        repo_root=tmp_path,
    )
    # `base_url` matters: TestClient defaults to a `testserver` Host, which
    # the loopback guard correctly refuses. Pointing it at 127.0.0.1 is what
    # makes these tests exercise the routes rather than the guard -- and the
    # guard has its own test below that deliberately does NOT do this.
    return TestClient(app, base_url="http://127.0.0.1")


# --- the tree and one node's detail ------------------------------------------------------


def test_the_tree_lists_every_node_with_its_depth(tmp_path):
    _node(tmp_path, "acme")
    _node(tmp_path, "acme/onboarding")
    with _client(tmp_path) as client:
        rows = client.get("/api/projects").json()
    assert [(r["path"], r["depth"]) for r in rows] == [("acme", 1), ("acme/onboarding", 2)]


def test_a_nodes_detail_carries_the_survey_hypotheses_fact_count_and_interviews(tmp_path):
    _node(
        tmp_path, "acme",
        survey=[{"id": "q1", "order": 1, "text": "What broke?"}],
        hypotheses=[{"id": "h1", "statement": "onboarding is the blocker", "falsifier": "they say setup was easy"}],
        facts=[{
            "id": "f1", "claim": "setup took a week", "provenance": "evidence",
            "session_id": "S1", "seqs": [1], "quoted_text": "setup took a week",
            "bearings": [],
        }],
        sessions=("20260810T090000", "20260810T120000"),
    )
    detail = _node_detail("acme", repo_root=tmp_path)
    assert [item["text"] for item in detail["survey"]] == ["What broke?"]
    assert detail["hypotheses"][0]["state"] == "untested"
    assert detail["fact_count"] == 1
    assert [s["id"] for s in detail["sessions"]] == ["20260810T120000", "20260810T090000"]


def test_the_detail_reports_a_fact_count_and_never_a_fact(tmp_path):
    # A fact carries a previous customer's verbatim speech in `quoted_text`,
    # and this is a browser page. The count answers the only question the
    # picker has: has this node accumulated anything?
    _node(
        tmp_path, "acme",
        survey=[{"id": "q1", "order": 1, "text": "What broke?"}],
        facts=[{
            "id": "f1", "claim": "a claim that must not appear", "provenance": "evidence",
            "session_id": "S1", "seqs": [1], "quoted_text": "words that must not appear",
            "bearings": [],
        }],
    )
    serialized = json.dumps(_node_detail("acme", repo_root=tmp_path))
    assert "must not appear" not in serialized
    assert '"fact_count": 1' in serialized


def test_past_interviews_are_newest_first_without_parsing_a_date(tmp_path):
    root = tmp_path / "sessions"
    for name in ("20260810T090000", "20260810T120000", "20260810T120000-1"):
        (root / name).mkdir(parents=True)
    assert [s["id"] for s in _past_interviews(root)] == [
        "20260810T120000-1", "20260810T120000", "20260810T090000",
    ]


def test_an_interview_reports_whether_it_has_been_rendered(tmp_path):
    root = tmp_path / "sessions"
    (root / "20260810T090000").mkdir(parents=True)
    (root / "20260810T120000").mkdir(parents=True)
    (root / "20260810T120000" / "transcript.md").write_text("x", encoding="utf-8")
    assert {s["id"]: s["rendered"] for s in _past_interviews(root)} == {
        "20260810T120000": True, "20260810T090000": False,
    }


def test_a_node_with_no_sessions_directory_lists_no_interviews(tmp_path):
    _node(tmp_path, "acme", survey=[])
    assert _node_detail("acme", repo_root=tmp_path)["sessions"] == []


# --- one broken node must not close the console -----------------------------------------


def test_a_malformed_survey_is_reported_beside_the_rest_rather_than_raising(tmp_path):
    # The opposite of `start`'s discipline, deliberately. `start` fails loudly
    # on a malformed artifact because it is about to ground a live coach in
    # it. A console that refuses to open because one project among twenty is
    # broken cannot be used to reach the other nineteen -- and the operator is
    # most likely opening it BECAUSE something is wrong.
    _node(tmp_path, "acme", hypotheses=[])
    (tmp_path / "projects" / "acme" / "context" / "survey.json").write_text("{ not json", encoding="utf-8")

    detail = _node_detail("acme", repo_root=tmp_path)
    assert detail["survey"] == []
    assert detail["survey_error"] is not None
    assert detail["hypotheses"] == [] and detail["hypotheses_error"] is None


def test_a_malformed_corpus_reports_an_unknown_count_not_a_wrong_one(tmp_path):
    _node(tmp_path, "acme", survey=[])
    (tmp_path / "projects" / "acme" / "context" / "facts.json").write_text("[{}]", encoding="utf-8")
    detail = _node_detail("acme", repo_root=tmp_path)
    assert detail["fact_count"] is None and detail["facts_error"] is not None


def test_a_hostile_project_path_is_a_400_not_a_traceback(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/api/project/../../etc").status_code in (400, 404)


# --- Start: the server re-checks what the page already filtered -------------------------


def test_a_valid_selection_is_handed_to_the_caller(tmp_path):
    seen: "list[ConsoleSelection]" = []
    with _client(tmp_path, seen) as client:
        response = client.post(
            "/api/start", json={"project": "acme/onboarding", "read_from": ["acme/billing"]}
        )
    assert response.status_code == 200
    assert seen == [ConsoleSelection(project="acme/onboarding", read_from=("acme/billing",))]


def test_a_source_under_a_different_root_is_refused_by_the_server(tmp_path):
    # The page filters its picker to in-branch nodes. That is a convenience
    # for the operator, never the rule -- this request never went through it.
    seen: "list[ConsoleSelection]" = []
    with _client(tmp_path, seen) as client:
        response = client.post(
            "/api/start", json={"project": "acme/onboarding", "read_from": ["globex/billing"]}
        )
    assert response.status_code == 400
    assert "different root" in response.json()["error"]
    assert seen == [], "a refused selection must never reach the caller"


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"project": "../../etc"}, "'.'"),
        ({"project": 7}, "must be a string"),
        ({"read_from": ["acme"]}, "must be a string"),
        ({"project": "acme", "read_from": "acme/billing"}, "must be a list"),
        ({"project": "acme", "read_from": ["acme"]}, "filing node itself"),
        ({"project": "acme", "read_from": ["acme/b", "acme/b"]}, "twice"),
        ([], "JSON object"),
    ],
)
def test_a_malformed_or_hostile_start_request_is_a_400(body, fragment, tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/api/start", json=body)
    assert response.status_code == 400
    assert fragment in response.json()["error"]


def test_a_start_with_no_extra_sources_is_normal(tmp_path):
    seen: "list[ConsoleSelection]" = []
    with _client(tmp_path, seen) as client:
        assert client.post("/api/start", json={"project": "acme"}).status_code == 200
    assert seen == [ConsoleSelection(project="acme", read_from=())]


def test_the_validator_normalizes_the_paths_it_returns():
    selection, error = _validate_start_request({"project": "  acme/onboarding  ", "read_from": []})
    assert error is None and selection.project == "acme/onboarding"


# --- the same guards the authoring page has, for a worse consequence --------------------


def test_a_non_loopback_host_header_is_refused(tmp_path):
    # DNS rebinding: an attacker-controlled domain resolved to 127.0.0.1, so
    # CORS never applies and the loopback bind is already satisfied.
    with _client(tmp_path) as client:
        response = client.get("/api/projects", headers={"Host": "evil.example.com"})
    assert response.status_code == 403
    assert response.content == b"", "a 403 here must never carry a detail an attacker can iterate on"


def test_a_cross_origin_post_without_a_json_content_type_is_refused(tmp_path):
    # Starlette parses the body whatever the Content-Type says, and this app
    # has no token and no CORS middleware. Without this guard any page open in
    # the operator's browser could choose which customer's knowledge grounds
    # the next interview.
    seen: "list[ConsoleSelection]" = []
    with _client(tmp_path, seen) as client:
        response = client.post(
            "/api/start",
            content=json.dumps({"project": "acme"}),
            headers={"Content-Type": "text/plain"},
        )
    assert response.status_code == 403
    assert seen == []


def test_the_console_binds_to_loopback_and_says_so_in_one_place():
    from app.web import console as console_mod

    assert console_mod._CONSOLE_HOST == "127.0.0.1"
    source = (STATIC_CONSOLE_DIR.parent.parent / "web" / "console.py").read_text(encoding="utf-8")
    assert source.count('host=_CONSOLE_HOST') == 1, (
        "the loopback bind must be established in exactly one visible place"
    )


# --- it never writes ---------------------------------------------------------------------


def test_no_route_changes_anything_on_disk(tmp_path):
    _node(
        tmp_path, "acme",
        survey=[{"id": "q1", "order": 1, "text": "What broke?"}],
        hypotheses=[{"id": "h1", "statement": "s", "falsifier": "f"}],
        sessions=("20260810T090000",),
    )

    def _snapshot():
        return {
            str(p.relative_to(tmp_path)): p.read_bytes()
            for p in sorted((tmp_path / "projects").rglob("*"))
            if p.is_file()
        }

    before = _snapshot()
    with _client(tmp_path) as client:
        client.get("/api/projects")
        client.get("/api/project/acme")
        client.post("/api/start", json={"project": "acme"})
    assert _snapshot() == before


def test_the_module_contains_no_write_primitive():
    # Gate 19 pins hypothesis WRITERS textually and would trip on a module
    # that both names the artifact and writes. This module names neither --
    # asserted here rather than left to the gate, because the gate only
    # notices the hypothesis file specifically.
    source = (STATIC_CONSOLE_DIR.parent.parent / "web" / "console.py").read_text(encoding="utf-8")
    for forbidden in ("write_text", "write_bytes", "os.replace", "open("):
        assert forbidden not in source, f"the console must never write: found {forbidden!r}"


# --- the page's own posture ---------------------------------------------------------------


def test_the_client_never_uses_innerhtml():
    # Project labels, survey questions, hypothesis statements and session ids
    # all reach the DOM here, and several are customer-derived.
    #
    # COMMENT LINES ARE STRIPPED FIRST, and the first draft of this test did
    # not do that -- it failed against the file's own comment explaining why
    # innerHTML is never used. That is this project's S13 lesson again (a
    # regex gate false-positiving on the prose describing it), in its
    # smallest possible form. Stripping is the fix rather than rewording the
    # comment: the comment is why the rule survives.
    source = (STATIC_CONSOLE_DIR / "console.js").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("//")
    )
    assert "innerHTML" not in code
    assert "textContent" in code


def test_the_interview_list_is_capped_and_says_so(tmp_path):
    # Found by running the console against the real `_eval` node, which has
    # 273 sessions: the uncapped list buried the Start button several screens
    # down. The TOTAL is always reported, so the cap is visible rather than a
    # silent truncation that would misreport how much a node has accumulated.
    from app.web.console import MAX_LISTED_INTERVIEWS

    ids = [f"2026081{n // 10}T{n:06d}" for n in range(MAX_LISTED_INTERVIEWS + 5)]
    _node(tmp_path, "acme", survey=[], sessions=tuple(ids))
    detail = _node_detail("acme", repo_root=tmp_path)
    assert len(detail["sessions"]) == MAX_LISTED_INTERVIEWS
    assert detail["session_total"] == MAX_LISTED_INTERVIEWS + 5
    # And the ones kept are the NEWEST, not the first N found on disk.
    assert [s["id"] for s in detail["sessions"]] == sorted(ids, reverse=True)[:MAX_LISTED_INTERVIEWS]
