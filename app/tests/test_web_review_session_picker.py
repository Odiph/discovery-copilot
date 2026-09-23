"""Choosing the interview in the page rather than on the command line.

The hub is scoped to a PROJECT; which interview it shows is picked in-page.
That change moves a session id from the operator's own command line onto the
wire, and `sessions_root / <name>` with a `..` in it resolves outside the
project — so the id is checked by MEMBERSHIP of the real listing rather than by
pattern. An allowlist built from what is actually on disk cannot be talked
round by traversal, a symlink, or a case difference.

The second property defended here is that the proposals card FOLLOWS the
picker. It is the one place in the product that writes, and left behind it
would offer a decision about a different interview than the one on screen.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import STATIC_REVIEW_DIR
from app.session.reader import list_session_ids
from app.web.review import create_review_app, resolve_session, sessions_at


def _project(tmp_path, *, sessions=(), rendered=()):
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    for session_id in sessions:
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(
            json.dumps({"kind": "segment", "seq": 0, "speaker": "interviewee", "text": "hi"}) + "\n",
            encoding="utf-8",
        )
        if session_id in rendered:
            (session_dir / "transcript.md").write_text("# rendered", encoding="utf-8")
    return root


def _client(tmp_path, session=None):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    # Loopback base_url, because the app's own middleware refuses anything
    # else outright -- see `test_web_review.py::test_a_non_loopback_host_header_is_refused`.
    return TestClient(app, base_url="http://127.0.0.1")


# --- the listing ----------------------------------------------------------------------------


def test_sessions_are_listed_newest_first(tmp_path):
    # Directory names are `%Y%m%dT%H%M%S`, so reverse lexicographic IS
    # newest-first without parsing a date.
    _project(tmp_path, sessions=("20260722T193140", "20260806T210533", "20260730T201205"))
    with _client(tmp_path) as client:
        listed = client.get("/api/sessions").json()["sessions"]
    assert [row["id"] for row in listed] == [
        "20260806T210533", "20260730T201205", "20260722T193140",
    ]


def test_an_unrendered_session_is_listed_and_flagged_not_hidden(tmp_path):
    # The hub reads the JSONL directly and does not need the rendered
    # document, so hiding un-rendered sessions would drop real interviews from
    # the only list of them.
    root = _project(tmp_path, sessions=("S1", "S2"), rendered=("S1",))
    listed = sessions_at(root / "sessions")
    assert {row["id"]: row["rendered"] for row in listed} == {"S1": True, "S2": False}


def test_a_project_with_no_sessions_lists_nothing_rather_than_raising(tmp_path):
    _project(tmp_path)
    with _client(tmp_path) as client:
        body = client.get("/api/sessions").json()
    assert body["sessions"] == [] and body["selected"] is None


def test_the_command_line_session_is_the_one_selected_by_default(tmp_path):
    _project(tmp_path, sessions=("S1", "S2"))
    with _client(tmp_path, session="S1") as client:
        assert client.get("/api/sessions").json()["selected"] == "S1"


# --- the id arriving from the browser -------------------------------------------------------


def test_a_session_outside_the_project_is_refused(tmp_path):
    # The whole reason the check is membership of the listing: this name
    # resolves ABOVE the sessions root, and no amount of path joining makes
    # that safe on its own.
    _project(tmp_path, sessions=("S1",))
    with _client(tmp_path, session="S1") as client:
        response = client.get("/api/review", params={"session": "../../context"})
    assert response.status_code == 400
    assert "not one of this project's interviews" in response.json()["error"]


def test_the_refusal_does_not_echo_the_requested_value_back(tmp_path):
    # It is attacker-controlled text heading for a page that renders it, and
    # repeating it buys the operator nothing.
    _project(tmp_path, sessions=("S1",))
    with _client(tmp_path, session="S1") as client:
        body = client.get("/api/review", params={"session": "<script>x</script>"}).json()
    assert "<script>" not in json.dumps(body)


def test_a_session_that_exists_is_served(tmp_path):
    _project(tmp_path, sessions=("S1", "S2"))
    with _client(tmp_path, session="S1") as client:
        body = client.get("/api/review", params={"session": "S2"}).json()
    assert body["session_id"] == "S2"


def test_no_session_parameter_falls_back_to_the_command_line_choice(tmp_path):
    _project(tmp_path, sessions=("S1", "S2"))
    with _client(tmp_path, session="S1") as client:
        assert client.get("/api/review").json()["session_id"] == "S1"


def test_resolve_session_admits_only_names_found_on_disk(tmp_path):
    root = _project(tmp_path, sessions=("S1",))
    sessions_root = root / "sessions"
    assert resolve_session(sessions_root, "S1", default=None) == ("S1", None)
    assert resolve_session(sessions_root, None, default="S1") == ("S1", None)
    chosen, error = resolve_session(sessions_root, "S9", default="S1")
    assert chosen is None and error is not None


def test_listing_a_missing_sessions_root_is_empty_not_an_error(tmp_path):
    # A project authored but never recorded into is normal.
    assert list_session_ids(tmp_path / "nope") == ()


# --- the proposals card follows the picker ---------------------------------------------------


def test_the_proposals_route_follows_the_selected_session(tmp_path):
    root = _project(tmp_path, sessions=("S1", "S2"))
    (root / "sessions" / "S2" / "transcript.jsonl").write_text(
        json.dumps({
            "kind": "suggestion", "seq": 1, "trigger_seq": 0, "text": "ask this",
            "target_hypothesis_id": "h1", "proposed_state": "supported", "shown": True,
        }) + "\n",
        encoding="utf-8",
    )
    with _client(tmp_path, session="S1") as client:
        assert client.get("/api/proposals").json()["proposals"] == []
        followed = client.get("/api/proposals", params={"session": "S2"}).json()["proposals"]
    assert [p["target_hypothesis_id"] for p in followed] == ["h1"]


def test_the_proposals_route_refuses_a_session_outside_the_project(tmp_path):
    # Re-checked here rather than trusted from the mounting app: this module
    # owns the one write in the product and must not depend on a caller having
    # remembered the guard.
    _project(tmp_path, sessions=("S1",))
    with _client(tmp_path, session="S1") as client:
        response = client.get("/api/proposals", params={"session": "../.."})
    assert response.status_code == 400
