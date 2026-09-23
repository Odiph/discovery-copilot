"""Building a hypothesis from the hub, without a second writer.

The hub is the post-interview surface, and the thing an operator most wants to
do there is write down the decision rule they just realised they needed. That
is a WRITE to `context/hypotheses.json`, which gate 19 pins to
`{author.py, accept.py}` textually and per file.

So the route is not reimplemented here: `register_hypothesis_routes` is mounted
onto the hub app from `app/web/author.py`. What this file defends is that the
mounting really does carry the whole save path with it -- above all the
pre-registration stamp, which is the difference between a rule written before
the answers were in and one typed afterwards. A second, plausible-looking copy
of that rule would be wrong in a way nothing on the page could show.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import STATIC_REVIEW_DIR
from app.web.review import create_review_app

HYPOTHESIS = {
    "id": "h1",
    "statement": "Players defer lookups to the end of the session",
    "falsifier": "they alt-tab mid-match and call it cheap",
}


def _project(tmp_path, *, hypotheses=None, sessions=()):
    root = tmp_path / "projects" / "acme"
    context_dir = root / "context"
    context_dir.mkdir(parents=True)
    if hypotheses is not None:
        (context_dir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    for session_id in sessions:
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text("", encoding="utf-8")
    return context_dir


def _client(tmp_path, session=None):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    return TestClient(app, base_url="http://127.0.0.1")


def _stored(context_dir):
    return json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))


# --- the route is really there ----------------------------------------------------------------


def test_the_hub_serves_the_hypotheses_it_would_edit(tmp_path):
    _project(tmp_path, hypotheses=[dict(HYPOTHESIS, criterion="")])
    with _client(tmp_path) as client:
        body = client.get("/api/hypotheses").json()
    assert [h["id"] for h in body] == ["h1"]


def test_a_hypothesis_can_be_created_from_the_hub(tmp_path):
    context_dir = _project(tmp_path, hypotheses=[])
    with _client(tmp_path) as client:
        response = client.post("/api/hypotheses", json=[dict(HYPOTHESIS, criterion="")])
    assert response.status_code == 200
    assert [h["id"] for h in _stored(context_dir)] == ["h1"]


def test_the_hub_refuses_a_write_from_a_non_loopback_host(tmp_path):
    _project(tmp_path, hypotheses=[])
    with _client(tmp_path) as client:
        response = client.post(
            "/api/hypotheses", json=[HYPOTHESIS], headers={"Host": "evil.example.com"}
        )
    assert response.status_code == 403


def test_a_hypothesis_with_no_falsifier_is_refused(tmp_path):
    # The validator comes across with the route rather than being re-typed --
    # `Hypothesis` has no default for `falsifier` and that rule must still bite
    # here.
    context_dir = _project(tmp_path, hypotheses=[])
    with _client(tmp_path) as client:
        response = client.post("/api/hypotheses", json=[{"id": "h1", "statement": "x"}])
    assert response.status_code == 400
    assert not (context_dir / "hypotheses.json").read_text(encoding="utf-8").strip() == "[{"


# --- the pre-registration stamp comes across with the route -------------------------------------


def test_a_criterion_written_from_the_hub_records_the_interviews_it_did_not_predate(tmp_path):
    # The whole point of doing this on the POST-interview surface: a rule typed
    # here is typed AFTER the calls already on disk, and the record has to say
    # so. Empty text becoming real text is a criterion edit like any other.
    context_dir = _project(
        tmp_path, hypotheses=[dict(HYPOTHESIS, criterion="")], sessions=("S1", "S2")
    )
    with _client(tmp_path) as client:
        client.post("/api/hypotheses", json=[dict(HYPOTHESIS, criterion="3 of 6 say so")])
    assert _stored(context_dir)[0]["criterion_registered_after"] == ["S1", "S2"]


def test_an_unchanged_criterion_keeps_its_original_boundary(tmp_path):
    # Otherwise every unrelated save would decay the field into "when did you
    # last press Save", which is the timestamp it was shaped as a session tuple
    # to avoid being.
    stored = dict(HYPOTHESIS, criterion="3 of 6 say so", criterion_registered_after=[])
    context_dir = _project(tmp_path, hypotheses=[stored], sessions=("S1", "S2"))
    with _client(tmp_path) as client:
        client.post("/api/hypotheses", json=[dict(stored, statement="reworded, same rule")])
    assert _stored(context_dir)[0]["criterion_registered_after"] == []


def test_an_edited_criterion_loses_its_pre_registration(tmp_path):
    # Moving the boundary on an edit looks wrong and is not: an edited
    # criterion is a DIFFERENT rule, written later. Freezing it would let a
    # rule be rewritten to fit the data while still displaying the original
    # pre-registration.
    stored = dict(HYPOTHESIS, criterion="3 of 6 say so", criterion_registered_after=[])
    context_dir = _project(tmp_path, hypotheses=[stored], sessions=("S1", "S2"))
    with _client(tmp_path) as client:
        client.post("/api/hypotheses", json=[dict(stored, criterion="2 of 6 say so")])
    assert _stored(context_dir)[0]["criterion_registered_after"] == ["S1", "S2"]


def test_a_boundary_claimed_by_the_client_is_ignored(tmp_path):
    # A client-supplied boundary is evidence of nothing at all -- it is the
    # same unbacked claim the bare criterion already was, wearing a credential.
    context_dir = _project(
        tmp_path, hypotheses=[dict(HYPOTHESIS, criterion="")], sessions=("S1", "S2")
    )
    with _client(tmp_path) as client:
        client.post("/api/hypotheses", json=[
            dict(HYPOTHESIS, criterion="new rule", criterion_registered_after=[]),
        ])
    assert _stored(context_dir)[0]["criterion_registered_after"] == ["S1", "S2"]


# --- what the page needs in order to price an edit ----------------------------------------------


def test_the_review_payload_carries_the_boundary_so_the_page_can_price_an_edit(tmp_path):
    from app.web.review import build_review

    _project(
        tmp_path,
        hypotheses=[dict(HYPOTHESIS, criterion="3 of 6", criterion_registered_after=["S1"])],
        sessions=("S1", "S2"),
    )
    standing = build_review("acme", "S2", repo_root=tmp_path)["standings"][0]
    assert standing["criterion_registered_after"] == ["S1"]
    # S2 is on disk and NOT in the boundary, so the rule currently pre-dates
    # it -- and that is exactly what an edit would cost.
    assert standing["is_scorable"] is True
