"""The authoring page must never overwrite a file it could not read.

The sequence that loses work, reachable by following the tool's own advice:

  1. `context/survey.json` is malformed (a hand edit, a half-finished paste).
  2. `python -m app start` refuses to run and says the survey cannot be loaded.
     Correct, deliberate, and pinned by a standing test.
  3. The operator does exactly what they are told and opens `python -m app
     author` to fix it.
  4. `_read_existing` returns `[]` for a malformed file just as it does for a
     missing one, so the page renders an EMPTY FORM -- identical to a node
     nobody has ever authored.
  5. They retype the one question they remember and save.
  6. `_atomic_write_json` -> `os.replace`. The other N questions are gone, and
     the file that still contained them -- recoverable by hand a moment
     earlier -- no longer exists.

Nothing in that chain is a mistake by the operator. The tolerance in
`_read_existing` is right for a READ (a page that will not load is useless for
fixing anything); it is the WRITE that must not proceed on a file whose
contents were never understood.

The sneaky cases, which a naive fix misses:

  - Valid JSON of the wrong SHAPE (`{"items": [...]}`) never raises
    `JSONDecodeError`, so a guard that only catches that exception still
    clobbers. This is the more likely real corruption: a JSON file usually
    stays valid JSON.
  - Over-correction is its own bug. A file containing exactly `[]` is
    LEGITIMATE -- an empty survey is a real state -- and refusing to save over
    it would block the ordinary "author a node from scratch" path, which is the
    page's main job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app.web.author import create_author_app

_STATIC = Path(__file__).resolve().parents[1] / "static" / "author"
_VALID_SURVEY = [{"id": "q1", "text": "What broke?"}]


def _client(context_dir: Path) -> TestClient:
    app = create_author_app(context_dir=context_dir, static_dir=_STATIC)
    return TestClient(app, headers={"host": "127.0.0.1:8766"})


def _bytes_under(root: Path) -> "dict[str, bytes]":
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize(
    "corrupt, why",
    [
        ('{"id": "q1", ', "truncated mid-write"),
        ('{"items": [{"id": "q1", "text": "What broke?"}]}', "valid JSON, wrong shape"),
        ("null", "valid JSON, not a container"),
        ('"[]"', "a string that looks like a list"),
    ],
)
def test_a_survey_that_cannot_be_understood_is_never_overwritten(tmp_path, corrupt, why):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    target = context_dir / "survey.json"
    target.write_text(corrupt, encoding="utf-8")
    before = _bytes_under(tmp_path)

    response = _client(context_dir).post("/api/survey", json=_VALID_SURVEY)

    assert response.status_code == 409, f"{why}: expected a refusal, got {response.status_code}"
    assert _bytes_under(tmp_path) == before, (
        f"{why}: the on-disk file changed -- this is the data loss the test exists to prevent"
    )
    # The operator has to be able to act on this. "invalid" is not actionable;
    # the path and the fact that their edit was NOT saved are.
    detail = response.json().get("error", "")
    assert "survey.json" in detail and "not saved" in detail.lower(), detail


def test_the_refusal_leaves_no_temp_file_behind(tmp_path):
    # The sneaky version of "nothing was written": staging the temp file before
    # the guard runs would leave `survey.json.tmp` sitting next to the original,
    # so a later save could find it, and a whole-tree comparison is the only
    # check that sees it.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "survey.json").write_text("{", encoding="utf-8")

    _client(context_dir).post("/api/survey", json=_VALID_SURVEY)

    assert not list(context_dir.glob("*.tmp")), "a refused save staged a temp file"


def test_an_empty_list_on_disk_is_a_real_state_and_saves_normally(tmp_path):
    # The over-correction guard. `[]` is a legitimate file, not a broken one.
    # A fix that refuses anything `_read_existing` renders as empty would block
    # the page's main job -- authoring a node from scratch -- and would look
    # exactly like the bug it replaced.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "survey.json").write_text("[]", encoding="utf-8")

    response = _client(context_dir).post("/api/survey", json=_VALID_SURVEY)

    assert response.status_code == 200, response.text
    assert json.loads((context_dir / "survey.json").read_text(encoding="utf-8")) == [
        {"id": "q1", "text": "What broke?", "order": 0}
    ]


def test_a_missing_file_still_saves_so_a_fresh_node_can_be_authored(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    response = _client(context_dir).post("/api/survey", json=_VALID_SURVEY)

    assert response.status_code == 200, response.text
    assert (context_dir / "survey.json").is_file()


def test_every_artifact_is_protected_not_only_the_survey(tmp_path):
    # The forcing function. Three artifacts share one save path; a guard wired
    # into only the route that prompted it would leave the other two clobbering.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    for name in ("brief.json", "survey.json", "hypotheses.json"):
        (context_dir / name).write_text('{"nope": 1}', encoding="utf-8")
    before = _bytes_under(tmp_path)
    client = _client(context_dir)

    for route, payload in (
        # A VALID payload for each route on purpose. The guard runs AFTER
        # validation by design -- a bad payload must report its own fault
        # rather than being masked by the state of the file on disk -- so an
        # invalid one here would pass this test for the wrong reason (400, not
        # 409) while proving nothing about clobbering.
        ("/api/brief", [{"id": "b1", "text": "context", "provenance": "evidence"}]),
        ("/api/survey", _VALID_SURVEY),
        ("/api/hypotheses", [{"id": "h1", "statement": "s", "falsifier": "f"}]),
    ):
        assert client.post(route, json=payload).status_code == 409, route
    assert _bytes_under(tmp_path) == before


def test_the_page_can_tell_the_three_states_apart(tmp_path):
    # Refusing at save is the safety net; it still leaves the operator typing
    # into an empty form before finding out. `/api/state` is additive -- the
    # existing GETs keep returning bare arrays -- and lets the page say "this
    # file is on disk and unreadable" instead of showing the same blank form it
    # shows for a node nobody has authored.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "survey.json").write_text("{", encoding="utf-8")
    (context_dir / "hypotheses.json").write_text("[]", encoding="utf-8")

    state = _client(context_dir).get("/api/state").json()

    assert state["survey"] == "unreadable"
    assert state["hypotheses"] == "present"
    assert state["brief"] == "missing"
