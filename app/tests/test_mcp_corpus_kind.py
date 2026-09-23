"""Over MCP, a demonstration claim must not read as customer research.

`projects/demo-claudictive/context/facts.json` holds 25 claims about a "he" and
a "she" -- what they play, what they installed, what they said they would never
do again. Thirteen were typed by hand; the rest came from a mock where the
operator played the interviewee. Nothing in the payload said so. `kind` is
evidence-vs-inference, which is how firmly a claim follows from what was said,
and is silent on whether a person said it; `status: extracted` says only that
the extractor ran. A model asked "what have we learned about our users" got
twenty-five confident sentences and no way to know that none of them came from
a user.

**Why this surface and not the hub.** The same argument
`test_mcp_extraction_status` makes: the hub's reader is the operator, who knows
which of their own projects is a demo. This reader is a MODEL, in another
process, with read-only tools and no way to go and look. If the payload does
not say it, it is not knowable -- and the failure is not a missing answer but a
confident wrong one, which is the kind that gets repeated downstream.

**What these tests are aimed at.** Not "does the field exist". The
discriminating cases are that a demo node and an undeclared node do not
collapse into each other, that a subtree holding one of each reports each
correctly rather than letting one speak for both, and that the marker survives
the explicit key projection in `_extraction_block` -- which silently dropped
the field on the first attempt at this change and is exactly the seam a naive
implementation gets wrong.
"""

from __future__ import annotations

import io
import json

import pytest

from app import project_meta
from app.mcp import server as mcp_server

SESSION = "20260812-101500-demo00"

FACT_READING_TOOLS = ("get_facts", "get_standing", "get_interview_summary")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fact(fact_id, claim):
    return {
        "id": fact_id,
        "session_id": SESSION,
        "seqs": [1],
        "claim": claim,
        "provenance": "evidence",
        "quoted_text": "corpus-kind-sentinel-9f31",
        "bearings": [{"hypothesis_id": "h1", "stance": "supports"}],
    }


def _node(projects, path, *, corpus, claim):
    """One extracted node, optionally declaring what its corpus is."""
    _write_json(projects / path / "context" / "hypotheses.json",
                [{"id": "h1", "statement": "Players look things up mid-match",
                  "falsifier": "three players never leave the game"}])
    _write_json(projects / path / "context" / "facts.json", [_fact(f"{path}-1", claim)])
    _write_json(projects / path / "sessions" / SESSION / "session.json",
                {"started_at": "2026-08-12T10:15:00+00:00"})
    if corpus is not None:
        _write_json(projects / path / "project.json", {"corpus": corpus})


@pytest.fixture()
def repo(tmp_path):
    """Three nodes that used to render identically.

    - `alpha`          -- declares `real`. Actual customer research.
    - `alpha/demo`     -- declares `demo`. Invented, and reads exactly like the
                          one above: a specific person doing a specific thing.
    - `alpha/quiet`    -- declares nothing. Not the same as either.
    """
    projects = tmp_path / "projects"
    _node(projects, "alpha", corpus="real",
          claim="She alt-tabbed out mid-match to check an error and lost the round.")
    _node(projects, "alpha/demo", corpus="demo",
          claim="He keeps a running list of things to look up after the session.")
    _node(projects, "alpha/quiet", corpus=None,
          claim="He plays windowed borderless so that leaving the game is cheap.")
    return tmp_path


def _call(name, arguments, *, repo_root):
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}}) + "\n"
    stdout = io.BytesIO()
    mcp_server.serve(io.BytesIO(payload.encode("utf-8")), stdout, repo_root=repo_root)
    response = json.loads(stdout.getvalue().decode("utf-8").splitlines()[0])
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


def _corpus_for(payload, project):
    matching = [e for e in payload["extraction"] if e["project"] == project]
    assert len(matching) == 1, f"{project} in {payload['extraction']}"
    return matching[0]["corpus"]


# --- the three that must not collapse into each other ---------------------------------


@pytest.mark.parametrize("tool", FACT_READING_TOOLS)
@pytest.mark.parametrize("project,expected", [
    ("alpha", project_meta.CORPUS_REAL),
    ("alpha/demo", project_meta.CORPUS_DEMO),
    ("alpha/quiet", project_meta.CORPUS_UNDECLARED),
])
def test_every_fact_reading_tool_states_the_corpus(tool, project, expected, repo):
    arguments = {"project": project}
    if tool == "get_interview_summary":
        arguments = {**arguments, "session_id": SESSION}
    payload = _call(tool, arguments, repo_root=repo)
    assert _corpus_for(payload, project) == expected


@pytest.mark.parametrize("tool", FACT_READING_TOOLS)
def test_a_subtree_reports_each_node_rather_than_letting_one_speak_for_all(tool, repo):
    """The discriminating case. An implementation that decides the corpus once
    for the root and stamps it on the rollup passes every single-node test above
    and reports a demo child as real research here -- which is the merge this
    whole field exists to survive."""
    arguments = {"project": "alpha", "scope": "subtree"}
    if tool == "get_interview_summary":
        arguments = {**arguments, "session_id": SESSION}
    payload = _call(tool, arguments, repo_root=repo)

    assert _corpus_for(payload, "alpha") == project_meta.CORPUS_REAL
    assert _corpus_for(payload, "alpha/demo") == project_meta.CORPUS_DEMO
    assert _corpus_for(payload, "alpha/quiet") == project_meta.CORPUS_UNDECLARED


@pytest.mark.parametrize("tool", FACT_READING_TOOLS)
def test_the_marker_survives_the_key_projection(tool, repo):
    """`_extraction_block` rebuilds every entry field by field on purpose, so a
    producer growing a key does not silently widen the wire. That means adding
    a key to the producer alone is a no-op -- which is what happened on the
    first attempt at this change, and the tests still passed because they only
    asserted the keys they already knew about."""
    arguments = {"project": "alpha/demo"}
    if tool == "get_interview_summary":
        arguments = {**arguments, "session_id": SESSION}
    payload = _call(tool, arguments, repo_root=repo)
    assert "corpus" in payload["extraction"][0], (
        "the producer set it and the projection dropped it -- the two must be changed together"
    )


def test_list_projects_states_it_too(repo):
    """The entry point. A model told to `list_projects` first should be able to
    see which nodes are demonstrations before it reads a single claim."""
    payload = _call("list_projects", {}, repo_root=repo)
    by_path = {entry["path"]: entry["corpus"] for entry in payload["projects"]}
    assert by_path == {
        "alpha": project_meta.CORPUS_REAL,
        "alpha/demo": project_meta.CORPUS_DEMO,
        "alpha/quiet": project_meta.CORPUS_UNDECLARED,
    }


def test_a_broken_project_json_does_not_take_the_tool_down(repo):
    """One malformed node degrades THAT NODE ONLY, the rule `_load_facts_across`
    already states for a broken corpus file."""
    (repo / "projects" / "alpha" / "demo" / "project.json").write_text(
        "{not json", encoding="utf-8")
    payload = _call("get_facts", {"project": "alpha", "scope": "subtree"}, repo_root=repo)
    assert _corpus_for(payload, "alpha/demo") == project_meta.CORPUS_UNDECLARED
    assert _corpus_for(payload, "alpha") == project_meta.CORPUS_REAL
    assert payload["errors"] == []


def test_the_tool_descriptions_warn_about_demo_claims(repo):
    """The field is only half the fix: a model that does not know to look at it
    reads past it. The description is the other half, so it is pinned."""
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    stdout = io.BytesIO()
    mcp_server.serve(io.BytesIO(payload.encode("utf-8")), stdout, repo_root=repo)
    listed = json.loads(stdout.getvalue().decode("utf-8").splitlines()[0])
    by_name = {tool["name"]: tool["description"] for tool in listed["result"]["tools"]}
    for name in ("get_facts", "get_standing", "list_projects"):
        assert "demo" in by_name[name], f"{name} does not mention the demo corpus"
