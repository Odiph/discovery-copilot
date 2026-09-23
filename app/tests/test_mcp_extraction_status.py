"""Over MCP, "nobody has extracted this yet" must not read as "nothing bears
on this belief".

This is the defect `18895a7` fixed on the hub, found again on the MCP surface
by the 2026-08-12 audit. `_load_facts_across` skipped a node with no
`facts.json` in silence -- no `errors` entry, no note of any kind -- so
`get_standing` answered a never-extracted project with a full set of
confident-looking zeros, byte-identical to the answer for a project whose
facts genuinely bear on nothing.

Only the second is a research finding. The first is an unfinished step.

**Why it is worse here than on the hub.** The hub's reader is the operator,
sitting at the terminal that would have printed the extraction command. This
surface's reader is a MODEL, in another process, with exactly seven read-only
tools and no way to go and check. If the payload does not say it, it is not
knowable.

**What these tests are aimed at.** Not "does the status field exist" -- that
fires on any implementation. The discriminating case is the pair: a node that
IS extracted and whose facts bear on nothing must NOT be reported the same way
as a node nobody has extracted, and a subtree containing one of each must
report each correctly rather than letting the populated node speak for both.
That last one is the mutation a naive fix misses, because the natural
implementation ("did we load any facts at all?") is right for a single node
and wrong for every subtree.
"""

from __future__ import annotations

import io
import json

import pytest

from app import project_meta
from app.mcp import server as mcp_server
from app.mcp import tools as mcp_tools

SENTINEL_QUOTE = "mcp-extraction-sentinel-quote-4c17e0aa"

SESSION_RICH = "20260812-101500-rich00"
SESSION_QUIET = "20260812-120000-quiet0"

# Every tool that reads facts, so a tool added later without the status block
# fails here rather than shipping the original defect on a new surface.
FACT_READING_TOOLS = ("get_facts", "get_standing", "get_interview_summary")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fact(fact_id, session_id, *, bearings=(("h1", "supports"),)):
    return {
        "id": fact_id,
        "session_id": session_id,
        "seqs": [1],
        "claim": f"claim behind {fact_id}",
        "provenance": "evidence",
        "quoted_text": SENTINEL_QUOTE,
        "bearings": [
            {"hypothesis_id": hypothesis_id, "stance": stance}
            for hypothesis_id, stance in bearings
        ],
    }


def _hypothesis(hypothesis_id="h1"):
    return {
        "id": hypothesis_id,
        "statement": "Activation stalls on invites",
        "falsifier": "three interviewees breeze through",
    }


@pytest.fixture()
def repo(tmp_path):
    """Five nodes, one per situation that used to render identically.

    - `alpha`         -- extracted, and its facts BEAR on `h1`.
    - `alpha/quiet`   -- extracted, hypotheses loaded, and the model found NO
                         bearing. This is the research finding, and it is the
                         node every other assertion here is measured against.
    - `alpha/never`   -- hypotheses authored, `extract` never run. No
                         `facts.json` on disk at all.
    - `alpha/empty`   -- `facts.json` exists and is an empty list. Extraction
                         RAN and filed nothing here. A third answer, not a
                         rounding of either neighbour.
    - `alpha/broken`  -- `facts.json` is not JSON. A fault, unlike the rest.
    """
    projects = tmp_path / "projects"

    _write_json(projects / "alpha" / "context" / "hypotheses.json", [_hypothesis()])
    _write_json(
        projects / "alpha" / "context" / "facts.json",
        [_fact("rich-1", SESSION_RICH)],
    )
    _write_json(
        projects / "alpha" / "sessions" / SESSION_RICH / "session.json",
        {"started_at": "2026-08-12T10:15:00+00:00"},
    )

    _write_json(projects / "alpha" / "quiet" / "context" / "hypotheses.json", [_hypothesis()])
    _write_json(
        projects / "alpha" / "quiet" / "context" / "facts.json",
        [_fact("quiet-1", SESSION_QUIET, bearings=())],
    )
    _write_json(
        projects / "alpha" / "quiet" / "sessions" / SESSION_QUIET / "session.json",
        {"started_at": "2026-08-12T12:00:00+00:00"},
    )

    _write_json(projects / "alpha" / "never" / "context" / "hypotheses.json", [_hypothesis()])

    _write_json(projects / "alpha" / "empty" / "context" / "hypotheses.json", [_hypothesis()])
    _write_json(projects / "alpha" / "empty" / "context" / "facts.json", [])

    _write_json(projects / "alpha" / "broken" / "context" / "hypotheses.json", [_hypothesis()])
    (projects / "alpha" / "broken" / "context" / "facts.json").write_text(
        "{not json at all", encoding="utf-8"
    )

    return tmp_path


def _drive(requests, *, repo_root):
    payload = "".join(json.dumps(request) + "\n" for request in requests)
    stdout = io.BytesIO()
    mcp_server.serve(io.BytesIO(payload.encode("utf-8")), stdout, repo_root=repo_root)
    return [json.loads(line) for line in stdout.getvalue().decode("utf-8").split("\n") if line]


def _scoped(name, arguments, *, repo_root):
    response = _drive(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": name, "arguments": arguments}}],
        repo_root=repo_root,
    )[0]
    assert "error" not in response, response
    return json.loads(response["result"]["content"][0]["text"])


def _status_for(payload, project):
    matching = [entry for entry in payload["extraction"] if entry["project"] == project]
    assert len(matching) == 1, f"{project} appears {len(matching)} times in {payload['extraction']}"
    return matching[0]


# --- the pair that is the whole point --------------------------------------------------


@pytest.mark.parametrize("tool", FACT_READING_TOOLS)
def test_a_never_extracted_project_is_named_as_such(tool, repo):
    arguments = {"project": "alpha/never"}
    if tool == "get_interview_summary":
        arguments = {**arguments, "session_id": SESSION_RICH}
    payload = _scoped(tool, arguments, repo_root=repo)

    assert payload["unextracted_projects"] == ["alpha/never"]
    assert _status_for(payload, "alpha/never") == {
        "project": "alpha/never",
        "status": mcp_tools.EXTRACTION_NOT_EXTRACTED,
        "facts": 0,
        # These fixtures declare no `project.json`, so nothing has been said
        # about whether their claims describe real people.
        "corpus": project_meta.CORPUS_UNDECLARED,
    }


@pytest.mark.parametrize("tool", FACT_READING_TOOLS)
def test_an_extracted_project_whose_facts_bear_on_nothing_is_not_called_unextracted(tool, repo):
    """The discriminating half. An implementation that reports "unextracted"
    whenever it ended up with no USABLE facts passes the test above and fails
    this one, and it would erase the only answer on this surface that is a
    research finding."""
    arguments = {"project": "alpha/quiet"}
    if tool == "get_interview_summary":
        arguments = {**arguments, "session_id": SESSION_QUIET}
    payload = _scoped(tool, arguments, repo_root=repo)

    assert payload["unextracted_projects"] == []
    assert _status_for(payload, "alpha/quiet")["status"] == mcp_tools.EXTRACTION_EXTRACTED
    assert _status_for(payload, "alpha/quiet")["facts"] == 1


def test_the_two_situations_no_longer_produce_the_same_get_standing_payload(repo):
    """Stated as the defect itself rather than as a property of the fix: the
    original report was that these two responses were indistinguishable. The
    counts are still identical -- correctly, both are zero -- so the assertion
    is that the payloads as a whole now differ, and specifically where."""
    never = _scoped("get_standing", {"project": "alpha/never"}, repo_root=repo)
    quiet = _scoped("get_standing", {"project": "alpha/quiet"}, repo_root=repo)

    assert [s["supporting_interviews"] for s in never["standings"]] == [0]
    assert [s["supporting_interviews"] for s in quiet["standings"]] == [0]
    assert never["unextracted_projects"] != quiet["unextracted_projects"], (
        "get_standing still answers 'never extracted' and 'nothing bears on it' identically"
    )


# --- the subtree case a single-node fix gets wrong --------------------------------------


def test_a_populated_sibling_does_not_make_an_unextracted_node_look_examined(repo):
    """The mutation a naive fix misses. "Did this query load any facts at
    all?" is right for one node and wrong for every subtree: `alpha` is rich,
    so a subtree rooted there loads plenty of facts while three of its five
    nodes have nothing to say. The status must be PER NODE."""
    payload = _scoped("get_standing", {"project": "alpha", "scope": "subtree"}, repo_root=repo)

    assert payload["unextracted_projects"] == ["alpha/never"]
    assert {entry["project"]: entry["status"] for entry in payload["extraction"]} == {
        "alpha": mcp_tools.EXTRACTION_EXTRACTED,
        "alpha/broken": mcp_tools.EXTRACTION_UNREADABLE,
        "alpha/empty": mcp_tools.EXTRACTION_EXTRACTED,
        "alpha/never": mcp_tools.EXTRACTION_NOT_EXTRACTED,
        "alpha/quiet": mcp_tools.EXTRACTION_EXTRACTED,
    }


def test_every_node_the_query_read_appears_in_the_status_block(repo):
    """A status block with a hole in it is the shape that started all this: a
    reader checking `unextracted_projects` and finding a node absent from it
    concludes that node was extracted."""
    payload = _scoped("get_facts", {"project": "alpha", "scope": "subtree"}, repo_root=repo)
    assert [entry["project"] for entry in payload["extraction"]] == payload["nodes"]


# --- the third and fourth answers -------------------------------------------------------


def test_an_empty_corpus_file_is_extracted_with_zero_facts_not_unextracted(repo):
    """`facts.json` present and empty means extraction RAN and filed nothing
    here. Collapsing it into `not-extracted` would tell the operator to re-run
    the command they just ran -- the remedy-for-the-wrong-problem failure the
    hub's own status block was criticised for."""
    payload = _scoped("get_facts", {"project": "alpha/empty"}, repo_root=repo)

    assert payload["unextracted_projects"] == []
    assert _status_for(payload, "alpha/empty") == {
        "project": "alpha/empty",
        "status": mcp_tools.EXTRACTION_EXTRACTED,
        "facts": 0,
        "corpus": project_meta.CORPUS_UNDECLARED,
    }


def test_an_unreadable_corpus_is_a_fault_and_lands_in_errors_as_well(repo):
    """`unreadable` is a fault and `not-extracted` is not, so the two must not
    share a code path. It is reported in BOTH lists deliberately: `errors`
    stays the key a caller can treat as "tell a human", and the status block
    stays complete."""
    payload = _scoped("get_facts", {"project": "alpha/broken"}, repo_root=repo)

    assert _status_for(payload, "alpha/broken")["status"] == mcp_tools.EXTRACTION_UNREADABLE
    assert payload["unextracted_projects"] == [], (
        "an unreadable corpus was reported as merely un-extracted, which understates it"
    )
    assert [entry["project"] for entry in payload["errors"]] == ["alpha/broken"]
    assert payload["error"] is not None


def test_the_unreadable_error_carries_a_type_and_never_the_quote(repo):
    """The module's standing rule, re-checked on the path this change added.
    `pydantic`'s `ValidationError` renders the whole offending record --
    quote included -- into its own message text."""
    payload = _scoped("get_facts", {"project": "alpha", "scope": "subtree"}, repo_root=repo)
    blob = json.dumps(payload)
    assert SENTINEL_QUOTE not in blob
    assert "not json at all" not in blob


# --- the vocabulary itself ---------------------------------------------------------------


def test_the_status_vocabulary_is_pinned_by_equality(repo):
    """Pinned by SET EQUALITY, not by membership. A fourth status added
    without a decision about how every consumer renders it is how a surface
    grows a state that nothing displays -- eight of those are already live in
    this product."""
    assert set(mcp_tools.ALL_EXTRACTION_STATUSES) == {
        "extracted",
        "not-extracted",
        "unreadable",
    }


def test_the_status_words_match_the_hub_so_the_two_surfaces_cannot_drift(repo):
    """The hub decides these words per INTERVIEW and this decides them per
    NODE -- different units, deliberately, because that is the unit each
    surface answers about. The WORDS are shared so a reader of both is never
    told the same situation in two vocabularies.

    Checked against the hub's exported vocabulary rather than by scanning its
    source, which is what this test did on its first cut -- a substring scan
    would have been satisfied by the word appearing in a comment explaining
    why it is no longer used."""
    from app.web import review as review_mod

    shared = set(mcp_tools.ALL_EXTRACTION_STATUSES)
    assert shared <= set(review_mod.ALL_EXTRACTION_STATUSES), (
        f"the hub dropped {sorted(shared - set(review_mod.ALL_EXTRACTION_STATUSES))}; the two "
        "surfaces now describe the same situation in two vocabularies"
    )
    # A SUBSET, not equality, and deliberately so: the hub answers per
    # INTERVIEW and has two statuses that are meaningless per NODE
    # (`no-hypotheses`, `no-facts-from-this-interview`). Asserting equality
    # would force one surface to carry the other's states.
    assert set(review_mod.ALL_EXTRACTION_STATUSES) - shared == {
        "no-hypotheses",
        "no-facts-from-this-interview",
    }
