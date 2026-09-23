"""`scope` -- reading a project NODE, or its whole SUBTREE, over MCP.

**The standing decision this implements, restated so no reader re-litigates
it:** cross-project reads are permitted only WITHIN the same hierarchy, and
only on EXPLICIT request; reads may merge, writes never merge. So `scope`
defaults to `"node"`, `"subtree"` is opt-in, and no scope of any kind reaches
outside one root.

Three failures this file is written to prevent, in descending order of how
much they matter:

1. **A subtree query escaping its root.** Not the obvious `..` -- which
   `app/projects.py::normalize` already made unrepresentable -- but the
   sneaky one: a SIBLING that merely shares a name PREFIX. `acme-corp` is a
   different customer from `acme`, and `"acme-corp".startswith("acme")` is
   `True`. A prefix match is the natural way to write this and it is wrong.
2. **Per-fact counting reappearing in the merge.** Covered in depth in
   `test_interview_summary.py`; asserted again HERE, end to end through real
   protocol bytes, because the shared module being right does not prove the
   server calls it.
3. **One malformed node killing the whole query.** A subtree read that dies
   because one child has a broken `facts.json` is strictly worse than five
   separate node reads.
"""

from __future__ import annotations

import io
import json

import pytest

from app.mcp import server as mcp_server
from app.mcp import tools as mcp_tools

SENTINEL_QUOTE = "mcp-scope-sentinel-quote-9d41f7ba"
SESSION_SHARED = "20260812-101500-shared"
SESSION_CHILD = "20260813-090000-child0"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fact(fact_id, session_id, *, stance="supports", hypothesis_id="h1", provenance="evidence"):
    return {
        "id": fact_id,
        "session_id": session_id,
        "seqs": [1],
        "claim": f"claim behind {fact_id}",
        "provenance": provenance,
        "quoted_text": SENTINEL_QUOTE,
        "bearings": [{"hypothesis_id": hypothesis_id, "stance": stance}],
    }


@pytest.fixture()
def repo(tmp_path):
    """Four nodes, chosen so every escape this file tests is REACHABLE:

    - `acme`               -- the subtree root under test, authors `h1`
    - `acme/onboarding`    -- a child, whose facts must roll up
    - `acme-corp`          -- A DIFFERENT CUSTOMER whose name shares a prefix
                              with `acme`. The whole point of the fixture.
    - `zeta`               -- an unrelated root, which no scope may reach.

    `SESSION_SHARED` is ONE interview whose facts were filed at BOTH `acme`
    and `acme/onboarding`. That is the counting trap, on disk.
    """
    projects = tmp_path / "projects"

    _write_json(
        projects / "acme" / "context" / "hypotheses.json",
        [{"id": "h1", "statement": "Activation stalls on invites",
          "falsifier": "three breeze through", "state": "contested",
          "criterion": "we are right if 3 of 10 raise it unprompted"}],
    )
    _write_json(
        projects / "acme" / "context" / "survey.json",
        [{"id": "q1", "order": 1, "text": "Tell me about onboarding"}],
    )
    _write_json(
        projects / "acme" / "context" / "facts.json",
        [_fact("parent-1", SESSION_SHARED), _fact("parent-2", SESSION_SHARED)],
    )
    _write_json(
        projects / "acme" / "sessions" / SESSION_SHARED / "session.json",
        {"started_at": "2026-08-12T10:15:00+00:00", "closed_at": "2026-08-12T11:00:00+00:00"},
    )

    _write_json(
        projects / "acme" / "onboarding" / "context" / "hypotheses.json",
        [{"id": "h2", "statement": "Invite emails land in spam",
          "falsifier": "nobody mentions email"}],
    )
    _write_json(
        projects / "acme" / "onboarding" / "context" / "facts.json",
        # THREE more SUPPORTING facts from the SAME interview as the parent's
        # two, plus a genuinely different interview that both supports and
        # undermines. Honest answer for h1: 2 supporting interviews over 6
        # supporting facts. The two wrong answers are 6 (per-fact) and 3
        # (de-duplicating on node+session: SHARED counted once per node it was
        # filed at, plus CHILD).
        [
            _fact("child-1", SESSION_SHARED),
            _fact("child-2", SESSION_SHARED),
            _fact("child-3", SESSION_SHARED),
            _fact("child-4", SESSION_CHILD, stance="undermines"),
            _fact("child-5", SESSION_CHILD),
        ],
    )
    _write_json(
        projects / "acme" / "onboarding" / "sessions" / SESSION_CHILD / "session.json",
        {"started_at": "2026-08-13T09:00:00+00:00"},
    )

    # The prefix-sharing SIBLING. Different customer. Its fact bears on `h1`
    # too, so if it is ever pulled in the counts move and the test can see it.
    _write_json(
        projects / "acme-corp" / "context" / "facts.json",
        [_fact("sibling-1", "20260101-000000-sibling")],
    )
    _write_json(
        projects / "acme-corp" / "context" / "hypotheses.json",
        [{"id": "h-sibling", "statement": "different customer entirely",
          "falsifier": "n/a but non-empty"}],
    )
    _write_json(
        projects / "acme-corp" / "sessions" / "20260101-000000-sibling" / "session.json",
        {"started_at": "2026-01-01T00:00:00+00:00"},
    )

    # An unrelated root.
    _write_json(
        projects / "zeta" / "context" / "facts.json",
        [_fact("zeta-1", "20260202-000000-zetaaa")],
    )
    return tmp_path


# --- the byte-level driver (the real stdio loop, as in test_mcp_server.py) -------------


def _drive(requests, *, repo_root):
    payload = "".join(json.dumps(request) + "\n" for request in requests)
    stdout = io.BytesIO()
    mcp_server.serve(io.BytesIO(payload.encode("utf-8")), stdout, repo_root=repo_root)
    return [json.loads(line) for line in stdout.getvalue().decode("utf-8").split("\n") if line]


def _call(name, arguments, *, repo_root):
    return _drive(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": name, "arguments": arguments}}],
        repo_root=repo_root,
    )[0]


def _payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def _scoped(name, arguments, *, repo_root):
    return _payload(_call(name, arguments, repo_root=repo_root))


# --- (a) a subtree query cannot escape its root ----------------------------------------


SCOPED_TOOLS = ("get_hypotheses", "list_interviews", "get_facts", "get_standing")


@pytest.mark.parametrize("tool", SCOPED_TOOLS)
def test_a_subtree_never_reaches_a_sibling_that_merely_shares_a_name_prefix(tool, repo):
    """THE escape this project is most likely to ship.

    `acme-corp` is a different customer. `"acme-corp".startswith("acme")` is
    `True`, so the natural one-line implementation -- filter
    `list_projects()` by `path.startswith(root)` -- silently merges two
    customers' interview corpora and reports the total as one number. The
    correct key is the SEPARATOR-terminated prefix, `"acme/"`.

    Asserted on the `nodes` list rather than only on the merged contents,
    because a leak with no facts in it would otherwise pass unnoticed until
    the day the sibling had data.
    """
    payload = _scoped(tool, {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert payload["nodes"] == ["acme", "acme/onboarding"], (
        f"{tool} pulled a prefix-sharing sibling into the subtree: {payload['nodes']}"
    )
    assert "acme-corp" not in payload["nodes"]
    blob = json.dumps(payload)
    assert "acme-corp" not in blob, f"{tool} leaked a different customer's node into the response"
    assert "sibling-1" not in blob, f"{tool} leaked a different customer's fact"
    assert "zeta" not in blob, f"{tool} reached an unrelated root"


@pytest.mark.parametrize("tool", SCOPED_TOOLS)
@pytest.mark.parametrize(
    "hostile",
    [
        "..",
        "../acme-corp",
        "acme/../acme-corp",
        "acme/../../secrets",
        "/etc",
        "C:/Windows",
        "\\\\server\\share",
        "acme\\onboarding",
        "",
        "   ",
    ],
)
def test_a_hostile_project_path_is_refused_under_subtree_scope_too(tool, hostile, repo):
    """`scope` must not become a second, laxer path resolver. Every one of
    these is refused by `app/projects.py::normalize`, and the point of running
    them again HERE is that a subtree implementation which built its member
    paths by string-joining rather than by re-resolving would bypass that
    module entirely."""
    response = _call(tool, {"project": hostile, "scope": "subtree"}, repo_root=repo)
    assert response.get("error", {}).get("code") == -32602, (
        f"{tool} did not refuse {hostile!r} under subtree scope"
    )


@pytest.mark.parametrize("tool", SCOPED_TOOLS)
def test_a_subtree_rooted_at_a_child_does_not_climb_to_its_parent(tool, repo):
    """Reads may merge DOWNWARD only. A subtree is a node and its
    descendants -- never its ancestors, and never its siblings."""
    payload = _scoped(tool, {"project": "acme/onboarding", "scope": "subtree"}, repo_root=repo)
    assert payload["nodes"] == ["acme/onboarding"]
    assert "parent-1" not in json.dumps(payload), "a subtree climbed to its parent"


@pytest.mark.parametrize("tool", SCOPED_TOOLS)
def test_every_node_in_a_resolved_subtree_shares_the_requested_root(tool, repo):
    """`app/projects.py::share_a_root` is where the same-hierarchy rule
    lives. Re-asserted over the tool's own reported `nodes` so the rule is
    checked against what the server actually READ, not against what the
    resolver intended to read."""
    from app import projects as projects_mod

    payload = _scoped(tool, {"project": "acme", "scope": "subtree"}, repo_root=repo)
    for node in payload["nodes"]:
        assert projects_mod.share_a_root("acme", node), f"{node} is outside the requested hierarchy"


def test_the_subtree_resolver_itself_refuses_a_prefix_sharing_sibling(repo):
    """The resolver, tested directly rather than only through a tool, and the
    mutation this catches that the tool-level tests do not: a tool could
    filter the sibling out of its own OUTPUT while the resolver still opened
    the sibling's files -- a read of another customer's corpus that never
    shows up in a response."""
    nodes = mcp_tools.resolve_scope("acme", "subtree", repo_root=repo)
    assert [node.path for node in nodes] == ["acme", "acme/onboarding"]


def test_an_unknown_scope_value_is_refused_rather_than_defaulted(repo):
    """Silently treating an unrecognized scope as `node` would make a typo
    ("subtre", "all", "recursive") read LESS than the caller asked for and
    say nothing -- the caller would draw a conclusion from a third of their
    corpus. Silently treating it as `subtree` would be worse: an
    accidental cross-project read, which the standing decision permits only
    on EXPLICIT request."""
    for bad in ("subtre", "all", "recursive", "SUBTREE", "", 1, None, True):
        response = _call("get_facts", {"project": "acme", "scope": bad}, repo_root=repo)
        assert response.get("error", {}).get("code") == -32602, f"scope={bad!r} was not refused"


def test_scope_defaults_to_node(repo):
    """The standing decision: cross-project reads happen only on EXPLICIT
    request, so the default can never be the merging one."""
    default = _scoped("get_facts", {"project": "acme"}, repo_root=repo)
    explicit = _scoped("get_facts", {"project": "acme", "scope": "node"}, repo_root=repo)
    assert default["scope"] == "node"
    assert default == explicit
    assert default["nodes"] == ["acme"]
    assert [fact["id"] for fact in default["facts"]] == ["parent-1", "parent-2"]


def test_a_subtree_is_a_superset_of_its_own_node_scope(repo):
    """An invariant worth pinning: whatever else the merge does, it may never
    LOSE a row the node-scoped read returned. A resolver that replaced the
    root with its children rather than prepending it would pass every
    escape test above."""
    node = _scoped("get_facts", {"project": "acme"}, repo_root=repo)
    subtree = _scoped("get_facts", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert set(fact["id"] for fact in node["facts"]) <= set(f["id"] for f in subtree["facts"])
    assert subtree["nodes"][0] == "acme", "the requested node must lead its own subtree"


def test_a_subtree_rooted_at_a_node_that_does_not_exist_is_empty_not_an_error(repo):
    """Resolution deliberately does not require existence -- `resolve()` says
    so. A subtree of a not-yet-created node is the node itself, empty."""
    payload = _scoped("get_facts", {"project": "acme/nothing_here", "scope": "subtree"},
                      repo_root=repo)
    assert payload["nodes"] == ["acme/nothing_here"]
    assert payload["facts"] == []


# --- node attribution ------------------------------------------------------------------


def test_every_merged_fact_says_which_node_it_came_from(repo):
    """A rolled-up answer that cannot be traced back to a node is worse than
    no rollup."""
    payload = _scoped("get_facts", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    attribution = {fact["id"]: fact["project"] for fact in payload["facts"]}
    assert attribution == {
        "parent-1": "acme", "parent-2": "acme",
        "child-1": "acme/onboarding", "child-2": "acme/onboarding",
        "child-3": "acme/onboarding", "child-4": "acme/onboarding",
        "child-5": "acme/onboarding",
    }


def test_node_scope_carries_the_same_attribution_key_as_subtree_scope(repo):
    """The attribution field is present in BOTH scopes rather than appearing
    only when merging. A response whose SHAPE depends on the mode invites a
    consumer to read `project` sometimes and infer it the rest of the time --
    and the inference is wrong exactly when it matters."""
    payload = _scoped("get_facts", {"project": "acme"}, repo_root=repo)
    assert all(fact["project"] == "acme" for fact in payload["facts"])
    hypotheses = _scoped("get_hypotheses", {"project": "acme"}, repo_root=repo)
    assert all(item["project"] == "acme" for item in hypotheses["hypotheses"])
    interviews = _scoped("list_interviews", {"project": "acme"}, repo_root=repo)
    assert all(item["project"] == "acme" for item in interviews["interviews"])


def test_merged_hypotheses_and_interviews_are_attributed_to_their_nodes(repo):
    hypotheses = _scoped("get_hypotheses", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert {item["id"]: item["project"] for item in hypotheses["hypotheses"]} == {
        "h1": "acme", "h2": "acme/onboarding",
    }
    interviews = _scoped("list_interviews", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert {item["session_id"]: item["project"] for item in interviews["interviews"]} == {
        SESSION_SHARED: "acme", SESSION_CHILD: "acme/onboarding",
    }


# --- (b) THE COUNTING TRAP, end to end -------------------------------------------------


def test_a_subtree_rollup_counts_interviews_not_facts(repo):
    """FIVE facts about `h1` from TWO interviews, split across two nodes --
    three of them from the SAME interview as the other node's two.

    The honest answer is 2 supporting interviews. Both wrong answers are
    asserted against by name because both are reachable:

    - **6** -- counting facts. The merge is where this comes back, because
      the union genuinely is a longer list.
    - **3** -- de-duplicating on `(node, session_id)`: the shared interview
      counted once per node it was filed at, plus the child's own.
    """
    payload = _scoped("get_standing", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    h1 = {item["hypothesis_id"]: item for item in payload["standings"]}["h1"]
    assert h1["supporting_interviews"] == 2, (
        "6 means facts were counted; 3 means the de-duplication key included the node"
    )
    assert h1["supporting_interviews"] not in (3, 6)
    assert h1["supporting_facts"] == 6, "the facts themselves must not be lost, only not counted"
    assert h1["contrary_interviews"] == 1


def test_the_same_interview_filed_at_two_nodes_is_listed_once_per_node_it_lives_at(repo):
    """The counterpart to the count: `list_interviews` reports where session
    DIRECTORIES are, and a session directory exists at exactly one node here.
    Stated explicitly so the interview LIST and the interview COUNT are not
    confused for one another."""
    payload = _scoped("list_interviews", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert sorted(item["session_id"] for item in payload["interviews"]) == sorted(
        [SESSION_SHARED, SESSION_CHILD]
    )


def test_a_hypothesis_at_the_parent_is_scored_against_a_childs_facts(repo):
    """The reason `subtree` exists at all: `h1` is authored at `acme` and
    most of its evidence was filed under `acme/onboarding`. Node scope must
    NOT see that evidence, and subtree scope must."""
    node = _scoped("get_standing", {"project": "acme"}, repo_root=repo)
    subtree = _scoped("get_standing", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert {s["hypothesis_id"]: s["supporting_interviews"] for s in node["standings"]}["h1"] == 1
    assert {s["hypothesis_id"]: s["supporting_interviews"] for s in subtree["standings"]}["h1"] == 2


def test_a_standing_names_the_nodes_that_contributed_to_it(repo):
    payload = _scoped("get_standing", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    h1 = {item["hypothesis_id"]: item for item in payload["standings"]}["h1"]
    assert h1["project"] == "acme", "the node that AUTHORED the hypothesis"
    assert h1["projects"] == ["acme", "acme/onboarding"], "the nodes that contributed evidence"


def test_get_standing_can_be_filtered_to_one_hypothesis(repo):
    payload = _scoped(
        "get_standing", {"project": "acme", "scope": "subtree", "hypothesis_id": "h1"},
        repo_root=repo,
    )
    assert [item["hypothesis_id"] for item in payload["standings"]] == ["h1"]


def test_get_standing_reports_a_duplicated_hypothesis_id_rather_than_merging_it(repo):
    """Two nodes may each author `h1` meaning different things. Merging them
    into one standing would invent a hypothesis nobody wrote; dropping one
    would hide it. Both are returned, each attributed, and the collision is
    NAMED so an agent does not read two rows as two independent results."""
    _write_json(
        repo / "projects" / "acme" / "onboarding" / "context" / "hypotheses.json",
        [{"id": "h1", "statement": "a DIFFERENT belief that reuses the id",
          "falsifier": "something else"}],
    )
    payload = _scoped("get_standing", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert payload["duplicate_hypothesis_ids"] == ["h1"]
    assert sorted(item["project"] for item in payload["standings"]) == ["acme", "acme/onboarding"]


def test_an_unscorable_hypothesis_is_marked_so_in_a_rollup(repo):
    payload = _scoped("get_standing", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    by_id = {item["hypothesis_id"]: item for item in payload["standings"]}
    assert by_id["h1"]["is_scorable"] is True
    assert by_id["h2"]["is_scorable"] is False


# --- get_interview_summary -------------------------------------------------------------


def test_an_interview_summary_counts_itself_as_one_interview_however_many_facts(repo):
    """The same trap, one level down: an interview summary over five facts is
    still ONE interview. Every count that is about claims is named `*_facts`
    so the two can never be read as the same number."""
    payload = _scoped(
        "get_interview_summary",
        {"project": "acme", "session_id": SESSION_SHARED, "scope": "subtree"},
        repo_root=repo,
    )
    assert payload["interviews"] == 1
    assert payload["fact_count"] == 5
    assert payload["projects"] == ["acme", "acme/onboarding"]
    touches = {item["hypothesis_id"]: item for item in payload["touches"]}
    assert touches["h1"]["supporting_facts"] == 5


def test_an_interview_summary_under_node_scope_sees_only_that_nodes_facts(repo):
    payload = _scoped(
        "get_interview_summary", {"project": "acme", "session_id": SESSION_SHARED},
        repo_root=repo,
    )
    assert payload["fact_count"] == 2
    assert payload["projects"] == ["acme"]
    assert payload["interviews"] == 1


def test_an_interview_summary_reports_where_the_session_was_found(repo):
    payload = _scoped(
        "get_interview_summary",
        {"project": "acme", "session_id": SESSION_CHILD, "scope": "subtree"},
        repo_root=repo,
    )
    assert payload["found_at"] == ["acme/onboarding"]
    assert payload["started_at"] == "2026-08-13T09:00:00+00:00"
    assert payload["closed"] is False


def test_an_unknown_session_id_is_an_empty_summary_that_says_so(repo):
    payload = _scoped(
        "get_interview_summary", {"project": "acme", "session_id": "20990101-000000-nosuch"},
        repo_root=repo,
    )
    assert payload["found_at"] == []
    assert payload["fact_count"] == 0
    assert payload["error"], "a session that does not exist must be NAMED, not silently empty"


def test_a_session_id_is_required_and_validated(repo):
    for bad in ({}, {"session_id": 4}, {"session_id": "../../../etc"}, {"session_id": "a/b"}):
        arguments = {"project": "acme", **bad}
        response = _call("get_interview_summary", arguments, repo_root=repo)
        assert response.get("error", {}).get("code") == -32602, f"{bad!r} was not refused"


# --- (c) the sentinel, over every tool including the new ones --------------------------


def _every_tool_invocation():
    """Every tool, in BOTH scopes, plus every optional argument. Enumerated
    here rather than derived from the registry so that a new tool added
    without a line in this list is caught by
    `test_the_privacy_sweep_covers_every_registered_tool` below -- an
    auto-derived list would silently "cover" a new tool with a call that
    passes no arguments and therefore reaches nothing."""
    return [
        ("list_projects", {}),
        ("get_survey", {"project": "acme"}),
        ("get_hypotheses", {"project": "acme"}),
        ("get_hypotheses", {"project": "acme", "scope": "subtree"}),
        ("list_interviews", {"project": "acme"}),
        ("list_interviews", {"project": "acme", "scope": "subtree"}),
        ("get_facts", {"project": "acme"}),
        ("get_facts", {"project": "acme", "scope": "subtree"}),
        ("get_facts", {"project": "acme", "scope": "subtree", "hypothesis_id": "h1"}),
        ("get_standing", {"project": "acme"}),
        ("get_standing", {"project": "acme", "scope": "subtree"}),
        ("get_standing", {"project": "acme", "scope": "subtree", "hypothesis_id": "h1"}),
        ("get_interview_summary", {"project": "acme", "session_id": SESSION_SHARED}),
        ("get_interview_summary",
         {"project": "acme", "session_id": SESSION_SHARED, "scope": "subtree"}),
    ]


def test_no_serialized_response_of_any_tool_in_any_scope_contains_the_verbatim_quote(repo):
    """THE failure: a past customer's verbatim speech leaving the machine.

    Asserted over `json.dumps` of the WHOLE response -- envelope, result,
    content, error text and all -- never a chosen sub-part, because the
    interesting leak is a quote arriving in a differently-named field
    (`evidence`, `excerpt`, a debug echo of the loaded record). A subtree
    rollup widens this surface: it now reads MORE files per call, and each
    one of them carries quotes.
    """
    assert len(SENTINEL_QUOTE) > 16, "a short sentinel proves nothing either way"
    raw = (repo / "projects" / "acme" / "context" / "facts.json").read_text(encoding="utf-8")
    assert SENTINEL_QUOTE in raw, "the fixture stopped carrying a quote -- this test is now vacuous"

    for name, arguments in _every_tool_invocation():
        blob = json.dumps(_call(name, arguments, repo_root=repo))
        assert SENTINEL_QUOTE not in blob, f"{name}{arguments} leaked the verbatim quote"


def test_the_privacy_sweep_covers_every_registered_tool(repo):
    """The mutation the sweep above cannot survive without. A tool added to
    the registry but not to `_every_tool_invocation` would be untested for the
    one property that matters most, and nothing would say so."""
    swept = {name for name, _arguments in _every_tool_invocation()}
    registered = {tool["name"] for tool in mcp_tools.TOOL_DEFINITIONS}
    assert registered <= swept, f"tools registered but never swept for quotes: {registered - swept}"


def test_a_degraded_subtree_node_reports_its_exception_type_never_its_message(repo):
    """The sneakiest leak in the merged surface: a malformed `facts.json` at
    ONE node of a subtree, whose parse or validation error message quotes the
    offending record -- which contains the quote. `pydantic` renders
    `input_value=` into its own message text, so `str(exc)` here would hand a
    customer's words to the caller as a diagnostic, from a node the caller
    may not even have named."""
    _write_json(
        repo / "projects" / "acme" / "onboarding" / "context" / "facts.json",
        [{"id": "f1", "session_id": SESSION_CHILD, "seqs": [1], "claim": "c",
          "provenance": "evidence", "quoted_text": "ordinary", "bearings": [],
          "verbatim": SENTINEL_QUOTE}],  # a drifted extra key; pydantic prints its VALUE back
    )
    for tool in ("get_facts", "get_standing"):
        response = _call(tool, {"project": "acme", "scope": "subtree"}, repo_root=repo)
        blob = json.dumps(response)
        assert SENTINEL_QUOTE not in blob, f"{tool} echoed the offending record's quote"
        assert _payload(response)["error"], "a malformed node must be NAMED, not silently empty"


# --- (d) one malformed node degrades that node only -------------------------------------


@pytest.mark.parametrize("tool", ("get_facts", "get_standing"))
def test_one_malformed_node_degrades_that_node_only(tool, repo):
    """A subtree read that dies because one child is broken is strictly worse
    than five separate node reads. The good node's rows must still arrive, and
    the bad node must be named with its own path -- "something failed
    somewhere in the subtree" is not an answer an agent can act on."""
    (repo / "projects" / "acme" / "onboarding" / "context" / "facts.json").write_text(
        "{{{ not json", encoding="utf-8"
    )
    payload = _scoped(tool, {"project": "acme", "scope": "subtree"}, repo_root=repo)

    errors = {entry["project"]: entry["error"] for entry in payload["errors"]}
    assert list(errors) == ["acme/onboarding"], (
        f"the failure was not attributed to the one node that failed: {payload['errors']}"
    )
    assert "acme/onboarding" in payload["error"], "the merged error must name the failing node"

    if tool == "get_facts":
        assert [fact["id"] for fact in payload["facts"]] == ["parent-1", "parent-2"], (
            "one broken node hid a healthy node's facts"
        )
    else:
        h1 = {item["hypothesis_id"]: item for item in payload["standings"]}["h1"]
        assert h1["supporting_interviews"] == 1, "the healthy node's evidence was lost"
        # h2 is authored at the broken node -- its HYPOTHESES file is fine, so
        # the hypothesis must still appear even though its facts did not load.
        assert "h2" in {item["hypothesis_id"] for item in payload["standings"]}


def test_a_malformed_node_at_the_ROOT_still_returns_the_healthy_children(repo):
    """The mirror image, and the one a naive implementation gets wrong: the
    root is the node the caller NAMED, so a failure there feels fatal. It is
    not -- the children are still readable and the caller asked for them."""
    (repo / "projects" / "acme" / "context" / "facts.json").write_text("nope", encoding="utf-8")
    payload = _scoped("get_facts", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert [entry["project"] for entry in payload["errors"]] == ["acme"]
    assert [fact["id"] for fact in payload["facts"]] == [
        "child-1", "child-2", "child-3", "child-4", "child-5",
    ]


def test_an_unreadable_interview_in_one_subtree_node_does_not_hide_the_others(repo):
    bad = repo / "projects" / "acme" / "sessions" / SESSION_SHARED / "transcript.jsonl"
    bad.write_text("not json\n{}\n", encoding="utf-8")
    payload = _scoped("list_interviews", {"project": "acme", "scope": "subtree"}, repo_root=repo)
    assert [item["session_id"] for item in payload["interviews"]] == [SESSION_CHILD]
    assert [entry["project"] for entry in payload["errors"]] == ["acme"]


def test_a_clean_subtree_reports_no_errors_at_all(repo):
    """The vacuity check on every degradation test above: if `errors` were
    always populated, those tests would prove nothing about attribution."""
    for tool in SCOPED_TOOLS:
        payload = _scoped(tool, {"project": "acme", "scope": "subtree"}, repo_root=repo)
        assert payload["errors"] == [], f"{tool} reported an error on a healthy subtree"
        assert payload["error"] is None


# --- the server still writes nothing ----------------------------------------------------


def test_no_tool_in_any_scope_writes_anything(repo):
    """Gate 19 pins the hypothesis writers by equality; an MCP write is the
    drift it exists to stop. A subtree read touches many more files than a
    node read, which is many more chances to open one for writing."""
    def _snapshot():
        return {
            str(path.relative_to(repo)): path.read_bytes()
            for path in sorted(repo.rglob("*")) if path.is_file()
        }

    before = _snapshot()
    for name, arguments in _every_tool_invocation():
        _call(name, arguments, repo_root=repo)
    assert _snapshot() == before


def test_the_two_formerly_deferred_tools_are_now_served_and_the_seam_is_closed(repo):
    """`get_standing` and `get_interview_summary` were withheld until
    `app/coach/interview_summary.py` existed, so the per-interview counting
    rule could not be forked. It exists now, both tools route through it, and
    the seam constant is emptied rather than deleted -- so a reader of the old
    commit finds the reason the wait ended."""
    response = _drive([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], repo_root=repo)[0]
    served = {tool["name"] for tool in response["result"]["tools"]}
    assert {"get_standing", "get_interview_summary"} <= served
    assert mcp_tools.TOOLS_AWAITING_THE_SHARED_SUMMARY_MODULE == ()


def test_the_scoped_tools_declare_scope_in_their_input_schema(repo):
    """A parameter a client cannot discover is a parameter no agent will
    use. Pinned by equality against the tools that actually accept it, so a
    new scoped tool that forgets to declare it fails here."""
    response = _drive([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], repo_root=repo)[0]
    declaring = {
        tool["name"] for tool in response["result"]["tools"]
        if "scope" in tool["inputSchema"]["properties"]
    }
    assert declaring == set(SCOPED_TOOLS) | {"get_interview_summary"}
    for tool in response["result"]["tools"]:
        schema = tool["inputSchema"]["properties"].get("scope")
        if schema is not None:
            assert schema["enum"] == ["node", "subtree"]
            assert schema["default"] == "node"
