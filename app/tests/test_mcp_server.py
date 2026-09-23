"""The MCP server: JSON-RPC 2.0 over stdio, read-only, and quote-free.

Three things this file is written to prove, in descending order of how much
they matter:

1. **No verbatim customer speech leaves through any tool.** Asserted against
   `json.dumps` of the WHOLE response -- never a chosen sub-part -- because
   the interesting leak is a quote arriving in a differently-named field. The
   same shape gate 21 already uses for the vendor request bodies, pointed at
   the one surface in this project that crosses the machine's trust boundary
   by design.
2. **The framing is right, at the byte level.** A server whose handlers are
   perfect and whose framing is wrong is the classic failure here, and it is
   invisible to any test that calls the handler functions directly. Every
   protocol test below drives real bytes in and reads real bytes out.
3. **It degrades rather than dies.** A missing or malformed artifact, an
   unknown tool, an unparseable line -- each is an answer, and the server is
   still answering afterwards.
"""

from __future__ import annotations

import io
import json

import pytest

from app.mcp import server as mcp_server
from app.mcp import tools as mcp_tools

# --- fixtures -----------------------------------------------------------------------

SENTINEL_QUOTE = "mcp-privacy-sentinel-quote-4b91d0ac"
SENTINEL_TAIL = "mcp-privacy-sentinel-truncated-tail-77e1"
SESSION_ID = "20260812-101500-abcdef"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def repo(tmp_path):
    """A two-node project tree with one interview and one fact whose
    `quoted_text` is a sentinel no response may ever contain."""
    node = tmp_path / "projects" / "acme"
    _write_json(
        node / "context" / "survey.json",
        [{"id": "q1", "order": 1, "text": "Tell me about onboarding"}],
    )
    _write_json(
        node / "context" / "hypotheses.json",
        [
            {
                "id": "h1",
                "statement": "Activation stalls on the invite step",
                "falsifier": "three interviewees breeze through invites",
                "state": "contested",
                "criterion": "we are right if 3 of 10 raise it unprompted",
            },
            {
                "id": "h2",
                "statement": "Pricing is the blocker",
                "falsifier": "nobody mentions price unprompted",
            },
        ],
    )
    _write_json(
        node / "context" / "facts.json",
        [
            {
                "id": "f1",
                "session_id": SESSION_ID,
                "seqs": [4],
                "claim": "the invite step is where they stall",
                "provenance": "evidence",
                "quoted_text": SENTINEL_QUOTE,
                "bearings": [{"hypothesis_id": "h1", "stance": "supports"}],
            },
            {
                "id": "f2",
                "session_id": SESSION_ID,
                "seqs": [9],
                "claim": "price never came up",
                "provenance": "inference",
                "quoted_text": SENTINEL_QUOTE,
                "bearings": [{"hypothesis_id": "h2", "stance": "undermines"}],
            },
        ],
    )
    session_dir = node / "sessions" / SESSION_ID
    _write_json(
        session_dir / "session.json",
        {"started_at": "2026-08-12T10:15:00+00:00", "closed_at": "2026-08-12T11:00:00+00:00",
         "coach_enabled": True},
    )
    # A complete line plus a truncated final line -- the crash tail is raw
    # customer speech, which is why `truncated_tail` is reported as a BOOLEAN.
    (session_dir / "transcript.jsonl").write_text(
        json.dumps({"kind": "segment", "seq": 4, "speaker": "interviewee", "text": SENTINEL_QUOTE})
        + "\n"
        + '{"kind": "segment", "seq": 9, "text": "' + SENTINEL_TAIL,
        encoding="utf-8",
    )
    # A second node with nothing in it -- `list_projects` must still see it.
    (tmp_path / "projects" / "acme" / "onboarding").mkdir(parents=True)
    return tmp_path


# --- the byte-level driver ------------------------------------------------------------


def _drive(requests, *, repo_root, raw=None):
    """Runs the real stdio loop over real bytes and returns the decoded
    responses. `raw` bypasses `json.dumps` so a malformed line can be fed in."""
    if raw is None:
        payload = "".join(json.dumps(request) + "\n" for request in requests)
    else:
        payload = raw
    stdin = io.BytesIO(payload.encode("utf-8"))
    stdout = io.BytesIO()
    mcp_server.serve(stdin, stdout, repo_root=repo_root)
    return stdout.getvalue()


def _lines(raw_out):
    return [line for line in raw_out.decode("utf-8").split("\n") if line]


def _decoded(raw_out):
    return [json.loads(line) for line in _lines(raw_out)]


def _call(name, arguments, *, repo_root, request_id=1):
    out = _drive(
        [{"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
          "params": {"name": name, "arguments": arguments}}],
        repo_root=repo_root,
    )
    return _decoded(out)[0]


def _payload(response):
    """The tool's own JSON document, unwrapped from the MCP content envelope."""
    return json.loads(response["result"]["content"][0]["text"])


# --- framing --------------------------------------------------------------------------


def test_each_response_is_exactly_one_newline_delimited_json_object(repo):
    raw_out = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": mcp_server.PROTOCOL_VERSION, "capabilities": {},
                        "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        repo_root=repo,
    )
    assert raw_out.endswith(b"\n"), "every message must be newline-terminated"
    lines = _lines(raw_out)
    assert len(lines) == 2, f"expected one line per request, got {lines}"
    for line in lines:
        parsed = json.loads(line)
        assert parsed["jsonrpc"] == "2.0"


def test_a_response_never_contains_an_embedded_newline(repo):
    """The stdio transport delimits on newlines, so a message that embeds one
    desynchronizes every subsequent message on the stream. A survey question
    containing a real newline is the sneaky version of this."""
    _write_json(
        repo / "projects" / "acme" / "context" / "survey.json",
        [{"id": "q1", "order": 1, "text": "line one\nline two\r\nline three"}],
    )
    raw_out = _drive(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
          "params": {"name": "get_survey", "arguments": {"project": "acme"}}}],
        repo_root=repo,
    )
    assert len(_lines(raw_out)) == 1, "an embedded newline split one message into two frames"
    assert "line two" in json.dumps(_payload(_decoded(raw_out)[0]))


def test_blank_lines_between_messages_are_ignored(repo):
    raw = (
        "\n\n"
        + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        + "\n\n"
    )
    out = _drive(None, repo_root=repo, raw=raw)
    assert len(_decoded(out)) == 1


def test_initialize_reports_the_protocol_version_and_a_tools_capability(repo):
    response = _decoded(
        _drive(
            [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": mcp_server.PROTOCOL_VERSION, "capabilities": {}}}],
            repo_root=repo,
        )
    )[0]
    result = response["result"]
    assert result["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert result["capabilities"]["tools"] == {}
    assert result["serverInfo"]["name"]
    assert result["serverInfo"]["version"]


def test_an_unsupported_client_protocol_version_is_answered_with_the_servers_own(repo):
    response = _decoded(
        _drive(
            [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "1999-01-01", "capabilities": {}}}],
            repo_root=repo,
        )
    )[0]
    assert response["result"]["protocolVersion"] == mcp_server.PROTOCOL_VERSION


def test_a_notification_gets_no_response_at_all(repo):
    """A JSON-RPC notification has no `id`, and answering one is a protocol
    violation that desynchronizes a strict client.

    Both shapes are exercised, and the second is the one that matters: a
    notification in the `notifications/` namespace is refused by a namespace
    check that would still hold if the general rule were broken, whereas a
    notification naming a method the server DOES serve (`ping`, a known
    method that would otherwise produce a result) can only be silenced by
    the general rule itself. The one-shape version of this test survives
    deleting that rule entirely."""
    out = _drive(
        [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "ping"},
            {"jsonrpc": "2.0", "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}},
        ],
        repo_root=repo,
    )
    responses = _decoded(out)
    assert len(responses) == 1, f"a notification was answered: {responses}"
    assert responses[0]["id"] == 7


def test_a_malformed_line_is_a_parse_error_and_the_server_keeps_serving(repo):
    raw = (
        "{not json at all\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        + "\n"
    )
    responses = _decoded(_drive(None, repo_root=repo, raw=raw))
    assert responses[0]["error"]["code"] == -32700
    assert responses[0]["id"] is None
    assert responses[1]["id"] == 2, "the server stopped serving after a bad line"


def test_a_non_object_message_is_an_invalid_request(repo):
    responses = _decoded(_drive(None, repo_root=repo, raw="[1, 2, 3]\n"))
    assert responses[0]["error"]["code"] == -32600


def test_an_unknown_method_is_method_not_found(repo):
    response = _decoded(
        _drive([{"jsonrpc": "2.0", "id": 3, "method": "resources/list"}], repo_root=repo)
    )[0]
    assert response["error"]["code"] == -32601
    assert "result" not in response


def test_an_unknown_tool_name_is_a_json_rpc_error_not_a_crash(repo):
    out = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "delete_everything", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        repo_root=repo,
    )
    responses = _decoded(out)
    assert responses[0]["error"]["code"] == -32602
    assert "delete_everything" in responses[0]["error"]["message"]
    assert responses[1]["id"] == 2, "the server died on an unknown tool"


def test_a_tool_call_with_a_missing_required_argument_is_an_invalid_params_error(repo):
    response = _decoded(
        _drive(
            [{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "get_survey", "arguments": {}}}],
            repo_root=repo,
        )
    )[0]
    assert response["error"]["code"] == -32602


# --- the tool surface -----------------------------------------------------------------


def test_tools_list_is_exactly_the_seven_read_only_tools(repo):
    """By EQUALITY and in order, the shape gates 8/18/19 already use here: the
    protocol surface grows by decision, never by drift. Widened from five to
    seven in the commit that added `get_standing` and `get_interview_summary`
    -- both read-only, both routed through `app/coach/interview_summary.py`
    rather than counting anything themselves."""
    response = _decoded(
        _drive([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], repo_root=repo)
    )[0]
    names = [tool["name"] for tool in response["result"]["tools"]]
    assert names == [
        "list_projects", "get_survey", "get_hypotheses", "list_interviews", "get_facts",
        "get_standing", "get_interview_summary",
    ]
    for tool in response["result"]["tools"]:
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"
        assert tool["inputSchema"]["additionalProperties"] is False
    assert not {name for name in names if name.startswith(("set_", "create_", "update_",
                                                           "delete_", "write_", "mark_"))}, (
        "a write tool appeared on a surface gate 19 exists to keep read-only"
    )


def test_the_deferred_tool_seam_is_closed_rather_than_deleted(repo):
    """`get_standing` and `get_interview_summary` were withheld from the first
    cut until `app/coach/interview_summary.py` existed, so the per-INTERVIEW
    counting rule could not be forked across two surfaces. It exists now and
    both tools route through it.

    The constant is EMPTIED rather than removed: a reader arriving from the
    commit that introduced it finds out what happened to the seam rather than
    finding nothing where the answer used to be."""
    response = _decoded(
        _drive([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}], repo_root=repo)
    )[0]
    served = {tool["name"] for tool in response["result"]["tools"]}
    assert mcp_tools.TOOLS_AWAITING_THE_SHARED_SUMMARY_MODULE == ()
    assert {"get_standing", "get_interview_summary"} <= served


def test_no_tool_writes_anything(repo):
    """Gate 19 pins the hypothesis writers by equality; an MCP write is the
    exact drift it exists to stop. Proven behaviourally too: calling every
    tool leaves the tree byte-identical."""
    before = {
        str(path.relative_to(repo)): path.read_bytes()
        for path in sorted(repo.rglob("*")) if path.is_file()
    }
    for name, arguments in _every_tool_invocation():
        _call(name, arguments, repo_root=repo)
    after = {
        str(path.relative_to(repo)): path.read_bytes()
        for path in sorted(repo.rglob("*")) if path.is_file()
    }
    assert before == after


def _every_tool_invocation():
    """Every tool, with every optional argument, in BOTH scopes.

    Extended in the commit that added `scope`, `get_standing` and
    `get_interview_summary` -- a new tool left out of this list would be
    untested for the one property that matters most on this surface, so
    `test_the_privacy_sweep_covers_every_registered_tool` below pins the list
    against the registry itself."""
    return [
        ("list_projects", {}),
        ("get_survey", {"project": "acme"}),
        ("get_hypotheses", {"project": "acme"}),
        ("get_hypotheses", {"project": "acme", "scope": "subtree"}),
        ("list_interviews", {"project": "acme"}),
        ("list_interviews", {"project": "acme", "scope": "subtree"}),
        ("get_facts", {"project": "acme"}),
        ("get_facts", {"project": "acme", "scope": "subtree"}),
        ("get_facts", {"project": "acme", "hypothesis_id": "h1"}),
        ("get_standing", {"project": "acme"}),
        ("get_standing", {"project": "acme", "scope": "subtree"}),
        ("get_standing", {"project": "acme", "hypothesis_id": "h1"}),
        ("get_interview_summary", {"project": "acme", "session_id": SESSION_ID}),
        ("get_interview_summary",
         {"project": "acme", "session_id": SESSION_ID, "scope": "subtree"}),
    ]


def test_the_privacy_sweep_covers_every_registered_tool(repo):
    """The mutation the sentinel sweep cannot survive without: a tool added to
    the registry but never added to `_every_tool_invocation` is swept by
    nothing, and no test says so. Pinned against the live registry rather than
    against a literal list, so the two cannot drift apart."""
    swept = {name for name, _arguments in _every_tool_invocation()}
    registered = {tool["name"] for tool in mcp_tools.TOOL_DEFINITIONS}
    assert registered <= swept, f"tools registered but never swept for quotes: {registered - swept}"


def test_list_projects_reports_every_node_with_its_shape(repo):
    payload = _payload(_call("list_projects", {}, repo_root=repo))
    assert payload["projects"] == [
        {"path": "acme", "label": "acme", "depth": 1, "has_context": True, "has_sessions": True,
         "corpus": "undeclared"},
        {"path": "acme/onboarding", "label": "onboarding", "depth": 2,
         "has_context": False, "has_sessions": False, "corpus": "undeclared"},
    ]


def test_get_survey_returns_the_items_in_order(repo):
    payload = _payload(_call("get_survey", {"project": "acme"}, repo_root=repo))
    assert payload["project"] == "acme"
    assert payload["error"] is None
    assert payload["survey"] == [{"id": "q1", "order": 1, "text": "Tell me about onboarding"}]


def test_get_hypotheses_carries_the_state_the_criterion_and_scorability(repo):
    payload = _payload(_call("get_hypotheses", {"project": "acme"}, repo_root=repo))
    by_id = {item["id"]: item for item in payload["hypotheses"]}
    assert by_id["h1"]["state"] == "contested"
    assert by_id["h1"]["criterion"] == "we are right if 3 of 10 raise it unprompted"
    assert by_id["h1"]["is_scorable"] is True
    assert by_id["h2"]["state"] == "untested"
    assert by_id["h2"]["criterion"] == ""
    assert by_id["h2"]["is_scorable"] is False, (
        "a hypothesis with no pre-registered rule must not look scorable"
    )


def test_get_hypotheses_can_report_all_four_states(repo):
    """The fourth state is the one a naive three-state consumer drops."""
    from app.coach import corpus as coach_corpus

    assert coach_corpus.HYPOTHESIS_STATE_CONTESTED in coach_corpus.ALL_HYPOTHESIS_STATES
    _write_json(
        repo / "projects" / "acme" / "context" / "hypotheses.json",
        [
            {"id": f"h-{state}", "statement": "s", "falsifier": "f", "state": state}
            for state in sorted(coach_corpus.ALL_HYPOTHESIS_STATES)
        ],
    )
    payload = _payload(_call("get_hypotheses", {"project": "acme"}, repo_root=repo))
    assert {item["state"] for item in payload["hypotheses"]} == set(
        coach_corpus.ALL_HYPOTHESIS_STATES
    )


def test_list_interviews_reports_closure_and_the_coach_flag(repo):
    payload = _payload(_call("list_interviews", {"project": "acme"}, repo_root=repo))
    # Still EXACT equality, which is the property worth keeping: an extra key
    # is exactly how a quote arrives under a different name. `project` was
    # added in the commit that added `scope`, and it is present in BOTH
    # scopes -- a response whose shape depends on the mode invites a consumer
    # to read the attribution sometimes and infer it the rest of the time.
    assert payload["interviews"] == [
        {
            "session_id": SESSION_ID,
            "project": "acme",
            "started_at": "2026-08-12T10:15:00+00:00",
            "closed": True,
            "truncated_tail": True,
            "coach_enabled": True,
        }
    ]


def test_list_interviews_reports_truncated_tail_as_a_boolean_never_the_tail_text(repo):
    """The tail is a half-written transcript line -- i.e. raw customer speech.
    Reporting the string itself would be a quote leak wearing a diagnostic's
    name."""
    payload = _payload(_call("list_interviews", {"project": "acme"}, repo_root=repo))
    assert payload["interviews"][0]["truncated_tail"] is True
    assert SENTINEL_TAIL not in json.dumps(payload)


def test_an_open_interview_reports_closed_false(repo):
    _write_json(
        repo / "projects" / "acme" / "sessions" / SESSION_ID / "session.json",
        {"started_at": "2026-08-12T10:15:00+00:00"},
    )
    payload = _payload(_call("list_interviews", {"project": "acme"}, repo_root=repo))
    assert payload["interviews"][0]["closed"] is False
    assert payload["interviews"][0]["coach_enabled"] is False


def test_get_facts_returns_claims_and_bearings_and_no_quote(repo):
    payload = _payload(_call("get_facts", {"project": "acme"}, repo_root=repo))
    # EXACT equality, and that is the load-bearing part: this assertion is
    # what fails if a SEVENTH key ever appears on a projected fact, which is
    # how `quoted_text` would arrive under a name nobody grepped for. The
    # `project` key was added deliberately, in the commit that added `scope`,
    # and it names the node the fact was filed at.
    assert payload["facts"] == [
        {
            "id": "f1",
            "project": "acme",
            "claim": "the invite step is where they stall",
            "kind": "evidence",
            "bearings": [{"hypothesis_id": "h1", "stance": "supports"}],
            "session_id": SESSION_ID,
        },
        {
            "id": "f2",
            "project": "acme",
            "claim": "price never came up",
            "kind": "inference",
            "bearings": [{"hypothesis_id": "h2", "stance": "undermines"}],
            "session_id": SESSION_ID,
        },
    ]
    assert payload["count"] == 2


def test_get_facts_filters_by_hypothesis(repo):
    payload = _payload(
        _call("get_facts", {"project": "acme", "hypothesis_id": "h1"}, repo_root=repo)
    )
    assert [fact["id"] for fact in payload["facts"]] == ["f1"]
    assert payload["count"] == 1


def test_get_facts_for_an_unknown_hypothesis_is_empty_not_an_error(repo):
    payload = _payload(
        _call("get_facts", {"project": "acme", "hypothesis_id": "nope"}, repo_root=repo)
    )
    assert payload["facts"] == []
    assert payload["error"] is None


# --- privacy: the load-bearing guard --------------------------------------------------


def test_no_serialized_response_of_any_tool_contains_the_verbatim_quote(repo):
    """THE failure this must prevent: a past customer's verbatim speech
    leaving the machine through the widest egress surface in the project.

    Asserted over `json.dumps` of the WHOLE response -- envelope, result,
    content, error text and all -- never a chosen sub-part, because the
    interesting leak is the quote arriving in a differently-named field
    (`evidence`, `excerpt`, `citation`, a debug echo of the loaded record).
    Gate 21's own shape, pointed at the trust boundary."""
    for name, arguments in _every_tool_invocation():
        response = _call(name, arguments, repo_root=repo)
        blob = json.dumps(response)
        assert SENTINEL_QUOTE not in blob, f"{name} leaked the verbatim quote"
        assert SENTINEL_TAIL not in blob, f"{name} leaked the truncated transcript tail"


def test_the_quote_sentinel_is_really_present_in_the_fixture(repo):
    """The mutation the privacy test above cannot survive without: if the
    fixture ever stopped carrying a quote, that test would pass vacuously
    forever.

    The sentinels are asserted to be DISTINCTIVE, not merely present -- an
    empty or one-character sentinel is `in` almost every string ever
    serialized, so `assert SENTINEL in raw` alone still passes while the
    privacy assertion it guards has become meaningless."""
    assert len(SENTINEL_QUOTE) > 16 and len(SENTINEL_TAIL) > 16, (
        "a sentinel short enough to occur by accident proves nothing either way"
    )
    raw = (repo / "projects" / "acme" / "context" / "facts.json").read_text(encoding="utf-8")
    assert SENTINEL_QUOTE in raw
    tail = (repo / "projects" / "acme" / "sessions" / SESSION_ID / "transcript.jsonl").read_text(
        encoding="utf-8"
    )
    assert SENTINEL_TAIL in tail


def test_an_error_message_never_echoes_file_content(repo):
    """The sneakiest leak in the whole surface: a malformed `facts.json` whose
    parse error message quotes the offending line, which contains the quote.
    The degraded `error` string must name the failure, never the content."""
    facts_path = repo / "projects" / "acme" / "context" / "facts.json"
    facts_path.write_text(
        '[{"id": "f1", "quoted_text": "' + SENTINEL_QUOTE + '", BROKEN', encoding="utf-8"
    )
    response = _call("get_facts", {"project": "acme"}, repo_root=repo)
    blob = json.dumps(response)
    assert SENTINEL_QUOTE not in blob, "the parse-error message echoed the file's content"
    assert _payload(response)["error"] is not None


def test_a_validation_error_message_never_echoes_the_offending_record(repo):
    """The sneakier version of the test above, and the reason the degraded
    `error` string carries only an exception TYPE and never its message:
    `pydantic.ValidationError` renders `input_value=` -- the offending value
    itself -- into its own message text, and `Fact` is `extra="forbid"`, so a
    corpus written by a drifted schema produces an `extra_forbidden` error
    whose `input_value` IS the customer's words. A file that is perfectly
    valid JSON and merely fails validation therefore hands the quote straight
    to any handler that does `str(exc)`.

    Note which mutation this survives and the obvious version does not: a bad
    `provenance` renders `input_value='not-a-real-provenance'` and echoes
    nothing interesting. The extra-key shape is the one that carries the
    quote."""
    _write_json(
        repo / "projects" / "acme" / "context" / "facts.json",
        [
            {
                "id": "f1", "session_id": SESSION_ID, "seqs": [4],
                "claim": "c", "provenance": "evidence",
                "quoted_text": "an ordinary quote", "bearings": [],
                # The drifted key. Its VALUE is what pydantic prints back.
                "verbatim": SENTINEL_QUOTE,
            }
        ],
    )
    response = _call("get_facts", {"project": "acme"}, repo_root=repo)
    assert SENTINEL_QUOTE not in json.dumps(response), (
        "the validation-error message echoed the offending record's quote"
    )
    assert _payload(response)["error"] is not None


# --- degradation ----------------------------------------------------------------------


def test_an_absent_facts_file_degrades_rather_than_killing_the_server(repo):
    (repo / "projects" / "acme" / "context" / "facts.json").unlink()
    out = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "get_facts", "arguments": {"project": "acme"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        repo_root=repo,
    )
    responses = _decoded(out)
    payload = _payload(responses[0])
    assert payload["facts"] == []
    assert payload["error"] is None, "an un-extracted corpus is empty, not broken"
    assert responses[1]["id"] == 2, "the server died on a missing artifact"


def test_a_malformed_facts_file_degrades_rather_than_killing_the_server(repo):
    (repo / "projects" / "acme" / "context" / "facts.json").write_text("{{{", encoding="utf-8")
    out = _drive(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "get_facts", "arguments": {"project": "acme"}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ],
        repo_root=repo,
    )
    responses = _decoded(out)
    payload = _payload(responses[0])
    assert payload["facts"] == []
    assert payload["error"], "a malformed corpus must be NAMED, not silently empty"
    assert responses[1]["id"] == 2


def test_a_missing_survey_is_named_rather_than_silently_empty(repo):
    (repo / "projects" / "acme" / "context" / "survey.json").unlink()
    payload = _payload(_call("get_survey", {"project": "acme"}, repo_root=repo))
    assert payload["survey"] == []
    assert payload["error"]


def test_an_unreadable_session_is_skipped_rather_than_killing_the_listing(repo):
    """A corrupt INTERIOR transcript line makes `read_session` raise. One bad
    interview must not hide every other interview in the project."""
    good = repo / "projects" / "acme" / "sessions" / "20260813-090000-zzzzzz"
    _write_json(good / "session.json", {"started_at": "2026-08-13T09:00:00+00:00"})
    bad = repo / "projects" / "acme" / "sessions" / SESSION_ID
    (bad / "transcript.jsonl").write_text("not json\n{}\n", encoding="utf-8")
    payload = _payload(_call("list_interviews", {"project": "acme"}, repo_root=repo))
    ids = [interview["session_id"] for interview in payload["interviews"]]
    assert "20260813-090000-zzzzzz" in ids
    assert payload["error"], "the skipped interview must be reported, not silently dropped"


def test_a_project_with_no_sessions_directory_lists_nothing(repo):
    payload = _payload(_call("list_interviews", {"project": "acme/onboarding"}, repo_root=repo))
    assert payload["interviews"] == []
    assert payload["error"] is None


def test_a_traversing_project_path_is_refused_without_crashing(repo):
    for hostile in ("../../etc", "C:/Windows", "acme/../../secrets", ""):
        response = _call("get_facts", {"project": hostile}, repo_root=repo)
        assert response["error"]["code"] == -32602, f"{hostile!r} was not refused"


def test_a_non_string_project_argument_is_refused(repo):
    response = _call("get_facts", {"project": 17}, repo_root=repo)
    assert response["error"]["code"] == -32602
