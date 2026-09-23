"""`app/mcp/` -- this project's interview assets, reachable by the operator's
OTHER AI agents over the Model Context Protocol.

**Read-only, and that is a decision rather than a first cut.** Gate 19 pins
the set of modules that write `context/hypotheses.json` to exactly
`{app/web/author.py}`, BY EQUALITY. An MCP write tool -- "let the agent mark
the hypothesis supported" -- is precisely the shortest-path drift that gate
exists to stop: a hypothesis's state is a human's judgement about their own
falsifier, and a second writer would let a model reach it without one. So
this package writes nothing at all, and the absence is asserted
behaviourally in `app/tests/test_mcp_server.py`.

**No verbatim customer speech crosses this boundary.** Every other consumer
of the corpus in this repository stays on one machine: the CLI prints to the
operator's own terminal, the review hub is loopback-bound. This package is
the first surface that hands the corpus to a process the operator did not
write, and it is therefore the widest egress surface in the project. It
exposes CLAIMS (the abstracted sentence a fact asserts), never QUOTES
(`Fact.quoted_text`), never a raw `transcript.jsonl` segment, never audio.
Gate 20 pins the set of modules that read `Fact.quoted_text` by equality and
this package must never join it; a new gate at the end of
`app/tests/test_phase_gates.py` asserts the same property scoped to
`app/mcp/` alone, so a leak here fails on its own name rather than by
widening a set shared with four legitimate offline readers.

**Reads may merge; writes never merge.** Five of the seven tools take a
`scope`: `"node"` (the default) reads the named project alone, `"subtree"`
also reads its descendants. `subtree` is opt-in because it is a
cross-project read, which the standing project decision permits only WITHIN
one hierarchy and only on EXPLICIT request. A subtree never climbs to a
parent, never reaches a sibling -- `acme` does not include `acme-corp` --
and every item in a merged answer names the node it came from, because a
rolled-up number that cannot be traced back to a node is worse than no
rollup at all.

**Counts in a merged answer are per-INTERVIEW, never per-fact**, and this
package does not implement that rule: `app/coach/interview_summary.py` does,
and `get_standing`/`get_interview_summary` call it. A rollup is exactly
where per-fact counting reappears, because the union of two nodes' facts
genuinely is a longer list.

**Zero new dependencies.** MCP is JSON-RPC 2.0 over stdio -- newline-
delimited JSON objects on stdin/stdout -- so it is hand-rolled in
`server.py` against the standard library. Nothing here imports an SDK.
`.mcp.json` at the repository root registers it for Claude Code; the README
shows the Claude Desktop equivalent. Neither carries a credential, and
`app/tests/test_mcp_harness.py` fails if either ever does.

Run it as `python -m app.mcp`. Deliberately its own entry point rather than
a subcommand of `app/main.py`: this is a long-lived process that owns stdout
absolutely (a stray `print` anywhere in its import graph corrupts the
protocol stream), and that is a different lifecycle from every command
`main.py` hosts.
"""
