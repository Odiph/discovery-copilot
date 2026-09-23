---
name: discovery-evidence
description: Ground product decisions in customer-interview evidence from the Discovery Copilot. Use when prioritising, scoping or designing a user-facing feature, writing a PRD, roadmap item or positioning, or when anyone asks what users need, want or struggle with. Also use when a claim about users is about to be written down without a source.
---

# Discovery evidence

This repository is a product. The customer interviews behind it live in a
separate tool, the Discovery Copilot, which records interviews, extracts claims
from them and tracks where each hypothesis stands. You reach it through the
`discovery-copilot` MCP server this plugin registers. It is **read-only** and
returns abstracted claims, never what anyone actually said.

## 1. Find the project this repository belongs to

Read `.discovery-copilot.json` at the repository root:

```json
{ "project": "acme", "scope": "subtree" }
```

- `project` is a node path from `list_projects`. `scope` is passed to every
  call that takes one (`subtree` includes the node's children).
- **No file: do not guess a project.** Say the repository is not linked yet
  and suggest `/link-discovery`. Picking a node by name similarity can ground a
  decision in another product's customers.

If the server is missing or fails to start, the operator has not set
`DISCOVERY_COPILOT_HOME` to the Discovery Copilot checkout, or has not restarted
Claude Code since. Say that; do not work around it.

## 2. Read the evidence, in this order

1. `list_projects`: confirm the node exists and read its `corpus`.
2. `get_standing` for the node: where every hypothesis stands.
3. `get_facts`, filtered to the hypothesis that matters, for the claims behind
   a count.
4. `get_hypotheses`, `get_survey` and `list_interviews` when you need the
   statements, the questions asked or how many interviews there are.
   `get_interview_summary` for one interview.

## 3. What counts as evidence

- **`corpus` decides whether anything is a finding.** It is on each node in
  `list_projects` and on each entry of a response's `extraction` list; a fact
  takes the `corpus` of the node named in its `project`. `real` is customer
  evidence. `demo` is hand-authored demonstration material that reads exactly
  like research: never report it as a finding. `undeclared` means nobody has
  said, so treat it as not established. With `scope: subtree`, check every
  node: one response can mix them.
- **Check `unextracted_projects` before reading a zero.** A project whose
  interviews were never extracted returns zero for everything; that means
  "not looked at yet", not "users don't care".
- **Do not tally against a hypothesis with `is_scorable: false`.** It has no
  pre-registered rule for what would count.
- **Counts are per interview.** One person repeating a point is one
  interview. Report `supporting_interviews` and `contrary_interviews` out of
  `total_interviews`, not the fact counts. Three interviews is a signal, not a
  result. A hypothesis's `criterion` is the rule it is judged by: quote that
  rather than inventing your own threshold.
- **`kind` is `evidence` or `inference`**: how directly the claim follows from
  what was said. Say which when it matters to the decision.

## 4. How to report it

Next to each conclusion, cite the claims behind it by their `id` (the first
eight characters are enough) and the interview each came from (`session_id`),
and name the hypothesis (`hypothesis_id`) it bears on. Separate what the
interviews show from your own reasoning. When the evidence is thin, mixed or
missing, say so plainly and still give your recommendation, marked as
unevidenced.

## 5. When the evidence is missing

Do not fill the gap with assumptions about users. Propose what to learn:

- **A hypothesis**: a falsifiable statement, a falsifier (what an interviewee
  would have to report for it to be false) and a decision criterion written
  before any interview (e.g. "at least 3 of 6 interviewees describe…").
- **Survey questions** about past behaviour: "walk me through the last time…",
  "what happened when…". No hypotheticals ("would you use…"), no pitching, no
  questions that invite a compliment.

Append proposals to `discovery-proposals.md` at this repository's root, dated.
The operator enters the ones they accept through the Discovery Copilot's own
`author` page: `python -m app author --project <project>`, run from the
Discovery Copilot checkout. **You propose; only `author` writes** a project's
hypotheses and survey.

## Boundaries

- **Never read the Discovery Copilot's files directly**: its `projects/`,
  `sessions/` or `context/` directories. They hold transcripts and verbatim
  customer speech that the MCP server deliberately withholds. Asking for a
  quote is a signal to use the claim instead.
- **Never start a recording** (`python -m app start`, the console or the
  overlay). An interview is a live event the operator runs.
- Nothing here writes to the Discovery Copilot. If a task seems to need that,
  hand the operator the command.
