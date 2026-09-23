---
description: Link this repository to one Discovery Copilot project, so evidence questions are answered from the right customers
argument-hint: <project path, e.g. acme or acme/onboarding>
allowed-tools: Bash(printenv DISCOVERY_COPILOT_HOME)
---

Link this repository to the Discovery Copilot project `$ARGUMENTS`. Follow the
`discovery-evidence` skill's rules throughout.

1. **Check the server is reachable.** Call `list_projects`. If the
   `discovery-copilot` server is missing or fails, stop and tell the operator to
   set `DISCOVERY_COPILOT_HOME` to their Discovery Copilot checkout (for example
   `setx DISCOVERY_COPILOT_HOME "C:\path\to\discovery-copilot"` on Windows) and
   restart Claude Code.

2. **Check the project exists.** If `$ARGUMENTS` is empty or is not a path in
   the listing, show the available paths and ask which one; do not pick one
   by resemblance. If the product has no project yet, say that projects are
   created on the Discovery Copilot side, never from here, and give the steps:
   create `projects/<name>/project.json` in the Discovery Copilot checkout
   containing `{"corpus": "real"}`, then run
   `python -m app author --project <name>` there to write the brief, survey
   and hypotheses.

3. **Check its `corpus`.** If it is `demo`, warn that nothing in it is
   customer evidence. If it is `undeclared`, say the operator should declare
   it in that node's `project.json` before its claims are relied on. Link it
   either way if the operator confirms.

4. **Write `.discovery-copilot.json`** at this repository's root, replacing any
   existing one:

   ```json
   { "project": "<path>", "scope": "subtree" }
   ```

   This file is meant to be committed: it says which customers this product's
   decisions answer to.

5. **Keep the raw interview files out of reach.** Get the checkout path with
   the Bash tool, exactly `printenv DISCOVERY_COPILOT_HOME` (pre-approved); if it
   prints nothing, skip this step and say so. Merge deny rules into
   `.claude/settings.local.json` (create it if needed; keep any existing
   content) so this repository's
   sessions cannot read the Discovery Copilot's `projects/`, `sessions/` and
   `context/` directories, where transcripts and verbatim quotes live. Write
   each as `Read(//<path>/<dir>/**)`, where `<path>` is the checkout path with
   forward slashes and the drive letter as a lowercase first segment
   (`C:\x\y` becomes `//c/x/y`). This file stays local and uncommitted
   because the path is specific to this machine.

6. **Add a pointer to `CLAUDE.md`** at the repository root (create it if
   missing). If a block between `<!-- discovery-copilot:start -->` and
   `<!-- discovery-copilot:end -->` exists, replace it; otherwise append:

   ```markdown
   <!-- discovery-copilot:start -->
   ## Customer evidence
   This product's interview evidence is the Discovery Copilot project `<path>`
   (see `.discovery-copilot.json`). Before prioritising, scoping or designing a
   user-facing feature, or stating what users need, use the
   `discovery-evidence` skill: cite claim and interview IDs, never treat
   `demo` or `undeclared` claims as findings, and put proposed hypotheses and
   survey questions in `discovery-proposals.md` rather than assuming.
   <!-- discovery-copilot:end -->
   ```

7. **Report**: the linked project, its `corpus`, how many hypotheses and
   interviews it has (`get_standing` with `scope: subtree`), whether anything
   is in `unextracted_projects`, and the files you wrote.
