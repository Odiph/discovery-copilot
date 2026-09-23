"""The seven read-only tools, their JSON schemas, and the mapping from this
project's loaders onto a wire payload.

**This module counts nothing.** `get_standing` and `get_interview_summary`
report numbers, and every one of them comes from
`app/coach/interview_summary.py`. That is not tidiness: the rule is
per-INTERVIEW and never per-fact, and a second implementation here would fork
it invisibly -- two surfaces each rendering a confident-looking integer, only
one of them right. A SUBTREE rollup is where the fork would be most tempting
and most wrong, because the merged list of facts genuinely is longer.

**Scope resolution happens in exactly one place.** `resolve_scope` is the only
function that turns a request into a set of nodes, and every node it returns
came out of `projects_mod.resolve`. A second resolver would be a second home
for the `..`/absolute/drive-letter rules to drift apart in, and the escape
that matters is a SIBLING SHARING A NAME PREFIX -- `acme-corp` is a different
customer from `acme` -- which only a separator-terminated prefix rejects.

**Every loader takes its directory as a PARAMETER, and so does every function
here.** No module under `app/coach/`, `app/session/` or `app/web/` reads
`app/config.py` internally -- `app/projects.py`'s own docstring states the
rule and the two gates that depend on it. This module keeps it: `repo_root`
arrives from the caller (`app/mcp/server.py`, injected in turn from
`app/mcp/__main__.py`), which is what makes the whole surface testable
against a temporary tree.

**The mapping is written out field by field, never `model_dump()`.** This is
the single most important line in this module. `Fact` is a pydantic model
whose SIXTH field is `quoted_text`; `fact.model_dump()` would put a verbatim
customer quote on the wire while mentioning the field name nowhere -- so it
would satisfy gate 20's AST scan, this package's own gate, and any reviewer
skimming for the string. An explicit projection is the only form of this
code whose safety is visible.

Since `get_standing` arrived, facts reach this module as
`interview_summary.FactRef` rather than as `Fact`, and that is a second
independent reason the quote cannot be here: `FactRef` HAS no `quoted_text`
field and no `seqs` field, so there is nothing for a later `model_dump()` or
a careless `**` to pick up. Both defences are kept. One is the right number
for a surface that hands data to a process the operator did not write.

**Error strings carry an exception TYPE, never its message.** `pydantic`'s
`ValidationError` renders `input_value=` -- the whole offending record,
quote included -- into its own message text, so a handler that did
`f"{exc}"` would hand the quote to the caller as a diagnostic. The caller
learns WHICH artifact failed and HOW (the class of failure); the operator
reads the detail on their own machine, where the file is.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from app import project_meta as project_meta_mod
from app import projects as projects_mod
from app.coach import context as coach_context
from app.coach import corpus as coach_corpus
from app.coach import interview_summary as summary_mod
from app.session import reader as session_reader

# --- the seam this cut deliberately left open, now closed -----------------------------
#
# `get_standing` and `get_interview_summary` were withheld from the first cut
# because both need the per-INTERVIEW counting rule, and re-deriving it here
# would have forked it invisibly -- two surfaces each rendering a
# confident-looking integer, only one of them right.
#
# `app/coach/interview_summary.py` is that shared home, and both tools below
# route through it: this module does no counting of its own, it only projects
# what that module returns onto the wire. The constant is EMPTIED rather than
# deleted, so a reader arriving from the commit that introduced it finds out
# what happened rather than finding nothing.
TOOLS_AWAITING_THE_SHARED_SUMMARY_MODULE: "tuple[str, ...]" = ()


# --- scope ----------------------------------------------------------------------------
#
# The standing decision this implements, stated once here so no caller
# re-litigates it: CROSS-PROJECT READS ARE PERMITTED ONLY WITHIN THE SAME
# HIERARCHY, AND ONLY ON EXPLICIT REQUEST; READS MAY MERGE, WRITES NEVER MERGE.
#
# So `node` is the default and `subtree` is opt-in. An unrecognized value is
# REFUSED rather than defaulted, in either direction: defaulting a typo to
# `node` would silently answer from a third of the corpus the caller asked
# about, and defaulting it to `subtree` would perform the cross-project read
# the rule permits only on explicit request.
SCOPE_NODE = "node"
SCOPE_SUBTREE = "subtree"
ALL_SCOPES = (SCOPE_NODE, SCOPE_SUBTREE)


class ToolArgumentError(ValueError):
    """A caller's argument is wrong -- an unknown tool name, a missing or
    mistyped argument, a project path that would escape `projects/`. Distinct
    from a DATA condition (a missing or malformed artifact), which is not the
    caller's fault and is reported inside the tool's own payload as a named
    `error` rather than as a protocol-level failure."""


def _describe(exc: BaseException) -> str:
    """The class of failure, never the message. See the module docstring."""
    return type(exc).__name__


def _require_project(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]"):
    raw = arguments.get("project")
    if not isinstance(raw, str):
        raise ToolArgumentError(
            f"'project' is required and must be a string, got {type(raw).__name__}"
        )
    try:
        return projects_mod.resolve(raw, repo_root=repo_root)
    except projects_mod.ProjectPathError as exc:
        # The one exception message that IS forwarded: it is about the
        # caller's own argument, which the caller already has, and it never
        # touches a file.
        raise ToolArgumentError(str(exc)) from exc


def _require_scope(arguments: "Mapping[str, Any]") -> str:
    raw = arguments.get("scope", SCOPE_NODE)
    # `is True` / `is False` first: `bool` is a subclass of `int`, not of
    # `str`, so the isinstance check below already rejects it -- but a caller
    # sending `true` deserves the scope message rather than a type name.
    if not isinstance(raw, str) or raw not in ALL_SCOPES:
        raise ToolArgumentError(
            f"'scope' must be one of {list(ALL_SCOPES)!r}, got {raw!r}. "
            "'node' reads the named project only; 'subtree' also reads its descendants, "
            "and is opt-in because a cross-project read happens only on explicit request"
        )
    return raw


def resolve_scope(
    project: str, scope: str, *, repo_root: "Optional[Path]"
) -> "tuple[projects_mod.ProjectPaths, ...]":
    """The nodes one call may read, parents before children.

    THE ONE PATH RESOLVER. Every member is produced by `projects_mod.resolve`,
    never by string-joining a discovered directory name onto the requested
    path -- a second resolver would be a second place for the `..`/absolute/
    drive-letter rules to be subtly different, and the interesting escape is
    always the one that only the second copy allows.

    **The escape this function is written against is not `..`.** That one is
    unrepresentable already: `normalize()` rejects it before a path is ever
    built. The reachable escape is a SIBLING THAT SHARES A NAME PREFIX --
    `acme-corp` is a different customer from `acme`, and
    `"acme-corp".startswith("acme")` is `True`. The membership test below is
    therefore against the SEPARATOR-TERMINATED prefix `"acme/"`, which is the
    only form that distinguishes a descendant from a namesake.

    `share_a_root` is then applied to every member as a backstop. It is where
    the same-hierarchy rule lives, so a future change to the descendant test
    that reintroduced an escape has to get past the rule's own home too."""
    root = projects_mod.resolve(project, repo_root=repo_root)
    if scope == SCOPE_NODE:
        return (root,)

    # The requested node ALWAYS leads its own subtree, and it is included
    # whether or not it exists on disk: `resolve()` deliberately does not
    # require existence, and a subtree that could return FEWER rows than the
    # node scope would be a merge that loses data.
    # `normcase` on BOTH sides, not `.lower()`: it is identity on POSIX and
    # case-folding on Windows, so the comparison matches whatever the
    # filesystem underneath actually does rather than guessing.
    #
    # The defect this closes is the mirror image of the namesake escape above,
    # and it is quieter. Windows folds case, so `projects/ACME` IS
    # `projects/acme`: a subtree query spelled `ACME` opened the ROOT node's
    # files perfectly well while every DESCENDANT failed a case-sensitive
    # `"acme/onboarding".startswith("ACME/")`. The call returned a PARTIAL
    # subtree with `error` set to None. An under-reported rollup answers "no
    # contrary evidence" when contrary evidence exists, which is the
    # confirmation failure `Bearing` was created to make impossible.
    #
    # Folding case must not loosen the SEPARATOR-TERMINATED prefix -- trading
    # a silent omission for a silent leak between two customers would be
    # strictly worse. `test_mcp_scope_case.py` pins both directions.
    prefix = os.path.normcase(f"{root.path}/")
    nodes = [root]
    for node in projects_mod.list_projects(repo_root=repo_root):
        if not os.path.normcase(node.path).startswith(prefix):
            continue
        resolved = projects_mod.resolve(node.path, repo_root=repo_root)
        if not projects_mod.share_a_root(root.path, resolved.path):
            # Unreachable through the prefix test above, and kept anyway: the
            # rule has one home and this is a read of it, not a copy.
            continue
        nodes.append(resolved)
    return tuple(nodes)


def _scoped_nodes(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]"):
    """The (root, scope, nodes) triple every scoped tool opens with."""
    root = _require_project(arguments, repo_root=repo_root)
    scope = _require_scope(arguments)
    return root, scope, resolve_scope(root.path, scope, repo_root=repo_root)


def _envelope(root, scope: str, nodes, errors: "list[dict[str, str]]") -> "dict[str, Any]":
    """The four keys every scoped response carries, in both scopes.

    Present in NODE scope too, deliberately. A response whose shape depends on
    the mode invites a consumer to read `nodes` sometimes and infer it the
    rest of the time, and the inference is wrong exactly when it matters.

    `error` stays a single string for compatibility with the unscoped first
    cut, and every entry inside it NAMES ITS NODE -- "something failed
    somewhere in the subtree" is not an answer an agent can act on."""
    return {
        "project": root.path,
        "scope": scope,
        "nodes": [node.path for node in nodes],
        "errors": errors,
        "error": (
            "; ".join(f"{entry['project']}: {entry['error']}" for entry in errors)
            if errors
            else None
        ),
    }


EXTRACTION_EXTRACTED = "extracted"
EXTRACTION_NOT_EXTRACTED = "not-extracted"
EXTRACTION_UNREADABLE = "unreadable"
ALL_EXTRACTION_STATUSES = (
    EXTRACTION_EXTRACTED,
    EXTRACTION_NOT_EXTRACTED,
    EXTRACTION_UNREADABLE,
)


def _load_facts_across(
    nodes, errors: "list[dict[str, str]]"
) -> "tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]":
    """Every node's facts, projected onto `FactRef` and tagged with its node,
    AND a per-node statement of whether that node has been extracted at all.

    The second half of that return is the whole point, and the signature was
    changed rather than added to so that every call site had to be touched.
    Before it, a node with no `facts.json` was `continue`d in silence, so a
    project nobody has ever run `extract` on and a project whose facts bear on
    nothing came back byte-identical: zero counts, empty `errors`, no way to
    tell which. That is the defect `18895a7` fixed on the hub, and this is the
    same defect on the surface whose reader is a MODEL -- which, unlike the
    operator, cannot go and look at the terminal to find out which of the two
    it is holding.

    A missing corpus is reported here rather than in `errors` on purpose, for
    the reason the hub's banner is amber and not red: an un-extracted project
    is an unfinished step, not a fault. `errors` stays for things that went
    WRONG, so that a caller can keep treating a non-empty `errors` as a
    problem to report to a human.

    Node semantics, stated because they differ from the hub's: `extracted`
    means this node has a readable corpus file, and `facts` says how many
    claims are in it -- so `extracted` with `facts: 0` is a real and distinct
    answer (extraction ran and filed nothing here). The hub decides the same
    word per INTERVIEW; this decides it per NODE, because that is the unit a
    scoped tool answers about.

    One malformed node degrades THAT NODE ONLY. A subtree read that died
    because one child had a broken `facts.json` would be strictly worse than
    the five separate node reads it replaces -- including when the broken node
    is the ROOT, which is the case a naive implementation treats as fatal
    because it is the node the caller named."""
    refs: "list[Any]" = []
    extraction: "list[dict[str, Any]]" = []
    for node in nodes:
        if not (node.context_dir / coach_corpus.FACTS_FILENAME).is_file():
            extraction.append(
                {"project": node.path, "status": EXTRACTION_NOT_EXTRACTED, "facts": 0,
                 "corpus": project_meta_mod.corpus_kind_of(node.node_dir)}
            )
            continue
        try:
            # `sessions_root=None`: a READ of an already-validated corpus, not
            # the extraction path. Passing the root would make every listing
            # re-parse every cited session's transcript, and would turn one
            # unreadable interview into a total failure of a tool that never
            # needed the transcript at all.
            facts = coach_corpus.load_facts(node.context_dir, sessions_root=None)
        except Exception as exc:  # noqa: BLE001
            errors.append({"project": node.path, "error": _describe(exc)})
            # An unreadable corpus IS a fault, so it lands in `errors` too.
            # It is also named here so that a caller reading one list gets a
            # complete answer about every node it asked about -- a status
            # block with a hole in it is the shape that started all this.
            extraction.append(
                {"project": node.path, "status": EXTRACTION_UNREADABLE, "facts": 0,
                 "corpus": project_meta_mod.corpus_kind_of(node.node_dir)}
            )
            continue
        refs.extend(summary_mod.fact_ref(fact, project=node.path) for fact in facts)
        extraction.append(
            {"project": node.path, "status": EXTRACTION_EXTRACTED, "facts": len(facts),
             "corpus": project_meta_mod.corpus_kind_of(node.node_dir)}
        )
    return tuple(refs), tuple(extraction)


def _extraction_block(extraction: "tuple[dict[str, Any], ...]") -> "dict[str, Any]":
    """The two keys every fact-reading tool publishes about what it KNOWS.

    `unextracted_projects` is derived from `extraction` and adds no
    information -- it exists because the reader of this wire is a model
    scanning a payload for the answer to the question it was asked, and a
    per-node list of statuses is exactly the shape that gets skimmed past when
    every count below it is a confident-looking zero. One list is the source;
    the other is the thing that is hard to miss."""
    return {
        # Field by field, like every other projection here. `dict(entry)`
        # would read the same today and would be a bulk copy of whatever the
        # producer grows next -- which is the exact move the module docstring
        # forbids for models, and there is no reason to keep one of them
        # around for a reader to mistake as sanctioned.
        "extraction": [
            {
                "project": entry["project"],
                "status": entry["status"],
                "facts": entry["facts"],
                # Whether these claims describe real people. `status` says
                # whether extraction RAN, which a reader can otherwise mistake
                # for a warrant that what it produced is customer research.
                "corpus": entry["corpus"],
            }
            for entry in extraction
        ],
        "unextracted_projects": [
            entry["project"]
            for entry in extraction
            if entry["status"] == EXTRACTION_NOT_EXTRACTED
        ],
    }


def _load_hypotheses_across(nodes, errors: "list[dict[str, str]]") -> "tuple[tuple[Any, str], ...]":
    """`(hypothesis, node_path)` pairs. The node is carried alongside rather
    than derived later: under a subtree rollup a hypothesis authored at the
    parent is scored against a child's facts, so which node AUTHORED it stops
    being inferable from anything else in the answer."""
    found: "list[tuple[Any, str]]" = []
    for node in nodes:
        try:
            # `missing_ok=True`: a node whose hypotheses have not been
            # authored yet is empty, not broken.
            hypotheses = coach_corpus.load_hypotheses(node.context_dir, missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append({"project": node.path, "error": _describe(exc)})
            continue
        found.extend((hypothesis, node.path) for hypothesis in hypotheses)
    return tuple(found)


# --- list_projects --------------------------------------------------------------------


def list_projects(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    nodes = projects_mod.list_projects(repo_root=repo_root)
    return {
        "projects": [
            {
                "path": node.path,
                "label": node.label,
                "depth": node.depth,
                "has_context": node.has_context,
                "has_sessions": node.has_sessions,
                "corpus": project_meta_mod.corpus_kind(node.path, repo_root=repo_root),
            }
            for node in nodes
        ]
    }


# --- get_survey -----------------------------------------------------------------------


def get_survey(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    paths = _require_project(arguments, repo_root=repo_root)
    try:
        survey = coach_context.load_survey(paths.context_dir)
    except Exception as exc:  # noqa: BLE001 -- a data condition, never a crash
        return {"project": paths.path, "survey": [], "error": _describe(exc)}
    return {
        "project": paths.path,
        "survey": [{"id": item.id, "order": item.order, "text": item.text} for item in survey],
        "error": None,
    }


# --- get_hypotheses -------------------------------------------------------------------


def get_hypotheses(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    root, scope, nodes = _scoped_nodes(arguments, repo_root=repo_root)
    errors: "list[dict[str, str]]" = []
    found = _load_hypotheses_across(nodes, errors)
    return {
        **_envelope(root, scope, nodes, errors),
        "hypotheses": [
            {
                "id": hypothesis.id,
                # WHICH NODE this hypothesis was authored at. A rolled-up
                # answer that cannot be traced back to a node is worse than no
                # rollup: two nodes may author the same id meaning different
                # things, and without this the reader cannot tell them apart.
                "project": node_path,
                "statement": hypothesis.statement,
                "falsifier": hypothesis.falsifier,
                "state": hypothesis.state,
                "criterion": hypothesis.criterion,
                # Carried explicitly rather than left for the caller to infer
                # from an empty `criterion`: a hypothesis with no
                # pre-registered rule cannot be honestly SCORED, and an agent
                # that tallied evidence against it anyway would be painting
                # the target around the arrow.
                "is_scorable": hypothesis.is_scorable,
            }
            for hypothesis, node_path in found
        ],
    }


# --- list_interviews ------------------------------------------------------------------


def list_interviews(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    root, scope, nodes = _scoped_nodes(arguments, repo_root=repo_root)
    errors: "list[dict[str, str]]" = []
    interviews: "list[dict[str, Any]]" = []

    for node in nodes:
        unreadable: "list[str]" = []
        for entry in _session_dirs(node):
            try:
                result = session_reader.read_session(entry)
            except Exception as exc:  # noqa: BLE001 -- one bad interview must not hide the rest
                unreadable.append(_describe(exc))
                continue
            metadata = result.metadata
            interviews.append(
                {
                    "session_id": entry.name,
                    "project": node.path,
                    "started_at": metadata.get("started_at"),
                    # `closed` is `SessionReadResult`'s own reading of whether
                    # `closed_at` is present -- never inferred from a
                    # timestamp gap (phase_specification U1).
                    "closed": result.closed,
                    # A BOOLEAN, never the tail itself. The tail is a
                    # half-written transcript line, i.e. raw customer speech
                    # -- reporting the string would be a quote leak wearing a
                    # diagnostic's name.
                    "truncated_tail": result.truncated_tail is not None,
                    "coach_enabled": bool(metadata.get("coach_enabled", False)),
                }
            )
        if unreadable:
            # Named, per node, rather than silently dropped: an agent
            # reasoning over "how many interviews support this" must know one
            # was skipped, AND where.
            errors.append(
                {
                    "project": node.path,
                    "error": f"{len(unreadable)} interview(s) could not be read "
                             f"({sorted(set(unreadable))})",
                }
            )

    return {**_envelope(root, scope, nodes, errors), "interviews": interviews}


def _session_dirs(node) -> "list[Path]":
    if not node.sessions_root.is_dir():
        return []
    return [
        entry
        for entry in sorted(node.sessions_root.iterdir(), key=lambda item: item.name)
        if entry.is_dir()
    ]


# --- get_facts ------------------------------------------------------------------------


def get_facts(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    root, scope, nodes = _scoped_nodes(arguments, repo_root=repo_root)
    hypothesis_id = _optional_string(arguments, "hypothesis_id")

    errors: "list[dict[str, str]]" = []
    refs, extraction = _load_facts_across(nodes, errors)
    if hypothesis_id is not None:
        refs = tuple(
            ref for ref in refs
            if any(bearing[0] == hypothesis_id for bearing in ref.bearings)
        )

    return {
        **_envelope(root, scope, nodes, errors),
        **_extraction_block(extraction),
        "hypothesis_id": hypothesis_id,
        "facts": [_project_fact(ref) for ref in refs],
        "count": len(refs),
    }


def _optional_string(arguments: "Mapping[str, Any]", name: str) -> "Optional[str]":
    value = arguments.get(name)
    if value is not None and not isinstance(value, str):
        raise ToolArgumentError(
            f"{name!r} must be a string when given, got {type(value).__name__}"
        )
    return value


def _project_fact(ref: Any) -> "dict[str, Any]":
    """The six fields a fact may cross this boundary as -- written out, one by
    one, on purpose.

    The input is a `summary_mod.FactRef`, not a `Fact`, and that is now the
    first line of the defence rather than a convenience: `FactRef` HAS no
    `quoted_text` field and no `seqs` field, so there is nothing here for a
    later `model_dump()` or a careless `**` to pick up. The explicit
    projection is kept anyway -- two independent reasons the quote cannot be
    on the wire is the right number for the widest egress surface in the
    project.

    `kind` is `provenance` ("evidence" | "inference") under the name the
    protocol surface uses. `project` is the node the fact was filed at."""
    return {
        "id": ref.fact_id,
        "project": ref.project,
        "claim": ref.claim,
        "kind": ref.provenance,
        "bearings": [
            {"hypothesis_id": hypothesis_id, "stance": stance}
            for hypothesis_id, stance in ref.bearings
        ],
        "session_id": ref.session_id,
    }


# --- get_standing ---------------------------------------------------------------------


def get_standing(arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]") -> "dict[str, Any]":
    """Where each hypothesis stands, counted PER INTERVIEW.

    This function does no counting. Every number below comes from
    `app/coach/interview_summary.py`, which is the one home for the rule -- a
    second implementation here is exactly the fork this tool waited to avoid,
    and a subtree rollup is precisely where per-fact counting silently
    reappears, because the union genuinely is a longer list of facts."""
    root, scope, nodes = _scoped_nodes(arguments, repo_root=repo_root)
    hypothesis_id = _optional_string(arguments, "hypothesis_id")

    errors: "list[dict[str, str]]" = []
    refs, extraction = _load_facts_across(nodes, errors)
    found = _load_hypotheses_across(nodes, errors)
    if hypothesis_id is not None:
        found = tuple(pair for pair in found if pair[0].id == hypothesis_id)

    standings = [
        summary_mod.standing(hypothesis, refs, project=node_path)
        for hypothesis, node_path in found
    ]

    # Two nodes may each author `h1` meaning different things. Merging them
    # into one standing would invent a hypothesis nobody wrote; dropping one
    # would hide it. Both are returned, each attributed to its own node, and
    # the collision is NAMED -- otherwise an agent reads two rows as two
    # independent results about the same belief.
    seen: "dict[str, int]" = {}
    for hypothesis, _node_path in found:
        seen[hypothesis.id] = seen.get(hypothesis.id, 0) + 1

    return {
        **_envelope(root, scope, nodes, errors),
        # Above the standings, and not optional. Every count in a standing is
        # zero for a project nobody has extracted, and zero is the same
        # integer as "we looked and nothing bears on this" -- which is the one
        # of the two that is a research finding.
        **_extraction_block(extraction),
        "hypothesis_id": hypothesis_id,
        "duplicate_hypothesis_ids": sorted(key for key, count in seen.items() if count > 1),
        "standings": [_project_standing(standing) for standing in standings],
    }


def _project_standing(standing: Any) -> "dict[str, Any]":
    """Field by field, and every count named for WHAT IT COUNTS.

    `*_interviews` and `*_facts` sit next to each other on the wire on
    purpose: the honest pair is the answer, and a surface that published only
    one of them would leave a reader to guess which kind of number they had."""
    return {
        "hypothesis_id": standing.hypothesis_id,
        "project": standing.project,
        "projects": list(standing.projects),
        "statement": standing.statement,
        "falsifier": standing.falsifier,
        "state": standing.state,
        "criterion": standing.criterion,
        "is_scorable": standing.is_scorable,
        "supporting_interviews": standing.supporting_interviews,
        "contrary_interviews": standing.contrary_interviews,
        "unclear_interviews": standing.unclear_interviews,
        "total_interviews": standing.total_interviews,
        "supporting_facts": standing.supporting_facts,
        "contrary_facts": standing.contrary_facts,
        "unclear_facts": standing.unclear_facts,
        "supporting": [_project_fact(ref) for ref in standing.supporting],
        "contrary": [_project_fact(ref) for ref in standing.contrary],
        "unclear": [_project_fact(ref) for ref in standing.unclear],
    }


# --- get_interview_summary ------------------------------------------------------------


def get_interview_summary(
    arguments: "Mapping[str, Any]", *, repo_root: "Optional[Path]"
) -> "dict[str, Any]":
    """One interview, summarized deterministically. No model call: the
    judgement was bought and paid for upstream in `Bearing.stance`."""
    root, scope, nodes = _scoped_nodes(arguments, repo_root=repo_root)

    session_id = arguments.get("session_id")
    if not isinstance(session_id, str):
        raise ToolArgumentError(
            f"'session_id' is required and must be a string, got {type(session_id).__name__}"
        )
    # A session id becomes a DIRECTORY NAME below, so it is untrusted input
    # turned into a filesystem location -- the same category `project` is, and
    # it gets the same allowlist. `normalize()` is reused rather than a second
    # rule being written here: one segment, letters/digits/'-'/'_' only, so
    # `..`, a separator and a drive letter are all unrepresentable.
    try:
        if len(projects_mod.normalize(session_id)) != 1:
            raise projects_mod.ProjectPathError(
                f"a session id is a single label, never a path: {session_id!r}"
            )
    except projects_mod.ProjectPathError as exc:
        raise ToolArgumentError(str(exc)) from exc

    errors: "list[dict[str, str]]" = []
    refs, extraction = _load_facts_across(nodes, errors)
    summary = summary_mod.summarize_interview(session_id, refs)

    found_at: "list[dict[str, Any]]" = []
    for node in nodes:
        session_dir = node.sessions_root / session_id
        if not session_dir.is_dir():
            continue
        try:
            result = session_reader.read_session(session_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append({"project": node.path, "error": _describe(exc)})
            continue
        found_at.append(
            {
                "project": node.path,
                "started_at": result.metadata.get("started_at"),
                "closed": result.closed,
                "truncated_tail": result.truncated_tail is not None,
                "coach_enabled": bool(result.metadata.get("coach_enabled", False)),
            }
        )

    envelope = _envelope(root, scope, nodes, errors)
    if not found_at:
        # NAMED rather than silently empty. An empty summary for a session
        # that does not exist and an empty summary for a session with no
        # extracted facts are different answers, and an agent acting on the
        # second would conclude the interview said nothing.
        missing = f"no interview {session_id!r} under {envelope['nodes']}"
        envelope["errors"] = [*envelope["errors"], {"project": root.path, "error": missing}]
        envelope["error"] = "; ".join(
            f"{entry['project']}: {entry['error']}" for entry in envelope["errors"]
        )

    return {
        **envelope,
        # This tool already NAMES a session it could not find. It says nothing
        # about a session that exists and was never extracted, which reads as
        # an interview that produced no claims.
        **_extraction_block(extraction),
        "session_id": summary.session_id,
        # 1 by definition, on the wire on purpose: a caller aggregating
        # several of these must add INTERVIEWS, never facts.
        "interviews": summary.interviews,
        "found_at": [entry["project"] for entry in found_at],
        "sessions": found_at,
        "started_at": found_at[0]["started_at"] if found_at else None,
        "closed": found_at[0]["closed"] if found_at else None,
        "projects": list(summary.projects),
        "fact_count": summary.fact_count,
        "evidence_facts": summary.evidence_facts,
        "inference_facts": summary.inference_facts,
        "touches": [
            {
                "hypothesis_id": touch.hypothesis_id,
                "supporting_facts": touch.supporting_facts,
                "contrary_facts": touch.contrary_facts,
                "unclear_facts": touch.unclear_facts,
            }
            for touch in summary.touches
        ],
    }


# --- the registry ---------------------------------------------------------------------


def _project_argument(description: str) -> "dict[str, Any]":
    return {"type": "string", "description": description}


_PROJECT_DESCRIPTION = (
    "A project path from list_projects, '/'-separated, e.g. 'acme/onboarding'. "
    "Letters, digits, '-' and '_' only; never absolute, never containing '..'."
)

_SCOPE_ARGUMENT = {
    "type": "string",
    "enum": list(ALL_SCOPES),
    "default": SCOPE_NODE,
    "description": (
        "'node' (default) reads only the named project. 'subtree' also reads every "
        "descendant of it and merges the results, and every returned item carries the "
        "'project' it came from. A subtree never reaches a parent, a sibling, or another "
        "hierarchy -- 'acme' does not include 'acme-corp'. Counts in a merged answer are "
        "per-INTERVIEW, never per-fact."
    ),
}

_HYPOTHESIS_ID_ARGUMENT = {
    "type": "string",
    "description": "Optional. Return only results bearing on this hypothesis id.",
}


def _scoped_schema(**extra: "dict[str, Any]") -> "dict[str, Any]":
    return {
        "type": "object",
        "properties": {
            "project": _project_argument(_PROJECT_DESCRIPTION),
            "scope": _SCOPE_ARGUMENT,
            **extra,
        },
        "required": ["project"],
        "additionalProperties": False,
    }

_TOOL_ORDER: "tuple[tuple[str, str, dict[str, Any], Callable[..., dict[str, Any]]], ...]" = (
    (
        "list_projects",
        "Every project node on this machine, parents before children: its path, label, "
        "nesting depth, whether it has authored context and recorded interviews, and whether "
        "its 'corpus' is demo, real or undeclared. A demo node holds hand-authored "
        "demonstration claims that read exactly like customer research; undeclared means "
        "nobody has said either way, which is not the same as real. "
        "Start here -- every other tool takes one of these paths.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        list_projects,
    ),
    (
        "get_survey",
        "The survey for one project: the ordered questions the interviewer set out to ask. "
        "Authored by the operator, not spoken by anyone.",
        {
            "type": "object",
            "properties": {"project": _project_argument(_PROJECT_DESCRIPTION)},
            "required": ["project"],
            "additionalProperties": False,
        },
        get_survey,
    ),
    (
        "get_hypotheses",
        "The hypotheses under test in one project: statement, required falsifier, validation "
        "state (untested | supported | contradicted | contested), the pre-registered decision "
        "criterion, and whether that criterion exists at all (is_scorable). A hypothesis that "
        "is not scorable has no rule to be judged against -- do not tally evidence against one. "
        "Each result names the project node that authored it.",
        _scoped_schema(),
        get_hypotheses,
    ),
    (
        "list_interviews",
        "Every recorded interview in one project: session id, the node it is filed at, start "
        "time, whether it was closed cleanly, whether its transcript ends in a crash-truncated "
        "line, and whether the live coach was running. Returns no transcript text of any kind.",
        _scoped_schema(),
        list_interviews,
    ),
    (
        "get_facts",
        "The extracted claims for one project, optionally only those bearing on one hypothesis. "
        "Each fact carries its abstracted CLAIM, its kind (evidence | inference), which "
        "hypotheses it bears on and in which direction (supports | undermines | unclear), the "
        "interview it came from and the project node it is filed at. The verbatim customer "
        "quote behind a claim is deliberately not exposed and never will be -- read it on the "
        "operator's own machine. Check 'unextracted_projects' before concluding anything from an "
        "empty result: a project nobody has run extraction on has no facts to return, which is "
        "not the same answer as an interview that produced none."
        "Every entry in 'extraction' carries a 'corpus' of demo, real or undeclared, and it is NOT a formality: some nodes on this machine hold hand-authored demonstration claims that read exactly like customer research, naming a 'he' or a 'she' and citing a session id. Nothing else in a fact distinguishes them -- 'kind' says evidence-vs-inference, which is how firmly the claim follows from what was said, not whether a person said it. Never report a demo claim as a finding, and treat undeclared as not-yet-established rather than real. ",
        _scoped_schema(hypothesis_id=_HYPOTHESIS_ID_ARGUMENT),
        get_facts,
    ),
    (
        "get_standing",
        "Where each hypothesis stands: how many INTERVIEWS support it, how many undermine it, "
        "how many were unclear, plus the claims behind each column and the nodes they came "
        "from. Counts are per-INTERVIEW and never per-fact -- one interviewee who made the same "
        "point five times moves a count by one, not five -- so supporting_interviews and "
        "supporting_facts are different numbers and both are returned. A hypothesis with "
        "is_scorable false has no pre-registered decision rule and must not be tallied against. "
        "READ 'unextracted_projects' FIRST: every count here is zero for a project whose "
        "interviews have never been extracted, and that zero means 'not looked at yet', not "
        "'nothing bears on this'. Only the second is a research finding."
        "Every entry in 'extraction' carries a 'corpus' of demo, real or undeclared, and it is NOT a formality: some nodes on this machine hold hand-authored demonstration claims that read exactly like customer research, naming a 'he' or a 'she' and citing a session id. Nothing else in a fact distinguishes them -- 'kind' says evidence-vs-inference, which is how firmly the claim follows from what was said, not whether a person said it. Never report a demo claim as a finding, and treat undeclared as not-yet-established rather than real. ",
        _scoped_schema(hypothesis_id=_HYPOTHESIS_ID_ARGUMENT),
        get_standing,
    ),
    (
        "get_interview_summary",
        "One interview, summarized deterministically from the extracted corpus: how many claims "
        "it produced, how many were evidence versus inference, and which hypotheses it touched "
        "in which direction. Counts inside a summary are per-CLAIM and named *_facts; the "
        "interview count is 1 by definition. No transcript text and no quotes. A summary of zero "
        "claims means the interview yielded none ONLY if its project is absent from "
        "'unextracted_projects'; otherwise extraction has simply not been run.",
        _scoped_schema(
            session_id={
                "type": "string",
                "description": (
                    "A session id from list_interviews, e.g. '20260812-101500-abcdef'. A single "
                    "label: letters, digits, '-' and '_' only, never a path."
                ),
            }
        )
        | {"required": ["project", "session_id"]},
        get_interview_summary,
    ),
)

TOOL_DEFINITIONS = tuple(
    {"name": name, "description": description, "inputSchema": schema}
    for name, description, schema, _handler in _TOOL_ORDER
)

_HANDLERS = {name: handler for name, _description, _schema, handler in _TOOL_ORDER}


def call_tool(
    name: str, arguments: "Optional[Mapping[str, Any]]", *, repo_root: "Optional[Path]"
) -> "dict[str, Any]":
    """Dispatches one tool call. Raises `ToolArgumentError` for anything the
    CALLER got wrong; a data condition is reported inside the returned
    payload instead."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ToolArgumentError(
            f"unknown tool {name!r} -- this server serves exactly "
            f"{sorted(_HANDLERS)}, and is read-only"
        )
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise ToolArgumentError(
            f"'arguments' must be an object, got {type(arguments).__name__}"
        )
    return handler(arguments, repo_root=repo_root)

