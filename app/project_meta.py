"""What a project IS, as opposed to where it lives.

**Why this is not in `app/projects.py`.** That module is a pure path
resolver: read its docstring -- it opens no files, names no artifact
filename, and two standing gates (19 and 20) detect their offenders
TEXTUALLY, so its restraint is load-bearing rather than stylistic. Everything
here has to touch the disk: whether the unfiled bucket exists yet, and
whether a node has been archived. So it lives beside that module rather than
inside it, and `projects.py` keeps handing out directories while this one
opens what is in them.

**The unfiled bucket is a REAL node, `projects/_unfiled/`.** A leading
underscore already satisfies `projects.py`'s segment pattern (`projects/_eval/`
is the precedent), so it resolves through the existing path code with no new
resolution logic at all. The alternative -- a synthetic picker entry meaning
"no project" -- would need a null-project branch in `main._apply_project`, in
`InterviewSession.command()`, in the console's session lister and in `render`,
and each of those four is a place a `None` could leak through as a path.

It carries no `context/`. There is no brief and no survey for a project that
does not exist yet, which is exactly why an interview filed here runs with no
coach.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from app import projects as projects_mod

logger = logging.getLogger(__name__)

# The node every recording lands in when the operator has no project yet.
UNFILED_PROJECT = "_unfiled"

# Local-only and uncommittable. Gate 29 already asserts `projects/**` is
# ignored in full, `projects/acme/project.json` named explicitly, and that is
# on purpose: whether a project is finished is one operator's view of their
# own tree, not a fact about the repository.
PROJECT_META_FILENAME = "project.json"

# The ONE status that hides a node from a picker, compared as an exact
# literal. See `is_archived` for what that rules out and why.
STATUS_ARCHIVED = "archived"

# --- Whether a node's corpus describes real people ------------------------------------
#
# **The one-way door this closes.** A fact record carries exactly seven fields
# and `provenance` is not one of the answers here: it says evidence-vs-inference
# -- how firmly the claim follows from what was said -- and says nothing about
# whether anyone said it. Today `projects/demo-claudictive/context/facts.json`
# holds 25 claims about "he" and "she", 13 of them typed by hand and the rest
# extracted from a mock where the operator played the interviewee, and NOTHING
# in the record, the file or the MCP payload distinguishes any of them from a
# customer. The only tell is that whoever wrote the fixtures used ids starting
# `1111`, which is an accident of authoring, not a property of the data.
#
# That is survivable only while every project is fake. The moment one real
# interview is extracted anywhere, the two kinds are mixed in a store with no
# field that separates them and no timestamp to separate them by -- and the
# repair becomes archaeology over claim text. Declaring it now costs a line of
# JSON per project.
#
# **Three states, because `provenance` set the precedent: never silently
# defaulted.** An absent declaration is reported as UNDECLARED rather than
# assumed real. Assuming real is the failure this exists to prevent, and
# assuming demo would mark every genuine project until someone remembered to
# say otherwise, which trains a reader to ignore the marker. "I was not told"
# is the honest third answer and the payloads carry it as such.
CORPUS_DEMO = "demo"
CORPUS_REAL = "real"
CORPUS_UNDECLARED = "undeclared"
# Pinned by equality, like every other closed set in this codebase.
ALL_CORPUS_KINDS = frozenset({CORPUS_DEMO, CORPUS_REAL, CORPUS_UNDECLARED})
CORPUS_KEY = "corpus"


def is_unfiled(project_path: str) -> bool:
    """Whether a path names the unfiled bucket ITSELF.

    Never raises: it is asked about whatever the picker returned, and a path
    that is not a project path at all is simply not the bucket. A CHILD of the
    bucket is not the bucket either -- that is a real filed project that
    happens to live under it, and it gets a coach like any other."""
    try:
        return projects_mod.normalize(project_path) == (UNFILED_PROJECT,)
    except projects_mod.ProjectPathError:
        return False


def ensure_unfiled(*, repo_root: "Optional[Path]" = None) -> Path:
    """Creates `projects/_unfiled/` if it is not there, and returns it.

    The directory ONLY. Creating a `context/` here would be a brief and a
    survey for a project that does not exist -- and with a key present,
    `start` would load them and ground a coach in another customer's
    material.

    Idempotent because it runs on every press of the record button."""
    node_dir = projects_mod.resolve(UNFILED_PROJECT, repo_root=repo_root).node_dir
    node_dir.mkdir(parents=True, exist_ok=True)
    return node_dir


def _declared_string(node_dir: Path, key: str) -> "Optional[str]":
    """The string a node declares under `key`, or `None` in every other case.

    Every failure returns `None`: an absent file, an unreadable one, malformed
    JSON, a JSON document that is not an object, a status that is not a
    string. The caller turns `None` into "active", and that direction is the
    entire point. Hiding a project the operator has on disk is unrecoverable
    by inspection -- nothing on the page says it dropped anything, so there is
    nowhere to look -- while showing one extra row is untidy.

    The same discipline `app/web/console.py::_node_detail` already applies
    per-section, for the same reason: one bad file among twenty must not cost
    the operator the other nineteen, and they are most likely reaching for the
    picker BECAUSE something is wrong."""
    path = node_dir / PROJECT_META_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        # Said out loud once, never raised. `OSError` and not just
        # `JSONDecodeError`: a directory where the file should be, a
        # permission failure, or a file being rewritten right now all arrive
        # here, and any of them escaping would take the whole picker down.
        logger.warning(
            "could not read a project's %s error_type=%s -- treating it as active",
            PROJECT_META_FILENAME, type(exc).__name__,
        )
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(key)
    return value if isinstance(value, str) else None


def _declared_status(node_dir: Path) -> "Optional[str]":
    """The `status` string a node declares. See `_declared_string`."""
    return _declared_string(node_dir, "status")


def corpus_kind(project_path: str, *, repo_root: "Optional[Path]" = None) -> str:
    """Whether this node's claims describe real people: one of `ALL_CORPUS_KINDS`.

    **An unrecognised value reads as UNDECLARED, never as real.** A typo
    (`"Demo"`, `"fixture"`, `"true"`) is someone TRYING to say this corpus is
    not real and failing to spell it. Falling back to real there would turn a
    near-miss into the exact misreading the field exists to stop, so the
    conservative answer is that nothing was successfully declared -- the same
    direction `Fact.provenance` takes when it refuses to guess.

    Note this is the OPPOSITE lean from `is_archived`, deliberately. There the
    recoverable mistake is showing one row too many; here it is doubting a
    corpus that was real all along, which a glance at `project.json` settles.
    Believing a fabricated claim about a customer is the mistake that does not
    announce itself.

    Never raises. Like `is_archived`, it is asked about whatever the walker
    found on disk, and an unreadable `project.json` must not take down the tool
    that asked."""
    try:
        node_dir = projects_mod.resolve(project_path, repo_root=repo_root).node_dir
    except projects_mod.ProjectPathError:
        return CORPUS_UNDECLARED
    return corpus_kind_of(node_dir)


def corpus_kind_of(node_dir: Path) -> str:
    """`corpus_kind` for a node whose directory the caller already holds.

    The MCP fact loader walks `ProjectPaths` objects that carry `node_dir`
    precisely so a caller can address the node without re-deriving it (see that
    field's own docstring), and it has no `repo_root` to re-resolve a path
    with. Splitting the lookup here keeps that call site from either growing a
    parameter it does not otherwise need or rebuilding a path it was handed."""
    declared = _declared_string(node_dir, CORPUS_KEY)
    if declared in (CORPUS_DEMO, CORPUS_REAL):
        return declared
    return CORPUS_UNDECLARED


def is_archived(project_path: str, *, repo_root: "Optional[Path]" = None) -> bool:
    """Whether a picker should hide this node.

    **Only the exact literal `"archived"`.** Not case-folded, not stripped,
    and emphatically not `status != "active"`. Both of those hide LIVE
    projects: the first accepts spellings this project never agreed on, and
    the second hides every node that has a `project.json` for any other reason
    at all -- a label, a colour, a key some later feature adds -- which is most
    of them. The conservative comparison is the recoverable one.

    **Archiving is a picker filter, never deletion.** `projects.resolve()`,
    `--project` and `list_projects()` are all untouched, so an archived node
    still opens, still records and still renders. Nothing here is reachable
    from a read path.

    **The unfiled bucket is exempt, explicitly.** The operator chose to show
    every node in the picker (no `_`-prefix filtering), so the exemption that
    filtering would have granted for free has to be written down instead --
    otherwise `_unfiled` could be archived into invisibility and there would
    be no way left to record without a project.

    Never raises. It is asked about whatever the walker found on disk."""
    if is_unfiled(project_path):
        return False
    try:
        node_dir = projects_mod.resolve(project_path, repo_root=repo_root).node_dir
    except projects_mod.ProjectPathError:
        return False
    return _declared_status(node_dir) == STATUS_ARCHIVED


def for_picker(
    nodes: "Iterable[projects_mod.ProjectNode]", *, repo_root: "Optional[Path]" = None
) -> "tuple[projects_mod.ProjectNode, ...]":
    """The nodes a picker should offer, in the order it should offer them.

    Two rules, and nothing else. This never re-sorts, so `list_projects()`'s
    parents-before-children, siblings-in-name-order guarantee survives intact
    -- the console indents against exactly that order.

    **An archived node takes its whole subtree with it.** Hiding `acme` while
    keeping `acme/onboarding` would be worse than not filtering at all: with
    no parent above it, the child reads as a root project. Implemented as a
    path-PREFIX skip (`path + "/"`, never the bare string, or archiving
    `acme` would also hide `acme-corp`), which is why the input order matters:
    a parent is always seen before its children.

    **The unfiled bucket sits LAST.** `_` (0x5F) sorts before `a` (0x61), so
    the name-ordered walk would otherwise put it at the TOP, where the first
    row reads as the default choice. It is a fallback, not a default, and the
    operator asked for it last and unselected."""
    hidden_prefixes: "list[str]" = []
    kept: "list[projects_mod.ProjectNode]" = []
    for node in nodes:
        if any(node.path.startswith(prefix + "/") for prefix in hidden_prefixes):
            continue
        if is_archived(node.path, repo_root=repo_root):
            hidden_prefixes.append(node.path)
            continue
        kept.append(node)
    return tuple(
        [node for node in kept if not is_unfiled(node.path)]
        + [node for node in kept if is_unfiled(node.path)]
    )
