"""The project tree: a project is a nested LABEL, and resolving one is a path
decision rather than a refactor.

The user runs several lines of enquiry at once, and a "project" for them is a
hierarchy -- a product, a feature within it, a hypothesis within that. Nesting
is directory nesting, so `acme/onboarding/activation-delay` is a real path
under `projects/`.

**Why this module is small, and must stay small.** No module under
`app/coach/`, `app/session/` or `app/web/` reads config internally -- every
loader and writer already takes its directory as a PARAMETER, and
`cfg.context_dir` / `cfg.sessions_root` are referenced only in `app/main.py`.
That makes `main.py` the single injection point and this module a pure
path resolver. It opens no files, writes nothing, and imports nothing from
`app.coach` or `app.session`. Two standing gates depend on that restraint:

- **Gate 19** pins the modules that WRITE `context/hypotheses.json` to exactly
  `{app/web/author.py}`, and detects them textually -- a module that merely
  builds a path string naming that artifact *and* contains any write primitive
  trips it. This module therefore names no artifact filename at all; it
  resolves DIRECTORIES and lets the existing loaders name their own files.
- **Gate 20** pins the modules that may read `Fact.quoted_text`. Nothing here
  reads a fact.

**Anchored to the repository root, never the working directory.** `APP_CONTEXT_DIR`
and `APP_SESSIONS_ROOT` default to the bare relative paths `context` and
`sessions` and are never resolved, so today both roots follow the process's
CWD. That is tolerable for a single pair of directories a user cd's into; it
is not tolerable for a tree the operator selects by name, where the same
`--project acme` would silently mean different customers' data depending on
where the shell happened to be. Every path this module returns is absolute.

**Path confinement is the security boundary.** A project path arrives from a
CLI argument and, once the console exists, from an HTTP request body. It is
therefore untrusted input that gets turned into a filesystem location holding
verbatim customer speech. `resolve()` rejects anything that could escape
`projects/` -- absolute paths, drive letters, `..`, and anything outside a
deliberately narrow character set -- and then verifies containment against the
resolved real path as a backstop, so a symlink planted inside the tree cannot
walk out of it either.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECTS_DIR_NAME = "projects"
CONTEXT_DIR_NAME = "context"
SESSIONS_DIR_NAME = "sessions"
# A node's CONTENTS, exactly like the two above, and skipped for the same
# reason. `benchmark/eval/rehearse.py` writes synthetic runs to
# `projects/<p>/rehearsals/<timestamp>/` -- deliberately NOT to `sessions/`, so
# that a rehearsal can never be mistaken for a recorded interview or displace
# the newest real session `analyse_eval.py` scores. That separation held on
# disk and then leaked at the surface: the walk below descends any directory
# whose name parses, so `rehearsals` and every timestamp under it came back as
# project NODES -- to the console, the hub picker and the MCP `list_projects`
# alike. Synthetic runs appearing as projects in a discovery corpus is the same
# class of error the directory split existed to prevent.
REHEARSALS_DIR_NAME = "rehearsals"
# Everything a project node CONTAINS rather than a node in its own right. A new
# per-project directory added anywhere must be listed here or it becomes a
# project.
NON_NODE_DIR_NAMES = frozenset({CONTEXT_DIR_NAME, SESSIONS_DIR_NAME, REHEARSALS_DIR_NAME})

# `app/projects.py` -> `app/` -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Deliberately narrow: a project path segment is a LABEL, and a label is
# something the operator types and reads back on the overlay. Letters, digits,
# hyphen and underscore only. This is a allowlist rather than a denylist of
# dangerous characters on purpose -- the denylist form has to anticipate
# every path-hostile construct on every platform (drive letters, `..`, NUL,
# trailing dots and spaces that Windows silently strips, reserved device
# names), and the interesting failure is always the one nobody anticipated.
_SEGMENT_PATTERN = re.compile(r"\A[A-Za-z0-9_-]+\Z")

# Windows resolves these to devices regardless of the directory they appear
# in, so a segment named `con` would be a path that cannot be created and
# whose failure mode is confusing rather than obvious. Excluded explicitly
# even though they satisfy the pattern above.
_RESERVED_SEGMENTS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in "123456789"}
    | {f"lpt{digit}" for digit in "123456789"}
)


class ProjectPathError(ValueError):
    """A project path that is malformed, or that would resolve outside
    `projects/`. Raised rather than silently normalized: quietly turning
    `../../etc` into something safe would hide a caller that is either
    confused or hostile."""


@dataclass(frozen=True)
class ProjectPaths:
    """Everything `main.py` needs to point the existing loaders at one node.

    `context_dir` and `sessions_root` are the same two things the config
    already carries, which is what makes selecting a project a substitution
    rather than a rewrite."""

    path: str
    """The normalized path, forward-slash separated: `acme/onboarding`."""

    label: str
    """The leaf segment -- what the overlay shows."""

    ancestors: "tuple[str, ...]"
    """Every enclosing node, outermost first, as normalized paths. For
    `acme/onboarding/activation-delay` this is
    `('acme', 'acme/onboarding')`. Empty for a root node.

    Carried because cross-project reads are permitted only WITHIN the same
    hierarchy: the same-root check is a comparison of first segments, and the
    picker needs the chain to render it."""

    root: str
    """The outermost segment. Two nodes may share knowledge only if their
    roots are equal."""

    node_dir: Path
    """Absolute. The node's own directory -- the PARENT of the two below.

    Carried so a caller that needs something else in the node (today:
    `app/project_meta.py`, reading whether it has been archived) can address it
    without re-deriving the location or reaching for `context_dir.parent`. This
    module still names no filename and still opens nothing: it hands over a
    DIRECTORY, exactly as it does for the two roots."""

    context_dir: Path
    """Absolute. Holds this node's brief, survey, hypotheses and facts."""

    sessions_root: Path
    """Absolute. Interviews filed at THIS node."""


@dataclass(frozen=True)
class ProjectNode:
    """One node of the tree as `list_projects()` found it on disk."""

    path: str
    label: str
    depth: int
    has_context: bool
    has_sessions: bool


def _projects_root(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or REPO_ROOT) / PROJECTS_DIR_NAME


def normalize(path: str) -> "tuple[str, ...]":
    """Validates a project path and returns its segments.

    Separated from `resolve()` so the console can validate a path the operator
    is still typing without touching the filesystem, and so the rejection
    rules can be tested without a directory existing."""
    if not isinstance(path, str):
        raise ProjectPathError(f"a project path must be a string, got {type(path).__name__}")

    candidate = path.strip()
    if not candidate:
        raise ProjectPathError("a project path cannot be empty")

    # Backslash is rejected rather than accepted as a separator. On Windows it
    # IS one, so tolerating it would mean this module's own validation and the
    # filesystem's interpretation could disagree about how many segments a
    # path has -- and the segment count is what every rule below is written
    # against.
    if "\\" in candidate:
        raise ProjectPathError(
            f"a project path must use '/' between labels, never '\\': {path!r}"
        )
    if candidate.startswith("/"):
        raise ProjectPathError(f"a project path must be relative, never absolute: {path!r}")
    if ":" in candidate:
        raise ProjectPathError(f"a project path may not name a drive or stream: {path!r}")

    segments = tuple(candidate.split("/"))
    for segment in segments:
        if not segment:
            raise ProjectPathError(f"a project path has an empty label: {path!r}")
        if segment in (".", ".."):
            raise ProjectPathError(
                f"a project path may not contain '.' or '..': {path!r}"
            )
        if not _SEGMENT_PATTERN.match(segment):
            raise ProjectPathError(
                f"project label {segment!r} may use only letters, digits, '-' and '_': {path!r}"
            )
        if segment.lower() in _RESERVED_SEGMENTS:
            raise ProjectPathError(
                f"project label {segment!r} is a reserved device name on Windows: {path!r}"
            )
    return segments


def resolve(path: str, *, repo_root: Optional[Path] = None) -> ProjectPaths:
    """Turns a project path into the two absolute directories the existing
    loaders take as parameters.

    Does NOT require the node to exist -- creating a project is writing to
    these paths, so resolution has to come first. It does require that the
    result be inside `projects/`, checked twice: once structurally on the
    segments, and once on the resolved real path, which is the check that
    catches a symlink planted inside the tree."""
    segments = normalize(path)
    projects_root = _projects_root(repo_root)
    node_dir = projects_root.joinpath(*segments)

    # The backstop. `normalize()` already makes `..` unrepresentable, so this
    # exists for what it cannot see: a symlink INSIDE projects/ pointing out.
    # `Path.resolve()` follows links; comparing the followed path against the
    # followed root is the only form of this check that is not fooled by one.
    resolved_node = node_dir.resolve()
    resolved_root = projects_root.resolve()
    if resolved_node != resolved_root and resolved_root not in resolved_node.parents:
        raise ProjectPathError(
            f"project path {path!r} resolves outside the project tree -- "
            f"refusing to read or write there"
        )

    normalized = "/".join(segments)
    ancestors = tuple(
        "/".join(segments[: index + 1]) for index in range(len(segments) - 1)
    )
    return ProjectPaths(
        path=normalized,
        label=segments[-1],
        ancestors=ancestors,
        root=segments[0],
        node_dir=node_dir,
        context_dir=node_dir / CONTEXT_DIR_NAME,
        sessions_root=node_dir / SESSIONS_DIR_NAME,
    )


def share_a_root(left: str, right: str) -> bool:
    """Whether two project paths sit in the same hierarchy.

    The user's rule, stated once here so no caller re-implements it: cross-
    project reads are allowed only within the same hierarchy and only on
    explicit request. Both paths are validated, so a malformed path is an
    error rather than a quiet `False` -- a rejected read must be
    distinguishable from an unparseable one.

    Compared under `normcase` -- identity on POSIX, case-folding on Windows --
    because the hierarchy this rule is ABOUT is a directory tree, and Windows
    folds case: `projects/ACME` and `projects/acme` are one directory. A
    case-sensitive comparison here made a subtree read return the root node's
    own data (the filesystem opened it happily) while dropping every
    descendant, and report no error while doing it. An under-reported rollup
    says "no contrary evidence" where contrary evidence exists, which is the
    exact confirmation failure `corpus.Bearing` exists to make impossible.

    Folding case cannot widen the rule: it compares the FIRST SEGMENT only, so
    `acme` and `acme-corp` remain different roots however either is spelled."""
    return os.path.normcase(normalize(left)[0]) == os.path.normcase(normalize(right)[0])


def list_projects(*, repo_root: Optional[Path] = None) -> "tuple[ProjectNode, ...]":
    """Walks `projects/` and returns every node, parents before children,
    siblings in name order.

    A node is any directory whose own path is a valid project path; everything
    in `NON_NODE_DIR_NAMES` is a node's CONTENTS, never a node itself, so those
    are not descended into. A directory whose name is not
    a valid label is skipped rather than raising -- the tree is on disk and a
    user may have put anything there, and one stray directory must not make
    the console unopenable."""
    projects_root = _projects_root(repo_root)
    if not projects_root.is_dir():
        return ()

    found: "list[ProjectNode]" = []

    def _walk(directory: Path, prefix: "tuple[str, ...]") -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if child.name in NON_NODE_DIR_NAMES:
                continue
            try:
                normalize(child.name)
            except ProjectPathError:
                continue
            segments = prefix + (child.name,)
            found.append(
                ProjectNode(
                    path="/".join(segments),
                    label=child.name,
                    depth=len(segments),
                    has_context=(child / CONTEXT_DIR_NAME).is_dir(),
                    has_sessions=(child / SESSIONS_DIR_NAME).is_dir(),
                )
            )
            _walk(child, segments)

    _walk(projects_root, ())
    return tuple(found)
