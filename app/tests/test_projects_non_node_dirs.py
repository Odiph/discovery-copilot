"""A project node's own directories must never come back as projects.

`rehearsals/` did. `benchmark/eval/rehearse.py` writes synthetic runs to
`projects/<p>/rehearsals/<timestamp>/` rather than to `sessions/` precisely so
a rehearsal cannot be mistaken for a recorded interview -- and then the walk in
`list_projects` descended it anyway, so `rehearsals` and every timestamp under
it were returned as project nodes to the console, the hub picker and the MCP
surface. The separation was correct on disk and leaked at every reader.

The test that matters here is the LAST one: pinning the three names known
today would pass forever while the next per-project directory someone adds
becomes a project, which is exactly how this one shipped.
"""

from __future__ import annotations

import pytest

from app.projects import (
    CONTEXT_DIR_NAME,
    NON_NODE_DIR_NAMES,
    REHEARSALS_DIR_NAME,
    SESSIONS_DIR_NAME,
    list_projects,
)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "projects"
    node = root / "demo"
    (node / CONTEXT_DIR_NAME).mkdir(parents=True)
    (node / SESSIONS_DIR_NAME / "20260101T000000").mkdir(parents=True)
    (node / REHEARSALS_DIR_NAME / "20260814T172816").mkdir(parents=True)
    (node / REHEARSALS_DIR_NAME / "20260814T173624").mkdir(parents=True)
    (node / "sub").mkdir()
    return tmp_path


def paths(repo) -> "set[str]":
    return {node.path for node in list_projects(repo_root=repo)}


def test_rehearsals_is_not_a_project(repo):
    assert "demo/rehearsals" not in paths(repo)


def test_individual_rehearsal_runs_are_not_projects(repo):
    """The timestamps are the visible symptom -- a discovery corpus listing
    `demo/rehearsals/20260814T172816` as a project to interview against."""
    assert not {p for p in paths(repo) if "rehearsals" in p}


def test_a_real_child_node_is_still_returned(repo):
    """The fix must not be 'stop descending'. A genuine sub-project alongside a
    rehearsals directory has to survive, or the hierarchy the whole corpus is
    built on collapses to one level."""
    assert "demo/sub" in paths(repo)
    assert "demo" in paths(repo)


def test_the_node_still_reports_its_context_and_sessions(repo):
    """Skipping a directory must not also hide what it tells us about its
    parent -- `has_sessions` is what the picker greys out on."""
    demo = next(n for n in list_projects(repo_root=repo) if n.path == "demo")
    assert demo.has_context and demo.has_sessions


def test_rehearsals_does_not_make_a_node_look_like_it_has_sessions(repo):
    """A rehearsal is not an interview. If `has_sessions` were computed from
    either directory, a project with only synthetic runs would advertise
    recorded ones."""
    bare = repo / "projects" / "synthetic-only"
    (bare / REHEARSALS_DIR_NAME / "20260814T180000").mkdir(parents=True)
    node = next(n for n in list_projects(repo_root=repo) if n.path == "synthetic-only")
    assert not node.has_sessions
    assert not node.has_context


def test_every_non_node_directory_is_excluded_by_the_same_rule(repo):
    """Whatever is in the set is skipped -- so adding a name to the set is the
    whole fix for the next one, and forgetting to is the whole bug."""
    for name in NON_NODE_DIR_NAMES:
        assert f"demo/{name}" not in paths(repo)


def test_the_set_is_the_only_place_these_names_are_decided():
    """`list_projects` must consult the set rather than restate its members.
    The original defect was a hard-coded pair that a third directory could not
    be added to without someone noticing the call site."""
    import inspect

    from app import projects

    source = inspect.getsource(projects.list_projects)
    assert "NON_NODE_DIR_NAMES" in source
    assert CONTEXT_DIR_NAME in NON_NODE_DIR_NAMES
    assert SESSIONS_DIR_NAME in NON_NODE_DIR_NAMES
    assert REHEARSALS_DIR_NAME in NON_NODE_DIR_NAMES
