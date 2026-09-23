"""Phase A: resolving a project path.

The reason this file is mostly rejection cases: a project path arrives from a
CLI argument today and from an HTTP request body once the console exists, and
it is turned into a filesystem location holding verbatim customer speech. The
interesting behaviour of `resolve()` is not what it accepts.
"""

from __future__ import annotations

import pytest

from app.projects import (
    ProjectPathError,
    list_projects,
    normalize,
    resolve,
    share_a_root,
)


# --- what a project path means ---------------------------------------------------------


def test_a_nested_path_resolves_to_that_nodes_own_two_directories(tmp_path):
    paths = resolve("acme/onboarding/activation-delay", repo_root=tmp_path)
    assert paths.context_dir == tmp_path / "projects" / "acme" / "onboarding" / "activation-delay" / "context"
    assert paths.sessions_root == tmp_path / "projects" / "acme" / "onboarding" / "activation-delay" / "sessions"
    assert paths.label == "activation-delay"
    assert paths.root == "acme"
    assert paths.ancestors == ("acme", "acme/onboarding")


def test_a_root_node_has_no_ancestors_and_is_its_own_root(tmp_path):
    paths = resolve("acme", repo_root=tmp_path)
    assert paths.ancestors == ()
    assert paths.root == "acme" and paths.label == "acme"


def test_resolution_does_not_require_the_node_to_exist(tmp_path):
    # Creating a project IS writing to these paths, so resolution has to come
    # first. If this ever starts raising, `--project` can only ever address
    # projects that somehow created themselves.
    paths = resolve("brand-new", repo_root=tmp_path)
    assert not paths.context_dir.exists()


def test_every_returned_path_is_absolute(tmp_path):
    # The whole point of anchoring to the repo root. Both roots are bare
    # relative paths today (`Path("context")`, `Path("sessions")`) and are
    # never resolved, so they follow the process CWD -- tolerable for one
    # fixed pair, not for a tree selected by name, where the same
    # `--project acme` would mean different customers' data depending on
    # where the shell happened to be.
    paths = resolve("acme", repo_root=tmp_path)
    assert paths.context_dir.is_absolute() and paths.sessions_root.is_absolute()


def test_surrounding_whitespace_is_tolerated_because_it_comes_from_a_form_field(tmp_path):
    assert resolve("  acme/onboarding  ", repo_root=tmp_path).path == "acme/onboarding"


# --- path confinement: the security boundary -------------------------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "..",
        "../secrets",
        "acme/../../etc",
        "acme/..",
        "/etc/passwd",
        "/acme",
        "C:/Windows",
        "c:\\Windows",
        "acme\\onboarding",
        "acme//onboarding",
        "",
        "   ",
        ".",
        "acme/.",
        "acme/./onboarding",
        "acme/onboarding/../../../..",
        "\\\\server\\share",
        "acme:stream",
    ],
)
def test_a_path_that_could_escape_the_tree_is_refused(hostile, tmp_path):
    with pytest.raises(ProjectPathError):
        resolve(hostile, repo_root=tmp_path)


@pytest.mark.parametrize("reserved", ["con", "CON", "nul", "com1", "lpt9", "acme/aux"])
def test_a_windows_reserved_device_name_is_refused(reserved, tmp_path):
    # These satisfy the label pattern but resolve to devices regardless of
    # which directory they appear in. Refused for a comprehensible error
    # rather than an inscrutable OSError at first write.
    with pytest.raises(ProjectPathError):
        resolve(reserved, repo_root=tmp_path)


@pytest.mark.parametrize("odd", ["acme project", "acme.onboarding", "acme;rm", "acme\x00x", "acmé"])
def test_a_label_outside_the_allowlist_is_refused(odd, tmp_path):
    # A allowlist, not a denylist of dangerous characters. The denylist form
    # has to anticipate every path-hostile construct on every platform, and
    # the interesting failure is always the one nobody anticipated.
    with pytest.raises(ProjectPathError):
        resolve(odd, repo_root=tmp_path)


def test_a_non_string_is_refused_rather_than_coerced(tmp_path):
    with pytest.raises(ProjectPathError):
        resolve(None, repo_root=tmp_path)  # type: ignore[arg-type]


def test_a_symlink_planted_inside_the_tree_cannot_walk_out_of_it(tmp_path):
    # What `normalize()` structurally CANNOT see: it makes `..`
    # unrepresentable in the path string, but says nothing about what a
    # directory inside the tree actually points at. This is the case the
    # resolved-real-path backstop exists for.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    try:
        (projects_root / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/user cannot create a directory symlink")

    with pytest.raises(ProjectPathError):
        resolve("escape", repo_root=tmp_path)


# --- cross-hierarchy: reads may merge, only within one root ----------------------------


def test_two_nodes_under_the_same_root_share_a_hierarchy():
    assert share_a_root("acme/onboarding", "acme/billing/dunning") is True


def test_two_nodes_under_different_roots_do_not():
    assert share_a_root("acme/onboarding", "globex/onboarding") is False


def test_a_malformed_path_is_an_error_not_a_quiet_false():
    # A rejected read and an unparseable path must be distinguishable. A bare
    # `False` here would let the console report "different hierarchy" for a
    # path that was never valid, and the operator would go looking for the
    # wrong problem.
    with pytest.raises(ProjectPathError):
        share_a_root("acme", "../etc")


# --- listing the tree ------------------------------------------------------------------


def _make(tmp_path, *relative_paths):
    for relative in relative_paths:
        (tmp_path / "projects" / relative).mkdir(parents=True, exist_ok=True)


def test_an_absent_project_tree_lists_as_empty_rather_than_raising(tmp_path):
    assert list_projects(repo_root=tmp_path) == ()


def test_the_tree_lists_parents_before_children_and_siblings_in_name_order(tmp_path):
    _make(tmp_path, "globex", "acme/onboarding/activation-delay", "acme/billing")
    assert [node.path for node in list_projects(repo_root=tmp_path)] == [
        "acme",
        "acme/billing",
        "acme/onboarding",
        "acme/onboarding/activation-delay",
        "globex",
    ]


def test_context_and_sessions_are_a_nodes_contents_never_nodes_themselves(tmp_path):
    _make(tmp_path, "acme/context", "acme/sessions/20260810T120000")
    nodes = list_projects(repo_root=tmp_path)
    assert [node.path for node in nodes] == ["acme"]
    assert nodes[0].has_context is True and nodes[0].has_sessions is True


def test_a_node_reports_which_of_the_two_directories_it_actually_has(tmp_path):
    _make(tmp_path, "acme/sessions")
    (node,) = list_projects(repo_root=tmp_path)
    assert node.has_context is False and node.has_sessions is True


def test_a_stray_directory_is_skipped_rather_than_making_the_tree_unopenable(tmp_path):
    # The tree is on disk and a user may have put anything there. One
    # directory the resolver would refuse must not cost the console every
    # other project.
    _make(tmp_path, "acme", "not a project", "acme/also bad")
    assert [node.path for node in list_projects(repo_root=tmp_path)] == ["acme"]


def test_a_file_sitting_in_the_tree_is_not_mistaken_for_a_project(tmp_path):
    _make(tmp_path, "acme")
    (tmp_path / "projects" / "README.md").write_text("notes", encoding="utf-8")
    assert [node.path for node in list_projects(repo_root=tmp_path)] == ["acme"]


def test_depth_is_reported_so_the_console_can_render_the_hierarchy(tmp_path):
    _make(tmp_path, "acme/onboarding/activation-delay")
    assert [(node.path, node.depth) for node in list_projects(repo_root=tmp_path)] == [
        ("acme", 1),
        ("acme/onboarding", 2),
        ("acme/onboarding/activation-delay", 3),
    ]


def test_every_listed_node_resolves(tmp_path):
    # The two halves of this module must agree: anything the walker reports as
    # a project must be addressable by `--project`. A label rule tightened on
    # one side only would list nodes that cannot be opened.
    _make(tmp_path, "acme/onboarding", "globex_2/billing-eu")
    for node in list_projects(repo_root=tmp_path):
        assert resolve(node.path, repo_root=tmp_path).path == node.path


# --- normalize() is usable without touching the filesystem ------------------------------


def test_normalize_returns_segments_and_needs_no_repo_root():
    assert normalize("acme/onboarding") == ("acme", "onboarding")
