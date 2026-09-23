"""Marking a project archived.

A tree that accumulates work eventually offers a picker full of finished
projects. `projects/<node>/project.json` holding `{"status": "archived"}`
takes one out of the picker.

Three properties this file exists to protect, and they are all about the
FAILURE direction rather than the feature:

1. **Default is active.** Missing, unreadable, malformed, an unrecognised
   status, a status under the wrong key -- every one of them lists normally.
   Hiding a project the operator has on disk is unrecoverable by inspection:
   the page says nothing about what it dropped, so there is nowhere to look.
   One extra row is untidy.
2. **Archiving is a picker filter, never deletion.** `resolve()` and
   `--project` still open an archived node; it still records and still
   renders.
3. **`app/projects.py` gains no file read.** Its docstring explains why that
   purity is load-bearing (gate 19 detects its offenders textually), which is
   the whole reason this reading lives in `app/project_meta.py`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import project_meta
from app import projects as projects_mod
from app.main import STATIC_CONSOLE_DIR
from app.web.console import _node_detail, create_console_app

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make(tmp_path, *relative_paths):
    for relative in relative_paths:
        (tmp_path / "projects" / relative).mkdir(parents=True, exist_ok=True)


def _mark(tmp_path, relative, payload):
    path = tmp_path / "projects" / relative / project_meta.PROJECT_META_FILENAME
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def _picker(tmp_path):
    return [
        node.path
        for node in project_meta.for_picker(
            projects_mod.list_projects(repo_root=tmp_path), repo_root=tmp_path
        )
    ]


# --- the one status that hides a node ----------------------------------------------------


def test_the_exact_archived_literal_takes_a_node_out_of_the_picker(tmp_path):
    _make(tmp_path, "acme", "globex")
    _mark(tmp_path, "globex", {"status": "archived"})
    assert _picker(tmp_path) == ["acme"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"status": "Archived"}, id="capitalised"),
        pytest.param({"status": "ARCHIVED"}, id="shouted"),
        pytest.param({"status": " archived "}, id="padded"),
        pytest.param({"state": "archived"}, id="wrong-key"),
        pytest.param({"archived": True}, id="a-boolean-under-another-name"),
        pytest.param({"status": "paused"}, id="an-unrecognised-status"),
        pytest.param({"status": "active"}, id="explicitly-active"),
        pytest.param({"status": None}, id="a-null-status"),
        pytest.param({"label": "acme"}, id="no-status-at-all"),
        pytest.param({}, id="an-empty-object"),
    ],
)
def test_anything_that_is_not_that_exact_literal_lists_normally(tmp_path, payload):
    # The two implementations this rules out, both of which would look right
    # on the happy path and both of which hide LIVE projects:
    #
    #   status.strip().lower() == "archived"   -- hides `"Archived"`, which the
    #       operator may have typed meaning something else entirely, and
    #       silently accepts spellings this project never agreed on.
    #   status != "active"                     -- hides EVERY node that has a
    #       project.json for any other reason: a label, a colour, a key some
    #       later feature adds. That is most of them.
    #
    # `" archived "` is in here on purpose: it is the closest miss, and it is
    # still a miss.
    _make(tmp_path, "acme", "globex")
    _mark(tmp_path, "globex", payload)
    assert _picker(tmp_path) == ["acme", "globex"]


# --- an unreadable file must never cost the whole tree -----------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("{not json at all", id="truncated-mid-write"),
        pytest.param("", id="empty-file"),
        pytest.param('["archived"]', id="a-list-rather-than-an-object"),
        pytest.param('"archived"', id="a-bare-string"),
        pytest.param("null", id="a-json-null"),
    ],
)
def test_a_malformed_project_file_still_lists_its_own_node(tmp_path, raw):
    # The failure this prevents is not "the flag is ignored" -- it is the
    # console or the overlay picker refusing to open at all because ONE node
    # among twenty has a half-written file. The operator is most likely
    # reaching for the picker precisely BECAUSE something is wrong, and
    # `_node_detail` already established this discipline per-section.
    _make(tmp_path, "acme", "globex")
    _mark(tmp_path, "globex", raw)
    assert _picker(tmp_path) == ["acme", "globex"]
    assert project_meta.is_archived("globex", repo_root=tmp_path) is False


def test_a_project_file_that_cannot_be_opened_at_all_still_lists(tmp_path):
    # A DIRECTORY where the file should be: `read_text` raises an OSError that
    # is not FileNotFoundError, which is the branch a `try/except
    # json.JSONDecodeError` alone would miss entirely -- and missing it means
    # an exception out of the picker rather than a project treated as active.
    _make(tmp_path, "acme", "globex")
    (tmp_path / "projects" / "globex" / project_meta.PROJECT_META_FILENAME).mkdir()
    assert _picker(tmp_path) == ["acme", "globex"]


def test_asking_about_a_node_that_is_not_on_disk_is_not_an_error(tmp_path):
    assert project_meta.is_archived("never-created", repo_root=tmp_path) is False
    assert project_meta.is_archived("../escape", repo_root=tmp_path) is False


# --- a subtree goes with its parent ------------------------------------------------------


def test_archiving_a_parent_hides_its_whole_subtree(tmp_path):
    # Hiding the parent and keeping the children is worse than not filtering:
    # `acme/onboarding` with no `acme` above it reads as a root project, and
    # the operator would file an interview under what looks like a live tree.
    _make(tmp_path, "acme/onboarding/activation-delay", "acme/billing", "globex")
    _mark(tmp_path, "acme", {"status": "archived"})
    assert _picker(tmp_path) == ["globex"]


def test_archiving_a_branch_leaves_its_siblings_and_its_parent_alone(tmp_path):
    _make(tmp_path, "acme/onboarding/activation-delay", "acme/billing")
    _mark(tmp_path, "acme/onboarding", {"status": "archived"})
    assert _picker(tmp_path) == ["acme", "acme/billing"]


def test_a_sibling_whose_name_merely_starts_with_an_archived_one_survives(tmp_path):
    # The prefix skip is on `path + "/"`, not on the bare string. Without the
    # separator, archiving `acme` would also hide `acme-corp` -- a different
    # customer entirely, hidden by a shared spelling.
    _make(tmp_path, "acme", "acme-corp", "acme-corp/billing")
    _mark(tmp_path, "acme", {"status": "archived"})
    assert _picker(tmp_path) == ["acme-corp", "acme-corp/billing"]


def test_the_emitted_order_stays_parents_before_children(tmp_path):
    # The filter must never re-sort. `list_projects` guarantees
    # parents-before-children and siblings in name order, and the console
    # renders the hierarchy by INDENTING against that order -- a filter that
    # rebuilt the list (say, by grouping the survivors) would draw children
    # above their parents while every membership test above still passed.
    _make(
        tmp_path,
        "acme/billing", "acme/onboarding/activation-delay", "globex/eu", "zulu",
        project_meta.UNFILED_PROJECT,
    )
    _mark(tmp_path, "globex", {"status": "archived"})
    assert _picker(tmp_path) == [
        "acme",
        "acme/billing",
        "acme/onboarding",
        "acme/onboarding/activation-delay",
        "zulu",
        project_meta.UNFILED_PROJECT,
    ]


# --- archiving is a picker filter, never a deletion --------------------------------------


def test_an_archived_node_still_resolves_and_still_opens(tmp_path):
    # `--project acme` must keep working on an archived node: the recordings,
    # the facts and the hypotheses are all still there, and re-opening one is
    # how it stops being archived. If this ever starts raising, archiving
    # becomes a one-way door with no key.
    _make(tmp_path, "acme/context")
    _mark(tmp_path, "acme", {"status": "archived"})
    paths = projects_mod.resolve("acme", repo_root=tmp_path)
    assert paths.context_dir == tmp_path / "projects" / "acme" / "context"
    assert projects_mod.list_projects(repo_root=tmp_path)[0].path == "acme", (
        "list_projects() itself must stay unfiltered -- the filter belongs to the picker, "
        "and anything that reads the tree for another reason must still see every node"
    )


def test_an_archived_node_still_renders_its_detail(tmp_path):
    _make(tmp_path, "acme")
    (tmp_path / "projects" / "acme" / "context").mkdir()
    (tmp_path / "projects" / "acme" / "context" / "survey.json").write_text(
        json.dumps([{"id": "q1", "order": 1, "text": "What broke?"}]), encoding="utf-8"
    )
    _mark(tmp_path, "acme", {"status": "archived"})
    detail = _node_detail("acme", repo_root=tmp_path)
    assert [item["text"] for item in detail["survey"]] == ["What broke?"]


def test_the_unfiled_bucket_cannot_be_archived_into_invisibility(tmp_path):
    # An EXPLICIT exemption, and it has to be written down: the operator chose
    # to show every node in the picker (no `_`-prefix filtering), so the
    # exemption that filtering would have given for free no longer exists. If
    # `_unfiled` could be archived there would be no way to record without a
    # project at all -- and the picker would offer no clue why.
    _make(tmp_path, "acme", project_meta.UNFILED_PROJECT)
    _mark(tmp_path, project_meta.UNFILED_PROJECT, {"status": "archived"})
    assert project_meta.is_archived(project_meta.UNFILED_PROJECT, repo_root=tmp_path) is False
    assert _picker(tmp_path) == ["acme", project_meta.UNFILED_PROJECT]


# --- both pickers apply it ---------------------------------------------------------------


def test_the_console_tree_hides_an_archived_node(tmp_path):
    _make(tmp_path, "acme", "acme/onboarding", "globex")
    _mark(tmp_path, "acme", {"status": "archived"})
    app = create_console_app(
        static_dir=STATIC_CONSOLE_DIR, on_start=lambda _s: None, repo_root=tmp_path
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        rows = client.get("/api/projects").json()
    assert [row["path"] for row in rows] == ["globex"]


def test_the_console_still_opens_an_archived_node_that_is_asked_for_by_path(tmp_path):
    # The picker hides it; the address bar does not. Same rule as `--project`.
    _make(tmp_path, "acme")
    _mark(tmp_path, "acme", {"status": "archived"})
    app = create_console_app(
        static_dir=STATIC_CONSOLE_DIR, on_start=lambda _s: None, repo_root=tmp_path
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/project/acme").status_code == 200


def _functions_calling(module_relative: str, function_name: str, callee: str) -> bool:
    tree = ast.parse((REPO_ROOT / module_relative).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == function_name):
            continue
        return any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == callee
            for inner in ast.walk(node)
        )
    raise AssertionError(f"{module_relative} has no {function_name}")


def test_the_overlay_picker_routes_through_the_same_filter():
    # The overlay's list is built against the REAL repo root, so it cannot be
    # driven from a tmp tree. What can be checked is that it does not build its
    # own list: a second, unfiltered path to the picker would show archived
    # projects on the surface the operator actually uses all day, while every
    # test above stayed green.
    assert _functions_calling("app/overlay/__main__.py", "active_projects", "for_picker")


# --- the purity `app/projects.py` depends on ---------------------------------------------


def test_the_path_resolver_still_opens_no_files():
    # Read `app/projects.py`'s own docstring: gate 19 detects its offenders
    # TEXTUALLY -- a module that names the hypothesis artifact AND contains a
    # write primitive trips it -- and that module's answer is to name no
    # filename and open nothing at all. Adding this feature's file read there
    # would have been the obvious place for it and would have started eroding
    # that argument. This is the guard that says so out loud.
    tree = ast.parse((REPO_ROOT / "app" / "projects.py").read_text(encoding="utf-8"))
    # `iterdir` is deliberately NOT in here: `list_projects` walks DIRECTORIES,
    # which is what a path resolver does. The line this guard draws is at file
    # CONTENT -- opening something and reading what is inside it.
    readers = {"open", "read_text", "read_bytes", "load", "loads", "write_text", "write_bytes"}
    found = sorted(
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr in readers)
            or (isinstance(node.func, ast.Name) and node.func.id in readers)
        )
    )
    assert found == [], (
        f"app/projects.py now touches file CONTENT ({found}) -- it is a pure path resolver, "
        "and app/project_meta.py exists precisely so it can stay one"
    )
