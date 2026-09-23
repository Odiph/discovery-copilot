"""A node has to say whether its claims describe real people.

**The one-way door.** `Fact` carries exactly seven fields and none of them
answers this. `provenance` is evidence-vs-inference -- how firmly a claim
follows from what was said -- and is silent on whether anyone said it. So a
hand-authored demonstration claim and a customer's own words are, in the store
and on every surface that reads it, the same shape. That is harmless only while
every project is fake, which is true today and stops being true at the first
real interview; afterwards the two kinds are interleaved in one file with no
field to separate them and no timestamp to separate them by.

The declaration lives in `project.json`, which `is_archived` already reads and
which gate 29 keeps uncommittable -- whether a corpus is a demo is one
operator's statement about their own tree, not a fact about the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import project_meta


def make_node(root: Path, path: str, meta: "dict | str | None") -> None:
    node = root / "projects" / Path(path)
    node.mkdir(parents=True, exist_ok=True)
    if meta is None:
        return
    text = meta if isinstance(meta, str) else json.dumps(meta)
    (node / "project.json").write_text(text, encoding="utf-8")


def test_a_node_that_declares_demo_reads_as_demo(tmp_path: Path) -> None:
    make_node(tmp_path, "demo-claudictive", {"corpus": "demo"})
    assert project_meta.corpus_kind("demo-claudictive", repo_root=tmp_path) == "demo"


def test_a_node_that_declares_real_reads_as_real(tmp_path: Path) -> None:
    make_node(tmp_path, "acme", {"corpus": "real"})
    assert project_meta.corpus_kind("acme", repo_root=tmp_path) == "real"


def test_saying_nothing_is_undeclared_and_NOT_real(tmp_path: Path) -> None:
    """The whole point of a third state.

    Defaulting an undeclared node to real is the exact misreading this field
    exists to prevent -- it would let a demo corpus nobody got round to
    labelling be reported as customer research. Defaulting it to demo would
    label every genuine project until someone remembered otherwise, which
    teaches a reader to ignore the marker. "Nobody said" is the true answer."""
    make_node(tmp_path, "acme", None)
    assert project_meta.corpus_kind("acme", repo_root=tmp_path) == "undeclared"
    make_node(tmp_path, "beta", {"status": "active", "label": "Beta"})
    assert project_meta.corpus_kind("beta", repo_root=tmp_path) == "undeclared"


@pytest.mark.parametrize("declared", ["Demo", "DEMO", "fixture", "synthetic", "fake", "true", ""])
def test_a_near_miss_is_undeclared_never_real(tmp_path: Path, declared: str) -> None:
    """Someone writing `"corpus": "fixture"` is TRYING to say this is not real
    and spelling it wrong. Reading that as real would convert a near-miss into
    the one misreading that matters, so an unrecognised value is treated as
    nothing having been successfully declared."""
    make_node(tmp_path, "acme", {"corpus": declared})
    assert project_meta.corpus_kind("acme", repo_root=tmp_path) == "undeclared"


@pytest.mark.parametrize("declared", [True, 1, ["demo"], {"kind": "demo"}, None])
def test_a_non_string_declaration_is_undeclared(tmp_path: Path, declared) -> None:
    make_node(tmp_path, "acme", {"corpus": declared})
    assert project_meta.corpus_kind("acme", repo_root=tmp_path) == "undeclared"


@pytest.mark.parametrize("meta", ["{not json", "[]", '"a string"', "null"])
def test_an_unreadable_project_json_never_raises(tmp_path: Path, meta: str) -> None:
    """Asked about whatever the walker found on disk, exactly like `is_archived`.
    A broken file in one node must not take down a tool that was asked about
    the whole tree."""
    make_node(tmp_path, "acme", meta)
    assert project_meta.corpus_kind("acme", repo_root=tmp_path) == "undeclared"


def test_a_path_that_is_not_a_project_is_undeclared_not_an_error(tmp_path: Path) -> None:
    assert project_meta.corpus_kind("../escape", repo_root=tmp_path) == "undeclared"
    assert project_meta.corpus_kind("", repo_root=tmp_path) == "undeclared"


def test_the_declaration_is_per_node_and_does_not_leak_to_a_sibling(tmp_path: Path) -> None:
    """`demo-acme` must not answer for `demo-acme-corp`, the same prefix trap
    `for_picker` documents for archiving."""
    make_node(tmp_path, "demo-acme", {"corpus": "demo"})
    make_node(tmp_path, "demo-acme-corp", {"corpus": "real"})
    assert project_meta.corpus_kind("demo-acme", repo_root=tmp_path) == "demo"
    assert project_meta.corpus_kind("demo-acme-corp", repo_root=tmp_path) == "real"


def test_archiving_still_works_after_the_reader_was_generalised(tmp_path: Path) -> None:
    """`_declared_status` became a thin wrapper over a key-agnostic reader.
    Pinned here because the refactor is invisible at its own call sites: a
    regression would silently un-archive every archived node rather than fail
    loudly."""
    make_node(tmp_path, "old", {"status": "archived"})
    make_node(tmp_path, "live", {"status": "active"})
    make_node(tmp_path, "both", {"status": "archived", "corpus": "demo"})
    assert project_meta.is_archived("old", repo_root=tmp_path) is True
    assert project_meta.is_archived("live", repo_root=tmp_path) is False
    assert project_meta.is_archived("both", repo_root=tmp_path) is True
    assert project_meta.corpus_kind("both", repo_root=tmp_path) == "demo"


def test_the_two_keys_are_independent(tmp_path: Path) -> None:
    """Declaring a corpus must not archive anything, and vice versa."""
    make_node(tmp_path, "demo-only", {"corpus": "demo"})
    assert project_meta.is_archived("demo-only", repo_root=tmp_path) is False
    make_node(tmp_path, "archived-only", {"status": "archived"})
    assert project_meta.corpus_kind("archived-only", repo_root=tmp_path) == "undeclared"


def test_the_kind_set_is_closed(tmp_path: Path) -> None:
    assert project_meta.ALL_CORPUS_KINDS == {"demo", "real", "undeclared"}
