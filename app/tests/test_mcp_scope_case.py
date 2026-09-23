"""A subtree query must not lose descendants to a difference in CASE.

`resolve_scope`'s own docstring names the escape it was designed against -- a
sibling sharing a name prefix, where `"acme-corp".startswith("acme")` is True --
and defends against it with a separator-terminated prefix. That guard is sound.

It is also aimed at OVER-inclusion, and the reachable defect here is the
opposite one. Windows filesystems are case-insensitive: `projects/ACME` and
`projects/acme` are the same directory. So a subtree query spelled `ACME`:

  - reads the ROOT node fine, because `Path.open` does not care about case, and
  - silently drops every DESCENDANT, because membership was a case-sensitive
    string comparison of `"acme/onboarding"` against the prefix `"ACME/"`.

The call succeeds. `error` is `None`. The caller receives a PARTIAL subtree
presented as a complete one. That is worse than an exception in exactly the way
this product cares about most: a rollup that under-reports says "no contrary
evidence" when contrary evidence exists, which is the confirmation failure the
whole `Bearing` design exists to prevent.

`os.path.normcase` is the right primitive rather than `.lower()`: it is
identity on POSIX and case-folding on Windows, so the comparison matches
whatever the underlying filesystem actually does instead of guessing.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from app.mcp import tools


def _tree() -> Path:
    root = Path(tempfile.mkdtemp())
    for node, hypothesis_id in (("acme", "h-root"), ("acme/onboarding", "h-child")):
        context = root / "projects" / Path(node) / "context"
        context.mkdir(parents=True, exist_ok=True)
        (context / "hypotheses.json").write_text(
            json.dumps([{"id": hypothesis_id, "statement": node, "falsifier": "f"}]),
            encoding="utf-8",
        )
    return root


def _ids(project: str, root: Path) -> "list[str]":
    result = tools.call_tool(
        "get_hypotheses", {"project": project, "scope": "subtree"}, repo_root=root
    )
    assert result.get("error") is None, result.get("error")
    return sorted(h["id"] for h in result["hypotheses"])


def test_a_subtree_query_finds_the_same_nodes_however_the_root_is_cased():
    # The exact-case spelling is the control: if this ever stops returning both,
    # the test below is passing for the wrong reason.
    root = _tree()
    assert _ids("acme", root) == ["h-child", "h-root"]


@pytest.mark.skipif(
    os.path.normcase("A") == "A",
    reason="only a case-insensitive filesystem can address one node by two spellings",
)
@pytest.mark.parametrize("spelling", ["ACME", "AcMe"])
def test_a_differently_cased_root_does_not_silently_drop_its_descendants(spelling):
    # The defect. Before the fix this returned ['h-root'] alone, with no error:
    # the root's own file opened (the filesystem folded the case) while every
    # descendant failed a case-sensitive prefix test.
    root = _tree()
    assert _ids(spelling, root) == ["h-child", "h-root"], (
        "a subtree query returned a PARTIAL tree and reported no error -- an "
        "under-reported rollup is a confirmation machine"
    )


@pytest.mark.skipif(os.path.normcase("A") == "A", reason="case-insensitive filesystems only")
def test_case_folding_the_membership_test_does_not_readmit_a_namesake_sibling():
    # The mutation guarding the fix. Folding case on both sides must not weaken
    # the separator-terminated prefix that keeps `acme-corp` -- a different
    # customer -- out of `acme`'s subtree. Fixing under-inclusion by loosening
    # the test into over-inclusion would trade a silent omission for a silent
    # LEAK ACROSS CUSTOMERS, which is strictly worse.
    root = _tree()
    context = root / "projects" / "acme-corp" / "context"
    context.mkdir(parents=True, exist_ok=True)
    (context / "hypotheses.json").write_text(
        json.dumps([{"id": "h-namesake", "statement": "other customer", "falsifier": "f"}]),
        encoding="utf-8",
    )
    for spelling in ("acme", "ACME"):
        assert "h-namesake" not in _ids(spelling, root), (
            f"{spelling!r}'s subtree pulled in the namesake sibling acme-corp"
        )
