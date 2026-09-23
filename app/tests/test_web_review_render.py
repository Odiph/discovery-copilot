"""The hub's CLIENT, driven for real against a minimal DOM.

Every other guard on `review.js` in this repo is a source assertion -- a grep
for `innerHTML`, a grep for a field name. A grep cannot tell the difference
between a page that computes a delta and a page that computes a delta and then
draws a running total, which is precisely the defect this workstream exists to
fix: `standing_for` already tagged every entry `from_this_interview` and the
client rendered one bold font weight from it.

So the real `review.js` is executed by node against a hand-written DOM shim and
the assertions are made on what a human would READ. The test bodies live in
`app/tests/js/review_render.test.mjs`; this module exists so they run inside the
ordinary pytest run rather than out of band.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_SUITE = Path(__file__).parent / "js" / "review_render.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_hub_client_renders_the_change_this_interview_made():
    result = subprocess.run(  # noqa: S603 -- a fixed argv, no shell, no user input
        [shutil.which("node"), "--test", str(_SUITE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_client_suite_is_not_quietly_empty():
    """The mutation this file cannot survive without. A suite that stopped
    exporting tests -- a bad path, a renamed asset, a `node --test` that
    matched nothing -- exits 0 and would make the guard above pass forever."""
    assert _SUITE.is_file()
    source = _SUITE.read_text(encoding="utf-8")
    assert source.count("\ntest(") >= 8, "the client suite lost its tests"
