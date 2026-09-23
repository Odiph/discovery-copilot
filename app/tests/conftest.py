"""Test-suite-wide fixtures.

**One job, and it is a hermeticity job: a developer's `.env` must never change
a test result.**

`app/config.py::_load_dotenv_once` reads the real `.env` and fills in any
variable "not already present". It is called from `load_config()`, so the
moment any test in a run touches configuration, every credential in that file
is in `os.environ` for the rest of the process -- and for every subprocess a
test spawns, since those inherit it. Nothing resets it between tests.

That has now silently disarmed the same class of test three separate times, and
each time it was diagnosed as a one-off:

- Phase 4 (`e4c883c`): a key-absent test popped `ANTHROPIC_API_KEY`, the loader
  handed it straight back, and the test passed for the wrong reason for months.
- 2026-08-09: the same shape again, recorded in `test_coach_context.py`'s own
  docstring, fixed by setting the variable EMPTY rather than popping it.
- 2026-08-14: `ce1f5bb` gave `build_client()` a SECOND route to a client --
  `COACH_BASE_URL` wins before the Anthropic key is even read -- so the day an
  OpenAI-compatible endpoint appeared in `.env`, four tests broke at once. Two
  asserted a coach DISABLED without a key and got an enabled one; gate 13
  asserted `client.api_key == dummy` and got an OpenAI client holding a
  different key entirely; and a parametrized test's `with_key=True` half was
  quietly exercising the OpenAI path while claiming to test the Anthropic one.

The pattern is not "somebody forgot". It is that the number of routes to a
client grows, and every existing test that closes route A stays green while
silently stopping testing anything the day route B appears. Blanking per-test
scales with the number of tests; blanking here scales with the number of
routes, and there is one place to update when a third arrives.

**Blanked, never popped** -- an absent variable is exactly what the loader
refills. An empty one is "already present", so the loader's own
"the environment always wins" rule keeps `.env` out, and
`build_client()`'s falsy checks still take the no-credential branch.

**Deliberately NOT blanked here: `ANTHROPIC_API_KEY`.** Existing tests already
handle it explicitly and several set a dummy value they then assert on; taking
it over globally would silently change what those tests mean. The seam
variables below are new, no test predating `ce1f5bb` knows they exist, and none
asserts on them -- which is what makes them safe to own here.

A test that WANTS the OpenAI-compatible seam sets these itself with
`monkeypatch.setenv`; monkeypatch applies after this fixture and wins.
"""

from __future__ import annotations

import pytest

# Every environment variable that can route `reasoning_client.build_client()`
# away from the plain Anthropic path. Grow this list when a route is added --
# that is the whole point of it being a list in one file.
_SEAM_VARIABLES = ("COACH_BASE_URL", "COACH_API_KEY", "COACH_MODEL")


@pytest.fixture(autouse=True)
def _no_ambient_coach_seam(monkeypatch):
    """Neutralise the OpenAI-compatible seam for every test, unconditionally.

    Autouse and unconditional on purpose. An opt-in fixture would be applied by
    exactly the tests whose authors already knew about the problem, which is
    the set that does not need it."""
    for name in _SEAM_VARIABLES:
        monkeypatch.setenv(name, "")
