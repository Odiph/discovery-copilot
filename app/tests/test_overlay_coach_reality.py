"""The overlay must show coaching panels because a coach IS RUNNING, never
because of how the project happens to be filed.

**The defect, found by the 2026-08-12 audit.** `shows_coaching` was computed at
`overlay/controller.py` from `project_meta.is_unfiled(project)` -- a string,
known before the child process was spawned. So a filed project with hypotheses
loaded and no `ANTHROPIC_API_KEY` showed "NEXT QUESTION" and a resting coach
line for the entire interview, while the child printed "Coach disabled: ..." to
a terminal nobody was looking at. `UNCOACHED_TEXT`'s own comment says a panel
that promises something for a whole interview teaches the operator that the
surface lies. The guard existed and was wired to the wrong signal.

**Why it survived a test suite with 1100 tests in it.** Both sides of the seam
were tested and nothing crossed it. `test_overlay_unfiled.py` proved the
overlay renders correctly for a given `coached` flag; `test_coach_context.py`
proved `main.py` prints "Coach disabled" when the key is absent. No test asked
whether the value the overlay renders has anything to do with what the
interview did. That is the same shape as `CTRL_BREAK` (a fake process asserted
the parent's intent, no real child was ever stopped) and as the coach panel
reading `message["candidates"]` while the wire said `primary` -- the third
instance of it in this project, so the crossing test is the point of this file.

**The signal chosen, and why not the other two.** The fix rides on `hello`:
`main.py` already computes `coach_enabled`, already writes it into
`session.json`, and `hello` is re-sent on every reconnect, so a phone that
slept through the start still learns it. Parsing "Coach disabled" out of the
child's stdout was rejected -- it would make a human-readable sentence
load-bearing, and `overlay/session.py` currently drops those lines for a good
reason. `server_status` was rejected too, and this is the sharp one: its
`coach_unhealthy` input is `None` when no pipeline exists, and `main.py`'s own
docstring says `None` "never treated as unhealthy" -- so the channel that looks
like it reports coach health is structurally incapable of reporting the coach
that never started.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.overlay import model
from app.web.hub import hello_payload

REPO_ROOT = Path(__file__).resolve().parents[2]


def _hello(**overrides):
    payload = {
        "session_id": "s", "server_started_at": "t",
        "heartbeat_interval_ms": 1, "heartbeat_timeout_ms": 2,
        "reconnect_backoff_base_ms": 3, "reconnect_backoff_cap_ms": 4,
        "project": "acme", "grounding_sources": (), "coach_enabled": True,
    }
    payload.update(overrides)
    return hello_payload(**payload)


# --- the crossing test: the REAL producer against the REAL consumer ---------------------


def test_the_real_hello_a_coachless_interview_sends_turns_the_panels_off():
    """The whole point of the file. Builds the payload with the REAL
    `hello_payload` -- not a hand-written dict that could agree with the
    overlay while disagreeing with the server -- and folds it in with the REAL
    `applied`."""
    state = model.OverlayState().recording("acme")
    assert state.shows_coaching is True, "a filed project starts out expecting a coach"

    state = state.applied(_hello(coach_enabled=False))

    assert state.coach_confirmed is False
    assert state.shows_coaching is False, (
        "the overlay promised coaching for a run that has no coach -- the defect this file exists "
        "for. Note the project is still filed, so the old filing-derived signal still says True"
    )
    assert state.coaching_absence_line == model.COACH_UNAVAILABLE_TEXT


def test_the_real_hello_a_coached_interview_sends_leaves_the_panels_on():
    state = model.OverlayState().recording("acme").applied(_hello(coach_enabled=True))
    assert state.coach_confirmed is True
    assert state.shows_coaching is True


def test_hello_actually_carries_the_key_the_overlay_reads():
    """Pinned against the producer's own output rather than against the
    string. If `hello_payload` renames the key, this fails HERE, at the
    producer, instead of the overlay silently reading `None` forever and
    treating it as 'unknown' -- which is the failure mode this key's own
    absent-means-unknown rule creates."""
    assert "coach_enabled" in _hello()


# --- the two situations must not be worded the same --------------------------------------


def test_a_deliberate_no_coach_run_and_a_broken_one_say_different_things():
    """`--no-coach` is a choice that took effect. A missing key is a filed
    interview that is quietly worth less than the operator thinks. Collapsing
    them into one sentence hides the second inside the first, which is exactly
    what the four-way extraction status was criticised for on the hub."""
    deliberate = model.OverlayState().recording("_unfiled", coached=False)
    broken = model.OverlayState().recording("acme").applied(_hello(coach_enabled=False))

    assert deliberate.shows_coaching is False and broken.shows_coaching is False
    assert deliberate.coaching_absence_line != broken.coaching_absence_line
    assert deliberate.coaching_absence_line == model.UNCOACHED_TEXT
    assert broken.coaching_absence_line == model.COACH_UNAVAILABLE_TEXT


def test_an_unfiled_run_stays_uncoached_even_if_hello_claimed_otherwise():
    """Either signal saying no is enough. A `--no-coach` run cannot be talked
    back into showing panels by a stray `coach_enabled: true`, because the
    operator's own instruction is not something a wire message may overrule."""
    state = model.OverlayState().recording("_unfiled", coached=False)
    state = state.applied(_hello(coach_enabled=True))
    assert state.shows_coaching is False
    assert state.coaching_absence_line == model.UNCOACHED_TEXT


# --- unknown is a third state, and it is not "no" -----------------------------------------


def test_before_hello_arrives_the_panels_stay_up_rather_than_flickering():
    """The gap between pressing record and the socket connecting is a second
    or two. Treating unknown as "no coach" would blank the panels on every
    single run and then restore them, which reads as a fault."""
    state = model.OverlayState().recording("acme")
    assert state.coach_confirmed is None
    assert state.shows_coaching is True


def test_a_hello_with_no_coach_key_leaves_the_answer_unknown_not_false():
    """`message.get("coach_enabled")` returning `None` must not be coerced by
    `bool()`. An interview that predates the key would otherwise make the
    overlay announce a broken coach on a run whose coach is working -- a false
    alarm on the honest surface, which costs more trust than the original
    silence did."""
    state = model.OverlayState().recording("acme").applied(
        {"type": "hello", "project": "acme"}
    )
    assert state.coach_confirmed is None
    assert state.shows_coaching is True


def test_a_reconnect_re_delivers_the_answer():
    """`hello` is re-sent on every fresh connect, which is the reason this
    rides on `hello` at all. A phone or overlay that joined late must not be
    stuck at unknown."""
    state = model.OverlayState().recording("acme")
    state = state.applied(_hello(coach_enabled=False))
    state = state.applied(_hello(coach_enabled=False))
    assert state.shows_coaching is False


# --- and the end of the chain: a real key-less run reports it -----------------------------


@pytest.mark.parametrize("with_key", [False, True])
def test_a_real_key_less_run_records_coach_enabled_false_in_its_own_session(
    tmp_path, with_key, monkeypatch
):
    """The last link, run as a real subprocess. The tests above prove the
    overlay reads what `hello_payload` writes; this proves `main.py` computes
    the value honestly in the first place, against a real absent key.

    `session.json` is the artifact checked rather than the wire, because it is
    the same variable (`main.py` passes one `coach_enabled` to both) and a
    subprocess test that had to hold a WebSocket open would be flaky for no
    extra coverage. `ANTHROPIC_API_KEY` is set to the EMPTY STRING, never
    popped: this repo's dotenv loader fills in anything "not already present",
    so a popped variable is handed straight back and the test passes for the
    wrong reason -- which it did, undetected, from Phase 4 until 2026-08-10.
    """
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "brief.json").write_text(
        json.dumps([{"id": "b1", "text": "they churn at week two", "source": "hand",
                     "provenance": "evidence"}]),
        encoding="utf-8",
    )
    (context_dir / "survey.json").write_text(
        json.dumps([{"id": "q1", "order": 1, "text": "What broke?"}]), encoding="utf-8"
    )
    sessions_root = tmp_path / "sessions"

    env = dict(os.environ)
    # EMPTY, never popped. See the docstring -- a popped key is handed back by
    # the dotenv loader and the test passes for the wrong reason.
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-never-billed-0000" if with_key else ""
    # `build_client()` checks `COACH_BASE_URL` BEFORE the Anthropic key and
    # answers with an OpenAI-compatible client when one is named, so a real
    # `COACH_BASE_URL` in the developer's `.env` makes BOTH halves of this
    # parametrization lie: the `False` case gets a working coach despite the
    # blank key, and the `True` case exercises the OpenAI path while claiming
    # to test the Anthropic one. Blanked here, empty-not-popped, for the same
    # dotenv reason the key above is.
    env["COACH_BASE_URL"] = ""
    env["COACH_API_KEY"] = ""
    env["APP_CONTEXT_DIR"] = str(context_dir)
    env["APP_SESSIONS_ROOT"] = str(sessions_root)

    result = subprocess.run(
        [sys.executable, "-m", "app", "start", "--engine", "fake", "--no-serve",
         "--seconds", "1"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    (session_dir,) = [p for p in sessions_root.iterdir() if p.is_dir()]
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    assert metadata["coach_enabled"] is with_key, (
        f"a run with{'' if with_key else 'out'} a key recorded "
        f"coach_enabled={metadata['coach_enabled']!r} -- this is the value hello carries, so a "
        "wrong answer here is a wrong panel on the overlay for the whole interview"
    )
