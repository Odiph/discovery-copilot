"""Recording with no project and no coach.

The user's request: *"a default project in case I do not have an active
project yet (in this case the overlay does not have a coach and next questions
just a recording for a future project)"*.

Two things this file protects, and they are separate failures:

1. **The unfiled bucket is a REAL node.** `projects/_unfiled/` resolves
   through the existing path code with no new resolution logic, which is what
   keeps `_apply_project`, `InterviewSession.command()`, the console's session
   lister and `render` free of a null-project branch each. A synthetic
   non-node entry would need all four.
2. **The overlay must not assert a process that will never happen.** With no
   coach, no `survey` and no `suggestion` message ever arrives, so the panel
   would show "loading the survey..." and "listening..." for the whole
   interview. Both are claims about something in flight.

`window.py` needs a real tkinter root and cannot run headless, so -- exactly
as `test_overlay_gestures.py` established -- the RULE lives in `model.py` and
is tested here, and the window is checked by scanning its source.
"""

from __future__ import annotations

import ast
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

import pytest

from app import project_meta
from app import projects as projects_mod
from app.main import build_arg_parser
from app.overlay import model
from app.overlay.controller import OverlayController
from app.overlay.session import InterviewSession

REPO_ROOT = Path(__file__).resolve().parents[2]
_WINDOW_SOURCE = (REPO_ROOT / "app" / "overlay" / "window.py").read_text(encoding="utf-8")
_MAIN_SOURCE = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")


# --- the unfiled bucket is a real node --------------------------------------------------


def test_the_unfiled_node_resolves_through_the_existing_path_code(tmp_path):
    # The whole reason `_unfiled` was chosen over a synthetic picker entry: a
    # leading underscore already satisfies `_SEGMENT_PATTERN` (and
    # `projects/_eval/` is the precedent), so this needs ZERO new resolution
    # logic. If the label rule is ever tightened, the unfiled bucket becomes
    # unaddressable by `--project` while still being offered in the picker --
    # and the failure would surface as a rejected path mid-interview.
    paths = projects_mod.resolve(project_meta.UNFILED_PROJECT, repo_root=tmp_path)
    assert paths.path == project_meta.UNFILED_PROJECT
    assert paths.node_dir == tmp_path / "projects" / project_meta.UNFILED_PROJECT


def test_ensure_unfiled_creates_the_node_but_never_a_context_directory(tmp_path):
    # No `context/` on purpose. A context directory there would be a brief and
    # survey for a project that does not exist yet -- and with a key present,
    # `start` would load it and run a coach grounded in nothing.
    node_dir = project_meta.ensure_unfiled(repo_root=tmp_path)
    assert node_dir.is_dir()
    assert not (node_dir / "context").exists()


def test_ensure_unfiled_is_idempotent_and_never_clobbers_what_is_there(tmp_path):
    # It runs on every press of the record button. A second call that raised,
    # or that reset anything, would break the overlay on its second use.
    first = project_meta.ensure_unfiled(repo_root=tmp_path)
    (first / "sessions").mkdir()
    (first / "sessions" / "20260812T090000").mkdir()
    second = project_meta.ensure_unfiled(repo_root=tmp_path)
    assert second == first
    assert (first / "sessions" / "20260812T090000").is_dir()


def test_is_unfiled_recognises_only_that_exact_node():
    assert project_meta.is_unfiled("_unfiled") is True
    assert project_meta.is_unfiled("acme") is False
    # A CHILD of the bucket is not the bucket: it is a real filed project that
    # happens to live under it, and it must get a coach.
    assert project_meta.is_unfiled("_unfiled/acme") is False
    # Never raises on input that is not a project path at all -- this is asked
    # about whatever the picker returned.
    assert project_meta.is_unfiled("../etc") is False
    assert project_meta.is_unfiled("") is False


def test_the_picker_puts_the_unfiled_row_last(tmp_path):
    # `_` (0x5F) sorts BEFORE `a` (0x61), so the name-ordered walk puts
    # `_unfiled` FIRST -- at the top of the list, where it reads as the default
    # choice. The user asked for it last and unselected, and this is the guard
    # for that: it fails the moment the ordering is left to the walker.
    for relative in ("acme", "acme/onboarding", "globex", project_meta.UNFILED_PROJECT):
        (tmp_path / "projects" / relative).mkdir(parents=True)
    nodes = project_meta.for_picker(projects_mod.list_projects(repo_root=tmp_path), repo_root=tmp_path)
    assert [node.path for node in nodes] == [
        "acme", "acme/onboarding", "globex", project_meta.UNFILED_PROJECT,
    ]


# --- the overlay state carries whether there is a coach at all --------------------------


def test_a_recording_is_coached_by_default():
    # Every existing caller, and every legacy path.
    assert model.OverlayState().recording("acme").coached is True


def test_recording_does_not_silently_drop_the_coached_flag():
    # `recording()` constructs a FRESH `OverlayState` rather than `replace`-ing
    # the current one -- deliberately, so a cancelled selection cannot leak
    # into the next interview. That same construction silently drops any field
    # not passed explicitly, which would leave an uncoached run rendering the
    # coached panel and no test would notice.
    state = model.OverlayState().picking().recording("_unfiled", coached=False)
    assert state.coached is False
    assert state.phase == model.STATE_RECORDING and state.project == "_unfiled"


def test_an_uncoached_recording_shows_no_coaching_regions():
    uncoached = model.OverlayState().recording("_unfiled", coached=False)
    assert uncoached.shows_coaching is False
    coached = model.OverlayState().recording("acme")
    assert coached.shows_coaching is True


def test_the_uncoached_notice_promises_nothing_that_is_coming():
    # The two resting strings this replaces both assert an ongoing process:
    # "loading the survey..." and "listening...". Neither will ever resolve in
    # an uncoached run, so the notice must not be a third one of those.
    assert "..." not in model.UNCOACHED_TEXT, model.UNCOACHED_TEXT
    assert model.UNCOACHED_TEXT != model.QUESTION_RESTING_TEXT
    assert model.UNCOACHED_TEXT != model.COACH_RESTING_TEXT


def test_the_coach_unavailable_notice_is_held_to_the_same_bar():
    # Extended when the second notice arrived. The original version of this
    # test named ONE constant, so a second absence string could have been
    # added as a third "loading..." and nothing would have objected -- the
    # scan-shaped blind spot, in a test about wording.
    assert "..." not in model.COACH_UNAVAILABLE_TEXT, model.COACH_UNAVAILABLE_TEXT
    assert model.COACH_UNAVAILABLE_TEXT != model.QUESTION_RESTING_TEXT
    assert model.COACH_UNAVAILABLE_TEXT != model.COACH_RESTING_TEXT
    assert model.COACH_UNAVAILABLE_TEXT != model.UNCOACHED_TEXT, (
        "a coach that was asked for and did not come up reads as a deliberate no-coach run"
    )


def _question_line_reads_outside_the_coaching_guard() -> "list[int]":
    """Every `state.question_line` / `state.coach_line` read in `window.py`
    that is NOT inside an `if <something>.shows_coaching:` body.

    A source scan for the reason `test_overlay_gestures.py` gives: the window
    cannot be constructed without a display, so the only way to check that the
    panel is actually guarded is to read the file. AST rather than a regex,
    per this project's own S13 precedent."""
    tree = ast.parse(_WINDOW_SOURCE)
    guarded: "set[int]" = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Attribute) and node.test.attr == "shows_coaching"):
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                guarded.add(id(inner))
    offenders: "list[int]" = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("question_line", "coach_line"):
            if id(node) not in guarded:
                offenders.append(node.lineno)
    return offenders


def test_the_window_draws_the_next_question_panel_only_when_there_is_a_coach():
    # The sneaky regression is not deleting the guard -- it is drawing the
    # panel with a different resting string ("no coach"), which keeps the
    # region on screen and keeps every model test green. The panel has to be
    # ABSENT: an empty box where the next question goes trains the eye to skip
    # the place the question will appear once there IS one.
    assert _question_line_reads_outside_the_coaching_guard() == [], (
        "window.py reads the coached panel text outside the `shows_coaching` guard -- an "
        "uncoached interview would show 'loading the survey...' for its whole duration"
    )
    assert "coaching_absence_line" in _WINDOW_SOURCE, (
        "the uncoached recording draws nothing in place of the panel -- say what this run IS. "
        "The SENTENCE is chosen in model.py: there are two reasons for no coaching, and "
        "picking between them at the point of drawing is how they collapse back into one"
    )


def test_the_scan_finds_an_unguarded_read_when_there_is_one():
    # The test above passes when the scan finds nothing, which is also what a
    # broken scan returns (gate 17/20/30's shape, applied to the walk).
    tree = ast.parse(
        "def draw(state):\n"
        "    label(state.question_line)\n"
        "    if state.shows_coaching:\n"
        "        label(state.coach_line)\n"
    )
    guarded: "set[int]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute) and node.test.attr == "shows_coaching":
            for statement in node.body:
                for inner in ast.walk(statement):
                    guarded.add(id(inner))
    unguarded = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in ("question_line", "coach_line")
        and id(node) not in guarded
    ]
    assert unguarded == ["question_line"]


# --- the child is told there is no coach ------------------------------------------------


def test_the_command_line_asks_for_no_coach_only_when_uncoached():
    coached = InterviewSession("acme", events=queue.Queue()).command()
    assert "--no-coach" not in coached

    uncoached = InterviewSession("_unfiled", events=queue.Queue(), coached=False).command()
    assert "--no-coach" in uncoached
    # Still filed under the real node -- `--project` is NOT dropped. A run with
    # no `--project` would write `"project": null` into session.json, which is
    # indistinguishable from a legacy pre-project recording, and filing it
    # later would stop being a rename.
    assert uncoached[uncoached.index("--project") + 1] == "_unfiled"


def test_choosing_the_unfiled_bucket_starts_an_uncoached_interview():
    started: "list[tuple[str, bool]]" = []

    class _Window:
        def set_projects(self, projects): ...
        def render(self, state): ...

    # `**_` swallows keywords this double does not care about (e.g.
    # `loopback_only`). The controller's own contract with the real
    # factory is pinned in test_overlay_capture_mode.py; a double that
    # had to be edited for every new keyword would just get **_ added
    # by whoever hit it first, so it may as well say why.
    def _start(project, events, *, coached=True, **_):
        started.append((project, coached))
        return object()

    controller = OverlayController(
        window=_Window(), list_projects=lambda: ("acme", "_unfiled"), start_session=_start
    )
    controller.on_project_chosen("_unfiled")
    assert started == [("_unfiled", False)]
    assert controller.state.coached is False

    controller.on_project_chosen("acme")
    assert started[-1] == ("acme", True)
    assert controller.state.coached is True


# --- `start --no-coach` ------------------------------------------------------------------


def test_start_accepts_no_coach_and_it_defaults_off():
    assert build_arg_parser().parse_args(["start"]).no_coach is False
    assert build_arg_parser().parse_args(["start", "--no-coach"]).no_coach is True


def _coach_construction_guards() -> "set[str]":
    """The names of the `if` tests enclosing every `CoverageDetector(...)` and
    `SuggestionPipeline(...)` construction in `app/main.py`."""
    tree = ast.parse(_MAIN_SOURCE)
    constructed = {"CoverageDetector", "SuggestionPipeline"}
    guards: "set[str]" = set()
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name):
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in constructed
                ):
                    guards.add(node.test.id)
                    found += 1
    assert found == 2, f"expected both coach threads to be constructed exactly once, found {found}"
    return guards


def test_both_coach_threads_are_constructed_only_under_the_coach_enabled_flag():
    # `coach_enabled` is the one thing `--no-coach` sets, so this is what makes
    # "no coach means neither thread" structural rather than a hope. A
    # construction moved out of the branch (or guarded by a second, different
    # flag) fails here.
    assert _coach_construction_guards() == {"coach_enabled"}


@pytest.mark.parametrize("flagged", [True, False])
def test_a_coachless_run_loads_no_context_and_starts_no_coach_thread(tmp_path, flagged):
    # Driven with a REAL-shaped key and a VALID context dir on purpose: the
    # degrade path already covered by test_coach_context.py is "no key", and a
    # `--no-coach` implementation that merely tolerated a missing brief would
    # pass that one while still loading a brief here. This is the case that
    # separates them.
    #
    # `flagged=False` is the control: the same directory, the same key, no
    # flag -- the coach must come up. Without it, an implementation that
    # simply never enables the coach would look correct.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "brief.json").write_text(
        json.dumps([{"id": "b1", "text": "they churn at week two", "source": "hand", "provenance": "evidence"}]),
        encoding="utf-8",
    )
    (context_dir / "survey.json").write_text(
        json.dumps([{"id": "q1", "order": 1, "text": "What broke?"}]), encoding="utf-8"
    )

    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-never-billed-0000"
    env["APP_CONTEXT_DIR"] = str(context_dir)
    env["APP_SESSIONS_ROOT"] = str(tmp_path / "sessions")
    argv = [sys.executable, "-m", "app", "start", "--engine", "fake", "--no-serve", "--seconds", "1"]
    if flagged:
        argv.append("--no-coach")

    result = subprocess.run(
        argv, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr

    (session_dir,) = [p for p in (tmp_path / "sessions").iterdir() if p.is_dir()]
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    if not flagged:
        assert metadata["coach_enabled"] is True
        assert metadata["context_survey"], "the control run did not load the survey"
        assert "coaching_latency_summary" in metadata
        return

    assert metadata["coach_enabled"] is False
    # Never LOADED, not merely never used: a brief read here is a previous
    # customer's material copied into an unfiled recording's provenance.
    assert metadata["context_brief"] == []
    assert metadata["context_survey"] == []
    # Written if and only if a `SuggestionPipeline` was constructed.
    assert "coaching_latency_summary" not in metadata


def test_the_no_coach_notice_is_distinguishable_from_the_missing_key_degrade(tmp_path):
    # An existing test greps stdout for "Coach disabled" to prove the no-key
    # degrade path ran. If `--no-coach` printed the same words, that test would
    # pass on a run that never even looked for a key -- and the operator could
    # not tell "I turned the coach off" from "your key is missing", which are
    # different problems with different fixes.
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-never-billed-0000"
    env["APP_CONTEXT_DIR"] = str(tmp_path / "nonexistent-context")
    env["APP_SESSIONS_ROOT"] = str(tmp_path / "sessions")

    result = subprocess.run(
        [sys.executable, "-m", "app", "start", "--engine", "fake", "--no-serve",
         "--seconds", "1", "--no-coach"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Coach disabled" not in result.stdout
    assert "Coach enabled" not in result.stdout
    assert "no-coach" in result.stdout, result.stdout
