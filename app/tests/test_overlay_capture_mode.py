"""A one-sided recording must not open the microphone at all.

**Measured, not hypothesised.** On 2026-08-12 the first real overlay recording
ever made was a Hebrew/English podcast played through SPEAKERS. Both tracks
captured it: **15 of 21 microphone segments were near-duplicates of a loopback
segment about 0.35s earlier** -- speaker, air, webcam mic -- so 71% of the
"interviewer" track was the podcast, attributed to a person who had not spoken.

Two consequences, and the second is worse than the duplicate text:

1. Fact extraction reads the interviewee side, so the claims are fine, but any
   later "who said what" is nonsense.
2. `CoverageDetector` is fed the INTERVIEWER stream and marks a survey question
   ASKED when it hears the interviewer ask it. Podcast speech arriving on that
   stream can therefore tick off questions nobody asked -- and a coverage match
   is irreversible for the rest of the run.

Headphones prevent it. Not opening the microphone prevents it in a way that
cannot be forgotten at the moment of pressing record, which is why the overlay
now offers it: `--only interviewee` already existed in the CLI and the overlay's
command builder simply never passed it.

**What this file will not let regress.** `--only` must be OFF by default -- a
default that quietly dropped the interviewer would silently halve every real
interview, which is a far worse failure than the one being fixed.
"""

from __future__ import annotations

import ast
import inspect
import queue
from pathlib import Path

import pytest

from app.overlay import model
from app.overlay.controller import OverlayController
from app.overlay.session import InterviewSession


class _Window:
    """The smallest thing the controller talks to."""

    def __init__(self) -> None:
        self.projects: "tuple[str, ...]" = ()
        self.rendered: "list[model.OverlayState]" = []

    def set_projects(self, projects) -> None:
        self.projects = tuple(projects)

    def render(self, state) -> None:
        self.rendered.append(state)


class _Session:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def stop(self) -> None:
        pass


def _controller(recorder: "list[dict]"):
    def _start(project, events, **kwargs):
        recorder.append({"project": project, **kwargs})
        return _Session(**kwargs)

    return OverlayController(
        window=_Window(), list_projects=lambda: ("acme",), start_session=_start
    )


# --- the command line ---------------------------------------------------------------------


def test_a_loopback_only_session_passes_only_interviewee():
    session = InterviewSession("acme", events=queue.Queue(), loopback_only=True)
    assert "--only" in session.command()
    assert session.command()[session.command().index("--only") + 1] == "interviewee"


def test_an_ordinary_session_passes_no_only_flag():
    """The control, and the one that matters most. A `--only` that leaked into
    every run would drop the interviewer from real interviews -- silently
    halving them, and disabling survey coverage, while every test about
    one-sided recording stayed green."""
    session = InterviewSession("acme", events=queue.Queue())
    assert "--only" not in session.command()


def test_loopback_only_is_off_unless_asked_for():
    assert InterviewSession("acme", events=queue.Queue()).loopback_only is False


def test_the_flag_the_overlay_sends_is_one_the_cli_accepts():
    """THE CROSSING TEST. Both halves of this seam are otherwise checked
    against themselves: the session builds an argv, and `start` parses one, and
    nothing compares them. `--only interviewee` is a string in two files.

    Parsed with the REAL parser rather than asserted against a literal -- a
    renamed choice or a renamed flag fails here instead of at the moment an
    operator presses record."""
    from app.main import build_arg_parser

    session = InterviewSession("acme", events=queue.Queue(), loopback_only=True)
    argv = session.command()[3:]  # drop python, -u, -m
    parsed = build_arg_parser().parse_args(argv[1:])  # drop "app"
    assert parsed.only == "interviewee"


# --- the controller threads it to BOTH the child and the state -----------------------------


def test_choosing_a_project_in_one_sided_mode_starts_a_one_sided_interview():
    started: "list[dict]" = []
    controller = _controller(started)
    controller.on_project_chosen("acme", loopback_only=True)

    assert started == [{"project": "acme", "coached": True, "loopback_only": True}]
    assert controller.state.loopback_only is True


def test_the_ordinary_path_stays_two_sided():
    started: "list[dict]" = []
    controller = _controller(started)
    controller.on_project_chosen("acme")

    assert started == [{"project": "acme", "coached": True, "loopback_only": False}]
    assert controller.state.loopback_only is False


def test_the_child_and_the_state_cannot_disagree():
    """Two independent derivations of one fact is how a panel ends up
    describing a recording that is not the one running -- the defect that made
    the overlay claim coaching on a key-less run. One value, threaded to both."""
    started: "list[dict]" = []
    controller = _controller(started)
    controller.on_project_chosen("acme", loopback_only=True)
    assert started[0]["loopback_only"] == controller.state.loopback_only


def test_the_real_start_factory_accepts_what_the_controller_sends():
    """The doubles above accept `**kwargs` and would swallow a keyword the
    REAL factory rejects -- which is exactly how a green suite ships a
    TypeError at the moment someone presses record. Checked against the real
    signature."""
    signature = inspect.signature(InterviewSession.__init__)
    for name in ("coached", "loopback_only"):
        assert name in signature.parameters, f"InterviewSession does not accept {name!r}"


# --- and the panel tells the truth about it ------------------------------------------------


def test_the_survey_panel_says_it_cannot_advance_without_a_microphone():
    """`CoverageDetector` reads the INTERVIEWER stream only, so a one-sided
    recording can never tick a question off. Left alone the panel would show
    question one for the whole recording and look live doing it -- the same
    lie `UNCOACHED_TEXT` exists to prevent, in the panel next door."""
    state = model.OverlayState().recording("acme", loopback_only=True)
    assert state.survey_can_advance is False
    assert state.question_line == model.QUESTION_NO_MIC_TEXT


def test_an_ordinary_recording_still_shows_the_survey_resting_state():
    state = model.OverlayState().recording("acme")
    assert state.survey_can_advance is True
    assert state.question_line == model.QUESTION_RESTING_TEXT


def test_the_no_microphone_notice_promises_nothing_that_is_coming():
    """Held to the same bar as the other two absence strings: it must not read
    as a process in flight, because nothing is in flight."""
    assert "..." not in model.QUESTION_NO_MIC_TEXT
    assert model.QUESTION_NO_MIC_TEXT != model.QUESTION_RESTING_TEXT
    assert model.QUESTION_NO_MIC_TEXT != model.UNCOACHED_TEXT


def test_the_coach_is_untouched_by_one_sided_capture():
    """Suggestions are generated from the INTERVIEWEE stream -- the one stream
    a one-sided recording does have. Turning the coach off here would remove
    the only half that still works."""
    state = model.OverlayState().recording("acme", loopback_only=True)
    assert state.shows_coaching is True


# --- the PRE-FLIGHT control is gone, and must not come back --------------------------------
#
# The checkbox that used to sit above the project list worked exactly as built,
# and the operator missed it on two consecutive recordings anyway -- shipping
# 67% bleed both times. Its replacement is `_draw_microphone`, on the recording
# panel, visible for the whole interview. Keeping both left ONE concept with TWO
# controls, which is the arrangement that caused the confusion rather than a fix
# for it. Removed on the user's explicit decision, 2026-08-13.
#
# The plumbing below it is deliberately UNTOUCHED: `--only interviewee` is still
# a CLI flag, `InterviewSession` still passes it, and every test above still
# pins that path. What was removed is the second CONTROL, not the capability.

_WINDOW_SOURCE = (Path(__file__).resolve().parents[1] / "overlay" / "window.py").read_text(
    encoding="utf-8"
)


def test_the_window_no_longer_draws_a_pre_flight_capture_mode_control():
    """By AST, not by string search: a test grepping for "system audio only"
    would pass the moment someone re-added the same control under a different
    label, which is precisely how a removed thing comes back."""
    tree = ast.parse(_WINDOW_SOURCE)
    methods = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_draw_capture_mode" not in methods
    assert "_toggle_loopback_only" not in methods


def test_the_window_holds_no_capture_mode_state_of_its_own():
    """The bug class this closes is a control that is invisible but still
    ACTIVE -- state left behind after its widget is deleted, still threaded into
    the child, so a run is one-sided for a reason nothing on screen explains."""
    assert "_loopback_only" not in _WINDOW_SOURCE
    assert "loopback_only" not in _WINDOW_SOURCE


def test_the_picker_starts_a_two_sided_interview():
    """THE CROSSING TEST, and the reason this file is not just two AST scans.

    The window now calls `self._on_project_chosen(p)` with ONE argument. The
    controller's `loopback_only` is keyword-with-a-default, so that call is
    legal -- but nothing else in this suite compares the two sides, and this
    project's defining defect is exactly that shape: a test proving intent on
    one side of a seam while the other side never runs. Drop the default from
    `OverlayController.on_project_chosen` and this fails here, rather than
    raising TypeError under an operator's finger at the moment they press a
    project."""
    started: "list[dict]" = []
    controller = _controller(started)
    controller.on_project_chosen("acme")  # exactly the window's call shape

    assert started == [{"project": "acme", "coached": True, "loopback_only": False}]
    assert controller.state.loopback_only is False
    assert controller.state.mic_on is True, "an interview must begin with a live microphone"


@pytest.mark.parametrize("loopback_only", [True, False])
def test_recording_never_silently_drops_the_capture_mode(loopback_only):
    """`recording()` builds a FRESH state, so any field not named on that line
    resets to its default. `coached` was nearly lost this way; this is the same
    trap one field over."""
    state = model.OverlayState().picking().recording("acme", loopback_only=loopback_only)
    assert state.loopback_only is loopback_only
    assert state.phase == model.STATE_RECORDING
