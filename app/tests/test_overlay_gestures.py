"""Starting and stopping an interview must both be deliberate.

The overlay floats above the call window, always on top, so a stray click
lands on IT rather than on whatever was behind it. Two consequences the user
asked to be designed against: a misclick that begins a recording, and -- much
worse -- a misclick that ends one, because `InterviewSession.stop()` closes the
session and the customer is still talking.

`window.py` needs a real tkinter root and cannot run headless, so the RULE
lives in `model.py` and is tested here; the window's job is only to call it.
The same split `model.py` already uses for everything else about what to show.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from app.overlay import model

_WINDOW_SOURCE = (Path(__file__).resolve().parents[1] / "overlay" / "window.py").read_text(
    encoding="utf-8"
)


@pytest.mark.parametrize("phase", [model.STATE_RECORDING, model.STATE_STOPPING])
def test_escape_alone_may_not_quit_while_an_interview_is_live(phase):
    # Escape is a reflex key -- far easier to hit by accident than the control
    # the Ctrl guard protects. Guarding the click and leaving Escape open would
    # defeat the guard entirely, because quitting mid-interview stops the
    # recording on its way out (`__main__.py::_quit`).
    assert model.quit_is_allowed(phase) is False


@pytest.mark.parametrize("phase", [model.STATE_IDLE, model.STATE_PICKING])
def test_escape_still_quits_when_nothing_is_being_recorded(phase):
    # The over-correction. A borderless window has no close button, so if
    # Escape stopped working outside a recording the overlay would have no
    # ordinary way to be dismissed at all.
    assert model.quit_is_allowed(phase) is True


def test_the_guard_covers_stopping_not_only_recording():
    # `stopping` is the window between pressing STOP and the child exiting --
    # up to GRACEFUL_STOP_TIMEOUT_S, twenty seconds, which is plenty of time
    # to hit Escape. A guard written only against `recording` would be open
    # for the whole shutdown, which is exactly when the store is being flushed.
    assert model.quit_is_allowed("stopping") is False


def test_every_hint_names_the_modifier_it_wants():
    # A guarded control that does nothing reads as a broken one. Each hint has
    # to say what was expected, or the operator learns "the overlay is
    # unreliable" rather than "this needs Ctrl".
    for hint in (
        model.RECORD_HINT_TEXT,
        model.STOP_HINT_TEXT,
        model.QUIT_WHILE_RECORDING_HINT_TEXT,
    ):
        assert "Ctrl" in hint, hint


def test_no_gesture_that_starts_or_stops_is_bound_to_an_unmodified_click():
    # The forcing function, and the reason it is a source scan: the binding
    # table is the thing that can silently regress, and it cannot be inspected
    # without a display. Every `<Button-1>` binding in window.py must be
    # dragging, picking, or a hint -- never a start or a stop.
    #
    # The sneaky regression this catches is not deleting the guard; it is
    # ADDING a second, convenient binding beside it ("also allow a plain click
    # when the window has focus"), which leaves the Control binding in place
    # and every test about it passing.
    unmodified = re.findall(r'bind\(\s*"<Button-1>"\s*,\s*(.+)', _WINDOW_SOURCE)
    for handler in unmodified:
        assert "_on_record_pressed" not in handler, handler
        assert "_on_stop_pressed" not in handler, handler
    assert '"<Control-Button-1>"' in _WINDOW_SOURCE, (
        "the armed binding disappeared -- the controls would now be unclickable"
    )


def _idle_drawn_strings() -> "list[str]":
    """Every string LITERAL `_draw_idle` puts on screen.

    Literals, via AST, not a text slice of the function -- mutation-tested and
    the first version failed: deleting the hint from the label left this
    file's own explanatory COMMENT about Escape sitting in the scanned text,
    and the guard passed over a surface that said nothing. The S13 precedent
    (a regex matching a module's own prose) with the identical shape."""
    tree = ast.parse(_WINDOW_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_draw_idle":
            return [
                inner.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            ]
    raise AssertionError("window.py no longer has a _draw_idle")


def test_the_idle_surface_says_how_to_close_the_overlay():
    # `overrideredirect(True)` removes the title bar and with it the close
    # button, so Escape is the ONLY way out of this window -- and it appeared
    # nowhere on screen, which is a control that exists and cannot be found.
    # Scanned rather than rendered, for this file's usual reason.
    assert any("Esc" in text for text in _idle_drawn_strings()), (
        "the idle surface draws nothing that says how to close the overlay"
    )


def test_the_close_hint_is_not_a_click_target():
    # A clickable close would sit beside the Ctrl-guarded record control, and
    # the entire point of that guard is that a stray click near this floating
    # window must not do anything expensive. The hint is a label.
    idle = _WINDOW_SOURCE.split("def _draw_idle")[1].split("\n    def ")[0]
    assert "_on_quit" not in idle, "the idle surface bound a click to quitting"


def test_a_live_interview_is_not_told_a_bare_escape_will_close_it():
    # `quit_is_allowed` REFUSES a bare Escape while recording or stopping.
    # Advertising it there would teach a gesture that visibly does nothing,
    # which is the exact failure the hint mechanism exists to prevent.
    recording = _WINDOW_SOURCE.split("def _draw_recording")[1].split("\n    def ")[0]
    assert "Esc to close" not in recording


def test_escape_is_bound_through_the_rule_and_not_straight_to_quit():
    # `self.root.bind("<Escape>", lambda _event: self._on_quit())` was the
    # original line. Restoring it would pass every other test in this file,
    # because `quit_is_allowed` would still exist and still be correct -- and
    # simply never be called.
    assert 'bind("<Escape>", self._escape_pressed)' in _WINDOW_SOURCE
    assert '"<Control-Escape>"' in _WINDOW_SOURCE
    assert "quit_is_allowed" in _WINDOW_SOURCE
