"""The tkinter window: three states, drawn flat and large.

Read this alongside `model.py`, which holds every rule about WHAT to show.
This file is only about putting it on screen and getting clicks back.

**Colour and type are constants at the top, not scattered literals.** The
window has no stylesheet, so this block is the stylesheet; a value used in two
places has to be named here or the two drift.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Optional, Sequence

from app.overlay.model import (
    HINT_VISIBLE_MS,
    MIC_OFF_HINT,
    MIC_ON_HINT,
    QUIT_WHILE_RECORDING_HINT_TEXT,
    RECORD_HINT_TEXT,
    STOP_HINT_TEXT,
    OverlayState,
    format_elapsed,
    quit_is_allowed,
)

# 0.85, not lower. Below about 0.8 the white text starts picking up whatever
# is behind it and the panel stops being readable at a glance, which is the
# only thing this surface is for.
WINDOW_ALPHA = 0.85

# The greys are WARM (a red-ish channel bias), not the neutral near-blacks
# they replaced. Over a video call -- which is what is behind this window --
# a neutral grey at 0.85 alpha picks up the blue of a screen-lit face and
# reads slightly cold and clinical; the warm steps sit against skin tones
# instead of competing with them.
BG = "#1f1e1d"
BG_PANEL = "#2b2a28"
# One step lighter than the panels, so the only control on the idle surface
# reads as raised rather than as another region of text.
BG_BUTTON = "#3d3c39"
FG = "#faf9f5"
FG_DIM = "#a8a49b"

# THREE accents, three meanings, and they must not be confusable at a glance
# on a translucent surface. The split is the same one this file has always
# made; only the values moved.
ACCENT = "#e5484d"      # the recording dot, and only that
# Clay. The coach's own colour, and the one thing on this surface that is
# meant to look like it comes from somewhere.
ACCENT_COACH = "#d97757"
# Hints are instructions, not errors. This used to be amber, which sat close
# enough to the clay above that a hint and a coach line could be taken for
# each other on a translucent panel at arm's length -- so it is pushed toward
# gold, away from the clay's red, while staying clearly not-red.
ACCENT_HINT = "#e3b341"

# What "bigger" means. 1.6 rather than a timid 1.25: the reason to enlarge
# this window is that a suggestion could not be read at a glance mid-sentence,
# and a size bump too small to fix that is worse than none -- it costs the
# screen space without buying the legibility.
EXPANDED_SCALE = 1.6

FONT_LABEL = ("Segoe UI", 8, "bold")
FONT_BODY = ("Segoe UI", 13)
FONT_QUESTION = ("Segoe UI", 15, "bold")
FONT_TIMER = ("Consolas", 12)
# The idle control carries the whole surface; letter-spacing is not available
# in tkinter, so weight and size do the work instead.
FONT_RECORD = ("Segoe UI", 12, "bold")

# Between the two gestures on the idle sub-label. A middle dot rather than a
# comma: they are two independent instructions, not a list.
IDLE_GESTURE_SEPARATOR = "   ·   "

PAD = 12
WRAP_PX = 380
IDLE_WIDTH = 260
RECORDING_WIDTH = 420


class OverlayWindow:
    """Owns the tkinter root and redraws it from an `OverlayState`.

    Every callback is injected. The window never starts a recording, never
    reads a project directory and never touches a subprocess -- it reports
    that a button was pressed, which is what keeps the whole surface
    driveable from a test without a display."""

    def __init__(
        self,
        *,
        on_record_pressed: "Callable[[], None]",
        on_project_chosen: "Callable[..., None]",
        on_stop_pressed: "Callable[[], None]",
        on_mic_toggled: "Callable[[], None]" = lambda: None,
        on_quit: "Callable[[], None]",
        root: "Optional[Any]" = None,
    ) -> None:
        self._on_record_pressed = on_record_pressed
        self._on_project_chosen = on_project_chosen
        self._on_stop_pressed = on_stop_pressed
        self._on_mic_toggled = on_mic_toggled
        self._on_quit = on_quit

        self.root = root if root is not None else tk.Tk()
        self.root.title("Interview")
        self.root.configure(bg=BG)
        # Borderless + always-on-top + translucent: the three attributes that
        # make this an overlay rather than a window. `overrideredirect` also
        # removes the close button, which is why `_bind_dragging` below has
        # to provide moving and quitting by hand.
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", WINDOW_ALPHA)

        self._body = tk.Frame(self.root, bg=BG)
        self._body.pack(fill="both", expand=True)
        self._drag_origin: "tuple[int, int]" = (0, 0)
        self._bind_dragging()

        # Bigger-on-click. Kept HERE and not in `OverlayState`, deliberately:
        # every field on that dataclass is something the interview produced,
        # and it is what `model.py`'s headless tests assert over. How large the
        # operator likes their window is a property of the window, survives no
        # session, and must never be able to change what a test of the coach's
        # own state can see.
        self._expanded = False

        self._projects: "tuple[str, ...]" = ()
        # There is NO pre-flight capture-mode state here any more -- see
        # `_draw_picking` for why the checkbox that used to live above the
        # project list was removed. The microphone has exactly one control now,
        # and it is on the recording panel (`_draw_microphone`).
        self._state = OverlayState()
        self._hint: "Optional[str]" = None
        self._hint_job: "Optional[Any]" = None

    # --- window chrome ----------------------------------------------------

    def _bind_dragging(self) -> None:
        """A borderless window has no title bar, so it cannot be moved or
        closed by the usual means. Drag anywhere to move; Escape quits.

        Moving matters more than it sounds: the operator has to be able to
        put this somewhere that is not on top of the customer's face in the
        call window, and where they want it depends on their layout."""
        self.root.bind("<ButtonPress-1>", self._remember_drag_origin)
        self.root.bind("<B1-Motion>", self._drag)
        # Escape alone must not end a live interview. It is a reflex key, and
        # far easier to hit by accident than the record control the Ctrl guard
        # protects -- guarding the click and leaving this open would defeat
        # the guard. `quit_is_allowed` holds the rule, in model.py, so it can
        # be tested without a display.
        self.root.bind("<Escape>", self._escape_pressed)
        self.root.bind("<Control-Escape>", lambda _event: self._on_quit())

    def _escape_pressed(self, _event: Any = None) -> None:
        if quit_is_allowed(self._state.phase):
            self._on_quit()
            return
        self._show_hint(QUIT_WHILE_RECORDING_HINT_TEXT)

    def _show_hint(self, text: str) -> None:
        """Say what the gesture wanted, briefly.

        An unmodified click on a Ctrl-guarded control does nothing, and a
        control that does nothing reads as a broken one. Showing what was
        expected teaches the gesture once instead of every time."""
        self._hint = text
        self.render(self._state)
        after = getattr(self.root, "after", None)
        if callable(after):
            if self._hint_job is not None:
                cancel = getattr(self.root, "after_cancel", None)
                if callable(cancel):
                    cancel(self._hint_job)
            self._hint_job = after(HINT_VISIBLE_MS, self._clear_hint)

    def _clear_hint(self) -> None:
        self._hint_job = None
        if self._hint is None:
            return
        self._hint = None
        self.render(self._state)

    def _arm(self, widget: Any, action: "Callable[[], None]", hint: str) -> None:
        """Binds a control so it fires only with Ctrl held.

        Two bindings, not one binding that inspects `event.state`: tkinter
        dispatches the most specific pattern, so a Ctrl+click fires ONLY the
        Control binding and never both. Reading the modifier bit by hand would
        have to get that mutual exclusion right itself, and would encode
        0x0004 in a place nobody would think to check."""
        widget.bind("<Control-Button-1>", lambda _e: action())
        widget.bind("<Button-1>", lambda _e: self._show_hint(hint))

    def _remember_drag_origin(self, event: Any) -> None:
        self._drag_origin = (event.x_root, event.y_root)

    def _drag(self, event: Any) -> None:
        start_x, start_y = self._drag_origin
        dx, dy = event.x_root - start_x, event.y_root - start_y
        self.root.geometry(f"+{self.root.winfo_x() + dx}+{self.root.winfo_y() + dy}")
        self._drag_origin = (event.x_root, event.y_root)

    def place_top_right(self, *, margin: int = 24) -> None:
        """Default position. Top-right because the call window's own controls
        and the speaker's video are usually centre and bottom."""
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        width = self.root.winfo_width()
        self.root.geometry(f"+{screen_w - width - margin}+{margin}")

    # --- drawing ----------------------------------------------------------

    def set_projects(self, projects: "Sequence[str]") -> None:
        self._projects = tuple(projects)

    def render(self, state: OverlayState) -> None:
        self._state = state
        for child in self._body.winfo_children():
            child.destroy()
        if state.phase == "recording" or state.phase == "stopping":
            self._draw_recording(state)
        elif state.phase == "picking":
            self._draw_picking()
        else:
            self._draw_idle(state)
        self.root.update_idletasks()

    # --- size ------------------------------------------------------------------------

    @property
    def _scale(self) -> float:
        return EXPANDED_SCALE if self._expanded else 1.0

    def _f(self, font):
        """A font tuple at the current scale.

        Rounds rather than truncates, and floors at the original size, so the
        collapsed state is byte-identical to what shipped before this existed
        -- an operator who never clicks the control sees no change at all."""
        if not self._expanded:
            return font
        family, size, *rest = font
        return (family, max(size, int(round(size * self._scale))), *rest)

    def _px(self, value: int) -> int:
        """A pixel dimension at the current scale -- wrap widths and panel
        widths, so text that got bigger also gets more room to wrap into
        rather than becoming a taller column of the same narrow measure."""
        return value if not self._expanded else int(round(value * self._scale))

    def _toggle_size(self, _event=None) -> None:
        """Grow or shrink, then redraw from the SAME state.

        Nothing about the interview changes here: no callback fires, no
        message goes to the coach, and `self._state` is untouched. That is
        what makes this safe to bind to a click during a live recording --
        the worst a misfire can do is resize a window."""
        self._expanded = not self._expanded
        result = "break"  # see the bind in `_size_control` -- swallow the drag
        # `overrideredirect` windows do not re-fit themselves when their
        # children change size, so the geometry has to be released explicitly
        # or the panel keeps its old width and clips the larger text.
        try:
            self.root.geometry("")
        except Exception:  # noqa: BLE001 -- a headless/fake root need not have it
            pass
        self.render(self._state)
        # Stops the root's own drag binding seeing this click. Without it a
        # click that lands a pixel off centre both resizes the window AND
        # starts a drag, so the panel jumps as it grows.
        return result

    def _size_control(self, parent, *, bg):
        """The affordance itself. Both glyphs are drawn in the dim ink rather
        than an accent: this is chrome, and an accent here would compete with
        the recording dot and the coach line for the same glance."""
        label = self._label(
            parent,
            "⤢" if not self._expanded else "⤡",
            font=self._f(FONT_LABEL), fg=FG_DIM, bg=bg, cursor="hand2",
        )
        label.bind("<Button-1>", self._toggle_size)
        return label

    def _label(self, parent, text, *, font, fg=FG, **kwargs):
        return tk.Label(
            parent, text=text, font=font, fg=fg, bg=kwargs.pop("bg", BG),
            anchor="w", justify="left", **kwargs,
        )

    def _draw_idle(self, state: OverlayState) -> None:
        # A filled, bordered block rather than a bare line of text. The old
        # version was a label that happened to be clickable, which left the
        # one control on the whole surface looking like a caption -- and this
        # is the only thing the operator ever does from the idle state.
        button = tk.Frame(self._body, bg=BG_BUTTON, highlightthickness=1,
                          highlightbackground=ACCENT, highlightcolor=ACCENT, cursor="hand2")
        button.pack(fill="x", padx=PAD, pady=(PAD, 6))
        inner = tk.Frame(button, bg=BG_BUTTON)
        inner.pack(fill="x", padx=10, pady=8)
        self._label(inner, "●", font=self._f(FONT_RECORD), fg=ACCENT, bg=BG_BUTTON).pack(side="left", padx=(0, 8))
        self._label(inner, "START RECORDING", font=self._f(FONT_RECORD), fg=FG, bg=BG_BUTTON).pack(side="left")

        # Armed on every part of the block, so the whole thing is the target
        # rather than the exact glyph. Still never on `self._body`: a drag
        # that begins on the background must not be able to start an
        # interview, and repositioning the window is the commonest gesture
        # here.
        for widget in (button, inner, *inner.winfo_children()):
            self._arm(widget, self._on_record_pressed, RECORD_HINT_TEXT)

        # Both gestures the idle surface has, said out loud. `overrideredirect`
        # removed the title bar and with it the close button, so Escape is the
        # ONLY way to dismiss this window -- and until this line it appeared
        # nowhere on screen. Shown ONLY here: `quit_is_allowed` refuses a bare
        # Escape while an interview is live, and advertising a key that will
        # refuse is worse than saying nothing.
        #
        # A label, never a button. A click target that closes the overlay would
        # sit beside the Ctrl-guarded record control, and the whole point of
        # that guard is that a stray click near this window costs nothing.
        self._label(
            self._body, f"hold Ctrl and click{IDLE_GESTURE_SEPARATOR}Esc to close",
            font=self._f(FONT_LABEL), fg=FG_DIM,
        ).pack(fill="x", padx=PAD, pady=(0, PAD if not state.error else 6))

        if state.error:
            # ACCENT red, and it does NOT expire -- unlike `_draw_hint`'s
            # amber, which clears itself after HINT_VISIBLE_MS. This carries
            # things like "the interview was cut short before it flushed",
            # which must still be on screen whenever the operator next looks
            # at the window. Two different colours for a persistent warning
            # and a momentary nudge, on purpose.
            self._label(
                self._body, state.error, font=self._f(FONT_LABEL), fg=ACCENT, wraplength=self._px(IDLE_WIDTH),
            ).pack(fill="x", padx=PAD, pady=(0, PAD))
        # A `failed` status that arrives late (the socket outliving the run)
        # is still worth showing here. It costs one call and no state-machine
        # change: `has_status_line` is False on a fresh state, so an ordinary
        # idle surface is unchanged.
        self._draw_status(state)
        self._draw_hint(IDLE_WIDTH)

    def _draw_status(self, state: OverlayState) -> None:
        """What the capture is doing, when it is not simply fine.

        Three situations, drawn distinguishably: healthy is ABSENT (a
        permanent status line is one more thing to learn to ignore), degraded
        is ACCENT_HINT amber, failure is ACCENT red.

        The colour split is not decoration. ACCENT is the recording dot and
        genuine failure; painting a muted microphone with it makes a degraded
        stream -- where capture is running perfectly and the audio reaching it
        is wrong -- look like a crash. Amber already means "attention, not
        failure" everywhere else on this surface."""
        if not state.has_status_line:
            return
        self._label(
            self._body, state.status_line, font=self._f(FONT_LABEL),
            fg=ACCENT if state.status_is_failure else ACCENT_HINT,
            wraplength=self._px(WRAP_PX),
        ).pack(fill="x", padx=PAD, pady=(0, 8))

    def _draw_hint(self, wrap: int) -> None:
        if self._hint is None:
            return
        self._label(
            self._body, self._hint, font=self._f(FONT_LABEL), fg=ACCENT_HINT, wraplength=wrap,
        ).pack(fill="x", padx=PAD, pady=(0, PAD))

    def _draw_picking(self) -> None:
        """The project list, and NOTHING above it but its own heading.

        **A "system audio only -- no microphone" checkbox used to sit here, and
        removing it is the point of this comment.** It worked exactly as built.
        The operator missed it on two consecutive recordings anyway and shipped
        67% speaker bleed both times, which is what produced `_draw_microphone`
        on the recording panel.

        Keeping both left one concept -- is the microphone open -- with two
        controls in two places, one of which is only reachable before the
        interview starts and resets on every visit. That is the configuration
        that caused the original confusion, not a fix for it: the operator used
        both and got neither.

        The one thing the pre-flight control uniquely guaranteed was that the
        device was never opened AT ALL, and that guarantee is genuinely gone --
        a run now opens the microphone and the operator closes it if they want
        it closed. Accepted deliberately: `--only interviewee` is still there on
        the CLI for a recording that must never touch the device, and
        `_draw_microphone` RELEASES the device rather than muting it, so the OS
        indicator goes off and the state is visible for the whole interview
        instead of for the ten seconds before it."""
        self._label(self._body, "FILE THIS INTERVIEW UNDER", font=self._f(FONT_LABEL), fg=FG_DIM).pack(
            fill="x", padx=PAD, pady=(PAD, 6)
        )
        if not self._projects:
            self._label(
                self._body, "No projects yet -- run `python -m app author --project <name>`",
                font=self._f(FONT_LABEL), fg=FG_DIM, wraplength=self._px(RECORDING_WIDTH),
            ).pack(fill="x", padx=PAD, pady=(0, PAD))
        for path in self._projects:
            item = tk.Label(
                self._body, text=path, font=self._f(FONT_BODY), fg=FG, bg=BG_PANEL,
                anchor="w", padx=8, pady=5, cursor="hand2",
            )
            item.pack(fill="x", padx=PAD, pady=1)
            item.bind("<Button-1>", lambda _e, p=path: self._on_project_chosen(p))
        cancel = tk.Label(self._body, text="cancel", font=self._f(FONT_LABEL), fg=FG_DIM, bg=BG, cursor="hand2")
        cancel.pack(anchor="w", padx=PAD, pady=(6, PAD))
        cancel.bind("<Button-1>", lambda _e: self._on_quit_picking())

    def _draw_microphone(self, state: OverlayState) -> None:
        """The microphone control, drawn on the RECORDING panel.

        **This is where the pre-flight checkbox failed.** That control was
        pre-flight only, so it was invisible at the exact moment the operator
        wanted to know what the microphone was doing -- and on 2026-08-12 a
        real operator missed it on two consecutive recordings and shipped 67%
        speaker bleed both times. A setting you must remember before you start
        is a setting you will forget.

        Every conferencing tool keeps microphone state on screen for the whole
        call, in one place, one click to change. This is that, in the panel
        that is already always visible.

        NOT Ctrl-armed, unlike record and stop. Those two are guarded because
        they are hard to undo -- an accidental stop ends an interview. Muting
        is instantly reversible, and putting a modifier on a control the
        operator may want twice a minute would make the guard the thing they
        fight rather than the thing that protects them.

        The state is coloured, not just worded: an open microphone is the
        condition that produces bleed, so it is the one that carries the
        accent."""
        row = tk.Frame(self._body, bg=BG)
        row.pack(fill="x", padx=PAD, pady=(0, 6))
        control = tk.Label(
            row, text=state.mic_line, font=self._f(FONT_LABEL),
            fg=ACCENT_HINT if state.mic_on else FG_DIM,
            bg=BG_PANEL, anchor="w", padx=8, pady=4, cursor="hand2",
        )
        control.pack(fill="x")
        control.bind("<Button-1>", lambda _e: self._on_mic_toggled())
        self._label(
            row, MIC_ON_HINT if state.mic_on else MIC_OFF_HINT,
            font=self._f(FONT_LABEL), fg=FG_DIM,
        ).pack(fill="x", pady=(2, 0))

    def _on_quit_picking(self) -> None:
        self._on_project_chosen("")

    def _draw_recording(self, state: OverlayState) -> None:
        header = tk.Frame(self._body, bg=BG)
        header.pack(fill="x", padx=PAD, pady=(PAD, 8))

        # The five things the operator asked to see while recording, in one
        # row: that it is running, for how long, which project, and how to
        # stop.
        self._label(header, "●", font=self._f(FONT_BODY), fg=ACCENT).pack(side="left", padx=(0, 6))
        self._label(header, "REC", font=self._f(FONT_LABEL), fg=ACCENT).pack(side="left", padx=(0, 10))
        self._label(header, format_elapsed(state.elapsed_s), font=self._f(FONT_TIMER), fg=FG).pack(side="left")
        self._label(
            header, state.project or "", font=self._f(FONT_LABEL), fg=FG_DIM,
        ).pack(side="left", padx=(10, 0))

        stopping = state.phase == "stopping"
        stop = tk.Label(
            header, text="stopping..." if stopping else "STOP", font=self._f(FONT_LABEL),
            fg=FG_DIM if stopping else FG, bg=BG_PANEL, padx=8, pady=3,
            cursor="" if stopping else "hand2",
        )
        stop.pack(side="right")
        # Between STOP and the timer, and NOT armed: resizing is the one
        # gesture on this panel that cannot cost anything, so putting a Ctrl
        # guard on it would teach the operator to Ctrl-click reflexively --
        # which is exactly the habit the guard on STOP depends on them NOT
        # having.
        self._size_control(header, bg=BG_PANEL).pack(side="right", padx=(0, 10))
        if not stopping:
            # Ctrl-armed like the record control, and this is the direction
            # that actually costs something: a stray click here ends a live
            # interview while the customer is mid-sentence, and the recording
            # cannot be resumed -- `stop()` closes the session.
            self._arm(stop, self._on_stop_pressed, STOP_HINT_TEXT)

        # Drawn BEFORE the coaching regions and outside their guard: what the
        # capture is doing outranks what to ask next, and an unfiled recording
        # draws no coaching regions at all -- the run with nothing else moving
        # on screen is the one that most needs to be told its audio stopped.
        self._draw_status(state)
        # Right below the capture status and ABOVE the coaching regions: it is
        # a fact about what is being recorded, not about what to ask next, and
        # it must be reachable in an uncoached run where no coaching regions
        # are drawn at all.
        self._draw_microphone(state)

        # Both coaching regions are ABSENT in an uncoached run, not emptied.
        # There is no coach process, so no `survey` and no `suggestion`
        # message will ever arrive -- a "loading the survey..." panel would
        # stand there claiming an in-flight request for the whole interview,
        # and a "listening..." line would claim a coach that was never
        # started. Drawn only under `state.shows_coaching`, whose rule lives
        # in model.py because this file cannot be tested headless.
        if state.shows_coaching:
            question = tk.Frame(self._body, bg=BG_PANEL)
            question.pack(fill="x", padx=PAD, pady=(0, 6))
            head = tk.Frame(question, bg=BG_PANEL)
            head.pack(fill="x", padx=8, pady=(6, 0))
            self._label(head, "NEXT QUESTION", font=self._f(FONT_LABEL), fg=FG_DIM, bg=BG_PANEL).pack(side="left")
            self._label(
                head, state.progress_line, font=self._f(FONT_LABEL), fg=FG_DIM, bg=BG_PANEL
            ).pack(side="right")
            self._label(
                question, state.question_line, font=self._f(FONT_QUESTION), fg=FG, bg=BG_PANEL,
                wraplength=self._px(WRAP_PX),
            ).pack(fill="x", padx=8, pady=(2, 8))

            # The coach region only exists when the coach has said something.
            # An always-present empty box trains the eye to skip the place the
            # suggestion will appear.
            if state.has_coach_line:
                coach = tk.Frame(self._body, bg=BG_PANEL)
                coach.pack(fill="x", padx=PAD, pady=(0, PAD))
                self._label(
                    coach, "COACH", font=self._f(FONT_LABEL), fg=ACCENT_COACH, bg=BG_PANEL,
                ).pack(fill="x", padx=8, pady=(6, 0))
                self._label(
                    coach, state.coach_line, font=self._f(FONT_BODY), fg=FG, bg=BG_PANEL, wraplength=self._px(WRAP_PX),
                ).pack(fill="x", padx=8, pady=(2, 8))
            else:
                self._label(
                    self._body, state.coach_line, font=self._f(FONT_LABEL), fg=FG_DIM,
                ).pack(fill="x", padx=PAD, pady=(0, PAD))
        else:
            # Says what this run IS, once, and makes no promise about anything
            # arriving later.
            #
            # The SENTENCE is chosen in model.py, not here. There are two
            # reasons for no coaching -- the operator asked for none, or one
            # was asked for and did not come up -- and picking between them at
            # the point of drawing is how they would quietly collapse back
            # into one, which is the defect this branch was fixed for.
            self._label(
                self._body, state.coaching_absence_line, font=self._f(FONT_LABEL), fg=FG_DIM,
            ).pack(fill="x", padx=PAD, pady=(0, PAD))
        self._draw_hint(self._px(WRAP_PX))
