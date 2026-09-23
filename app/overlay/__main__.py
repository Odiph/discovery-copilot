"""`python -m app overlay` -- the assembled overlay.

Nothing but wiring. Every decision lives in `model.py` (what to show),
`window.py` (how to draw it), `session.py` (how to supervise an interview) and
`controller.py` (the loop). This file exists so those four can each be tested
without any of the others.
"""

from __future__ import annotations

import logging
import queue
import sys
from typing import Optional, Sequence

from app import config as config_mod
from app import project_meta
from app import projects as projects_mod
from app.overlay.controller import OverlayController
from app.overlay.session import InterviewSession
from app.overlay.window import OverlayWindow

logger = logging.getLogger(__name__)


def active_projects() -> "tuple[str, ...]":
    """The projects offered in the picker.

    Every node in the tree, ordered by `project_meta.for_picker` so the
    unfiled bucket lands LAST rather than at the top, where `_`'s sort order
    would otherwise put it and where the first row reads as the default.

    The bucket is created here rather than at install time because this runs
    on every press of the record button: a fresh clone, or a tree the operator
    deleted, gets it back without a reinstall."""
    project_meta.ensure_unfiled()
    return tuple(node.path for node in project_meta.for_picker(projects_mod.list_projects()))


def run(argv: "Optional[Sequence[str]]" = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = config_mod.load_config()

    def _start(
        project: str,
        events: "queue.Queue",
        *,
        coached: bool = True,
        loopback_only: bool = False,
    ) -> InterviewSession:
        session = InterviewSession(
            project, events=events, port=cfg.app_port, coached=coached,
            loopback_only=loopback_only,
        )
        session.start()
        return session

    window = OverlayWindow(
        on_record_pressed=lambda: controller.on_record_pressed(),
        on_project_chosen=lambda p, loopback_only=False: controller.on_project_chosen(
            p, loopback_only=loopback_only
        ),
        on_stop_pressed=lambda: controller.on_stop_pressed(),
        on_mic_toggled=lambda: controller.on_mic_toggled(),
        # A lambda, like the three above it, and for a reason worth naming:
        # `_quit` is defined BELOW because it needs `window` and `controller`,
        # so passing the bare name here raises UnboundLocalError at
        # construction. The other three are already lambdas and hid it.
        on_quit=lambda: _quit(),
    )
    controller = OverlayController(
        window=window, list_projects=active_projects, start_session=_start
    )

    def _quit() -> None:
        # Escape while recording must still close the interview cleanly --
        # quitting the overlay is not a reason to abandon a session.
        if controller.state.phase in ("recording", "stopping"):
            controller.on_stop_pressed()
        window.root.destroy()

    window.render(controller.state)
    window.place_top_right()
    controller.attach(window.root)
    print("Overlay running. Drag to move, Escape to quit.")
    window.root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(run())
