"""Writes the asked-question critique artifact. Its only job is that write.

A separate module rather than four lines in `app/main.py`, for the reason gate
19 states about `app/web/accept.py`: that gate matches its two regexes PER
FILE, so any module holding both a write primitive and a reference to the
hypothesis artifact joins the pinned set of that artifact's writers.
`run_critique_questions` legitimately loads hypotheses (the critic needs their
falsifiers) and legitimately writes a file -- but the file it writes is a
critique, not that artifact, and the gate cannot tell those apart.

Admitting `app/main.py` to that set to resolve it would have been the wrong
trade by a wide margin: it is the largest module in the tree and dispatches
every command, so pinning it would retire gate 19's discriminating power over
everything `main.py` ever does again. Keeping the write here costs one small
file and leaves the gate at full strength.

**This docstring may not NAME the artifact's filename.** An earlier draft spelt
it out while explaining the gate, and the prose alone was enough to match --
the same lesson gate 1 already teaches about module docstrings: a scanner
reading the file sees explanation and code alike.
"""

from __future__ import annotations

import json
from pathlib import Path

CRITIQUE_DOC_FILENAME = "questions_critique.json"


def write_critique(session_dir: Path, payload: dict) -> Path:
    """Writes the critique beside the transcript it describes, and returns
    where it went.

    `ensure_ascii=False` because interviewer speech is not always English --
    the corpus holds Hebrew and mixed-language sessions, and escaping them
    would make the artifact unreadable to the one person most able to check
    it."""
    destination = session_dir / CRITIQUE_DOC_FILENAME
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination
