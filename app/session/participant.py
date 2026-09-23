"""WHO was interviewed -- one optional human-supplied label per session,
stored in that session's own `session.json`.

**Why this field exists at all, and why it is not cosmetic.** The product's
central counting rule is participants-by-session: `app/coach/interview_summary`
counts INTERVIEWS, never facts, and its stated justification is that four
separate people beat one repetitive one. Nothing anywhere recorded that four
people were involved. Sessions and people are only the same number while nobody
is re-interviewed and nobody records two calls in one sitting -- and the moment
either happens, the rule silently counts one person twice while claiming the
opposite. This label is what makes the claim checkable by a human.

**Absent is the NORMAL state, not a defect.** Every recording this product has
made so far predates this field, so `read_participant` returning `None` is the
common path and every consumer is written for it first. That is also why the
key is OMITTED from `session.json` rather than written as `null` when there is
no label: a session that was recorded before anyone could name a participant
and a session someone deliberately blanked are the same situation -- nobody
knows who it was -- and inventing a stored `null` to distinguish them would be
a confident sentence about something no byte on disk supports.

**This module never writes.** It owns the key, the validation rule and the
tolerant read; the write lives in `app/web/participant.py`, which is the one
module in the codebase that puts bytes into a `session.json` after the
interview is over. Splitting it that way keeps this module importable from
anywhere -- including `app/main.py` at `start` -- without dragging a FastAPI
dependency or a write primitive along with it.

No STT vendor is ever named in this module's executable code (STT-03), and
nothing here reads a transcript: naming the person in the room is not
displaying customer speech, and this module stays on the right side of that
line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.session.store import METADATA_FILENAME

logger = logging.getLogger(__name__)

# The one spelling of this field, named here rather than typed as a bare
# literal in the reader, the writer, the hub payload and `app/main.py`. That
# duplication is a real failure mode this repo has already paid for once: the
# 2026-08-12 audit found the extraction vocabulary emitted as four literals on
# one side and re-typed by hand on the other, with no test rendering either, so
# a rename on one side produced `undefined` with the whole suite green.
PARTICIPANT_KEY = "participant"

# A label, not a note. Long enough for "Dana Levi — ops lead, mid-size clinic"
# and short enough that nothing downstream has to decide how to truncate it.
# The cap is enforced at every entrance (the `start` flag and the hub's save)
# rather than at the exit, because a value already on disk cannot be refused
# without making an existing session unreadable.
MAX_PARTICIPANT_LENGTH = 120


def normalize_participant(value: Any) -> "tuple[Optional[str], Optional[str]]":
    """Turns whatever a human typed into the value that gets stored, or the
    reason it will not be stored. Returns `(label, None)` or `(None, error)`.

    `(None, None)` -- no label and no error -- is a LEGITIMATE and expected
    answer, not a failure: an absent flag, an empty box and a box the operator
    deliberately cleared all mean "nobody is named", which is the common path.
    A caller that treats a `None` label as an error will refuse the normal
    case, so the two `None`s are returned separately and must be checked
    separately.

    **Control characters are refused rather than stripped.** A newline in a
    one-line label is not a typo to be silently repaired -- it means the
    operator pasted something that is not a label, and quietly keeping the
    first line would store a value they never read back. What this is NOT
    guarding against is script injection: the hub renders every string through
    `textContent`, so the danger a filter here would be protecting against does
    not exist. This is about the field meaning what it says.
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, f"the participant label must be a string, not {type(value).__name__}"
    label = value.strip()
    if not label:
        # Whitespace-only is CLEARING, not an error. The operator who selects
        # the box and deletes what is in it is saying "I was wrong about who
        # this was", and refusing that would leave a wrong name on disk with no
        # way to take it off.
        return None, None
    if len(label) > MAX_PARTICIPANT_LENGTH:
        return None, (
            f"the participant label is {len(label)} characters; "
            f"the limit is {MAX_PARTICIPANT_LENGTH}"
        )
    if any(character < " " or character == "\x7f" for character in label):
        return None, "the participant label must be a single line with no control characters"
    return label, None


def read_participant(session_dir: Path) -> "Optional[str]":
    """The label recorded for one session, or `None`.

    **Tolerant by design, and it reads `session.json` DIRECTLY rather than
    going through `app.session.reader.read_session`.** That function is the one
    parser for the TRANSCRIPT and it parses the whole of it -- every segment,
    every event, every suggestion -- which is the right cost when a caller
    wants records and an absurd one when a caller wants one short string. The
    session picker asks this question once per interview on every page load, so
    routing it through the transcript parser would make listing twenty
    interviews read twenty transcripts to print twenty names.

    Every failure answers `None`, and each one is a genuinely different
    situation that a reader cannot act on differently anyway: no metadata file,
    metadata that will not parse, no key, or a key holding something that is
    not a string (a hand-edited `session.json` with a number or an object in
    it). In all four the honest answer to "who was interviewed" is "nothing on
    disk says", and the hub prints exactly that. Raising instead would let one
    hand-edited metadata file take down the picker for every other interview in
    the project, which is the failure mode the hub's per-section degradation
    exists to prevent.

    The one thing this does NOT do is repair anything. A malformed
    `session.json` is left exactly as it is; `app/web/participant.py` refuses
    to write over one for the same reason `app/web/accept.py` refuses to write
    over a corpus it could not read.
    """
    metadata_path = Path(session_dir) / METADATA_FILENAME
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.info(
            "no readable participant label for this session error_type=%s", type(exc).__name__
        )
        return None
    if not isinstance(metadata, dict):
        return None
    label = metadata.get(PARTICIPANT_KEY)
    if not isinstance(label, str):
        return None
    # Stripped on the way out as well as on the way in: a label hand-typed into
    # `session.json` never passed `normalize_participant`, and " Dana " on
    # screen would read as a rendering bug rather than as what someone typed.
    label = label.strip()
    return label or None
