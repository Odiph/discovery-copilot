"""The ONE writer of `context/rejections.json` -- what extraction threw away.

**Why this exists at all.** `extract_facts` has always produced rejections and
`app/main.py` has always printed them to stdout. Nothing kept them. The
project's own notes carry the finding "~30% of extractions reject as
`unlocatable-quote`", and when someone finally went to look at WHICH claims
those were, the answer was that the terminal they were printed to is gone and
the number is unreproducible. Re-deriving it costs a paid model call per
session; the rejections themselves cost nothing to keep.

So: the same session-slice-replace discipline `corpus_store` applies to
accepted facts, applied to the ones that did not make it. Same atomic write,
same "re-running a session supersedes only ITS slice" rule, same regenerable
derived data.

**One rule deliberately INVERTED from `corpus_store`.** `replace_session_facts`
is guarded on `extraction.accepted` -- a run that accepts nothing writes
nothing, so a fully-rejected first run never fabricates an empty
`facts.json`. This module has no such guard, and must not grow one. A run that
accepted NOTHING is the single most informative run there is: it is either a
broken extractor or a session with nothing in it, and those look identical
from a `facts.json` that was never created. Writing an empty rejection slice
is meaningful here (it says "this session was extracted and refused nothing"),
where writing an empty fact array there would have been a lie.

**What this file does and does not contain, structurally.** `quoted_text` is
attached to a rejection at exactly ONE site in `extract.py`, the
`unlocatable-quote` branch -- and that branch fires precisely BECAUSE the
quote could not be located in the cited span. So every quote this file can
hold is one the transcript does not contain: model-authored text that failed
to be verbatim. It is the reverse of the usual worry, and worth stating
plainly because the field name invites the opposite assumption. `claim` is
likewise the model's own words, truncated. Nothing here is customer speech,
and `test_coach_rejection_store.py` pins that as a property of the pipeline
rather than as a habit of the caller.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from app.coach.extract import Rejection

REJECTIONS_FILENAME = "rejections.json"


def _atomic_write_json(path: Path, payload: object) -> None:
    """Identical discipline to `corpus_store._atomic_write_json` -- temp file
    in the SAME directory, flush, fsync, `os.replace`. Not imported from
    there: that helper is private to the module that owns `facts.json`, and
    this writes a different artifact. Same precedent `corpus_store` itself
    cites for not importing `app/web/author.py`'s copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _as_row(session_id: str, rejection: Rejection) -> "dict[str, Any]":
    """`session_id` is stamped on every ROW rather than nesting rows under a
    session key. It makes the slice replace below a flat filter (the same one
    `corpus_store` does over `Fact.session_id`), and it makes the file
    answerable with a one-line comprehension -- which is the entire point of
    keeping it."""
    row = asdict(rejection)
    row["seqs"] = list(rejection.seqs) if rejection.seqs is not None else None
    return {"session_id": session_id, **row}


def _load_existing_or_empty(context_dir: "Path | str") -> "list[dict[str, Any]]":
    """A missing file is the normal first case and reads back as empty.

    A file that exists but does NOT parse reads back as empty TOO -- and this
    is the one place this module diverges from `corpus_store`'s loud loader,
    deliberately. There, a malformed corpus read as empty would silently
    destroy every past interview's facts, so it must raise. Here the stake is
    a diagnostic log: refusing to write would mean a corrupt rejection file
    could block extraction itself, making the record of what went wrong into
    a thing that can go wrong. The worst case is losing prior rejections,
    which are regenerable; blocking extraction is not a trade worth making
    for them."""
    path = Path(context_dir) / REJECTIONS_FILENAME
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(loaded, list):
        return []
    return [row for row in loaded if isinstance(row, dict)]


def replace_session_rejections(
    context_dir: "Path | str", session_id: str, rejections: Sequence[Rejection]
) -> Path:
    """Replace `session_id`'s whole rejection slice, leaving every other
    session's rows untouched and in order. Returns the path written.

    Called unconditionally, including with an EMPTY `rejections` -- see the
    module docstring for why that is the meaningful case rather than the
    skippable one."""
    existing = _load_existing_or_empty(context_dir)
    kept = [row for row in existing if row.get("session_id") != session_id]
    combined = kept + [_as_row(session_id, rejection) for rejection in rejections]
    path = Path(context_dir) / REJECTIONS_FILENAME
    _atomic_write_json(path, combined)
    return path


def load_rejections(context_dir: "Path | str") -> "tuple[dict[str, Any], ...]":
    """Every recorded rejection, newest run's rows last. Read-only, tolerant
    (same reasoning as `_load_existing_or_empty`), and returned as plain
    dicts rather than `Rejection`s: rows written by an older version of this
    file may carry fields this one does not know about, and a reader asking
    "what did we throw away and why" should see them rather than have them
    dropped on the way through a stricter type."""
    return tuple(_load_existing_or_empty(context_dir))


def count_by_reason(context_dir: "Path | str") -> "dict[str, int]":
    """Rejections per `reason`, descending. This is the function that answers
    the question that started this module -- "is it really ~30%
    unlocatable-quote?" -- from data instead of from memory."""
    counts: "dict[str, int]" = {}
    for row in load_rejections(context_dir):
        reason = row.get("reason")
        if isinstance(reason, str):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
