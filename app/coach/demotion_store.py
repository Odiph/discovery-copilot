"""The ONE writer of `context/demotions.json` -- the labels extraction OVERRODE.

**Why this exists at all, and why it is `rejection_store` twice.** That module
was written because rejections existed only on a terminal that is now gone,
which is how the finding "~30% of extractions reject as `unlocatable-quote`"
became a number nobody could go back and check. Demotions were left in exactly
that state: `extract_facts` produces them, `app/main.py` prints them, nothing
keeps them. So this file is deliberately its near-twin -- same session-slice
replace, same atomic write, same tolerant loader, same `count_by_reason`. The
symmetry IS the point. Two diagnostic logs that answer the same shape of
question should not have two shapes; a reader who has understood one has
understood both, and a future change to the discipline is visibly a change to
both files rather than a silent divergence in one.

**What a demotion is, and why it is NOT a rejection.** A demotion is a fact
whose `provenance` the pipeline overrode from `evidence` to `inference` and
then ACCEPTED. The claim is usually TRUE; what was wrong is the label. That
is why the two live in separate files rather than one file with a status
column: `rejections.json` answers "what did we throw away", and mixing kept
facts into it would corrupt the only number it exists to produce. It is also
why nothing in this module may ever be consulted for an exit code -- see
`app/main.py`'s own comment at the `return`.

**The state collapse this closes.** `corpus_store` writes a demoted fact into
`facts.json` as a plain `inference`. Nothing on that record says the model
had said `evidence`, and nothing in `corpus.Fact` forbids a quote on an
inference -- so a demoted fact and an honestly-labelled one can be identical
in every stored field but `id` and `claim`. The hub's evidence/inference panel
reading `facts.json` alone therefore CANNOT tell "the model mislabelled this
and we corrected it" from "the model labelled this correctly". This file is
the only record where that distinction survives, joinable back to the corpus
on `session_id` + `seqs` + the (truncated) claim.

**What this file can contain, and how that differs from its twin.** Every
`quoted_text` in `rejections.json` is by construction text the transcript does
NOT contain -- it is attached only on the unlocatable-quote branch. The
opposite is true here: a demoted quote WAS located in its cited span, so it is
real interviewee speech ("Yeah, exactly.", "Not really."). `facts.json` holds
verbatim speech too and is ignored for exactly that reason, but a reader who
carried `rejection_store`'s "nothing here is customer speech" over to this
file would be wrong -- which is precisely why nothing in this module logs, and
why callers must keep claim and quote on stdout rather than in a log record. The content is
contentless by definition -- that is what got it demoted -- but the rule is
about where speech may be written, not about how interesting the speech is.
`.gitignore`'s `context/*` and blanket `projects/` cover this file already
(read the comment above those lines before moving it); a demotions log
written anywhere OUTSIDE a context dir would be a committable file holding
verbatim speech, which this project's own ignore rules exist to prevent.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from app.coach.extract import Demotion

DEMOTIONS_FILENAME = "demotions.json"


def _atomic_write_json(path: Path, payload: object) -> None:
    """Identical discipline to `rejection_store._atomic_write_json` -- temp
    file in the SAME directory, flush, fsync, `os.replace`. Not imported from
    there for the reason that module itself gives for not importing
    `corpus_store`'s copy: that helper is private to the module that owns
    `rejections.json`, and this writes a different artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _as_row(session_id: str, demotion: Demotion) -> "dict[str, Any]":
    """`session_id` is stamped on every ROW rather than nesting rows under a
    session key -- same reasoning as `rejection_store._as_row`: it makes the
    slice replace below a flat filter and the file answerable with a one-line
    comprehension, which is the entire point of keeping it. Here it is load-
    bearing twice over, because `session_id` is also half of the join key back
    to `facts.json` that undoes the state collapse."""
    row = asdict(demotion)
    row["seqs"] = list(demotion.seqs) if demotion.seqs is not None else None
    return {"session_id": session_id, **row}


def _load_existing_or_empty(context_dir: "Path | str") -> "list[dict[str, Any]]":
    """A missing file is the normal first case and reads back as empty.

    A file that exists but does NOT parse reads back as empty TOO, matching
    `rejection_store` and diverging from `corpus_store`'s loud loader on
    purpose. There, a malformed corpus read as empty would silently destroy
    every past interview's facts, so it must raise. Here the stake is a
    diagnostic log: refusing to write would mean a corrupt demotion file could
    block extraction itself, making the record of what went wrong into a thing
    that can go wrong. The worst case is losing prior demotions, which are
    regenerable by re-extracting; blocking extraction is not."""
    path = Path(context_dir) / DEMOTIONS_FILENAME
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(loaded, list):
        return []
    return [row for row in loaded if isinstance(row, dict)]


def replace_session_demotions(
    context_dir: "Path | str", session_id: str, demotions: Sequence[Demotion]
) -> Path:
    """Replace `session_id`'s whole demotion slice, leaving every other
    session's rows untouched and in order. Returns the path written.

    Called unconditionally, including with an EMPTY `demotions`. That is the
    meaningful case, not the skippable one: "this session was extracted and no
    label was overridden" is the answer that says the extraction model got it
    right, and a file that was never created cannot say it -- it is
    indistinguishable from "this session was never extracted"."""
    existing = _load_existing_or_empty(context_dir)
    kept = [row for row in existing if row.get("session_id") != session_id]
    combined = kept + [_as_row(session_id, demotion) for demotion in demotions]
    path = Path(context_dir) / DEMOTIONS_FILENAME
    _atomic_write_json(path, combined)
    return path


def load_demotions(context_dir: "Path | str") -> "tuple[dict[str, Any], ...]":
    """Every recorded demotion, newest run's rows last. Read-only, tolerant
    (same reasoning as `_load_existing_or_empty`), and returned as plain dicts
    rather than `Demotion`s: rows written by an older version of this file may
    carry fields this one does not know about, and a reader asking "which
    labels did we override" should see them rather than have them dropped on
    the way through a stricter type."""
    return tuple(_load_existing_or_empty(context_dir))


def count_by_reason(context_dir: "Path | str") -> "dict[str, int]":
    """Demotions per `reason`, descending. This is the function that answers
    the question this module was built for -- "how often does the extraction
    model label an inference as evidence?" -- from data instead of from the
    impression left by whoever last watched a run scroll past."""
    counts: "dict[str, int]" = {}
    for row in load_demotions(context_dir):
        reason = row.get("reason")
        if isinstance(reason, str):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
