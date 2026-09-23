"""The ONE writer of `context/facts.json` (05-CONTEXT.md D-06/D-08;
05-01-PLAN.md §K1; 05-02-PLAN.md §K7/§K7a). Reuses the atomic-write
discipline from `app/web/author.py:90-104` VERBATIM -- temp file in the same
directory, write, flush, fsync, `os.replace`, `ensure_ascii=False` so Hebrew
stays readable on disk exactly as `app/session/store.py:384` already
requires. Not imported from `app/web/author.py` itself: that helper is
private to the web module, and this is a coach-side module writing a
different artifact.

`write_facts()` writes the WHOLE fact corpus it is handed, in one atomic
replace -- it does not read, merge, or de-duplicate anything itself.
`replace_session_facts()` is the D-08 mechanism built on top of it (§K7):
read the existing corpus with the LOUD validating loader, drop every fact
whose `session_id` matches the session being (re-)extracted, append the new
run's accepted facts, and atomic-write the whole array through `write_facts`.
A missing corpus file is D-01's normal first case (empty, not an error); an
EXISTING file that fails to parse raises `CorpusValidationError` and nothing
is written -- see `_load_existing_facts_or_empty`'s own docstring for why
the two branches must never collapse into one.

Concurrent extraction is NOT a supported shape, and no lock is built here for
it: this is a foreground CLI command on a single-user, single-process
machine (05-CONTEXT.md constraints). Two `extract` processes running at once
would race, last writer wins -- that shape does not occur, and building a
lock for it would be inventing a mechanism nothing needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from app.coach.corpus import FACTS_FILENAME, Fact, load_facts


def _atomic_write_json(path: Path, payload: object) -> None:
    """Temp file in the SAME directory, flush, fsync, `os.replace` --
    identical discipline to `app/web/author.py:_atomic_write_json`. If
    anything fails before `os.replace` runs, `path` itself was never opened
    for writing and is left exactly as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_facts(context_dir: "Path | str", facts: Sequence[Fact]) -> Path:
    """Atomically writes `facts` as the WHOLE `context/facts.json` array.
    Called only with a run's fully-resolved fact set (§K5: "nothing written
    on rejection", §K7: the slice-replaced whole array) -- a rejected fact
    never reaches this function at all. Returns the path written."""
    path = Path(context_dir) / FACTS_FILENAME
    payload = [fact.model_dump() for fact in facts]
    _atomic_write_json(path, payload)
    return path


def _load_existing_facts_or_empty(context_dir: "Path | str") -> "tuple[Fact, ...]":
    """The one read site shared by the pre-flight check and the replace
    write below (§K7a). A MISSING file is D-01's normal first case and reads
    back as empty -- checked here, BEFORE the loud loader is ever called, so
    that branch can never collapse into "malformed reads as empty" the way
    `app/web/author.py:_read_existing`'s tolerant reader deliberately does
    for the authoring page's form pre-fill. That tolerance is fine there (a
    page that shows nothing is recoverable); it would be catastrophic here --
    a corpus that failed to parse would be read as empty, and every fact from
    every past interview would be silently replaced by one run's output.

    An EXISTING file is read through `corpus.load_facts` -- the LOUD,
    validating loader -- with the citation resolver explicitly disabled
    (`sessions_root=None`): the corpus on disk was already resolved when it
    was written, and re-resolving every past session's citations on every
    extraction would make this read depend on session directories the user
    may since have archived. Raises `CorpusValidationError`, naming the file
    and the offending item, on anything that fails to parse or validate."""
    path = Path(context_dir) / FACTS_FILENAME
    if not path.is_file():
        return ()
    return load_facts(context_dir, sessions_root=None)


def preflight_check_corpus_parses(context_dir: "Path | str") -> None:
    """A read-only pre-flight (§K7a), meant to be called BEFORE the vendor
    call: raises `CorpusValidationError` if an EXISTING `context/facts.json`
    cannot be parsed, and does nothing at all if the file is simply missing
    (D-01's normal first case). A caller who runs this first and aborts on
    failure never pays for a vendor call whose result would be discarded --
    a user whose corpus is broken should not be billed for finding that out."""
    _load_existing_facts_or_empty(context_dir)


def replace_session_facts(
    context_dir: "Path | str", session_id: str, accepted_facts: Sequence[Fact]
) -> Path:
    """The D-08 mechanism (§K7): re-running extraction on `session_id`
    replaces that session's whole slice rather than doubling the corpus.
    Four steps, in this order:

    1. Read the existing corpus through `_load_existing_facts_or_empty` --
       the LOUD validating loader, missing-file branch kept distinct from
       the malformed-file branch. A crash or bug here leaves `context/
       facts.json` completely untouched: nothing about this function has
       opened it for writing yet.
    2. On a malformed existing corpus, `CorpusValidationError` propagates
       unchanged from step 1 -- nothing is written, exactly as if this
       function had never been called.
    3. Drop every fact whose `session_id` matches `session_id` -- every
       other session's facts are kept, in their existing order, byte-for-byte
       equal to how they were read.
    4. Append `accepted_facts` and atomic-write the WHOLE array via
       `write_facts()`. A crash between the temp-file write and `os.replace`
       (see `write_facts` -> `_atomic_write_json`) leaves the PREVIOUS
       `context/facts.json` byte-identical and still loadable -- nothing
       becomes visible until that one `os.replace` call succeeds.

    Deliberately, permanently NOT retained: this session's PREVIOUS facts,
    once superseded. `05-RESEARCH.md`'s append-only design kept them for
    auditability; that trade is given up knowingly here, because the fact
    corpus is derived data, regenerable at any time from `transcript.jsonl`
    -- which remains append-only, fsync'd, and untouched by this function.
    Dedup is by SESSION only: nothing in this function (or anywhere else in
    this codebase) may compare fact ids, claim text, or a content hash across
    runs (05-01-PLAN.md §K2, restated binding in 05-02-PLAN.md §K7)."""
    existing = _load_existing_facts_or_empty(context_dir)
    kept = tuple(fact for fact in existing if fact.session_id != session_id)
    combined = kept + tuple(accepted_facts)
    return write_facts(context_dir, combined)
