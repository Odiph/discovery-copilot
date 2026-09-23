"""The only place a session file is parsed (phase_specification U1/Task 1).
Everything downstream -- the renderer (plan 02-03 Task 2), any future surface,
any post-mortem -- reads through this module, so crash tolerance is
implemented once rather than re-invented per consumer.

The durability contract this module enforces on read: a truncated FINAL line
is the one thing a hard kill produces by definition, and is therefore
tolerated and named rather than raised; a malformed INTERIOR line is not
expected from any crash and is therefore rejected loudly, because silently
skipping it would hide real corruption behind the same code path that
legitimately tolerates a crash tail.

No STT vendor is ever named in this module's executable code (STT-03).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.session.store import METADATA_FILENAME, TRANSCRIPT_FILENAME


def list_session_ids(sessions_root: Path) -> "tuple[str, ...]":
    """The session ids under a node, NEWEST FIRST.

    Session directory names are `%Y%m%dT%H%M%S`, so a reverse lexicographic
    sort IS newest-first without parsing a date -- and it stays correct for a
    same-second collision, which `create_session_dir` disambiguates with a `-1`
    suffix that sorts after the bare name.

    **This is also the hub's ALLOWLIST for a session id arriving from a
    browser.** A page that can choose its session sends a name over the wire,
    and `sessions_root / name` with a `..` in it walks out of the project
    entirely. Checking membership of this listing is what makes that
    impossible -- it admits only names that were actually found on disk, so no
    traversal, no symlink and no case trick can construct a passing value that
    a pattern check might let through.

    Returns empty rather than raising when the node has no `sessions/` yet: a
    project authored but never recorded into is normal, not an error."""
    if not Path(sessions_root).is_dir():
        return ()
    return tuple(
        entry.name
        for entry in sorted(Path(sessions_root).iterdir(), key=lambda item: item.name, reverse=True)
        if entry.is_dir()
    )


class MalformedRecordError(Exception):
    """Raised when an INTERIOR line (not the file's final line) is not valid
    JSON. A truncated tail is the expected shape of a hard kill; a corrupt
    middle line is not, and is refused rather than silently skipped."""


@dataclass(frozen=True)
class SessionReadResult:
    """Everything a consumer needs after parsing a session directory once."""

    records: tuple[dict[str, Any], ...]
    truncated_tail: Optional[str]
    closed: bool
    metadata: dict[str, Any]

    def records_of_kind(self, kind: str) -> tuple[dict[str, Any], ...]:
        return tuple(r for r in self.records if r.get("kind") == kind)


def read_session(session_dir: Path) -> SessionReadResult:
    """Parses one session directory. Tolerates a session that is still being
    written (no exclusive lock is ever taken on the transcript file by the
    writer, so a read never blocks behind an open writer) and a session that
    was never closed (reported via `closed=False`, never inferred from a
    timestamp gap alone -- phase_specification U1)."""
    transcript_path = session_dir / TRANSCRIPT_FILENAME
    metadata_path = session_dir / METADATA_FILENAME

    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    closed = "closed_at" in metadata

    records: list[dict[str, Any]] = []
    truncated_tail: Optional[str] = None

    if transcript_path.is_file():
        # Read BYTES and decode per line, never the whole file at once. A hard
        # kill truncates at a byte offset, and nearly every character in this
        # product's transcripts is a two-byte Hebrew codepoint -- so the most
        # likely crash tail is one cut mid-codepoint, i.e. invalid UTF-8 and
        # not merely invalid JSON. Decoding the file up front would raise
        # UnicodeDecodeError before any line-level tolerance could run, which
        # would kill both `render` and the resume path on exactly the
        # interrupted interviews SESS-02 exists to protect.
        raw_lines = transcript_path.read_bytes().split(b"\n")
        # A file that ends with a trailing newline (the normal, complete
        # case) splits into one empty trailing element -- drop it so it is
        # never mistaken for a truncated final line.
        if raw_lines and raw_lines[-1] == b"":
            raw_lines.pop()

        last_index = len(raw_lines) - 1
        for index, raw_line in enumerate(raw_lines):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                if index == last_index:
                    # Cut mid-codepoint by the kill. Named, not raised. The
                    # replacement char is only for the human-readable report;
                    # no record is fabricated from a partial one.
                    truncated_tail = raw_line.decode("utf-8", errors="replace")
                    continue
                raise MalformedRecordError(
                    f"{transcript_path}: interior line {index + 1} of "
                    f"{len(raw_lines)} is not valid UTF-8 and is not the "
                    f"file's final line -- a mid-codepoint cut is expected "
                    f"only at the tail, so this is real corruption"
                )

            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == last_index:
                    # The one shape a hard kill produces by definition: a
                    # record whose write was interrupted mid-flush. Named,
                    # not raised -- an interrupted interview must stay
                    # readable (SESS-02).
                    truncated_tail = line
                else:
                    raise MalformedRecordError(
                        f"{transcript_path}: interior line {index + 1} of "
                        f"{len(raw_lines)} is not valid JSON and is not the "
                        f"file's final line -- refusing to silently skip a "
                        f"corrupt middle record"
                    )

    return SessionReadResult(
        records=tuple(records),
        truncated_tail=truncated_tail,
        closed=closed,
        metadata=metadata,
    )
