"""`.planning/FLOWMAP.md`'s header must be arithmetic over its own table.

**Why this is a test and not a habit.** On 2026-08-12 an audit found the header
wrong in three independent ways at once: `24 work, 10 partial` against a real
25/9, a flow total of 41 against 46 rows, and `9` never-live-verified against 3
rows actually carrying the marker. The mechanism is worth naming, because it is
not carelessness: **the header is one hand-edited sentence and the table is
edited row by row**, so the two drift the moment anyone touches one without the
other. Three separate sessions did exactly that.

The file's own preamble says a stale map is worse than none, "because it is read
as current". It was the thing being read as current.

`/post-impl` audit G already requires the file to be updated for every flow a
piece of work changes. That obligation is on the ROWS, which is the half a human
notices. This holds the half a human does not.

**What this deliberately does NOT check.** Whether a row's status is true. No
test can know that `WORKS` is honest -- that is what the audit is for, and
pretending otherwise here would be the same overstated-guard problem this
project keeps finding. This checks one thing: that the summary at the top counts
the table below it.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOWMAP = REPO_ROOT / ".planning" / "FLOWMAP.md"

# `.planning/` is the maintainer's design record and is not published, so a
# checkout without it has no map to check.
pytestmark = pytest.mark.skipif(not FLOWMAP.is_file(), reason=".planning/FLOWMAP.md is not in this checkout")

# The status vocabulary the file's own preamble defines. Pinned by equality
# rather than harvested, so a sixth word is a decision about what the map can
# say -- and the header sentence that would have to change with it.
KNOWN_STATUSES = ("WORKS", "PARTIAL", "BROKEN", "MISSING", "FUTURE")

NEVER_MARKER = "NEVER LIVE-VERIFIED"


def _rows() -> "list[tuple[str, str]]":
    """(flow, status) for every real table row, skipping headers and rules."""
    rows: "list[tuple[str, str]]" = []
    for line in FLOWMAP.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "Flow" or set(cells[0]) <= set("- "):
            continue
        rows.append((cells[0], cells[1]))
    return rows


def _counts() -> "collections.Counter":
    counter: "collections.Counter" = collections.Counter()
    for _flow, status in _rows():
        # "WORKS · **NEVER LIVE-VERIFIED**" and "MISSING (by design)" both
        # count as their base status; the marker is a separate axis and the
        # parenthetical is a reason, not a sixth state.
        base = status.split("·")[0].strip().split("(")[0].strip()
        base = base.strip("*")
        counter[base] += 1
    return counter


def _header_numbers() -> "dict[str, int]":
    text = FLOWMAP.read_text(encoding="utf-8")
    match = re.search(
        r"\*\*(\d+) flows — (\d+) work, (\d+) partial, (\d+) missing, "
        r"(\d+) future, (\d+) broken, and (\d+)\s*\n?\s*that have never been watched",
        text,
    )
    assert match is not None, (
        "the header sentence no longer matches the shape this test reads. It is the one "
        "sentence in the file that summarises the table, so if its wording changes, this "
        "test must change with it -- do not delete the check to make it pass"
    )
    total, works, partial, missing, future, broken, never = (int(g) for g in match.groups())
    return {
        "total": total, "WORKS": works, "PARTIAL": partial, "MISSING": missing,
        "FUTURE": future, "BROKEN": broken, "never": never,
    }


def test_the_scan_finds_a_real_table():
    """The emptiness guard gates 17/20/30 all carry. Every assertion below is
    satisfied by a scan that found nothing, so prove the walk works first."""
    rows = _rows()
    assert len(rows) > 30, f"the table scan found only {len(rows)} rows -- it is not reading the file"
    flows = [flow for flow, _status in rows]
    assert len(set(flows)) == len(flows), (
        f"two rows name the same flow: "
        f"{[f for f, n in collections.Counter(flows).items() if n > 1]}"
    )


def test_every_status_is_one_the_preamble_defines():
    unknown = sorted(
        {
            status for _flow, status in _rows()
            if status.split("·")[0].strip().split("(")[0].strip().strip("*") not in KNOWN_STATUSES
        }
    )
    assert unknown == [], (
        f"a row uses a status the preamble does not define: {unknown}. The vocabulary is five "
        "words; a sixth needs a definition at the top of the file, not a quiet appearance in a row"
    )


@pytest.mark.parametrize("status", KNOWN_STATUSES)
def test_the_header_count_for_each_status_equals_the_table(status):
    header = _header_numbers()
    actual = _counts()[status]
    assert header[status] == actual, (
        f"FLOWMAP's header says {header[status]} {status.lower()} and the table has {actual}. "
        "Recount from the table -- do not adjust the header by hand, which is exactly how it "
        "drifted three times before"
    )


def test_the_header_flow_total_equals_the_number_of_rows():
    assert _header_numbers()["total"] == len(_rows())


def test_the_never_live_verified_count_is_the_number_of_rows_that_say_so():
    """The one that mattered most. The previous header claimed 9 while THREE
    rows carried the marker and six more confessed it in prose with the Status
    column reading `WORKS` -- so the file was simultaneously overstating how
    much had been verified, row by row, and understating it in the summary."""
    marked = sum(1 for _flow, status in _rows() if NEVER_MARKER in status)
    assert _header_numbers()["never"] == marked, (
        f"the header claims {_header_numbers()['never']} never-live-verified flows and "
        f"{marked} rows carry the marker. The marker is the record; if a row has never been "
        "watched, say so IN THE ROW -- a count in the header that no row backs up is the "
        "opposite of the honesty this axis exists for"
    )


def test_a_row_does_not_confess_in_prose_what_its_status_hides():
    """The subtler half, and the reason the count was wrong rather than merely
    stale. Six rows said "nobody has watched", "never triggered", "never
    observed" in their own Gap column while the Status column read `WORKS` with
    no marker. A reader scanning the Status column -- which is what a status
    column is for -- was told the opposite of what the row said."""
    confession = re.compile(
        r"nobody has|never been (?:watched|observed|read)|never observed|never triggered|"
        r"never completed|never watched|never read by a human|not done|unverified",
        re.IGNORECASE,
    )
    offenders = []
    for line in FLOWMAP.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or NEVER_MARKER in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] == "Flow" or set(cells[0]) <= set("- "):
            continue
        if cells[1].startswith("WORKS") and confession.search(" ".join(cells[2:])):
            offenders.append(cells[0])
    assert offenders == [], (
        f"these rows read WORKS while their own text says nobody has ever watched them: "
        f"{offenders}. Add `· **NEVER LIVE-VERIFIED**` to the status -- the preamble says that "
        "axis must never be folded into a status, and a confession buried in the Gap column is "
        "exactly that folding, done quietly"
    )
