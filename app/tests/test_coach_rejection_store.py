"""`context/rejections.json` -- the record of what extraction threw away.

The defect this module exists to prevent is not a crash. It is that the
project once held the finding "~30% of extractions reject as
`unlocatable-quote`" and could not go back and check it, because rejections
were printed to a terminal and never written down. So these tests are mostly
about what SURVIVES a run, not about what a function returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.coach.extract import (
    REASON_SPAN_TOO_LONG,
    REASON_UNLOCATABLE_QUOTE,
    Rejection,
)
from app.coach.rejection_store import (
    REJECTIONS_FILENAME,
    count_by_reason,
    load_rejections,
    replace_session_rejections,
)


def _rejection(claim: str = "a claim", reason: str = REASON_UNLOCATABLE_QUOTE, **kw) -> Rejection:
    return Rejection(claim=claim, reason=reason, **kw)


def test_a_rejection_survives_the_run_that_produced_it(tmp_path: Path):
    replace_session_rejections(
        tmp_path, "s1", [_rejection(claim="they onboard in a week", seqs=(3, 4))]
    )
    rows = load_rejections(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["claim"] == "they onboard in a week"
    assert rows[0]["reason"] == REASON_UNLOCATABLE_QUOTE
    assert rows[0]["seqs"] == [3, 4]


def test_re_extracting_a_session_replaces_only_its_own_slice(tmp_path: Path):
    replace_session_rejections(tmp_path, "s1", [_rejection(claim="from s1")])
    replace_session_rejections(tmp_path, "s2", [_rejection(claim="from s2")])
    replace_session_rejections(tmp_path, "s1", [_rejection(claim="s1 again")])

    claims = [(row["session_id"], row["claim"]) for row in load_rejections(tmp_path)]
    assert ("s2", "from s2") in claims, "another session's rows were destroyed"
    assert ("s1", "s1 again") in claims
    assert ("s1", "from s1") not in claims, "the superseded slice survived"


def test_a_run_that_rejected_nothing_still_writes(tmp_path: Path):
    """The INVERSE of `corpus_store`'s rule, and deliberately so. There, a run
    that accepted nothing writes nothing, so an empty `facts.json` is never
    fabricated. Here, "extracted and refused nothing" is a real answer, and it
    must be distinguishable from "never extracted at all"."""
    replace_session_rejections(tmp_path, "s1", [_rejection()])
    replace_session_rejections(tmp_path, "s1", [])

    assert (tmp_path / REJECTIONS_FILENAME).is_file()
    assert load_rejections(tmp_path) == ()


def test_nothing_is_written_before_the_atomic_replace(tmp_path: Path):
    path = tmp_path / REJECTIONS_FILENAME
    replace_session_rejections(tmp_path, "s1", [_rejection(claim="first")])
    before = path.read_bytes()
    # A rejection whose `asdict` will blow up mid-write -- the file must be
    # byte-identical afterwards, not truncated or half-written.
    with pytest.raises(Exception):
        replace_session_rejections(tmp_path, "s2", [object()])  # type: ignore[list-item]
    assert path.read_bytes() == before
    assert not (tmp_path / (REJECTIONS_FILENAME + ".tmp")).exists()


# --- the tally that answers the original question ----------------------------------------


def test_the_reason_tally_is_computed_from_the_file_not_from_memory(tmp_path: Path):
    replace_session_rejections(
        tmp_path, "s1",
        [_rejection(), _rejection(), _rejection(reason=REASON_SPAN_TOO_LONG)],
    )
    replace_session_rejections(tmp_path, "s2", [_rejection()])
    assert count_by_reason(tmp_path) == {REASON_UNLOCATABLE_QUOTE: 3, REASON_SPAN_TOO_LONG: 1}


def test_the_tally_spans_sessions_because_one_session_proves_nothing(tmp_path: Path):
    for index in range(4):
        replace_session_rejections(tmp_path, f"s{index}", [_rejection()])
    assert count_by_reason(tmp_path)[REASON_UNLOCATABLE_QUOTE] == 4


def test_an_empty_corpus_tallies_to_nothing_rather_than_raising(tmp_path: Path):
    assert count_by_reason(tmp_path) == {}
    assert load_rejections(tmp_path) == ()


# --- what the file can and cannot contain --------------------------------------------------


def test_only_an_unlocatable_quote_can_carry_a_quote_at_all(tmp_path: Path):
    """`extract.py` attaches `quoted_text` at exactly one site: the branch
    that fires BECAUSE the quote was not found in the cited span. So every
    quote this file can hold is text the transcript does not contain -- model
    output that failed to be verbatim, not customer speech. Pinned here as a
    property rather than trusted as a habit of the caller."""
    import ast
    import inspect

    from app.coach import extract as extract_mod

    tree = ast.parse(inspect.getsource(extract_mod))
    quoting_sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Rejection"
        and any(kw.arg == "quoted_text" for kw in node.keywords)
    ]
    assert len(quoting_sites) == 1, (
        "a second site now attaches quoted_text to a rejection -- check it is "
        "still only the unlocatable-quote branch, or this file may now hold "
        "speech that IS in the transcript"
    )
    reasons = [
        kw.value.id for site in quoting_sites for kw in site.keywords
        if kw.arg == "reason" and isinstance(kw.value, ast.Name)
    ]
    assert reasons == ["REASON_UNLOCATABLE_QUOTE"]


def test_a_malformed_file_loses_old_rows_rather_than_blocking_extraction(tmp_path: Path):
    """The one place this diverges from `corpus_store`'s loud loader. A
    corrupt diagnostic log must not become a thing that can stop the run it
    was diagnosing."""
    (tmp_path / REJECTIONS_FILENAME).write_text("{not json", encoding="utf-8")
    replace_session_rejections(tmp_path, "s1", [_rejection(claim="still recorded")])
    rows = load_rejections(tmp_path)
    assert [row["claim"] for row in rows] == ["still recorded"]


@pytest.mark.parametrize("payload", ['{"not": "a list"}', "[1, 2, 3]", "null"])
def test_a_file_of_the_wrong_shape_is_ignored_rather_than_half_read(tmp_path: Path, payload):
    (tmp_path / REJECTIONS_FILENAME).write_text(payload, encoding="utf-8")
    replace_session_rejections(tmp_path, "s1", [_rejection()])
    assert len(load_rejections(tmp_path)) == 1


def test_non_ascii_claims_stay_readable_on_disk(tmp_path: Path):
    """`ensure_ascii=False`, matching every other writer in this project --
    a rejection nobody can read is a rejection nobody will examine."""
    replace_session_rejections(tmp_path, "s1", [_rejection(claim="הם לא השתמשו בזה")])
    raw = (tmp_path / REJECTIONS_FILENAME).read_text(encoding="utf-8")
    assert "הם לא השתמשו בזה" in raw
    assert json.loads(raw)[0]["claim"] == "הם לא השתמשו בזה"
