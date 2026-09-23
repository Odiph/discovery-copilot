"""`context/demotions.json` -- the record of the labels extraction OVERRODE.

The defect here is the one `rejection_store` was built to close, one field
over. `extract_facts` has always produced `demotions`, and `app/main.py` has
always printed them to stdout and kept nothing. That is exactly the state
rejections were in when the project found itself holding the number "~30% of
extractions reject as `unlocatable-quote`" and no way to go back and check
it. So these tests are mostly about what SURVIVES a run, not about what a
function returns.

There is a second defect, specific to demotions, that no amount of stdout
would have fixed: STATE COLLAPSE. A demoted fact is stored in `facts.json`
as a plain `inference`, byte-for-byte indistinguishable from one the model
labelled `inference` itself -- same provenance, and (since nothing forbids a
quote on an inference) possibly the same `quoted_text` too. The corpus
cannot tell "the model said evidence and we disagreed" from "the model said
inference". `demotions.json` is the only place that distinction exists, and
`test_a_demoted_inference_is_indistinguishable_in_facts_json_and_separable_here`
is the test that pins it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import main as main_mod
from app.coach import reasoning_client
from app.coach.demotion_store import (
    DEMOTIONS_FILENAME,
    count_by_reason,
    load_demotions,
    replace_session_demotions,
)
from app.coach.extract import (
    EXTRACT_TOOL_NAME,
    REASON_CONTENTLESS_QUOTE,
    Demotion,
)
from app.coach.rejection_store import REJECTIONS_FILENAME
from app.coach.stub import StubAnthropicClient, tool_use_response
from app.session.store import SessionStore, create_session_dir


def _demotion(
    claim: str = "a claim",
    reason: str = REASON_CONTENTLESS_QUOTE,
    seqs: "tuple[int, ...] | None" = None,
    quoted_text: str = "Yeah, exactly.",
) -> Demotion:
    return Demotion(claim=claim, reason=reason, seqs=seqs, quoted_text=quoted_text)


# --- the store, in isolation ---------------------------------------------------------------


def test_a_demotion_survives_the_run_that_produced_it(tmp_path: Path):
    replace_session_demotions(
        tmp_path, "s1", [_demotion(claim="they onboard in a week", seqs=(3, 4))]
    )
    rows = load_demotions(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["claim"] == "they onboard in a week"
    assert rows[0]["reason"] == REASON_CONTENTLESS_QUOTE
    assert rows[0]["seqs"] == [3, 4]


def test_re_extracting_a_session_replaces_only_its_own_slice(tmp_path: Path):
    replace_session_demotions(tmp_path, "s1", [_demotion(claim="from s1")])
    replace_session_demotions(tmp_path, "s2", [_demotion(claim="from s2")])
    replace_session_demotions(tmp_path, "s1", [_demotion(claim="s1 again")])

    claims = [(row["session_id"], row["claim"]) for row in load_demotions(tmp_path)]
    assert ("s2", "from s2") in claims, "another session's rows were destroyed"
    assert ("s1", "s1 again") in claims
    assert ("s1", "from s1") not in claims, "the superseded slice survived"


def test_a_run_that_demoted_nothing_still_writes(tmp_path: Path):
    """Same inversion of `corpus_store`'s rule that `rejection_store` makes,
    for the same reason. "Extracted and overrode no label" is a real answer
    about the extraction model, and it must be distinguishable from "never
    extracted at all" -- which is what a file that was never created says."""
    replace_session_demotions(tmp_path, "s1", [_demotion()])
    replace_session_demotions(tmp_path, "s1", [])

    assert (tmp_path / DEMOTIONS_FILENAME).is_file()
    assert load_demotions(tmp_path) == ()


def test_nothing_is_written_before_the_atomic_replace(tmp_path: Path):
    path = tmp_path / DEMOTIONS_FILENAME
    replace_session_demotions(tmp_path, "s1", [_demotion(claim="first")])
    before = path.read_bytes()
    # A demotion whose `asdict` will blow up mid-write -- the file must be
    # byte-identical afterwards, not truncated or half-written.
    with pytest.raises(Exception):
        replace_session_demotions(tmp_path, "s2", [object()])  # type: ignore[list-item]
    assert path.read_bytes() == before
    assert not (tmp_path / (DEMOTIONS_FILENAME + ".tmp")).exists()


def test_the_reason_tally_is_computed_from_the_file_not_from_memory(tmp_path: Path):
    replace_session_demotions(
        tmp_path, "s1", [_demotion(), _demotion(), _demotion(reason="some-future-reason")]
    )
    replace_session_demotions(tmp_path, "s2", [_demotion()])
    assert count_by_reason(tmp_path) == {REASON_CONTENTLESS_QUOTE: 3, "some-future-reason": 1}


def test_the_tally_spans_sessions_because_one_session_proves_nothing(tmp_path: Path):
    """The whole question this file exists to answer -- "how often does the
    extraction model mislabel an inference as evidence?" -- is a rate over the
    corpus. One session's demotion count is noise."""
    for index in range(4):
        replace_session_demotions(tmp_path, f"s{index}", [_demotion()])
    assert count_by_reason(tmp_path)[REASON_CONTENTLESS_QUOTE] == 4


def test_an_empty_corpus_tallies_to_nothing_rather_than_raising(tmp_path: Path):
    assert count_by_reason(tmp_path) == {}
    assert load_demotions(tmp_path) == ()


def test_a_malformed_file_loses_old_rows_rather_than_blocking_extraction(tmp_path: Path):
    """Same divergence from `corpus_store`'s loud loader that `rejection_store`
    makes: a corrupt diagnostic log must not become a thing that can stop the
    run it was diagnosing."""
    (tmp_path / DEMOTIONS_FILENAME).write_text("{not json", encoding="utf-8")
    replace_session_demotions(tmp_path, "s1", [_demotion(claim="still recorded")])
    assert [row["claim"] for row in load_demotions(tmp_path)] == ["still recorded"]


@pytest.mark.parametrize("payload", ['{"not": "a list"}', "[1, 2, 3]", "null"])
def test_a_file_of_the_wrong_shape_is_ignored_rather_than_half_read(tmp_path: Path, payload):
    (tmp_path / DEMOTIONS_FILENAME).write_text(payload, encoding="utf-8")
    replace_session_demotions(tmp_path, "s1", [_demotion()])
    assert len(load_demotions(tmp_path)) == 1


def test_non_ascii_claims_stay_readable_on_disk(tmp_path: Path):
    replace_session_demotions(tmp_path, "s1", [_demotion(claim="הם לא השתמשו בזה")])
    raw = (tmp_path / DEMOTIONS_FILENAME).read_text(encoding="utf-8")
    assert "הם לא השתמשו בזה" in raw
    assert json.loads(raw)[0]["claim"] == "הם לא השתמשו בזה"


def test_demotions_never_land_in_the_rejection_file(tmp_path: Path):
    """The two are separate artifacts on purpose: a rejection DELETED a fact
    and drives the exit code, a demotion KEPT one. A reader tallying
    `rejections.json` to answer "how much did we throw away" must not find
    kept facts mixed into the count."""
    replace_session_demotions(tmp_path, "s1", [_demotion()])
    assert not (tmp_path / REJECTIONS_FILENAME).exists()


# --- the real producer against the real consumer, through the real CLI ---------------------


def _session_with(tmp_path: Path, turns: "list[tuple[str, str]]") -> Path:
    """A REAL session directory via `SessionStore`, so `read_session` and the
    citation resolver see genuine `seq` values -- never a hand-rolled
    transcript file. Mirrors `test_coach_extract.py`'s own `_new_session`."""
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "demotion-store-test"})
    for speaker, text in turns:
        store.append_segment(
            speaker=speaker, text=text, audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
    store.close()
    return session_dir


def _fact_payload(*, claim: str, seq: int, provenance: str, quoted_text: str) -> dict:
    return {
        "claim": claim, "seqs": [seq], "provenance": provenance,
        "quoted_text": quoted_text, "bearings": [],
    }


def _run_extract_cli(tmp_path, monkeypatch, *, turns, facts) -> "tuple[int, Path]":
    """Drives `python -m app extract` end to end against the offline stub. The
    point is that NOTHING here fakes an `ExtractionResult`: the real
    `extract_facts` decides what is demoted, and the real store writes it. A
    test that hand-built a `Demotion` and handed it to the store would prove
    only that the two halves of the seam agree with the test's imagination."""
    session_dir = _session_with(tmp_path, turns)
    context_dir = tmp_path / "context"
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": facts})]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)
    exit_code = main_mod.main(["extract", str(session_dir)])
    return exit_code, context_dir


def test_a_demotion_the_real_extractor_made_is_on_disk_after_the_real_cli_run(
    tmp_path, monkeypatch, capsys
):
    exit_code, context_dir = _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")],
        facts=[_fact_payload(
            claim="a claim resting on agreement alone", seq=1,
            provenance="evidence", quoted_text="Yeah, exactly.",
        )],
    )
    assert exit_code == 0
    rows = load_demotions(context_dir)
    assert len(rows) == 1, "the demotion the extractor really made was not persisted"
    assert rows[0]["reason"] == REASON_CONTENTLESS_QUOTE
    assert "a claim resting on agreement alone" in rows[0]["claim"]
    assert capsys.readouterr().out.count("DEMOTED to inference") == 1


def test_a_demotion_alone_never_makes_the_extract_command_fail(tmp_path, monkeypatch):
    """CRITICAL, and the reason the exit code is keyed on rejections ALONE. A
    demoted fact was KEPT. If demotions moved the exit code, every CI run over
    a session where the model got one label slightly wrong would fail, and the
    pressure would be to stop demoting -- i.e. to accept the bad label."""
    exit_code, context_dir = _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")],
        facts=[_fact_payload(
            claim="a claim resting on agreement alone", seq=1,
            provenance="evidence", quoted_text="Yeah, exactly.",
        )],
    )
    assert exit_code == 0, "a demotion moved the exit code -- it must not"
    assert len(load_demotions(context_dir)) == 1, "and it must still have been recorded"


def test_the_exit_code_still_tracks_rejections_when_a_demotion_is_also_present(
    tmp_path, monkeypatch
):
    """The other half of the same pin: adding the demotion write must not have
    made the exit code STOP tracking rejections either."""
    exit_code, context_dir = _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")],
        facts=[
            _fact_payload(
                claim="a claim resting on agreement alone", seq=1,
                provenance="evidence", quoted_text="Yeah, exactly.",
            ),
            _fact_payload(
                claim="a claim resting on a quote nobody said", seq=1,
                provenance="evidence", quoted_text="we ship every Tuesday",
            ),
        ],
    )
    assert exit_code == 1
    assert len(load_demotions(context_dir)) == 1


def test_a_session_that_demoted_nothing_still_leaves_an_answerable_file(tmp_path, monkeypatch):
    exit_code, context_dir = _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "how long"), ("interviewee", "we onboard in a week")],
        facts=[_fact_payload(
            claim="onboarding takes a week", seq=1,
            provenance="evidence", quoted_text="we onboard in a week",
        )],
    )
    assert exit_code == 0
    assert (context_dir / DEMOTIONS_FILENAME).is_file(), (
        "no demotions file at all -- 'this session was extracted and no label "
        "was overridden' is now indistinguishable from 'never extracted'"
    )
    assert load_demotions(context_dir) == ()


def test_a_demoted_inference_is_indistinguishable_in_facts_json_and_separable_here(
    tmp_path, monkeypatch
):
    """STATE COLLAPSE, pinned. Two facts go in: one the model labelled
    `evidence` on a contentless quote (demoted), one it labelled `inference`
    itself, carrying the SAME quote. On disk in `facts.json` they differ in
    nothing but `id` and `claim` -- same provenance, same quoted_text --
    because nothing in `corpus.Fact` forbids a quote on an inference. The
    hub's evidence/inference panel therefore cannot tell an overridden label
    from an honest one from the corpus alone. `demotions.json` is the only
    record that can, and this test fails the moment it stops being written."""
    exit_code, context_dir = _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")],
        facts=[
            _fact_payload(
                claim="the overridden one", seq=1,
                provenance="evidence", quoted_text="Yeah, exactly.",
            ),
            _fact_payload(
                claim="the honest one", seq=1,
                provenance="inference", quoted_text="Yeah, exactly.",
            ),
        ],
    )
    assert exit_code == 0

    facts = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(facts) == 2
    collapsed = {
        (fact["provenance"], fact["quoted_text"], tuple(fact["seqs"])) for fact in facts
    }
    assert len(collapsed) == 1, (
        "the two facts are separable in facts.json after all -- if that is now "
        "true by design, this test is measuring the wrong thing; if it is an "
        "accident, do not rely on it"
    )

    demoted_claims = [row["claim"] for row in load_demotions(context_dir)]
    assert demoted_claims == ["the overridden one"], (
        "the demotion record is the ONLY thing that separates an overridden "
        "label from an honest inference, and it is gone"
    )


def test_the_standing_demotion_tally_is_reported_across_sessions(tmp_path, monkeypatch, capsys):
    """The tally, not just this run's count, is what makes "how often does the
    extraction model make this error?" answerable -- the same reason
    `rejection_store.count_by_reason` is printed rather than the per-run
    number alone."""
    context_dir = tmp_path / "context"
    replace_session_demotions(context_dir, "an-earlier-session", [_demotion(), _demotion()])

    session_dir = _session_with(
        tmp_path, [("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")]
    )
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="a claim resting on agreement alone", seq=1,
            provenance="evidence", quoted_text="Yeah, exactly.",
        )]})]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)
    assert main_mod.main(["extract", str(session_dir)]) == 0

    out = capsys.readouterr().out
    assert "Demotions on record (3 across all sessions)" in out, (
        "the standing tally is missing or counts only this run -- one session's "
        "demotion rate says nothing"
    )
    assert REASON_CONTENTLESS_QUOTE in out
    assert "Demotions written to:" in out


def test_neither_a_claim_nor_a_quote_reaches_a_log_record(tmp_path, monkeypatch, caplog):
    """Two-tier logging: counts, ids and reason codes may be logged; claim and
    quote text may go to STDOUT (the existing `REJECTED`/`DEMOTED` convention)
    and nowhere else. A demoted quote is transcript text the interviewee
    really said, so a log file is exactly where it must not be."""
    caplog.set_level(0)
    _run_extract_cli(
        tmp_path, monkeypatch,
        turns=[("interviewer", "did that bother you"), ("interviewee", "Yeah, exactly.")],
        facts=[_fact_payload(
            claim="a claim resting on agreement alone", seq=1,
            provenance="evidence", quoted_text="Yeah, exactly.",
        )],
    )
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Yeah, exactly." not in logged
    assert "a claim resting on agreement alone" not in logged
