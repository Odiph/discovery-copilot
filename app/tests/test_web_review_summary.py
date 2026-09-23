"""The hub's summary block: the numbers the server was already computing.

`summarize_interview` has always been called by `build_review` -- once per
hypothesis, from `_contribution`, which kept three integers of the returned
`InterviewSummary` and dropped the rest on the floor. The payload then carried
`summary: None` and the page printed "Not built yet". This file pins the whole
object onto the wire, and pins the three properties that made throwing it away
tempting in the first place:

- it is counted for THIS interview, never for the corpus a rich project
  accumulated from earlier calls;
- it carries no verbatim customer speech, because it is counts and ids only;
- it costs ONE pass over the corpus per page, not one per hypothesis.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.coach import interview_summary as summary_mod
from app.web.review import build_review

_HYPOTHESES = [
    {"id": "h1", "statement": "onboarding is the blocker", "falsifier": "setup was easy"},
    {"id": "h2", "statement": "they export to spreadsheets", "falsifier": "nobody exports"},
]


def _fact(fact_id, session_id, *, bearings=(), provenance="evidence", quoted_text=None):
    claim = "they rebuilt the export by hand"
    return {
        "id": fact_id,
        "claim": claim,
        "provenance": provenance,
        "session_id": session_id,
        "seqs": [1],
        "quoted_text": claim if quoted_text is None else quoted_text,
        "bearings": [{"hypothesis_id": hid, "stance": stance} for hid, stance in bearings],
    }


def _node(tmp_path: Path, *, facts=None, hypotheses=_HYPOTHESES, raw_facts=None) -> Path:
    context = tmp_path / "projects" / "acme" / "context"
    context.mkdir(parents=True, exist_ok=True)
    (tmp_path / "projects" / "acme" / "sessions").mkdir(parents=True, exist_ok=True)
    if hypotheses is not None:
        (context / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    if raw_facts is not None:
        (context / "facts.json").write_text(raw_facts, encoding="utf-8")
    elif facts is not None:
        (context / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    return tmp_path


# --- the object reaches the wire ----------------------------------------------------------


def test_the_computed_summary_reaches_the_payload(tmp_path):
    root = _node(tmp_path, facts=[
        _fact("f1", "s1", bearings=(("h1", "supports"),)),
        _fact("f2", "s1", bearings=(("h2", "undermines"),), provenance="inference"),
        _fact("f3", "s1"),
    ])
    summary = build_review("acme", "s1", repo_root=root)["summary"]
    assert summary["fact_count"] == 3
    assert summary["evidence_facts"] == 2 and summary["inference_facts"] == 1
    assert summary["hypotheses_touched"] == 2
    assert summary["projects"] == ["acme"]
    # Always 1, and carried anyway: it is the field that says these numbers
    # describe one interview, beside standings that count many.
    assert summary["interviews"] == 1


def test_the_summary_counts_this_interview_not_the_corpus(tmp_path):
    # The flattering failure the whole page is built against: a corpus made
    # rich by five earlier calls reported as the size of this afternoon's work.
    root = _node(tmp_path, facts=[
        _fact("f1", "older", bearings=(("h1", "supports"),)),
        _fact("f2", "older", bearings=(("h2", "supports"),)),
        _fact("f3", "s-new", bearings=(("h1", "undermines"),)),
    ])
    summary = build_review("acme", "s-new", repo_root=root)["summary"]
    assert summary["fact_count"] == 1
    assert summary["hypotheses_touched"] == 1


def test_a_bearing_on_a_deleted_hypothesis_is_not_counted_as_a_hypothesis_touched(tmp_path):
    # The same rule the extraction banner already applies. A bearing on an id
    # since renamed or deleted is not a connection the operator can see, and
    # counting it reports the interview as better connected than the page is.
    root = _node(tmp_path, facts=[
        _fact("f1", "s1", bearings=(("h1", "supports"), ("gone-h9", "supports"))),
    ])
    summary = build_review("acme", "s1", repo_root=root)["summary"]
    assert summary["fact_count"] == 1
    assert summary["hypotheses_touched"] == 1


def test_an_interview_that_produced_nothing_summarizes_as_zero_rather_than_vanishing(tmp_path):
    root = _node(tmp_path, facts=[_fact("f1", "older", bearings=(("h1", "supports"),))])
    summary = build_review("acme", "s-new", repo_root=root)["summary"]
    assert summary is not None
    assert summary["fact_count"] == 0 and summary["hypotheses_touched"] == 0


# --- when there is nothing to count ------------------------------------------------------


def test_no_session_selected_yields_no_summary(tmp_path):
    root = _node(tmp_path, facts=[_fact("f1", "s1")])
    assert build_review("acme", None, repo_root=root)["summary"] is None


def test_an_unreadable_corpus_yields_no_summary_rather_than_zeros(tmp_path):
    # Zeros here would read as a finding about the interview -- "nothing was
    # extracted" -- when the truth is that the file could not be parsed. The
    # extraction banner above already names which of the two happened.
    root = _node(tmp_path, raw_facts="[{}]")
    review = build_review("acme", "s1", repo_root=root)
    assert review["facts_error"] is not None
    assert review["summary"] is None


# --- the wire ------------------------------------------------------------------------------


def test_the_payload_carries_one_summary_field_and_no_dead_status_string(tmp_path):
    # THE WIRE DEFECT. The server published `summary_status` and the client read
    # `data.summary`, so the server's words never reached a screen and the
    # sentence a reader saw could not be changed from the server at all. One
    # field, read by the one client.
    root = _node(tmp_path, facts=[_fact("f1", "s1")])
    review = build_review("acme", "s1", repo_root=root)
    assert "summary" in review
    assert "summary_status" not in review


def test_the_summary_carries_no_verbatim_customer_speech(tmp_path):
    # Counts and node names only. `InterviewSummary` is computed from `FactRef`,
    # which structurally cannot hold a quote -- this asserts the payload built
    # from it did not reach around for one.
    sentinel = "we went back to paper because the export broke"
    root = _node(tmp_path, facts=[
        _fact("f1", "s1", bearings=(("h1", "supports"),), quoted_text=sentinel),
    ])
    summary = build_review("acme", "s1", repo_root=root)["summary"]
    assert sentinel not in json.dumps(summary)


def test_the_page_makes_one_pass_over_the_corpus_for_the_summary(tmp_path, monkeypatch):
    # `summarize_interview` reads every fact on disk, and `standing_for` runs
    # once per hypothesis. Computing it there made a project with twenty
    # hypotheses do twenty identical passes for an answer that cannot differ
    # between them -- invisible in every assertion above, because all twenty
    # answers are correct.
    calls = []
    real = summary_mod.summarize_interview

    def counted(session_id, refs):
        calls.append(session_id)
        return real(session_id, refs)

    monkeypatch.setattr(summary_mod, "summarize_interview", counted)
    root = _node(tmp_path, facts=[
        _fact("f1", "s1", bearings=(("h1", "supports"),)),
        _fact("f2", "s1", bearings=(("h2", "supports"),)),
    ])
    build_review("acme", "s1", repo_root=root)
    assert calls == ["s1"], f"the summary was computed {len(calls)} times for one page"
