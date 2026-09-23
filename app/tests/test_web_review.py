"""The hub: where each hypothesis stands after an interview.

The properties defended here are the ones the competitive research said the
category gets wrong. Chiefly: both evidence columns always exist, counts are
per-INTERVIEW rather than per-fact, and a hypothesis with no pre-registered
criterion is marked unscoreable rather than shown a confident-looking tally.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.coach.corpus import Bearing, Fact, Hypothesis
from app.main import STATIC_REVIEW_DIR
from app.web.review import build_review, create_review_app, standing_for


def _hypothesis(hid="h1", criterion=""):
    return Hypothesis(
        id=hid, statement=f"statement {hid}", falsifier=f"falsifier {hid}", criterion=criterion
    )


def _fact(fid, session_id, hid, stance, *, claim="a claim", seqs=(1,)):
    return Fact(
        id=fid, session_id=session_id, seqs=seqs, claim=claim,
        provenance="evidence", quoted_text=claim,
        bearings=(Bearing(hypothesis_id=hid, stance=stance),),
    )


# --- both columns always exist -----------------------------------------------------------


def test_supporting_and_contrary_are_separate_lists():
    standing = standing_for(
        _hypothesis(),
        (_fact("f1", "S1", "h1", "supports"), _fact("f2", "S2", "h1", "undermines")),
    )
    assert len(standing["supporting"]) == 1 and len(standing["contrary"]) == 1


def test_the_contrary_column_exists_even_with_nothing_in_it():
    # The monotonic count `Bearing` was introduced to kill. A supporting-only
    # view makes disconfirmation invisible rather than merely absent, and the
    # page draws this empty list as a question rather than a blank.
    standing = standing_for(_hypothesis(), (_fact("f1", "S1", "h1", "supports"),))
    assert standing["contrary"] == []
    assert "contrary" in standing and "contrary_interviews" in standing


def test_an_unclear_bearing_is_neither_for_nor_against():
    # Forcing a direction manufactures evidence, and the manufactured
    # direction is almost always `supports`.
    standing = standing_for(_hypothesis(), (_fact("f1", "S1", "h1", "unclear"),))
    assert standing["supporting"] == [] and standing["contrary"] == []
    assert len(standing["unclear"]) == 1


def test_a_fact_bearing_on_another_hypothesis_is_ignored():
    standing = standing_for(_hypothesis("h1"), (_fact("f1", "S1", "h2", "supports"),))
    assert standing["supporting"] == [] and standing["contrary"] == []


# --- counts are per interview, never per fact --------------------------------------------


def test_four_facts_from_one_interview_count_as_one_interview():
    # Counting facts would let one talkative interviewee who repeated
    # themselves four times outweigh four separate people -- the vote-counting
    # failure evidence synthesis abandoned decades ago.
    facts = tuple(
        _fact(f"f{n}", "S1", "h1", "supports", claim=f"claim {n}") for n in range(4)
    )
    standing = standing_for(_hypothesis(), facts)
    assert len(standing["supporting"]) == 4
    assert standing["supporting_interviews"] == 1


def test_two_interviews_count_as_two():
    standing = standing_for(
        _hypothesis(), (_fact("f1", "S1", "h1", "supports"), _fact("f2", "S2", "h1", "supports"))
    )
    assert standing["supporting_interviews"] == 2


def test_supporting_and_contrary_interviews_are_counted_separately():
    standing = standing_for(
        _hypothesis(),
        (
            _fact("f1", "S1", "h1", "supports"),
            _fact("f2", "S2", "h1", "undermines"),
            _fact("f3", "S3", "h1", "undermines"),
        ),
    )
    assert standing["supporting_interviews"] == 1
    assert standing["contrary_interviews"] == 2


# --- what THIS interview did -------------------------------------------------------------


def test_a_fact_from_this_interview_is_marked_as_new():
    # The entire point of the page. A running total cannot tell you whether
    # the call you just finished changed anything.
    standing = standing_for(
        _hypothesis(),
        (_fact("f1", "S1", "h1", "supports"), _fact("f2", "S2", "h1", "supports")),
        session_id="S2",
    )
    marked = [f for f in standing["supporting"] if f["from_this_interview"]]
    assert [f["fact_id"] for f in marked] == ["f2"]


def test_a_hypothesis_this_interview_did_not_touch_says_so():
    standing = standing_for(
        _hypothesis(), (_fact("f1", "S1", "h1", "supports"),), session_id="S9"
    )
    assert standing["touched_this_interview"] is False


def test_an_unclear_bearing_still_counts_as_touching_the_hypothesis():
    # "We asked and it did not settle anything" is a real outcome of an
    # interview, and reporting it as untouched would hide that the question
    # was put at all.
    standing = standing_for(
        _hypothesis(), (_fact("f1", "S1", "h1", "unclear"),), session_id="S1"
    )
    assert standing["touched_this_interview"] is True


# --- the DELTA: what this interview changed, not merely where things stand ---------------


def test_the_standing_carries_a_before_as_well_as_an_after():
    # A level is not a delta. "2 for / 1 against" cannot tell an operator
    # whether the call they just finished did anything; "was 2 for / 0 against
    # -> now 2 for / 1 against" is the only shape that can.
    standing = standing_for(
        _hypothesis(),
        (
            _fact("f1", "S1", "h1", "supports"),
            _fact("f2", "S2", "h1", "supports"),
            _fact("f3", "S3", "h1", "undermines"),
        ),
        session_id="S3",
    )
    assert standing["before"]["supporting_interviews"] == 2
    assert standing["before"]["contrary_interviews"] == 0
    assert standing["supporting_interviews"] == 2
    assert standing["contrary_interviews"] == 1


def test_a_talkative_interviewee_repeating_themselves_cannot_raise_the_count():
    # The sneaky version of the vote-counting failure: not four facts in the
    # corpus, but four facts from ONE session all landing in the delta. The
    # interview count must move by exactly one however many times they said it.
    facts = tuple(
        _fact(f"f{n}", "S1", "h1", "supports", claim=f"claim {n}") for n in range(4)
    )
    standing = standing_for(_hypothesis(), facts, session_id="S1")
    assert standing["before"]["supporting_interviews"] == 0
    assert standing["supporting_interviews"] == 1
    # The fact count is reported too, but NAMED as facts, beside a number named
    # interviews -- a reader must never have to remember which kind this is.
    assert standing["this_interview"]["supporting_facts"] == 4


def test_a_corpus_made_rich_by_earlier_interviews_cannot_look_like_this_interviews_work():
    # The failure this prevents is the flattering one: a page that shows a big
    # standing after a call that contributed nothing, and lets the operator
    # read the size of the corpus as the size of their afternoon.
    facts = tuple(_fact(f"f{n}", f"S{n}", "h1", "supports") for n in range(5))
    standing = standing_for(_hypothesis(), facts, session_id="S9")
    assert standing["before"]["supporting_interviews"] == 5
    assert standing["supporting_interviews"] == 5
    assert standing["this_interview"] == {
        "supporting_facts": 0, "contrary_facts": 0, "unclear_facts": 0
    }
    assert standing["touched_this_interview"] is False
    assert standing["moved_this_interview"] is False


def test_a_fact_from_this_interview_is_excluded_from_the_before():
    # The sneakiest wrong "before" is the one computed by subtracting a count
    # rather than by re-counting without this session: with the same
    # interviewee appearing twice in the corpus that subtraction is off by one.
    standing = standing_for(
        _hypothesis(),
        (_fact("f1", "S1", "h1", "supports"), _fact("f2", "S1", "h1", "supports")),
        session_id="S1",
    )
    assert standing["before"]["supporting_interviews"] == 0


def test_with_no_session_selected_the_before_is_the_standing_itself():
    # Nothing was subtracted, so nothing may be claimed as changed.
    standing = standing_for(_hypothesis(), (_fact("f1", "S1", "h1", "supports"),))
    assert standing["before"]["supporting_interviews"] == standing["supporting_interviews"]
    assert standing["moved_this_interview"] is False


# --- touched is not moved ----------------------------------------------------------------


def test_an_unclear_only_fact_touches_the_hypothesis_without_moving_it():
    # A call that produced one claim the model could not direct must not light
    # up the badge that carries the whole thesis of this product. "We asked and
    # it did not settle" is a real outcome and is reported -- as touched.
    standing = standing_for(
        _hypothesis(), (_fact("f1", "S1", "h1", "unclear"),), session_id="S1"
    )
    assert standing["touched_this_interview"] is True
    assert standing["moved_this_interview"] is False
    assert standing["before"]["unclear_interviews"] == 0
    assert standing["unclear_interviews"] == 1


def test_a_directed_fact_from_this_interview_does_move_it():
    for stance in ("supports", "undermines"):
        standing = standing_for(
            _hypothesis(), (_fact("f1", "S1", "h1", stance),), session_id="S1"
        )
        assert standing["moved_this_interview"] is True, stance


def test_an_unclear_fact_beside_a_directed_one_still_reads_as_moved():
    standing = standing_for(
        _hypothesis(),
        (_fact("f1", "S1", "h1", "unclear"), _fact("f2", "S1", "h1", "undermines")),
        session_id="S1",
    )
    assert standing["moved_this_interview"] is True
    assert standing["this_interview"]["unclear_facts"] == 1


# --- one home for the counting rule ------------------------------------------------------


def test_the_hub_counts_through_the_one_home_for_the_rule(monkeypatch):
    # Not a source grep: the hub is made to report a number only
    # `interview_summary.count_interviews` can produce. A re-forked count --
    # even one that happens to agree today -- cannot answer 41.
    from app.coach import interview_summary as summary_mod

    monkeypatch.setattr(summary_mod, "count_interviews", lambda refs: 41 if refs else 0)
    standing = standing_for(_hypothesis(), (_fact("f1", "S1", "h1", "supports"),))
    assert standing["supporting_interviews"] == 41


def test_the_unclear_column_is_counted_per_interview_like_the_other_two():
    standing = standing_for(
        _hypothesis(),
        (_fact("f1", "S1", "h1", "unclear"), _fact("f2", "S1", "h1", "unclear")),
    )
    assert standing["unclear_interviews"] == 1


# --- the pre-registered criterion --------------------------------------------------------


def test_a_hypothesis_without_a_criterion_is_reported_unscoreable():
    standing = standing_for(_hypothesis(), (_fact("f1", "S1", "h1", "supports"),))
    assert standing["is_scorable"] is False


def test_a_hypothesis_with_a_criterion_carries_it_to_the_page():
    standing = standing_for(_hypothesis(criterion="3 of 10 unprompted mention X"), ())
    assert standing["is_scorable"] is True
    assert standing["criterion"] == "3 of 10 unprompted mention X"


def test_the_falsifier_reaches_the_page_beside_the_statement():
    # The direction of every bearing was judged against it, so a reader
    # checking a stance needs it in front of them.
    assert standing_for(_hypothesis(), ())["falsifier"] == "falsifier h1"


# --- assembling the whole review ---------------------------------------------------------


def _project(tmp_path, *, hypotheses=None, facts=None, session=None, segments=()):
    context_dir = tmp_path / "projects" / "acme" / "context"
    context_dir.mkdir(parents=True)
    if hypotheses is not None:
        (context_dir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    if facts is not None:
        (context_dir / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    if session is not None:
        session_dir = tmp_path / "projects" / "acme" / "sessions" / session
        session_dir.mkdir(parents=True)
        lines = [json.dumps({"kind": "segment", "seq": i, "speaker": sp, "text": tx})
                 for i, (sp, tx) in enumerate(segments)]
        (session_dir / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return context_dir


def test_the_review_carries_the_transcript_with_readable_speaker_labels(tmp_path):
    _project(tmp_path, session="S1", segments=(("interviewee", "we went back to paper"),))
    review = build_review("acme", "S1", repo_root=tmp_path)
    assert review["transcript"][0]["speaker"] == "them"
    assert review["transcript"][0]["text"] == "we went back to paper"


def test_a_malformed_corpus_does_not_cost_the_transcript(tmp_path):
    # The operator has just finished an interview, and the transcript is the
    # part that cannot be regenerated.
    context_dir = _project(tmp_path, session="S1", segments=(("interviewer", "hello"),))
    (context_dir / "facts.json").write_text("[{}]", encoding="utf-8")
    review = build_review("acme", "S1", repo_root=tmp_path)
    assert review["facts_error"] is not None
    assert len(review["transcript"]) == 1


def test_a_project_with_no_hypotheses_reviews_without_raising(tmp_path):
    _project(tmp_path, session="S1", segments=(("interviewer", "hello"),))
    assert build_review("acme", "S1", repo_root=tmp_path)["standings"] == []


def test_a_project_with_no_corpus_at_all_still_summarizes(tmp_path):
    # No `facts.json` is not an error, and the summary must still be an object
    # of zeros rather than the `null` an unreadable corpus produces -- the two
    # render different sentences. `test_web_review_summary.py` holds the rest.
    _project(tmp_path, session="S1")
    review = build_review("acme", "S1", repo_root=tmp_path)
    assert review["summary"]["fact_count"] == 0
    assert "summary_status" not in review


def test_a_session_that_does_not_exist_yields_an_empty_transcript(tmp_path):
    _project(tmp_path)
    assert build_review("acme", "nope", repo_root=tmp_path)["transcript"] == []


# --- the same guards the other two loopback pages have -----------------------------------


def _client(tmp_path, session="S1"):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    return TestClient(app, base_url="http://127.0.0.1")


def test_a_non_loopback_host_header_is_refused(tmp_path):
    _project(tmp_path)
    with _client(tmp_path) as client:
        response = client.get("/api/review", headers={"Host": "evil.example.com"})
    assert response.status_code == 403 and response.content == b""


def test_the_page_and_its_assets_are_served(tmp_path):
    _project(tmp_path)
    with _client(tmp_path) as client:
        for path in ("/", "/review.js", "/review.css"):
            assert client.get(path).status_code == 200


def test_the_client_never_uses_innerhtml():
    # This page renders verbatim customer speech. Comment lines are stripped
    # first -- the first draft of the console's identical guard failed against
    # its own explanatory comment.
    source = (STATIC_REVIEW_DIR / "review.js").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("//")
    )
    assert "innerHTML" not in code and "textContent" in code


def _tree(tmp_path):
    """Every file anywhere under `tmp_path`, mapped to its bytes.

    The WHOLE tree, deliberately -- not `tmp_path / "projects"`. Gate 17 had
    exactly this defect: a check scoped to the directory the write was expected
    in cannot see a write that lands somewhere else, which is the interesting
    failure rather than the boring one.
    """
    return {
        str(p.relative_to(tmp_path)).replace("\\", "/"): p.read_bytes()
        for p in sorted(tmp_path.rglob("*")) if p.is_file()
    }


def test_a_get_touches_nothing(tmp_path):
    # NARROWED, not deleted, in the commit that added the accept/reject card.
    # The hub is no longer write-free as a whole -- POSTing a decision writes
    # `context/hypotheses.json`, which is what gate 19's pinned set was widened
    # to two modules for. What survives unchanged is that merely LOOKING at the
    # page changes nothing on disk: rendering must never have a side effect,
    # or an operator refreshing the hub after a call would silently mutate
    # their own corpus. The POST's blast radius is pinned separately below.
    _project(
        tmp_path,
        hypotheses=[{"id": "h1", "statement": "s", "falsifier": "f"}],
        session="S1", segments=(("interviewer", "hello"),),
    )
    before = _tree(tmp_path)
    with _client(tmp_path) as client:
        client.get("/api/review")
        client.get("/api/proposals")
        client.get("/")
    assert _tree(tmp_path) == before


def test_a_post_writes_the_hypotheses_file_and_nothing_else_anywhere(tmp_path):
    # The guard: an accept must move a hypothesis state and touch NOTHING else
    # -- no decision log, no session-directory scribble, no stray temp file
    # left behind by the atomic write, and nothing outside `context/` at all.
    # The comparison is over the whole tmp tree (see `_tree`), so a write that
    # lands in a directory this test never thought to look in still fails here.
    _project(
        tmp_path,
        hypotheses=[{"id": "h1", "statement": "s", "falsifier": "f"}],
        session="S1", segments=(("interviewer", "hello"),),
    )
    before = _tree(tmp_path)
    with _client(tmp_path) as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"},
        )
    assert response.status_code == 200
    after = _tree(tmp_path)
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    assert changed == {"projects/acme/context/hypotheses.json"}
    assert json.loads(after["projects/acme/context/hypotheses.json"])[0]["state"] == "supported"


def test_a_refused_post_writes_nothing_anywhere(tmp_path):
    # Validation happens entirely before the atomic write, so a payload the
    # corpus schema refuses must leave the tree byte-identical -- including no
    # half-written `.tmp` beside the target.
    _project(
        tmp_path,
        hypotheses=[{"id": "h1", "statement": "s", "falsifier": "f"}],
        session="S1", segments=(("interviewer", "hello"),),
    )
    before = _tree(tmp_path)
    with _client(tmp_path) as client:
        assert client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "nonsense", "decision": "accept"},
        ).status_code == 400
    assert _tree(tmp_path) == before


def test_the_module_contains_no_write_primitive():
    source = (STATIC_REVIEW_DIR.parent.parent / "web" / "review.py").read_text(encoding="utf-8")
    for forbidden in ("write_text", "write_bytes", "os.replace"):
        assert forbidden not in source, f"the hub must never write: found {forbidden!r}"
