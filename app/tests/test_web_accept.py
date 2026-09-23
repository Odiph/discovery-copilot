"""The accept/reject card: the one place a coach proposal becomes a change.

D-01/D-03 say the coach only ever PROPOSES a hypothesis state; the set changes
on an explicit human accept. Until now the proposal died in the transcript
(`render.py` prints "PROPOSED (not applied)") and the only way to act on it was
to retype the state on the authoring page from memory. This module is that
missing step, and every test here is about the two ways it could go wrong:
acting on something the human did not accept, or writing a file it could not
first read.

A third way was found later and has its own section below: doing NOTHING with
something the human explicitly rejected. A rejection is real information about
a hypothesis -- someone read the evidence and judged it insufficient -- and it
used to vanish the instant the response was sent, so the same proposal came
back in a later interview with no trace that the operator had already
adjudicated it once.
"""

from __future__ import annotations

import json
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import STATIC_REVIEW_DIR
from app.web.accept import apply_decision, proposals_for, register_accept_routes
from app.web.review import create_review_app


def _session(tmp_path, session_id="S1", records=()):
    session_dir = tmp_path / "projects" / "acme" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records]
    (session_dir / "transcript.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return tmp_path / "projects" / "acme" / "sessions"


def _suggestion(seq, *, target=None, state=None, text="did that actually happen?"):
    return {
        "kind": "suggestion", "seq": seq, "trigger_seq": seq - 1, "text": text,
        "move": "test_hypothesis", "band": "green", "shown": True,
        "target_hypothesis_id": target, "proposed_state": state,
    }


def _context(tmp_path, hypotheses):
    context_dir = tmp_path / "projects" / "acme" / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    return context_dir


def _tree(tmp_path):
    """Every file anywhere under `tmp_path`. Deliberately the whole tree and
    not the context directory: a refusal that still left something behind
    somewhere unexpected -- a staged temp file, a log beside the sessions --
    is the interesting failure, and a check scoped to where the write was
    expected cannot see it (the pathspec defect gate 17 was fixed for)."""
    return {
        str(p.relative_to(tmp_path)).replace("\\", "/"): p.read_bytes()
        for p in sorted(tmp_path.rglob("*")) if p.is_file()
    }


def _hyp(hid="h1", state="untested", criterion="3 of 10 mention it"):
    return {
        "id": hid, "statement": f"statement {hid}", "falsifier": f"falsifier {hid}",
        "state": state, "criterion": criterion,
    }


# --- reading the proposals out of the interview ------------------------------------------


def test_a_suggestion_carrying_a_proposal_becomes_a_proposal(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    proposals = proposals_for(root, "S1", context_dir=None)
    assert [(p["target_hypothesis_id"], p["proposed_state"]) for p in proposals] == [
        ("h1", "supported")
    ]
    assert proposals[0]["questions"] == ["did that actually happen?"]


def test_a_suggestion_with_no_proposed_state_is_not_a_proposal(tmp_path):
    # Most suggestions in a session carry no proposal at all. Treating every
    # suggestion as a proposal would bury the handful that are real ones.
    root = _session(tmp_path, records=[_suggestion(4), _suggestion(6, target="h1")])
    assert proposals_for(root, "S1", context_dir=None) == []


def test_a_proposal_with_no_target_hypothesis_is_not_a_proposal(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, state="supported")])
    assert proposals_for(root, "S1", context_dir=None) == []


def test_the_same_proposal_made_twice_is_one_card_carrying_both_questions(tmp_path):
    # A coach that asked twice about the same belief has not doubled the
    # evidence -- two identical cards would invite two clicks for one decision.
    root = _session(tmp_path, records=[
        _suggestion(4, target="h1", state="supported", text="first ask"),
        _suggestion(9, target="h1", state="supported", text="second ask"),
    ])
    proposals = proposals_for(root, "S1", context_dir=None)
    assert len(proposals) == 1
    assert proposals[0]["questions"] == ["first ask", "second ask"]
    assert proposals[0]["seqs"] == [4, 9]


def test_two_different_states_for_one_hypothesis_stay_two_decisions(tmp_path):
    # Collapsing these would silently pick a winner between two proposals that
    # disagree -- exactly the judgement this card exists to leave with a human.
    root = _session(tmp_path, records=[
        _suggestion(4, target="h1", state="supported"),
        _suggestion(9, target="h1", state="contested"),
    ])
    assert len(proposals_for(root, "S1", context_dir=None)) == 2


def test_no_session_selected_yields_no_proposals(tmp_path):
    assert proposals_for(tmp_path / "sessions", None, context_dir=None) == []


def test_a_session_that_does_not_exist_yields_no_proposals(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    assert proposals_for(root, "nope", context_dir=None) == []


def test_an_unreadable_transcript_is_no_proposals_rather_than_a_crash(tmp_path):
    # The hub degrades per section; a broken transcript must not take the
    # standing card down with it.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    (root / "S1" / "transcript.jsonl").write_text("{oops\n{also oops\n", encoding="utf-8")
    assert proposals_for(root, "S1", context_dir=None) == []


# --- the write itself --------------------------------------------------------------------


def test_accepting_moves_only_the_targeted_hypothesis(tmp_path):
    context_dir = _context(tmp_path, [_hyp("h1"), _hyp("h2")])
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert error is None and written == 2
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert [(h["id"], h["state"]) for h in on_disk] == [("h1", "supported"), ("h2", "untested")]


def test_accepting_preserves_every_other_field_of_the_moved_hypothesis(tmp_path):
    # The criterion is the pre-registered rule the state was judged against.
    # A write that dropped it would erase the only thing making the new state
    # defensible.
    context_dir = _context(tmp_path, [_hyp("h1", criterion="3 of 10 unprompted mention X")])
    apply_decision(context_dir, hypothesis_id="h1", proposed_state="contested")
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert on_disk[0]["criterion"] == "3 of 10 unprompted mention X"
    assert on_disk[0]["falsifier"] == "falsifier h1"


def test_accepting_preserves_the_criterions_pre_registration_boundary(tmp_path):
    # The criterion and its boundary are one claim, and only the pair of them
    # is worth anything: the rule, plus what already existed when it was
    # written. Dropping the boundary on an accept would leave a criterion that
    # still LOOKS pre-registered and can no longer be checked -- silently
    # converting a verifiable claim into an unverifiable one, at exactly the
    # moment the state it justifies is being applied.
    context_dir = _context(
        tmp_path,
        [dict(_hyp("h1"), criterion_registered_after=["20260101T090000"]), _hyp("h2")],
    )
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert error is None and written == 2
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert on_disk[0]["criterion_registered_after"] == ["20260101T090000"]


def test_accepting_never_re_stamps_a_boundary(tmp_path):
    # `accept.py` has no sessions_root and must never acquire one for this. An
    # accept happens AFTER the interview it was proposed by, so a re-stamp here
    # would move every accepted hypothesis's boundary past the very evidence
    # the accept was judged on -- destroying the pre-registration of precisely
    # the hypotheses that were actually tested. The boundary moves on a
    # CRITERION edit and on nothing else.
    context_dir = _context(tmp_path, [dict(_hyp("h1"), criterion_registered_after=[])])
    (tmp_path / "projects" / "acme" / "sessions" / "20260101T090000").mkdir(parents=True)
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert error is None and written == 1
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert on_disk[0]["criterion_registered_after"] == []


def test_accepting_for_an_unknown_hypothesis_writes_nothing(tmp_path):
    context_dir = _context(tmp_path, [_hyp("h1")])
    before = _tree(tmp_path)
    written, error = apply_decision(context_dir, hypothesis_id="ghost", proposed_state="supported")
    assert written is None and error is not None and "ghost" in error
    assert _tree(tmp_path) == before


def test_accepting_an_unrecognized_state_writes_nothing(tmp_path):
    # Rejected by `Hypothesis`'s own state validator, reached through
    # author.py's validator -- never by a second rule written here.
    context_dir = _context(tmp_path, [_hyp("h1")])
    before = _tree(tmp_path)
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="probably_true")
    assert written is None and error is not None
    assert _tree(tmp_path) == before


def test_a_corpus_with_one_unreadable_item_is_never_clobbered(tmp_path):
    # The sneaky version of "reads before it writes", and the reason this
    # module reads through `corpus.load_hypotheses` rather than author.py's
    # deliberately TOLERANT `_read_existing`. The target hypothesis here is
    # perfectly valid, so a tolerant read finds it, moves it, and writes the
    # list back -- silently deleting `h2`, which the user authored and which
    # merely failed to parse. A wholly unparseable file would be caught by any
    # implementation; a file where only the OTHER item is broken is the one
    # that quietly loses data. Strict read, or no write at all.
    context_dir = _context(tmp_path, [
        {"id": "h1", "statement": "s1", "falsifier": "f1"},
        {"id": "h2", "statement": "s2"},  # no falsifier -- invalid (D-04)
    ])
    before = _tree(tmp_path)
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert written is None and error is not None and "h2" in error
    assert _tree(tmp_path) == before


def test_a_corpus_that_is_not_even_a_list_is_never_clobbered(tmp_path):
    context_dir = tmp_path / "projects" / "acme" / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "hypotheses.json").write_text('{"h1": "supported"}', encoding="utf-8")
    before = _tree(tmp_path)
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert written is None and error is not None
    assert _tree(tmp_path) == before


def test_accepting_with_no_corpus_at_all_writes_no_file(tmp_path):
    context_dir = tmp_path / "projects" / "acme" / "context"
    context_dir.mkdir(parents=True)
    written, error = apply_decision(context_dir, hypothesis_id="h1", proposed_state="supported")
    assert written is None and error is not None
    assert not (context_dir / "hypotheses.json").exists()


# --- the routes --------------------------------------------------------------------------


def _bare_app(tmp_path, session="S1"):
    """Deliberately NO middleware -- the guard under test is the route's own."""
    app = FastAPI()
    register_accept_routes(app, project="acme", session_id=session, repo_root=tmp_path)
    return app


def test_the_decision_route_refuses_a_non_loopback_host_with_no_middleware_present(tmp_path):
    # The write route carries its own guard rather than trusting whichever app
    # it is mounted on to have installed one -- mounting it somewhere without
    # the hub's middleware must not open a DNS-rebinding write.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"},
            headers={"Host": "evil.example.com"},
        )
    assert response.status_code == 403 and response.content == b""


def test_rejecting_leaves_the_hypothesis_artifact_byte_identical(tmp_path):
    # The half of the old behaviour that was right and must stay right: a
    # rejection is DEFINED by the corpus not moving. The decline log is written
    # beside it, never through it.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    before = (context_dir / "hypotheses.json").read_bytes()
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    assert response.status_code == 200 and response.json()["applied"] is False
    assert (context_dir / "hypotheses.json").read_bytes() == before


def test_rejecting_touches_exactly_one_file_and_it_is_the_decline_log(tmp_path):
    # The whole tree, on `_tree`'s own reasoning: a write that also left a
    # staged temp file behind, or dropped something beside the sessions, is the
    # interesting failure and a context-scoped check cannot see it.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    before = _tree(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    after = _tree(tmp_path)
    assert set(after) - set(before) == {"projects/acme/context/declined_proposals.json"}
    assert {path: after[path] for path in before} == before


def test_a_decision_that_is_neither_accept_nor_reject_is_refused(tmp_path):
    # A typo'd or absent verb must never fall through to the accept branch.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    before = _tree(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        for body in (
            {"hypothesis_id": "h1", "proposed_state": "supported"},
            {"hypothesis_id": "h1", "proposed_state": "supported", "decision": "Accept "},
            {"hypothesis_id": "h1", "proposed_state": "supported", "decision": True},
            ["h1", "supported", "accept"],
        ):
            assert client.post("/api/decision", json=body).status_code == 400
    assert _tree(tmp_path) == before


def test_a_body_that_is_not_json_is_a_400_not_a_500(tmp_path):
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision", content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400


def test_accepting_over_the_route_moves_the_hypothesis(tmp_path):
    context_dir = _context(tmp_path, [_hyp("h1"), _hyp("h2")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "contested", "decision": "accept"},
        )
    assert response.status_code == 200 and response.json()["applied"] is True
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert [(h["id"], h["state"]) for h in on_disk] == [("h1", "contested"), ("h2", "untested")]


# --- mounted on the hub, so evidence and decision are one sitting ------------------------


def _hub(tmp_path, session="S1"):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    return TestClient(app, base_url="http://127.0.0.1")


def test_the_hub_serves_the_proposals_beside_the_standing(tmp_path):
    # One port, one page: a second port would force a cross-origin fetch and a
    # CORS surface for the sake of the one write in the product.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    with _hub(tmp_path) as client:
        proposals = client.get("/api/proposals").json()
    assert [p["target_hypothesis_id"] for p in proposals["proposals"]] == ["h1"]


def test_the_hub_accepts_a_proposal(tmp_path):
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    with _hub(tmp_path) as client:
        assert client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"},
        ).status_code == 200
    on_disk = json.loads((context_dir / "hypotheses.json").read_text(encoding="utf-8"))
    assert on_disk[0]["state"] == "supported"


def test_the_hubs_own_middleware_still_refuses_a_cross_origin_content_type(tmp_path):
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    before = _tree(tmp_path)
    with _hub(tmp_path) as client:
        response = client.post(
            "/api/decision",
            content=json.dumps(
                {"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"}
            ),
            headers={"Content-Type": "text/plain"},
        )
    assert response.status_code == 403 and response.content == b""
    assert _tree(tmp_path) == before


def test_an_unresolvable_project_is_a_400_on_both_routes(tmp_path):
    app = FastAPI()
    register_accept_routes(app, project="../escape", session_id="S1", repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/proposals").status_code == 400
        assert client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"},
        ).status_code == 400


# --- a rejection is a record, not a shrug -------------------------------------------------
#
# What these prove: the operator's judgement that the evidence did NOT move
# them survives the click, is findable by (hypothesis, proposed state) the next
# time the coach raises it, and never puts the corpus at risk to get there.
#
# What they explicitly DO NOT prove, stated here rather than left to be
# assumed:
#
# - Nothing here proves the operator reads the marker or acts on it. That is a
#   claim about a human and this suite cannot make it. The JS suite proves the
#   words reach the DOM; that is the end of what is checkable.
# - Nothing here proves the log is durable against a crash MID-write. It is
#   written through `author.py::_atomic_write_json` (temp file, fsync,
#   os.replace), so it inherits whatever that primitive gives, and these tests
#   only ever observe the after state of a completed call.
# - `test_the_route_stamps_a_time_that_parses` proves the stamp is a real UTC
#   instant, not that it is the CORRECT instant -- nothing here has an
#   independent clock to check it against.


def _declines(context_dir):
    return json.loads((context_dir / "declined_proposals.json").read_text(encoding="utf-8"))


def _decline(hid="h1", state="supported", at="2026-03-04T09:15:00Z", session="S1"):
    return {
        "hypothesis_id": hid, "proposed_state": state, "declined_at": at, "session_id": session,
    }


def _write_declines(context_dir, records):
    (context_dir / "declined_proposals.json").write_text(
        json.dumps(records), encoding="utf-8"
    )


def test_rejecting_records_the_decision_rather_than_dropping_it(tmp_path):
    # The defect this whole section exists for. A rejection is real information
    # about a hypothesis -- someone read the evidence and judged it
    # insufficient -- and it used to vanish on the wire.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "contested", "decision": "reject"},
        )
    assert response.json()["recorded"] is True
    records = _declines(context_dir)
    assert len(records) == 1
    assert records[0]["hypothesis_id"] == "h1"
    assert records[0]["proposed_state"] == "contested"
    assert records[0]["session_id"] == "S1"
    assert records[0]["declined_at"]


def test_accepting_writes_no_decline(tmp_path):
    # An acceptance leaves its trace in `state`; a second record of it in the
    # decline log would be a plain lie about what the operator decided.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "accept"},
        )
    assert not (context_dir / "declined_proposals.json").exists()


def test_the_route_stamps_a_time_that_parses_as_a_real_utc_instant(tmp_path):
    from datetime import datetime, timezone

    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    stamped = _declines(context_dir)[0]["declined_at"]
    # A bare `datetime.fromisoformat` on 3.10 rejects the trailing Z, which is
    # the point: the stamp has to be an offset-aware instant, and a naive local
    # string must not pass this.
    parsed = datetime.fromisoformat(stamped.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 300


def test_a_second_rejection_of_the_same_proposal_is_appended_not_collapsed(tmp_path):
    # Two judgements made at two times are two facts. Collapsing them to one
    # would throw away the same information this log exists to stop throwing
    # away -- "I looked at this twice and twice it did not move me" is a
    # stronger statement than "I looked at this".
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        for _ in range(2):
            client.post(
                "/api/decision",
                json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
            )
    assert len(_declines(context_dir)) == 2


def test_declining_a_proposal_for_a_vanished_hypothesis_is_still_recorded(tmp_path):
    # Unlike an accept, which must refuse: there is nothing to move. But the
    # judgement was still made by a human, and refusing to keep it would lose
    # exactly what this log was added for.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "ghost", "proposed_state": "supported", "decision": "reject"},
        )
    assert response.status_code == 200 and response.json()["recorded"] is True
    assert _declines(context_dir)[0]["hypothesis_id"] == "ghost"


def test_a_decline_is_filed_against_the_interview_on_screen_not_the_command_line(tmp_path):
    # The card follows the hub's picker everywhere else; the record it writes
    # has to follow it too, or a decision made while reading interview S2 is
    # filed against S1 forever.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path, session_id="S1")
    _session(tmp_path, session_id="S2")
    with TestClient(_bare_app(tmp_path, session="S1"), base_url="http://127.0.0.1") as client:
        client.post("/api/decision", json={
            "hypothesis_id": "h1", "proposed_state": "supported",
            "decision": "reject", "session": "S2",
        })
    assert _declines(context_dir)[0]["session_id"] == "S2"


def test_a_decline_against_a_session_outside_the_project_records_nothing(tmp_path):
    # Membership of the real listing, exactly as the read route validates it --
    # this is a write path and `sessions_root / session` with a `..` in it
    # leaves the project.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    before = _tree(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post("/api/decision", json={
            "hypothesis_id": "h1", "proposed_state": "supported",
            "decision": "reject", "session": "../../elsewhere",
        })
    assert response.status_code == 400
    assert _tree(tmp_path) == before


# --- reading it back, which is the whole point ---------------------------------------------


def test_a_previously_declined_proposal_is_marked_when_it_is_seen_again(tmp_path):
    # Counts and session ids chosen so no number here can coincide with
    # another: two declines, three interviews on disk, two of them named in
    # the log -- a fixture where the count equalled the number of sessions
    # would pass on an implementation that reported either one.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    _session(tmp_path, session_id="S2")
    _session(tmp_path, session_id="S3")
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [
        _decline(at="2026-03-04T09:15:00Z", session="S2"),
        _decline(at="2026-06-30T17:45:00Z", session="S3"),
    ])
    proposal = proposals_for(root, "S1", context_dir=context_dir)[0]
    assert proposal["previously_declined"] == {
        "count": 2,
        "last_declined_at": "2026-06-30T17:45:00Z",
        "sessions": ["S2", "S3"],
    }


def test_a_proposal_nobody_has_declined_carries_an_explicit_null(tmp_path):
    # Present and null, never absent: a missing key arrives in the client as
    # `undefined`, which is how a page ends up rendering nothing where a
    # warning belongs.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    proposal = proposals_for(root, "S1", context_dir=context_dir)[0]
    assert "previously_declined" in proposal
    assert proposal["previously_declined"] is None


def test_a_decline_of_a_different_state_does_not_mark_this_proposal(tmp_path):
    # Keyed on the PAIR, like the deduplication above it. Two different
    # proposed states for one hypothesis genuinely disagree, so a decline of
    # one must never be reported as a decline of the other.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [_decline(hid="h1", state="contested")])
    assert proposals_for(root, "S1", context_dir=context_dir)[0]["previously_declined"] is None


def test_a_decline_of_a_different_hypothesis_does_not_mark_this_proposal(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [_decline(hid="h2", state="supported")])
    assert proposals_for(root, "S1", context_dir=context_dir)[0]["previously_declined"] is None


def test_a_previously_declined_proposal_is_still_offered_for_decision(tmp_path):
    # Marked, never hidden. Hiding it would replace one silent loss with
    # another, and a proposal declined in March may be right in June.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [_decline()])
    proposals = proposals_for(root, "S1", context_dir=context_dir)
    assert len(proposals) == 1
    assert proposals[0]["questions"] == ["did that actually happen?"]


def test_a_decline_log_that_will_not_parse_costs_a_badge_not_the_card(tmp_path):
    # The hub's per-section degradation, one level down: a log this page
    # cannot read must not take the proposals with it.
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    (context_dir / "declined_proposals.json").write_text("{oops", encoding="utf-8")
    proposals = proposals_for(root, "S1", context_dir=context_dir)
    assert len(proposals) == 1 and proposals[0]["previously_declined"] is None


def test_a_decline_log_that_is_not_a_list_costs_a_badge_not_the_card(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    (context_dir / "declined_proposals.json").write_text('{"h1": "supported"}', encoding="utf-8")
    assert proposals_for(root, "S1", context_dir=context_dir)[0]["previously_declined"] is None


def test_an_unkeyable_record_is_skipped_for_display_without_losing_the_others(tmp_path):
    root = _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [
        "not even an object",
        {"hypothesis_id": "h1"},  # no proposed_state -- cannot be keyed
        _decline(at="2026-05-05T05:05:00Z", session="S9"),
    ])
    assert proposals_for(root, "S1", context_dir=context_dir)[0]["previously_declined"] == {
        "count": 1, "last_declined_at": "2026-05-05T05:05:00Z", "sessions": ["S9"],
    }


def test_the_hub_marks_a_previously_declined_proposal_over_the_route(tmp_path):
    # The classifier run over the REAL path the page uses, not just over the
    # function -- a route that forgot to pass its own context dir would leave
    # every marker null with the unit tests above still green.
    _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    context_dir = _context(tmp_path, [_hyp("h1")])
    _write_declines(context_dir, [_decline()])
    with _hub(tmp_path) as client:
        proposals = client.get("/api/proposals").json()["proposals"]
    assert proposals[0]["previously_declined"]["count"] == 1


def test_a_rejection_over_the_hub_is_visible_on_the_very_next_read(tmp_path):
    # End to end over the real app: click reject, re-read the card, and the
    # marker is there. This is the loop the whole feature is for.
    _context(tmp_path, [_hyp("h1")])
    _session(tmp_path, records=[_suggestion(4, target="h1", state="supported")])
    with _hub(tmp_path) as client:
        assert client.get("/api/proposals").json()["proposals"][0]["previously_declined"] is None
        client.post("/api/decision", json={
            "hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject",
        })
        marked = client.get("/api/proposals").json()["proposals"][0]["previously_declined"]
    assert marked["count"] == 1 and marked["sessions"] == ["S1"]


# --- never destroying what is already there -----------------------------------------------


def test_a_decline_log_that_will_not_parse_is_never_clobbered(tmp_path):
    # The same rule `apply_decision` follows for the corpus: a file we cannot
    # read is a file we refuse to replace. A tolerant read here would answer
    # "empty log", append one record, and put back a one-item array over
    # whatever the operator actually had.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    (context_dir / "declined_proposals.json").write_text("[{half a rec", encoding="utf-8")
    before = _tree(tmp_path)
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    body = response.json()
    assert response.status_code == 200
    assert body["applied"] is False and body["recorded"] is False and body["record_error"]
    assert _tree(tmp_path) == before


def test_a_record_written_by_a_future_version_survives_a_new_decline(tmp_path):
    # `record_decline` re-serializes the existing array UNREAD. A field this
    # version has never heard of must not be normalised away by this version
    # -- that is how a forward-compatible append-only log stops being one.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    _write_declines(context_dir, [
        dict(_decline(), reason_note="felt like one interviewee, not a pattern", schema=7),
    ])
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    records = _declines(context_dir)
    assert len(records) == 2
    assert records[0]["reason_note"] == "felt like one interviewee, not a pattern"
    assert records[0]["schema"] == 7


def test_an_unkeyable_record_is_preserved_on_disk_even_though_it_is_not_displayed(tmp_path):
    # The read is tolerant and the write is not allowed to act on that
    # tolerance. Skipping a record for DISPLAY is a rendering decision;
    # deleting it would be data loss dressed up as one.
    context_dir = _context(tmp_path, [_hyp("h1")])
    _session(tmp_path)
    _write_declines(context_dir, ["not even an object"])
    with TestClient(_bare_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.post(
            "/api/decision",
            json={"hypothesis_id": "h1", "proposed_state": "supported", "decision": "reject"},
        )
    assert _declines(context_dir)[0] == "not even an object"


# --- the two buttons stay the same weight --------------------------------------------------


def test_accept_and_reject_carry_no_styling_of_their_own(tmp_path):
    """The rule `review.css` states in prose, asserted rather than trusted:
    making accept the loud default is how a proposal becomes a rubber stamp.

    Checked as "neither class is ever a selector", which is stronger than
    comparing two declaration blocks -- it forbids the divergence rather than
    catching one particular shape of it.

    What it CANNOT prove: this reads `review.css` only, so a rule arriving in
    another stylesheet, an inline style written by `review.js`, or a
    difference produced by a `:hover`/`:focus` state on a shared selector
    would all pass. The button nodes get their classes from `proposalCard`
    and nothing in this suite renders them through a real CSS engine."""
    css = (STATIC_REVIEW_DIR / "review.css").read_text(encoding="utf-8")
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selectors = [block.split("{", 1)[0] for block in without_comments.split("}") if "{" in block]
    offenders = [s.strip() for s in selectors if ".accept" in s or ".reject" in s]
    assert offenders == [], (
        "accept and reject are the same visual weight on purpose -- a selector naming "
        f"one of them alone is how that stops being true: {offenders}"
    )
    # Selector EQUALITY, not substring. The first draft of this line asked
    # whether the string ".actions button" appeared anywhere, which
    # `.actions button:disabled` satisfies on its own -- so renaming the shared
    # rule out from under both buttons left the assertion green. It survived a
    # mutation, and this is the fix.
    stripped = {" ".join(s.split()) for s in selectors}
    assert ".actions button" in stripped, (
        "the ONE rule both buttons share is gone -- with no shared rule there is nothing "
        f"holding the two at the same weight: {sorted(stripped)}"
    )


def test_the_page_offers_accept_and_reject(tmp_path):
    source = (STATIC_REVIEW_DIR / "review.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.strip().startswith("//"))
    assert "/api/proposals" in code and "/api/decision" in code
    assert "Accept" in code and "Reject" in code
