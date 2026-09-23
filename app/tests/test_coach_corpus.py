"""Tests for `app/coach/corpus.py` -- the no-default discipline on
`provenance`/`falsifier` (D-04/D-07), the pinned hypothesis state set (D-05),
the shipped format samples, and the no-seed-path assertion (D-01).
05-01-PLAN.md Task 2. Written in matched PAIRS, following
`test_coach_context.py:79`/`:91`'s own convention: one test asserting the
rejection, one asserting specifically that the value was NOT coerced --
the second of each pair is what D-07/D-04 actually buy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.coach import corpus, corpus_store
from app.session.store import SessionStore, create_session_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONTEXT_DIR = REPO_ROOT / "context"


def _fact(*, id, session_id, seqs=(0,), claim="some claim", provenance="evidence", quoted_text="q", related=()):
    return corpus.Fact(
        id=id, session_id=session_id, seqs=tuple(seqs), claim=claim, provenance=provenance,
        quoted_text=quoted_text,
        bearings=tuple(corpus.Bearing(hypothesis_id=h, stance="supports") for h in related),
    )


def _write_facts(tmp_path: Path, entries: list) -> Path:
    context_dir = tmp_path / "context"
    context_dir.mkdir(exist_ok=True)
    (context_dir / corpus.FACTS_FILENAME).write_text(json.dumps(entries), encoding="utf-8")
    return context_dir


def _write_hypotheses(tmp_path: Path, entries: list) -> Path:
    context_dir = tmp_path / "context"
    context_dir.mkdir(exist_ok=True)
    (context_dir / corpus.HYPOTHESES_FILENAME).write_text(json.dumps(entries), encoding="utf-8")
    return context_dir


def _copy_example(tmp_path: Path, example_filename: str, target_filename: str) -> Path:
    context_dir = tmp_path / "context"
    context_dir.mkdir(exist_ok=True)
    (context_dir / target_filename).write_text(
        (REPO_CONTEXT_DIR / example_filename).read_text(encoding="utf-8"), encoding="utf-8"
    )
    return context_dir


def _base_fact_entry(**overrides) -> dict:
    entry = {
        "id": "f-bad", "session_id": "s1", "seqs": [0], "claim": "some claim",
        "quoted_text": "", "bearings": [],
    }
    entry.update(overrides)
    return entry


# --- D-07: provenance has no default and is never coerced -----------------------------


@pytest.mark.parametrize(
    "provenance_value",
    ["misspelled", "", "Evidence", "EVIDENCE", None],
    ids=["misspelled", "empty", "wrong-case", "upper", "absent-value"],
)
def test_load_facts_rejects_a_malformed_provenance_naming_the_fact_id(tmp_path, provenance_value):
    entry = _base_fact_entry()
    if provenance_value is not None:
        entry["provenance"] = provenance_value
    context_dir = _write_facts(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_facts(context_dir, sessions_root=None)
    assert "f-bad" in str(excinfo.value), "the error must name the offending fact's id, not just say 'invalid'"


def test_load_facts_never_coerces_a_malformed_provenance_to_evidence(tmp_path):
    # Directly asserts the D-07 property: a malformed entry never silently
    # becomes a validated Fact with provenance == "evidence" -- it is a
    # raised error every time, not a coerced value we could inspect.
    entry = _base_fact_entry(provenance="maybe")
    context_dir = _write_facts(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError):
        corpus.load_facts(context_dir, sessions_root=None)


def test_load_facts_absent_provenance_field_entirely_raises_naming_the_fact(tmp_path):
    entry = _base_fact_entry()  # no "provenance" key at all
    context_dir = _write_facts(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_facts(context_dir, sessions_root=None)
    assert "f-bad" in str(excinfo.value)


# --- D-04: falsifier has no default and is never coerced -------------------------------


def test_load_hypotheses_rejects_an_absent_falsifier_naming_the_id(tmp_path):
    entry = {"id": "h-bad", "statement": "some hypothesis"}
    context_dir = _write_hypotheses(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_hypotheses(context_dir, missing_ok=False)
    assert "h-bad" in str(excinfo.value)


def test_load_hypotheses_never_loads_a_hypothesis_carrying_an_empty_falsifier(tmp_path):
    # Directly asserts the D-04 property: an absent falsifier never silently
    # becomes a validated Hypothesis with falsifier == "" -- it is a raised
    # error every time, not a coerced empty string we could inspect.
    entry = {"id": "h-bad", "statement": "some hypothesis", "falsifier": ""}
    context_dir = _write_hypotheses(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError):
        corpus.load_hypotheses(context_dir, missing_ok=False)


def test_load_hypotheses_rejects_a_whitespace_only_falsifier_identically(tmp_path):
    # An empty string is a valid STRING and is not a valid falsifier -- both
    # must be refused the same way.
    entry = {"id": "h-bad", "statement": "some hypothesis", "falsifier": "   \t  "}
    context_dir = _write_hypotheses(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_hypotheses(context_dir, missing_ok=False)
    assert "h-bad" in str(excinfo.value)


# --- D-05: state defaults to untested (correctly, not tolerantly) ----------------------


def test_hypothesis_with_no_state_defaults_correctly_to_untested(tmp_path):
    entry = {"id": "h1", "statement": "x", "falsifier": "y"}
    context_dir = _write_hypotheses(tmp_path, [entry])
    loaded = corpus.load_hypotheses(context_dir, missing_ok=False)
    assert loaded[0].state == corpus.HYPOTHESIS_STATE_UNTESTED


@pytest.mark.parametrize("state", sorted(corpus.ALL_HYPOTHESIS_STATES))
def test_load_hypotheses_accepts_every_pinned_state(tmp_path, state):
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "state": state}
    context_dir = _write_hypotheses(tmp_path, [entry])
    loaded = corpus.load_hypotheses(context_dir, missing_ok=False)
    assert loaded[0].state == state


def test_hypothesis_state_set_is_pinned_by_exact_set_equality():
    # Containment would let a fifth state slip in unnoticed -- exact
    # equality is the shape gate 15 already established for a different
    # pinned set in this same codebase.
    #
    # `contested` was added here in the SAME commit that added it to
    # corpus.py, which is this pin working rather than being worked around:
    # the fourth state could not appear without a human editing this line and
    # writing down why. The why is that a hypothesis with real evidence on
    # both sides otherwise sits in `untested`, indistinguishable from one
    # nobody has looked at -- opposite situations needing opposite actions.
    assert corpus.ALL_HYPOTHESIS_STATES == {
        "untested", "supported", "contradicted", "contested",
    }


def test_load_hypotheses_rejects_an_unrecognized_state(tmp_path):
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "state": "confirmed"}
    context_dir = _write_hypotheses(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_hypotheses(context_dir, missing_ok=False)
    assert "h1" in str(excinfo.value)


# --- duplicate ids ----------------------------------------------------------------------


def test_load_facts_rejects_a_duplicate_id_naming_it(tmp_path):
    entries = [
        _base_fact_entry(id="dup", seqs=[0], provenance="evidence", quoted_text="a"),
        _base_fact_entry(id="dup", seqs=[1], provenance="inference"),
    ]
    context_dir = _write_facts(tmp_path, entries)
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_facts(context_dir, sessions_root=None)
    assert "dup" in str(excinfo.value)


def test_load_hypotheses_rejects_a_duplicate_id_naming_it(tmp_path):
    entries = [
        {"id": "dup", "statement": "first", "falsifier": "y"},
        {"id": "dup", "statement": "second", "falsifier": "z"},
    ]
    context_dir = _write_hypotheses(tmp_path, entries)
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_hypotheses(context_dir, missing_ok=False)
    assert "dup" in str(excinfo.value)


# --- extra="forbid": a field nobody designed is rejected, never absorbed ---------------


def test_load_facts_rejects_an_extra_field_nobody_designed(tmp_path):
    entry = _base_fact_entry(provenance="evidence", quoted_text="a", extraction_timestamp="2026-01-01T00:00:00Z")
    context_dir = _write_facts(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError):
        corpus.load_facts(context_dir, sessions_root=None)


def test_load_hypotheses_rejects_an_extra_field_nobody_designed(tmp_path):
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "priority": "high"}
    context_dir = _write_hypotheses(tmp_path, [entry])
    with pytest.raises(corpus.CorpusValidationError):
        corpus.load_hypotheses(context_dir, missing_ok=False)


# --- the shipped format samples load through the real loaders --------------------------


def test_hypotheses_example_sample_loads_through_the_real_loader_with_every_field_intact(tmp_path):
    context_dir = _copy_example(tmp_path, "hypotheses.example.json", corpus.HYPOTHESES_FILENAME)
    hypotheses = corpus.load_hypotheses(context_dir, missing_ok=False)
    assert hypotheses, "context/hypotheses.example.json must contain at least one item"
    for hypothesis in hypotheses:
        assert hypothesis.falsifier.strip(), "the worked example's falsifier must be genuinely falsifiable, not blank"
        assert hypothesis.state in corpus.ALL_HYPOTHESIS_STATES


def test_facts_example_sample_loads_through_the_real_loader_with_every_field_intact(tmp_path):
    context_dir = _copy_example(tmp_path, "facts.example.json", corpus.FACTS_FILENAME)
    # sessions_root=None disables the D-06 citation resolver -- the ONE
    # legitimate use of the nullable form: this sample's session_id is
    # synthetic and cites no real session directory that exists anywhere on
    # this machine, so there is nothing for the resolver to check it
    # against.
    facts = corpus.load_facts(context_dir, sessions_root=None)
    assert facts, "context/facts.example.json must contain at least one item"
    provenances = {fact.provenance for fact in facts}
    assert provenances == {"evidence", "inference"}, (
        "the shipped sample must show both values side by side, per the plan's own action text"
    )


# --- D-01: the corpus has no seed path, asserted positively ----------------------------


_ARTIFACT_FILENAME_IDENTIFIER = re.compile(r"\b(FACTS_FILENAME|HYPOTHESES_FILENAME)\b")


def test_the_corpus_has_no_seed_path_only_the_loader_and_the_store_reference_the_artifact_filenames():
    # D-01 is a negative requirement, and negatives rot silently -- proved
    # POSITIVELY here rather than by grepping for absent things. Enumerates
    # every module under app/ (excluding tests) that references either
    # artifact filename constant at all, and pins that set by equality: a
    # future import/migration/seed feature would have to add itself here to
    # pass, which is exactly the friction D-01 wants.
    #
    # `web/author.py` joined this set in 05-03-PLAN.md §K13/§K14 -- the
    # hypothesis-authoring surface this comment already anticipated, not an
    # import/migration/seed path. Its own dedicated pin is gate 19
    # (`app/tests/test_phase_gates.py`), which asserts by equality that it is
    # the ONLY module that both references HYPOTHESES_FILENAME and calls a
    # write primitive against it -- strictly narrower than this gate's plain
    # "references the name at all", so a future writer must clear BOTH.
    #
    # `coach/context.py` joined this set in 06-01-PLAN.md §F2 -- `load_brief`'s
    # v2 retrieval checks `(context_dir / corpus.FACTS_FILENAME).is_file()` to
    # tell an ABSENT corpus (D-02's normal empty case) apart from one it should
    # actually read via `corpus.load_facts`. This is a READ-ONLY reference, not
    # a seed/import/migration path -- `context.py` contains no write primitive
    # against either filename, and D-01's own rule (the corpus grows only from
    # this tool's own interviews) is untouched by a module that merely checks
    # whether a file created by that same growth process exists yet.
    #
    # `web/console.py` joined this set in Phase B for the IDENTICAL reason as
    # `coach/context.py` above, and by the identical mechanism: the console
    # reports a per-node fact COUNT, and checking `is_file()` first is what
    # distinguishes "this node has extracted nothing yet" (0, the normal case
    # for a new project) from "this node's corpus is malformed" (unknown,
    # shown as an error beside the count). It is read-only, contains no write
    # primitive at all -- `test_web_console.py` asserts that separately and
    # more strictly, by scanning for every write primitive rather than only
    # the two artifact names -- and it never reads a FIELD of a fact, so gate
    # 20 is untouched too.
    #
    # `web/review.py` joined for the identical `is_file()` reason: the hub
    # distinguishes "nothing extracted yet" from "the corpus is malformed",
    # and shows the second as an error beside the standings rather than
    # failing to open. It contains no write primitive -- asserted separately
    # and more strictly in test_web_review.py, which scans for every write
    # primitive rather than only the two artifact names.
    #
    # `web/accept.py` joined in the commit that added the accept/reject card,
    # and is the second module ever to WRITE `hypotheses.json` -- so it also
    # had to clear gate 19's strictly narrower pin, in that same commit, with
    # its own stated reason. It is not a seed/import/migration path: it moves
    # ONE already-authored hypothesis to a state a human explicitly accepted,
    # and creates no hypothesis and no fact. D-01 (the corpus grows only from
    # this tool's own interviews) and D-03 (the coach only proposes; the set
    # changes on an explicit accept) are what it implements, not exceptions to
    # it. It is a module of its own precisely because gate 19 matches per
    # FILE: putting the save in `web/review.py` instead would have retired
    # that gate's reach over the whole hub.
    #
    # `mcp/tools.py` joined in the commit that added the MCP server, for the
    # third instance of the IDENTICAL read-only `is_file()` reason: an agent
    # asking `get_facts` about a project that has never been interviewed must
    # get an empty list (D-01's normal first case), and one asking about a
    # project whose corpus is malformed must get a NAMED error -- and the two
    # are distinguishable only by checking existence before loading. It is
    # read-only by construction: the whole package contains no write
    # primitive (`test_mcp_server.py` proves it behaviourally, by calling
    # every tool and asserting the tree is byte-identical afterwards), and it
    # never reads `Fact.quoted_text`, so gate 20 is untouched and the new
    # gate 32 pins that emptiness for `app/mcp/` specifically.
    # `main.py` joined in the commit that added `python -m app summary` and
    # `python -m app standing`, for the fourth instance of the IDENTICAL
    # read-only `is_file()` reason `coach/context.py`, `web/console.py`,
    # `web/review.py` and `mcp/tools.py` each joined for: a corpus that was
    # never written and a corpus holding nothing from the named interview are
    # different answers with different remedies ("run extract" versus "either
    # extraction found nothing or it was never run here"), and an empty tuple
    # of facts cannot tell them apart. The reference is the existence check
    # alone. `main.py` has ALWAYS written facts -- `run_extract` calls
    # `corpus_store.replace_session_facts` -- but it did so through the store,
    # naming no filename, and that is unchanged: nothing here puts bytes
    # anywhere, and D-01 (the corpus grows only from this tool's own
    # interviews) is untouched by two commands that only read what those
    # interviews already produced.
    app_dir = REPO_ROOT / "app"
    touching = set()
    for py_file in sorted(app_dir.rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        if _ARTIFACT_FILENAME_IDENTIFIER.search(py_file.read_text(encoding="utf-8")):
            touching.add(py_file.relative_to(app_dir).as_posix())
    assert touching == {
        "coach/context.py", "coach/corpus.py", "coach/corpus_store.py",
        "main.py", "mcp/tools.py",
        "web/accept.py", "web/author.py", "web/console.py", "web/review.py",
    }, (
        f"a module besides the loader, the fact writer, the two hypothesis-writing surfaces "
        f"and the grounding seam's own retrieval check references an artifact filename "
        f"constant -- got {touching}"
    )


# --- app.coach.corpus_store.replace_session_facts: D-08 slice replacement (05-02-PLAN.md §K7) ---


def test_replace_session_facts_against_no_existing_corpus_writes_just_the_new_facts(tmp_path):
    # D-01's normal first case, at the store level directly.
    context_dir = tmp_path / "context"
    path = corpus_store.replace_session_facts(context_dir, "sess-a", [_fact(id="f1", session_id="sess-a")])
    assert path.is_file()
    loaded = corpus.load_facts(context_dir, sessions_root=None)
    assert [f.id for f in loaded] == ["f1"]


def test_replace_session_facts_drops_only_the_targeted_sessions_slice(tmp_path):
    context_dir = tmp_path / "context"
    corpus_store.write_facts(context_dir, [
        _fact(id="a1", session_id="sess-a", claim="old a"),
        _fact(id="b1", session_id="sess-b", claim="b claim"),
    ])
    corpus_store.replace_session_facts(context_dir, "sess-a", [_fact(id="a2", session_id="sess-a", claim="new a")])
    loaded = corpus.load_facts(context_dir, sessions_root=None)
    by_session = {f.session_id: f for f in loaded}
    assert len(loaded) == 2
    assert by_session["sess-a"].id == "a2"
    assert by_session["sess-a"].claim == "new a"
    assert by_session["sess-b"].id == "b1", "a different session's facts must survive untouched"
    assert by_session["sess-b"].claim == "b claim"


def test_replace_session_facts_with_an_empty_accepted_list_still_removes_the_old_slice(tmp_path):
    # Direct unit coverage of the MECHANISM: replace_session_facts always
    # performs all four §K7 steps when called, including "append zero
    # facts". app/main.py is what decides NOT to call this function at all
    # when a run accepts nothing -- see its own §K7 comment.
    context_dir = tmp_path / "context"
    corpus_store.write_facts(context_dir, [_fact(id="a1", session_id="sess-a")])
    corpus_store.replace_session_facts(context_dir, "sess-a", [])
    loaded = corpus.load_facts(context_dir, sessions_root=None)
    assert loaded == ()


def test_preflight_check_corpus_parses_does_nothing_when_the_file_is_missing(tmp_path):
    context_dir = tmp_path / "context"
    corpus_store.preflight_check_corpus_parses(context_dir)  # must not raise


def test_preflight_check_corpus_parses_raises_naming_the_file_for_a_malformed_corpus(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / corpus.FACTS_FILENAME).write_text("not json", encoding="utf-8")
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus_store.preflight_check_corpus_parses(context_dir)
    assert corpus.FACTS_FILENAME in str(excinfo.value)


def test_replace_session_facts_raises_and_writes_nothing_against_a_malformed_existing_corpus(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    path = context_dir / corpus.FACTS_FILENAME
    path.write_bytes(b"not valid json at all")
    with pytest.raises(corpus.CorpusValidationError):
        corpus_store.replace_session_facts(context_dir, "sess-a", [_fact(id="a2", session_id="sess-a")])
    assert path.read_bytes() == b"not valid json at all", "a malformed corpus must be left completely untouched"


def test_a_failure_during_replace_leaves_the_previous_corpus_intact_and_still_loadable(tmp_path, monkeypatch):
    # Same shape as test_web_author.py:296's own atomic-write monkeypatch test,
    # applied to the coach-side store (05-02-PLAN.md §K7's own naming of that
    # test as the brief this behaviour must match exactly).
    context_dir = tmp_path / "context"
    corpus_store.write_facts(context_dir, [_fact(id="survivor", session_id="sess-a")])
    original_bytes = (context_dir / corpus.FACTS_FILENAME).read_bytes()

    def _boom(*_args, **_kwargs):
        raise OSError("simulated failure between write and replace")

    monkeypatch.setattr(corpus_store.os, "replace", _boom)

    with pytest.raises(OSError):
        corpus_store.replace_session_facts(context_dir, "sess-a", [_fact(id="new", session_id="sess-a")])

    assert (context_dir / corpus.FACTS_FILENAME).read_bytes() == original_bytes, (
        "the previous corpus must still be intact and byte-identical after a simulated failure"
    )
    loaded = corpus.load_facts(context_dir, sessions_root=None)
    assert loaded[0].id == "survivor"


# =========================================================================================
# STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: Fact.seqs -- a citation SPAN
# =========================================================================================
#
# `Fact`'s own field validator enforces the two STRUCTURAL span conditions (non-empty,
# at most MAX_SPAN_SEGMENTS) with no session data at all; `validate_citation_span`
# enforces the three SESSION-DEPENDENT ones (resolution, same-speaker, contiguity within
# that speaker's own ordered segments). Both are exercised directly here, at the unit
# level -- `app/tests/test_coach_extract.py` exercises the SAME three conditions again,
# end to end, through the extraction pipeline that shares this exact logic.


# --- Fact.seqs: the two structural conditions ------------------------------------------


def test_fact_rejects_an_empty_seqs_tuple():
    with pytest.raises(Exception) as excinfo:
        corpus.Fact(
            id="f1", session_id="s1", seqs=(), claim="x", provenance="evidence",
            quoted_text="", bearings=(),
        )
    assert "non-empty" in str(excinfo.value)


def test_fact_accepts_a_span_exactly_at_the_segment_cap():
    fact = corpus.Fact(
        id="f1", session_id="s1", seqs=tuple(range(corpus.MAX_SPAN_SEGMENTS)), claim="x",
        provenance="evidence", quoted_text="", bearings=(),
    )
    assert len(fact.seqs) == corpus.MAX_SPAN_SEGMENTS


def test_fact_rejects_a_span_one_longer_than_the_segment_cap():
    with pytest.raises(Exception) as excinfo:
        corpus.Fact(
            id="f1", session_id="s1", seqs=tuple(range(corpus.MAX_SPAN_SEGMENTS + 1)), claim="x",
            provenance="evidence", quoted_text="", bearings=(),
        )
    assert str(corpus.MAX_SPAN_SEGMENTS) in str(excinfo.value)


# --- validate_citation_span: the three session-dependent conditions --------------------


def _speakers(*pairs) -> "dict[int, str]":
    """`_speakers(0, "interviewee", 1, "interviewer", ...)` -> `{0: "interviewee", 1:
    "interviewer", ...}` -- a terse builder for the `speaker_by_seq` maps these tests
    construct by hand rather than through a real session."""
    return dict(zip(pairs[0::2], pairs[1::2]))


def test_validate_citation_span_rejects_a_seq_that_does_not_resolve():
    speaker_by_seq = _speakers(0, "interviewee")
    with pytest.raises(corpus.SpanValidityError) as excinfo:
        corpus.validate_citation_span([0, 999], speaker_by_seq=speaker_by_seq)
    assert excinfo.value.code == corpus.SpanValidityError.SPAN_UNRESOLVED


def test_validate_citation_span_accepts_a_single_seq():
    speaker_by_seq = _speakers(0, "interviewee")
    corpus.validate_citation_span([0], speaker_by_seq=speaker_by_seq)  # must not raise


def test_validate_citation_span_rejects_a_span_mixing_two_speakers():
    speaker_by_seq = _speakers(0, "interviewee", 1, "interviewer")
    with pytest.raises(corpus.SpanValidityError) as excinfo:
        corpus.validate_citation_span([0, 1], speaker_by_seq=speaker_by_seq)
    assert excinfo.value.code == corpus.SpanValidityError.SPAN_MIXED_SPEAKER


def test_validate_citation_span_accepts_another_speakers_segment_interleaved_in_the_raw_gap():
    # The positive control paired with the mutation test right below --
    # identical raw-seq shape (a record at seq=1, between the cited 0 and 2),
    # differing only in who speaks at seq=1.
    speaker_by_seq = _speakers(0, "interviewee", 1, "interviewer", 2, "interviewee")
    corpus.validate_citation_span([0, 2], speaker_by_seq=speaker_by_seq)  # must not raise


def test_validate_citation_span_rejects_a_span_that_skips_the_same_speakers_own_segment():
    # The mutation shape that matters most (05-01-PLAN.md §K2's "join that
    # never existed" concern): seq=1 is the CITED speaker's OWN segment,
    # uncited -- the identical raw-seq shape as the accept test above, so a
    # naive `max(seqs) - min(seqs) + 1 == len(seqs)` check computes the SAME
    # number (3 != 2) for both and rejects both, which would ALSO wrongly
    # reject the valid interleaved case above; only resolving WHO occupies
    # the gap (via this speaker's own ordered position list) tells the two
    # apart correctly in both directions.
    speaker_by_seq = _speakers(0, "interviewee", 1, "interviewee", 2, "interviewee")
    with pytest.raises(corpus.SpanValidityError) as excinfo:
        corpus.validate_citation_span([0, 2], speaker_by_seq=speaker_by_seq)
    assert excinfo.value.code == corpus.SpanValidityError.SPAN_NONCONTIGUOUS


def test_validate_citation_span_rejects_a_three_element_span_skipping_the_middle_of_five():
    # A wider skip: five of the speaker's own segments (seq 0..4), citing
    # only the 1st, 3rd, and 5th -- skipping their own 2nd and 4th.
    speaker_by_seq = _speakers(0, "interviewee", 1, "interviewee", 2, "interviewee", 3, "interviewee", 4, "interviewee")
    with pytest.raises(corpus.SpanValidityError) as excinfo:
        corpus.validate_citation_span([0, 2, 4], speaker_by_seq=speaker_by_seq)
    assert excinfo.value.code == corpus.SpanValidityError.SPAN_NONCONTIGUOUS


def test_validate_citation_span_accepts_a_multi_segment_run_with_heavy_other_speaker_interleaving():
    # Three of the speaker's own CONSECUTIVE (in their own ordering) segments,
    # each separated from the next by two turns of the OTHER speaker -- a
    # realistic shape for a real back-and-forth conversation.
    speaker_by_seq = _speakers(
        0, "interviewee", 1, "interviewer", 2, "interviewer", 3, "interviewee",
        4, "interviewer", 5, "interviewer", 6, "interviewee",
    )
    corpus.validate_citation_span([0, 3, 6], speaker_by_seq=speaker_by_seq)  # must not raise


# --- load_facts(sessions_root=...): the loader wires validate_citation_span in ---------


def _new_real_session(tmp_path: Path, turns: "list[tuple[str, str]]") -> Path:
    """A REAL session directory via `SessionStore`, so `load_facts`'s own
    D-06 resolution sees genuine `seq` values -- never a hand-rolled
    transcript file (mirrors `test_coach_extract.py::_new_session`)."""
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "corpus-span-test"})
    for speaker, text in turns:
        store.append_segment(
            speaker=speaker, text=text, audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
    store.close()
    return session_dir


def test_load_facts_accepts_a_real_multi_segment_span_against_a_real_session(tmp_path):
    sessions_root = tmp_path / "sessions"
    session_dir = _new_real_session(
        sessions_root, [("interviewee", "we onboard a new customer"), ("interviewee", "in about three weeks total")]
    )
    context_dir = _write_facts(tmp_path, [
        _base_fact_entry(
            id="f1", session_id=session_dir.name, seqs=[0, 1], provenance="evidence",
            quoted_text="we onboard a new customer in about three weeks total",
        ),
    ])
    facts = corpus.load_facts(context_dir, sessions_root=sessions_root)
    assert facts[0].seqs == (0, 1)


def test_load_facts_rejects_a_real_span_that_skips_the_same_speakers_own_segment(tmp_path):
    sessions_root = tmp_path / "sessions"
    session_dir = _new_real_session(
        sessions_root,
        [
            ("interviewee", "we onboard a new customer"),
            ("interviewee", "and it is honestly a bit painful"),
            ("interviewee", "in about three weeks total"),
        ],
    )
    context_dir = _write_facts(tmp_path, [
        _base_fact_entry(id="f-skip", session_id=session_dir.name, seqs=[0, 2], provenance="evidence", quoted_text="x"),
    ])
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_facts(context_dir, sessions_root=sessions_root)
    assert "f-skip" in str(excinfo.value)


# --- the pre-registered criterion, and `contested` -------------------------------------


def test_a_hypothesis_may_be_contested_and_it_is_a_resting_state(tmp_path):
    # Not a way station. A hypothesis with real evidence on both sides is a
    # finding, and nothing may auto-resolve it to a winner -- resolving it is
    # a human deciding the question was wrong, or gathering evidence that
    # settles it.
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "state": "contested"}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].state == corpus.HYPOTHESIS_STATE_CONTESTED


def test_contested_is_distinguishable_from_untested():
    # The whole reason the fourth state exists. "Nobody has looked at this"
    # and "we looked and the evidence disagrees" need opposite next actions:
    # one wants an interview, the other wants a sharper question.
    assert corpus.HYPOTHESIS_STATE_CONTESTED != corpus.HYPOTHESIS_STATE_UNTESTED


def test_a_hypothesis_authored_before_the_criterion_existed_still_loads(tmp_path):
    # A compatibility decision, not an endorsement. Requiring the field would
    # make every previously-authored hypotheses.json MALFORMED, and in this
    # codebase malformed RAISES -- which would take the coach down over a
    # field added later.
    entry = {"id": "h1", "statement": "x", "falsifier": "y"}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].criterion == ""


def test_a_hypothesis_without_a_criterion_is_not_scorable(tmp_path):
    entry = {"id": "h1", "statement": "x", "falsifier": "y"}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].is_scorable is False


def test_a_whitespace_only_criterion_is_not_a_criterion(tmp_path):
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "criterion": "   \n "}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].is_scorable is False


def test_a_real_criterion_makes_a_hypothesis_scorable(tmp_path):
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y",
        "criterion": "3 of 10 unprompted mention reverting to manual notes",
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].is_scorable is True
    assert "unprompted" in loaded[0].criterion


def test_scorability_is_a_property_not_a_validator(tmp_path):
    # Stated as a test because the distinction is load-bearing: an absent
    # criterion is a fact about the JUDGEMENT, never about the file. A
    # criterion-less hypothesis is perfectly valid and merely cannot be
    # honestly scored.
    entry = {"id": "h1", "statement": "x", "falsifier": "y", "state": "supported"}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].state == "supported" and loaded[0].is_scorable is False


# --- the criterion's own PRE-REGISTRATION BOUNDARY --------------------------------------
#
# `criterion` alone is a claim about ORDER that carries no evidence of order:
# nothing on the record distinguishes a rule written before the interviewing
# from one typed afterwards to fit what arrived. `criterion_registered_after`
# is that evidence -- the session ids that ALREADY EXISTED at this node when
# the criterion text was written. Deliberately not a timestamp: see the field's
# own docstring, and `Fact`'s seven-field note it is argued against.


# A hypotheses.json exactly as the two PREVIOUS eras of this codebase wrote it:
# the first before `criterion` existed at all, the second after `criterion` and
# before this field. Kept as literal bytes rather than built from today's model
# -- a fixture built from the CURRENT model would silently acquire every new
# field and could never fail this test, which is the whole point of it.
_PRE_BOUNDARY_HYPOTHESES_JSON = """[
  {
    "id": "h-oldest",
    "statement": "Slow onboarding drives early churn",
    "falsifier": "If churned customers cite a different top reason more often."
  },
  {
    "id": "h-older",
    "statement": "Teams revert to manual notes by week three",
    "falsifier": "If nobody mentions manual notes unprompted.",
    "state": "supported",
    "criterion": "3 of 10 unprompted mention reverting to manual notes"
  }
]"""


def test_a_hypotheses_file_written_before_this_field_existed_still_loads(tmp_path):
    # The identical compatibility argument `criterion`'s own default makes, for
    # the identical reason: requiring the field would turn every hypotheses.json
    # already on disk into a MALFORMED file, and malformed RAISES here -- taking
    # the coach down over a field added later. `extra="forbid"` means this also
    # proves the loader tolerates the ABSENCE rather than merely ignoring it.
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / corpus.HYPOTHESES_FILENAME).write_text(
        _PRE_BOUNDARY_HYPOTHESES_JSON, encoding="utf-8"
    )
    loaded = corpus.load_hypotheses(context_dir, missing_ok=False)
    assert [h.id for h in loaded] == ["h-oldest", "h-older"]
    assert loaded[0].criterion_registered_after == ()
    assert loaded[1].criterion_registered_after == ()
    assert loaded[1].criterion.startswith("3 of 10")


def test_an_empty_boundary_means_the_criterion_predates_every_interview(tmp_path):
    # The STRONGEST claim available, and the one an empty tuple honestly makes:
    # written before any interview existed at this node, so it pre-dates all of
    # them -- including sessions that do not exist yet.
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y",
        "criterion": "3 of 10 unprompted mention X",
        "criterion_registered_after": [],
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].criterion_registered_after == ()
    assert loaded[0].pre_registered_for("20260101T090000") is True
    assert loaded[0].pre_registered_for("20990101T090000") is True


def test_an_interview_inside_the_boundary_is_not_one_the_criterion_predates(tmp_path):
    # Set MEMBERSHIP, never an ordering comparison on the id. The ids happen to
    # sort chronologically today (`%Y%m%dT%H%M%S`), and a check written as
    # `session_id > max(boundary)` would look right and would silently answer
    # for a session id nobody recorded -- the boundary is the RECORD of what
    # existed, not a rule for extrapolating past it.
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y",
        "criterion": "3 of 10 unprompted mention X",
        "criterion_registered_after": ["20260101T090000", "20260105T090000"],
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].pre_registered_for("20260101T090000") is False
    assert loaded[0].pre_registered_for("20260105T090000") is False
    assert loaded[0].pre_registered_for("20260210T090000") is True


def test_an_unrecorded_interview_older_than_a_recorded_one_is_not_pre_registered_for(tmp_path):
    # Written specifically because a mutation survived without it. Replacing
    # membership with `session_id > max(boundary)` passes every test above --
    # the ids sort chronologically, so the two agree for every id the boundary
    # actually contains. They diverge here, and only here: an interview the
    # boundary does NOT name, sitting between two it does. That is a real
    # query, not a contrivance -- a sibling node's session (cross-node reads
    # are legal within one hierarchy) or an interview filed out of order. The
    # ordering shortcut answers "the rule came first" for an interview the rule
    # never accounted for, which is the false claim in the dangerous direction.
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y",
        "criterion": "3 of 10 unprompted mention X",
        "criterion_registered_after": ["20260101T090000", "20260301T090000"],
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].pre_registered_for("20260201T090000") is True, (
        "an interview the boundary never named did not exist when the rule was written"
    )
    assert loaded[0].pre_registered_for("20260301T090000") is False, (
        "the LATEST recorded interview must still be inside the boundary"
    )


def test_a_hypothesis_with_no_criterion_is_pre_registered_for_nothing(tmp_path):
    # The sneakiest false positive this field could produce. A hypothesis with
    # NO rule has an empty boundary too -- and a membership test alone would
    # then report "pre-registered" for every interview ever run, which is the
    # exact unearned claim the field exists to make checkable. No rule cannot
    # pre-date anything.
    entry = {"id": "h1", "statement": "x", "falsifier": "y"}
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].is_scorable is False
    assert loaded[0].pre_registered_for("20260101T090000") is False


def test_a_whitespace_only_criterion_is_pre_registered_for_nothing_either(tmp_path):
    # Asserted separately from the absent case, on `test_web_author.py:424`'s
    # own stated precedent: a whitespace string is a valid STRING, and a check
    # that only tested presence would let this one claim pre-registration.
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y", "criterion": "  \n ",
        "criterion_registered_after": [],
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].pre_registered_for("20260101T090000") is False


def test_the_boundary_is_a_property_not_a_validator(tmp_path):
    # `is_scorable`'s own stated distinction, applied to the boundary: whether
    # a criterion pre-dates an interview is a fact about the JUDGEMENT, never
    # about the file. A boundary naming a session that no longer exists on disk
    # -- a deleted interview -- is still a perfectly valid record of what was
    # there when the rule was written, and must not raise.
    entry = {
        "id": "h1", "statement": "x", "falsifier": "y", "criterion": "3 of 10 mention X",
        "criterion_registered_after": ["a-session-nobody-can-find"],
    }
    loaded = corpus.load_hypotheses(_write_hypotheses(tmp_path, [entry]), missing_ok=False)
    assert loaded[0].criterion_registered_after == ("a-session-nobody-can-find",)


def test_the_boundary_carries_no_time_value_of_any_kind():
    # `Fact`'s seven-field note argues a record with somewhere to put a time
    # invites a later edit that fills it from the wrong source -- and here the
    # wrong source is worse than noise, because a fabricated clock would forge
    # the ONE claim this field exists to make checkable. Session ids cannot be
    # populated from a clock: they are the names of directories that either
    # exist or do not.
    fields = corpus.Hypothesis.model_fields
    assert set(fields) == {
        "id", "statement", "falsifier", "state", "criterion", "criterion_registered_after",
    }, (
        "a SEVENTH field on Hypothesis -- and if it is a timestamp, it retires the argument "
        "the boundary was shaped as a session-id tuple to make"
    )


# --- a fact bears on a hypothesis in a DIRECTION ----------------------------------------


def test_a_bearing_carries_a_direction_not_just_a_link(tmp_path):
    # The whole point. `related_hypothesis_ids` recorded that a claim "bears
    # on" a hypothesis and nothing else, so a claim that DEMOLISHED it and one
    # that confirmed it were the same integer -- and any count built from that
    # can only go up. Four shipped discovery tools have exactly that shape.
    entry = {
        "id": "f1", "session_id": "S1", "seqs": [1], "claim": "they went back to paper",
        "provenance": "evidence", "quoted_text": "they went back to paper",
        "bearings": [{"hypothesis_id": "h1", "stance": "undermines"}],
    }
    (fact,) = corpus.load_facts(_write_facts(tmp_path, [entry]), sessions_root=None)
    assert fact.bearings[0].stance == corpus.STANCE_UNDERMINES


@pytest.mark.parametrize("stance", sorted(corpus.ALL_STANCES))
def test_every_recognized_stance_round_trips(stance, tmp_path):
    entry = {
        "id": "f1", "session_id": "S1", "seqs": [1], "claim": "c", "provenance": "inference",
        "quoted_text": "", "bearings": [{"hypothesis_id": "h1", "stance": stance}],
    }
    (fact,) = corpus.load_facts(_write_facts(tmp_path, [entry]), sessions_root=None)
    assert fact.bearings[0].stance == stance


def test_an_unrecognized_stance_is_refused(tmp_path):
    entry = {
        "id": "f1", "session_id": "S1", "seqs": [1], "claim": "c", "provenance": "inference",
        "quoted_text": "", "bearings": [{"hypothesis_id": "h1", "stance": "probably"}],
    }
    with pytest.raises(corpus.CorpusValidationError):
        corpus.load_facts(_write_facts(tmp_path, [entry]), sessions_root=None)


def test_the_stance_set_is_pinned_and_includes_unclear():
    # `unclear` is not padding. Forcing a direction onto a claim that bears on
    # a hypothesis without settling it MANUFACTURES evidence -- and the
    # manufactured direction will almost always be `supports`, because the
    # model is completing a sentence about a hypothesis someone wrote down
    # because they already believed it.
    assert corpus.ALL_STANCES == {"supports", "undermines", "unclear"}


def test_related_hypothesis_ids_is_derived_so_the_two_cannot_drift(tmp_path):
    # Retrieval scoring asks only "is this fact linked to a live hypothesis"
    # and genuinely does not care which way -- a fact that DEMOLISHES an open
    # hypothesis is exactly as worth retrieving as one that supports it.
    # Storing both a link list and a stance list would let them disagree.
    entry = {
        "id": "f1", "session_id": "S1", "seqs": [1], "claim": "c", "provenance": "inference",
        "quoted_text": "",
        "bearings": [
            {"hypothesis_id": "h1", "stance": "undermines"},
            {"hypothesis_id": "h2", "stance": "supports"},
        ],
    }
    (fact,) = corpus.load_facts(_write_facts(tmp_path, [entry]), sessions_root=None)
    assert fact.related_hypothesis_ids == ("h1", "h2")
    assert "related_hypothesis_ids" not in corpus.Fact.model_fields, (
        "it must be a derived property, never a second stored field"
    )


def test_a_fact_is_still_exactly_seven_fields():
    # §K2's rule is untouched: `bearings` REPLACED `related_hypothesis_ids`
    # rather than joining it.
    assert len(corpus.Fact.model_fields) == 7


def test_disconfirming_evidence_is_representable_at_all(tmp_path):
    # Stated as its own test because it is the property that was MISSING, and
    # a regression here would be silent: the corpus would keep loading, counts
    # would keep rising, and nothing would ever be refutable.
    entry = {
        "id": "f1", "session_id": "S1", "seqs": [1], "claim": "the opposite happened",
        "provenance": "evidence", "quoted_text": "the opposite happened",
        "bearings": [{"hypothesis_id": "h1", "stance": "undermines"}],
    }
    (fact,) = corpus.load_facts(_write_facts(tmp_path, [entry]), sessions_root=None)
    against = [b for b in fact.bearings if b.stance == corpus.STANCE_UNDERMINES]
    assert against, "a corpus that cannot express disconfirmation is a confirmation machine"
