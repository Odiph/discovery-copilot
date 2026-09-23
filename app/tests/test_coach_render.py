"""Task 3 done criteria for 04-03-PLAN.md §E6 (COACH-04): the post-call
renderer extended with suggestion records (shown, ranked down, and dropped
alike) and the full coverage transition history, inline in session-time
order, plus a coaching summary -- and the pre-Phase-4 rendering behaviour
(`app/tests/test_render.py`) staying completely unbroken.

06-03-PLAN.md §H4: extended again for the post-call read -- resolving a
suggestion's `grounding_ids` to the past-interview facts that grounded it
(and their verbatim quotes, HERE ONLY), rendering a coach-proposed
hypothesis-state change as an explicit, un-applied proposal, and the two new
coaching-summary counts (grounded suggestions, proposed changes)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.coach.context import FACT_ID_PREFIX, SurveyItem
from app.session.render import render_session, write_rendered_transcript
from app.session.store import SessionStore, create_session_dir


def _new_store(tmp_path: Path, name: str = "session") -> SessionStore:
    session_dir = create_session_dir(tmp_path) if name == "session" else tmp_path / name
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    return store


def _judgment_fields(*, is_leading=False, is_hypothetical=False, is_pitch=False, is_opinion=False, is_confirmation_only=False, is_generic=False):
    # 06-03-PLAN.md §H1 (deviation, out of Task 1's own plan scope --
    # necessary consequence of `append_suggestion`'s new REQUIRED
    # `is_confirmation_only` keyword, mirroring 06-01/06-02-SUMMARY.md's
    # identical precedent for other files in this same position): every
    # direct `store.append_suggestion(...)` call in this file must supply it
    # or fail with a missing-keyword TypeError.
    return dict(
        is_leading=is_leading,
        is_hypothetical_or_future_tense=is_hypothetical,
        is_pitch=is_pitch,
        is_opinion_seeking=is_opinion,
        is_confirmation_only=is_confirmation_only,
        is_generic=is_generic, is_off_topic=False,
    )


# --- Every suggestion the coach ever produced, including dropped ones ------------------


def test_every_suggestion_record_renders_including_dropped_ones_marked_as_dropped(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="I struggled with onboarding", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="What made onboarding hard?", move="deepen", band="clean",
        **_judgment_fields(), rationale="grounded in what they just said", shown=True,
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="Wouldn't you agree onboarding is always hard?", move="advance",
        band="clear_violation", **_judgment_fields(is_leading=True), rationale="leading question", shown=False,
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "SHOWN" in doc
    assert "What made onboarding hard?" in doc
    assert "DROPPED" in doc
    assert "Wouldn't you agree onboarding is always hard?" in doc
    assert "is_leading=True" in doc
    assert "grounded in what they just said" in doc


def test_a_candidate_the_critique_never_vetted_is_marked_not_vetted(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="never vetted probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="an unvetted candidate", move="advance",
        band=None, is_leading=None, is_hypothetical_or_future_tense=None, is_pitch=None,
        is_opinion_seeking=None, is_confirmation_only=None, is_generic=None, is_off_topic=False, rationale=None, shown=False,
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "not vetted" in doc.lower()
    assert "an unvetted candidate" in doc


# --- Suggestions appear INLINE, in session-time order, not in a trailing appendix ------


def test_a_suggestion_renders_between_the_two_segments_around_it(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="segment before the suggestion", audio_start_s=0.0, audio_end_s=5.0,
        ready_wall_s=5.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="the suggested next question", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
    )
    store.append_segment(
        speaker="interviewer", text="segment after the suggestion", audio_start_s=5.0, audio_end_s=10.0,
        ready_wall_s=10.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    before_idx = doc.index("segment before the suggestion")
    suggestion_idx = doc.index("the suggested next question")
    after_idx = doc.index("segment after the suggestion")
    assert before_idx < suggestion_idx < after_idx, (
        "the suggestion must render BETWEEN the two segments, not gathered into an appendix"
    )


# --- Full coverage transition history: what matched against what ----------------------


def test_a_matched_coverage_item_shows_which_segment_matched_it(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({
        "coach_enabled": True,
        "context_survey": [{"id": "q1", "order": 0, "text": "What is your current role?"}],
    })
    segment = store.append_segment(
        speaker="interviewer", text="So tell me, what do you do these days?", audio_start_s=0.0,
        audio_end_s=3.0, ready_wall_s=3.1, detected_language="en", engine="fake",
    )
    store.append_event(
        event="coverage-asked", speaker="interviewer",
        detail={"question_id": "q1", "matched_seq": segment.seq, "matched_audio_end_s": segment.audio_end_s},
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "MATCHED" in doc
    assert "What is your current role?" in doc
    assert "So tell me, what do you do these days?" in doc


def test_a_non_match_is_also_rendered_as_part_of_the_full_history(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({"coach_enabled": True, "context_survey": []})
    segment = store.append_segment(
        speaker="interviewer", text="unrelated small talk", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_event(
        event="coverage-considered", speaker="interviewer",
        detail={"seq": segment.seq, "matched_question_id": None},
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "considered" in doc.lower()
    assert "unrelated small talk" in doc
    assert "no survey item matched" in doc.lower()


# --- Verbatim reproduction is unbroken by this extension --------------------------------


def test_mixed_hebrew_english_segment_text_still_renders_verbatim_alongside_a_suggestion(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({"coach_enabled": True})
    mixed_text = "שלום hello עולם world, מה קורה today?"
    trigger = store.append_segment(
        speaker="interviewee", text=mixed_text, audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="he", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="a suggestion beside the mixed segment", move="deepen", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert mixed_text in doc, "the mixed-language segment must still appear verbatim, unreordered, unreflowed"
    for forbidden in ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c"):
        assert forbidden not in doc, f"a bidi control character ({forbidden!r}) was inserted -- forbidden by U2"


# --- Coaching summary: zero-of-each without raising, and a real breakdown -------------


def test_coaching_summary_reports_zero_without_raising_when_the_coach_never_ran(tmp_path):
    store = _new_store(tmp_path)
    store.append_segment(
        speaker="interviewee", text="a session with no coach at all", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "## Coaching summary" in doc
    assert "did not run" in doc.lower()


def test_coaching_summary_breaks_down_dropped_candidates_by_which_rule_fired(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({
        "coach_enabled": True,
        "coaching_latency_summary": {"first_cycle_s": 1.8, "stats": {"count": 0}},
    })
    trigger = store.append_segment(
        speaker="interviewee", text="trigger for the summary test", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="shown one", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="leading drop", move="advance", band="clear_violation",
        **_judgment_fields(is_leading=True), rationale="r", shown=False,
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="pitch drop", move="advance", band="clear_violation",
        **_judgment_fields(is_pitch=True), rationale="r", shown=False,
    )
    store.append_event(event="coverage-asked", speaker="interviewer", detail={"question_id": "x", "matched_seq": 0, "matched_audio_end_s": 0.0})
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "Cycles completed: 1" in doc
    assert "Suggestions shown: 1" in doc
    assert "Candidates dropped: 2" in doc
    assert "is_leading=1" in doc
    assert "is_pitch=1" in doc
    assert "Coverage: 1 asked" in doc
    assert "1.80s" in doc  # the first-cycle latency figure


# --- Backward compatibility: a pre-Phase-4 session renders identically -----------------


def test_a_pre_phase_four_session_renders_the_same_transcript_with_new_sections_empty(tmp_path):
    """Simulates a session recorded before this plan existed: no
    `coach_enabled` metadata key at all, no suggestion records, no coverage
    events -- exactly the shape `app/tests/test_render.py`'s own fixtures
    already produce."""
    store = _new_store(tmp_path, name="pre-phase-4")
    store.append_segment(
        speaker="interviewee", text="a segment from before the coach existed", audio_start_s=0.0,
        audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "a segment from before the coach existed" in doc
    assert "## Coaching summary" in doc
    assert "did not run" in doc.lower()
    assert "SHOWN" not in doc
    assert "MATCHED" not in doc


# --- 06-01-PLAN.md §F4: the session-start grounding line -----------------------------


def test_the_grounding_line_prints_beside_the_existing_coaching_latency_lines(tmp_path):
    store = _new_store(tmp_path)
    store.update_metadata({
        "coach_enabled": True,
        "coaching_latency_summary": {
            "first_cycle_s": 1.8, "stats": {"count": 0}, "session_start_grounding_s": 0.427,
        },
    })
    store.append_segment(
        speaker="interviewee", text="trigger for the grounding-line test", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "0.43s" in doc, "the session-start grounding figure must print beside the other coaching-latency lines"


def test_a_pre_phase_six_session_with_no_grounding_key_renders_without_raising_or_inventing_a_number(tmp_path):
    # A session recorded before THIS phase existed (04-02/04-03) has a
    # `coaching_latency_summary` with no `session_start_grounding_s` key at
    # all -- `.get(...)`, like every other field in this section, so it
    # renders unchanged rather than raising a KeyError or printing a
    # fabricated 0.00s the coach never actually measured.
    store = _new_store(tmp_path)
    store.update_metadata({
        "coach_enabled": True,
        "coaching_latency_summary": {"first_cycle_s": 1.8, "stats": {"count": 0}},
    })
    store.append_segment(
        speaker="interviewee", text="trigger for the pre-phase-six test", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)  # must not raise
    assert "session-start grounding" not in doc.lower(), (
        "with no grounding key recorded, the line must be omitted entirely -- never a fabricated number"
    )


# =========================================================================================
# 06-03-PLAN.md §H4 (GC-02, GC-05, GC-08): the post-call read -- resolving grounding ids to
# real claims and quotes, rendering an un-applied hypothesis-state proposal, the two new
# coaching-summary counts, and the required-nullable context_dir keyword.
# =========================================================================================


def test_render_session_and_write_rendered_transcript_require_the_context_dir_keyword(tmp_path):
    store = _new_store(tmp_path)
    store.close()
    with pytest.raises(TypeError):
        render_session(store.session_dir)  # noqa: missing context_dir -- must raise, never silently skip resolution
    with pytest.raises(TypeError):
        write_rendered_transcript(store.session_dir)  # noqa: same


def test_a_suggestion_with_a_resolvable_grounding_id_renders_the_facts_claim_and_verbatim_quote(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "facts.json").write_text(json.dumps([{
        "id": "f1", "session_id": "sess-x", "seqs": [3], "claim": "onboarding takes three weeks",
        "provenance": "evidence", "quoted_text": "it takes us three weeks to onboard",
        "bearings": [],
    }]), encoding="utf-8")

    store = _new_store(tmp_path, name="session-grounded")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="does onboarding still take that long?", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True, grounding_ids=(f"{FACT_ID_PREFIX}f1",),
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=context_dir)
    assert "onboarding takes three weeks" in doc, "the resolved fact's claim must be printed"
    assert "it takes us three weeks to onboard" in doc, (
        "the resolved fact's verbatim quote must be printed -- this is the ONE place in the "
        "codebase it is ever shown"
    )


def test_a_suggestion_with_an_unresolvable_grounding_id_renders_a_named_note_and_does_not_raise(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "facts.json").write_text(json.dumps([]), encoding="utf-8")

    store = _new_store(tmp_path, name="session-unresolvable")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="a question", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True, grounding_ids=(f"{FACT_ID_PREFIX}gone",),
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=context_dir)  # must not raise
    assert f"{FACT_ID_PREFIX}gone" in doc
    assert "re-extraction" in doc.lower(), (
        "the note must explain fact ids are re-minted on re-extraction, so a reader does not "
        "diagnose this as corruption"
    )


def test_rendering_with_no_facts_corpus_at_all_still_produces_the_whole_document(tmp_path):
    context_dir = tmp_path / "context"  # deliberately never created -- no facts.json at all

    store = _new_store(tmp_path, name="session-no-corpus")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="the transcript body must still be complete", audio_start_s=0.0,
        audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="a question", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True, grounding_ids=(f"{FACT_ID_PREFIX}f1",),
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=context_dir)  # must not raise
    assert "the transcript body must still be complete" in doc
    assert "## Coaching summary" in doc
    assert f"{FACT_ID_PREFIX}f1" in doc, "with no corpus, the id is shown raw rather than blocking the document"


def test_rendering_with_a_malformed_facts_corpus_still_produces_the_whole_document(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "facts.json").write_text("not valid json at all", encoding="utf-8")

    store = _new_store(tmp_path, name="session-malformed-corpus")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="the transcript body must still be complete here too",
        audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="a question", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True, grounding_ids=(f"{FACT_ID_PREFIX}f1",),
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=context_dir)  # must not raise
    assert "the transcript body must still be complete here too" in doc
    assert "## Coaching summary" in doc


def test_a_proposed_state_renders_the_target_hypothesis_and_an_explicit_not_applied_statement(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "hypotheses.json").write_text(json.dumps([{
        "id": "h1", "statement": "onboarding delay drives churn",
        "falsifier": "churned customers cite something else more often", "state": "untested",
    }]), encoding="utf-8")

    store = _new_store(tmp_path, name="session-proposal")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="does the churn reason still hold?", move="test_hypothesis", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
        target_hypothesis_id="h1", proposed_state="supported",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=context_dir)
    assert "onboarding delay drives churn" in doc, "the target hypothesis's own statement must be resolved"
    assert "'supported'" in doc or '"supported"' in doc
    assert "not applied" in doc.lower(), "the proposal line must say explicitly it was not applied"
    assert "yourself" in doc.lower(), "the proposal line must say the change is the reader's own to make"


def test_coaching_summary_reports_the_grounded_and_proposed_counts(tmp_path):
    store = _new_store(tmp_path, name="session-summary-counts")
    store.update_metadata({"coach_enabled": True})
    trigger = store.append_segment(
        speaker="interviewee", text="probe", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="grounded question", move="advance", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
        grounding_ids=(f"{FACT_ID_PREFIX}a", f"{FACT_ID_PREFIX}b"),
    )
    store.append_suggestion(
        trigger_seq=trigger.seq, text="proposing question", move="test_hypothesis", band="clean",
        **_judgment_fields(), rationale="r", shown=True,
        target_hypothesis_id="h1", proposed_state="supported",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "Suggestions grounded in a past-interview fact: 1" in doc
    assert "2 distinct fact" in doc
    assert "Hypothesis-state changes proposed: 1" in doc
    assert "never applied" in doc.lower()


def test_coaching_summary_reports_zero_grounded_and_proposed_when_the_coach_ran_but_produced_nothing(tmp_path):
    store = _new_store(tmp_path, name="session-summary-zero")
    store.update_metadata({"coach_enabled": True})
    store.append_segment(
        speaker="interviewee", text="a session where the coach ran but never suggested anything",
        audio_start_s=0.0, audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)  # must not raise
    assert "Suggestions grounded in a past-interview fact: 0" in doc
    assert "Hypothesis-state changes proposed: 0" in doc


def test_a_suggestion_record_missing_every_06_03_key_entirely_still_renders_without_raising(tmp_path):
    # 06-03-PLAN.md §H4's backward-compatibility requirement, asserted against
    # a fixture in the OLD (pre-06-03) shape directly -- a raw transcript.jsonl
    # line whose suggestion record has none of `is_confirmation_only`/
    # `grounding_ids`/`target_hypothesis_id`/`proposed_state` at all, exactly
    # as a session recorded before this plan existed would read back. This is
    # a stronger proof than relying on `append_suggestion`'s own defaults,
    # since a REAL pre-06-03 record never had these keys as a JSON dict at all.
    session_dir = tmp_path / "pre-06-03-session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps({"engine": "fake", "coach_enabled": True, "closed_at": "2026-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    old_shape_suggestion = {
        "kind": "suggestion", "seq": 1, "trigger_seq": 0, "text": "an old-shape suggestion",
        "move": "advance", "band": "clean", "is_leading": False,
        "is_hypothetical_or_future_tense": False, "is_pitch": False, "is_opinion_seeking": False,
        "rationale": "r", "shown": True, "written_at": "2026-01-01T00:00:00+00:00",
        # Deliberately NO is_confirmation_only/grounding_ids/target_hypothesis_id/proposed_state.
    }
    old_shape_segment = {
        "kind": "segment", "seq": 0, "speaker": "interviewee", "text": "an old-shape segment",
        "audio_start_s": 0.0, "audio_end_s": 1.0, "ready_wall_s": 1.1, "latency_s": 0.1,
        "detected_language": "en", "engine": "fake", "written_at": "2026-01-01T00:00:00+00:00",
    }
    (session_dir / "transcript.jsonl").write_text(
        json.dumps(old_shape_segment) + "\n" + json.dumps(old_shape_suggestion) + "\n", encoding="utf-8",
    )

    doc = render_session(session_dir, context_dir=None)  # must not raise
    assert "an old-shape segment" in doc
    assert "an old-shape suggestion" in doc
    assert "SHOWN" in doc
    assert ">   grounded in:" not in doc, "no grounding_ids key at all -- no PER-SUGGESTION grounding line, ever"
    assert "PROPOSED" not in doc, "no proposed_state key at all -- no proposal line, ever"
