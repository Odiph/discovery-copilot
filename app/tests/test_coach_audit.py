"""Tests for `app/coach/pipeline.py`'s durable-record discipline (COACH-04,
§D8): every candidate the cycle ever produced -- shown, ranked down, or
dropped -- is durably recorded with its full field set, generate always runs
before critique and critique is proven to have classified every candidate it
was given, a critique failure still records what generate proposed, and no
candidate/transcript text ever reaches a log line or an event `detail` dict
(T-04-27). Behavioural test names, per this project's own convention."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from app.coach.context import ContextItem, FACT_ID_PREFIX
from app.coach.coverage import CoverageState
from app.coach.critic import BAND_CLEAN, BAND_CLEAR_VIOLATION, BAND_UNCERTAIN, CRITIC_TOOL_NAME
from app.coach.generate import GENERATE_TOOL_NAME
from app.coach.pipeline import EVENT_CYCLE_SKIPPED, EVENT_REASONING_FAILURE, SuggestionPipeline
from app.coach.stub import StubAnthropicClient, ToolUseResponse, tool_use_response
from app.session.reader import read_session
from app.session.store import SessionStore, create_session_dir

# 06-03-PLAN.md §H1: updated deliberately, in the SAME commit that adds the
# four new fields to `SuggestionRecord` -- this equality test is SUPPOSED to
# fail the moment the record grows, so a field nobody thought to add also
# fails it (store.py's own docstring states this discipline). The four new
# keys: `is_confirmation_only` (the fifth per-rule critic flag, 06-02-PLAN.md
# §G4), `grounding_ids`/`target_hypothesis_id`/`proposed_state` (the
# candidate's own self-reports, 06-02-PLAN.md §G3/§G5).
EXPECTED_SUGGESTION_KEYS = {
    "kind", "seq", "trigger_seq", "text", "move", "band", "is_leading",
    "is_hypothetical_or_future_tense", "is_pitch", "is_opinion_seeking",
    "is_confirmation_only", "is_generic", "is_off_topic", "rationale", "shown",
    "written_at",
    "grounding_ids", "target_hypothesis_id", "proposed_state",
}


def _new_store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"purpose": "coach-audit-test"})
    return store


def _make_pipeline(
    store, client, *, survey=(), brief=(), hypotheses=(), asked=frozenset(),
    max_shown=3, queue_maxsize=1, on_ranked=None,
):
    state = CoverageState(asked=frozenset(asked))
    return SuggestionPipeline(
        client=client, store=store, brief=brief, survey=survey, hypotheses=hypotheses,
        session_start_grounding_s=0.0,
        coverage_state_provider=lambda: state,
        model="m", generate_timeout_s=1.0, critique_timeout_s=1.0,
        generate_max_tokens=200, critique_max_tokens=200,
        max_shown=max_shown, failure_threshold=3,
        language_window_s=90.0, language_flip_margin=0.65, language_min_chars=40,
        queue_maxsize=queue_maxsize, on_ranked=on_ranked,
    )


def _gen_response(candidates: list) -> ToolUseResponse:
    return tool_use_response(GENERATE_TOOL_NAME, {"candidates": candidates})


def _judgment_payload(index: int, band: str) -> dict:
    # 06-02-PLAN.md §G4 (deviation, out of this file's own plan scope --
    # necessary consequence of the critique schema's fifth REQUIRED flag,
    # mirroring 06-01-SUMMARY.md's identical precedent for this exact file):
    # is_confirmation_only must be present or every critique response here
    # fails to parse.
    return {
        "index": index, "is_leading": False, "is_hypothetical_or_future_tense": False,
        "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
        "band": band, "rationale": "fine",
    }


def _crit_response(judgments: list) -> ToolUseResponse:
    return tool_use_response(CRITIC_TOOL_NAME, {"judgments": judgments})


def _trigger(store):
    return store.append_segment(
        speaker="interviewee", text="I usually just wing it", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )


# --- The full cycle: generate before critique, and critique sees everything -----------


def test_full_cycle_calls_generate_then_critique_in_order(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "clean q", "move": "advance"}, {"text": "violating q", "move": "deepen"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN), _judgment_payload(1, BAND_CLEAR_VIOLATION)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    assert len(stub.calls) == 2
    assert stub.calls[0].tool_name == GENERATE_TOOL_NAME
    assert stub.calls[1].tool_name == CRITIC_TOOL_NAME


def test_critique_call_classified_every_candidate_generate_produced_including_the_dropped_one(tmp_path):
    # A candidate the critic never saw is a candidate that was never vetted
    # -- proven here by checking every candidate ended up with a REAL band
    # (never None), which is only possible if the critique call classified
    # all of them, including the one that ends up dropped.
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "clean q", "move": "advance"}, {"text": "violating q", "move": "deepen"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN), _judgment_payload(1, BAND_CLEAR_VIOLATION)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = {r["text"]: r for r in result.records_of_kind("suggestion")}
    assert len(suggestions) == 2
    assert suggestions["clean q"]["band"] == BAND_CLEAN
    assert suggestions["clean q"]["shown"] is True
    assert suggestions["violating q"]["band"] == BAND_CLEAR_VIOLATION
    assert suggestions["violating q"]["shown"] is False, "a clear violation is dropped -- never shown at any rank"


# --- COACH-04: every candidate -- shown, ranked down, dropped -- is durably recorded ---


def test_every_candidate_shown_ranked_down_and_dropped_is_durably_recorded_with_full_fields(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response(
                [
                    {"text": "clean-shown", "move": "advance"},
                    {"text": "uncertain-ranked-down-1", "move": "deepen"},
                    {"text": "uncertain-ranked-down-2", "move": "deepen"},
                    {"text": "violation-dropped", "move": "fill_gap"},
                ]
            ),
            _crit_response(
                [
                    _judgment_payload(0, BAND_CLEAN),
                    _judgment_payload(1, BAND_UNCERTAIN),
                    _judgment_payload(2, BAND_UNCERTAIN),
                    _judgment_payload(3, BAND_CLEAR_VIOLATION),
                ]
            ),
        ]
    )
    # max_shown=1 forces "ranked down but not shown" to actually occur -- with
    # more shown slots than survivors every survivor would be shown, which
    # would not prove the distinction between "ranked down" and "dropped".
    pipeline = _make_pipeline(store, stub, max_shown=1)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = {r["text"]: r for r in result.records_of_kind("suggestion")}
    assert len(suggestions) == 4

    assert suggestions["clean-shown"]["shown"] is True
    assert suggestions["clean-shown"]["band"] == BAND_CLEAN

    assert suggestions["uncertain-ranked-down-1"]["shown"] is False
    assert suggestions["uncertain-ranked-down-1"]["band"] == BAND_UNCERTAIN
    assert suggestions["uncertain-ranked-down-2"]["shown"] is False
    assert suggestions["uncertain-ranked-down-2"]["band"] == BAND_UNCERTAIN

    assert suggestions["violation-dropped"]["shown"] is False
    assert suggestions["violation-dropped"]["band"] == BAND_CLEAR_VIOLATION

    for record in suggestions.values():
        assert record["move"] in ("advance", "deepen", "fill_gap")
        assert isinstance(record["is_leading"], bool)
        assert record["rationale"]


def test_suggestion_record_key_set_is_exactly_the_intended_set(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[_gen_response([{"text": "q", "move": "advance"}]), _crit_response([_judgment_payload(0, BAND_CLEAN)])]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert set(suggestions[0].keys()) == EXPECTED_SUGGESTION_KEYS, (
        "a field nobody thought to add must also fail this assertion"
    )


# --- 2026-08-15: a cycle that declines is recorded, and shows nothing ------------------


def _declining_pipeline(store):
    """One cycle where the model returns a single `no_question` candidate,
    banded clean. Clean deliberately: if the decline were banded a violation
    the drop would prove nothing about the move, since the band alone would
    have removed it."""
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "", "move": "no_question"}]),
            _crit_response([{
                "index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
                "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False,
                "is_generic": False, "is_off_topic": False, "band": BAND_CLEAN, "rationale": "a considered decline",
            }]),
        ]
    )
    shown_sets = []
    pipeline = _make_pipeline(store, stub, on_ranked=shown_sets.append)
    pipeline._process(_trigger(store))
    return pipeline, shown_sets


def test_a_declining_cycle_shows_nothing_but_is_still_durably_recorded(tmp_path):
    """Both halves matter. Nothing reaches the interviewer -- and the decision
    is still on disk, because a declining cycle that left no trace would be
    indistinguishable from a coach that had quietly died, which is the one
    question a quiet panel makes an interviewer ask."""
    store = _new_store(tmp_path)
    _pipeline, shown_sets = _declining_pipeline(store)

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1, "the decline is recorded like any other candidate"
    assert suggestions[0]["move"] == "no_question"
    assert suggestions[0]["shown"] is False, (
        "`shown` feeds uptake.py's P1 denominator -- a decline counted as shown "
        "would silently deflate the measure of whether the coach gets used"
    )
    assert shown_sets == [[]], "the phone must receive an empty ranked set, not the decline"

    events = [e for e in result.records_of_kind("event")
              if e.get("event") == "coach-suggestion-declined"]
    assert len(events) == 1, "a declining cycle must leave a named trace"
    assert events[0]["detail"] == {"declined": 1, "shown": 0}
    for value in events[0]["detail"].values():
        assert isinstance(value, int), "counts only, never candidate text (T-04-27)"


def test_a_declining_cycle_counts_as_a_success_not_a_failure(tmp_path):
    """The whole point. Under the old shape the only way to say nothing was an
    empty array, which was a parse failure counted against coach health -- so
    a coach exercising judgement looked exactly like a coach breaking."""
    store = _new_store(tmp_path)
    pipeline, _shown = _declining_pipeline(store)

    result = read_session(store.session_dir)
    failures = [e for e in result.records_of_kind("event")
                if e.get("event") == "reasoning-timeout"]
    assert failures == [], "declining is not a failure"
    assert pipeline._health.consecutive_failures == 0  # noqa: SLF001


def test_a_decline_alongside_a_real_question_removes_only_the_decline(tmp_path):
    """A fixture with the decline alone could not tell "declines are excluded"
    from "nothing was ever shown". This one can: two candidates in, one out."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([
                {"text": "", "move": "no_question"},
                {"text": "what did you do the last time that happened?", "move": "deepen"},
            ]),
            _crit_response([
                {"index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
                 "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False,
                 "is_generic": False, "is_off_topic": False, "band": BAND_CLEAN, "rationale": "decline"},
                {"index": 1, "is_leading": False, "is_hypothetical_or_future_tense": False,
                 "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False,
                 "is_generic": False, "is_off_topic": False, "band": BAND_CLEAN, "rationale": "fine"},
            ]),
        ]
    )
    shown_sets = []
    pipeline = _make_pipeline(store, stub, on_ranked=shown_sets.append)
    pipeline._process(_trigger(store))

    assert [c.move for c in shown_sets[0]] == ["deepen"], (
        "the real question survives; only the decline is removed"
    )
    records = {r["move"]: r for r in read_session(store.session_dir).records_of_kind("suggestion")}
    assert records["no_question"]["shown"] is False
    assert records["deepen"]["shown"] is True


# --- 06-03-PLAN.md §H1: the durable record carries what the coach claimed --------------


def test_a_full_cycle_records_grounding_ids_target_hypothesis_and_proposed_state_from_the_candidate(tmp_path):
    store = _new_store(tmp_path)
    fact_item = ContextItem(
        id=f"{FACT_ID_PREFIX}f1", text="onboarding takes three weeks", provenance="evidence",
    )
    hypothesis = SimpleNamespace(id="h1", statement="onboarding delay drives churn", falsifier="it does not")
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{
                "text": "does onboarding delay still bother you?", "move": "test_hypothesis",
                "target_hypothesis_id": "h1", "grounding_ids": [f"{FACT_ID_PREFIX}f1"],
                "proposed_state": "supported",
            }]),
            _crit_response([{
                "index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
                "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": True, "is_generic": True, "is_off_topic": False,
                "band": BAND_UNCERTAIN, "rationale": "borderline",
            }]),
        ]
    )
    pipeline = _make_pipeline(store, stub, brief=(fact_item,), hypotheses=(hypothesis,))
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    record = suggestions[0]
    assert record["grounding_ids"] == [f"{FACT_ID_PREFIX}f1"], (
        "grounding_ids must reach the durable record exactly as the candidate self-reported"
    )
    assert record["target_hypothesis_id"] == "h1"
    assert record["proposed_state"] == "supported"
    assert record["is_confirmation_only"] is True, "the fifth flag must come from the judgment, not be invented here"


def test_a_candidate_the_critique_never_classified_records_all_five_flags_none_together(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "proposed but never vetted", "move": "advance"}]),
            tool_use_response(CRITIC_TOOL_NAME, {}),  # missing "judgments" -- unparseable
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    record = suggestions[0]
    assert record["shown"] is False
    for field in (
        "band", "is_leading", "is_hypothetical_or_future_tense", "is_pitch",
        "is_opinion_seeking", "is_confirmation_only", "is_generic",
    ):
        assert record[field] is None, f"{field} must be None -- never vetted, extended to every flag"
    # Candidate-level self-reports are independent of whether critique ran at all.
    assert record["grounding_ids"] == []
    assert record["target_hypothesis_id"] is None
    assert record["proposed_state"] is None


def test_a_suggestion_record_round_trips_through_the_session_file_with_grounding_ids_intact_and_in_order(tmp_path):
    store = _new_store(tmp_path)
    trigger = _trigger(store)
    store.append_suggestion(
        trigger_seq=trigger.seq, text="q", move="test_hypothesis", band=BAND_CLEAN,
        is_leading=False, is_hypothetical_or_future_tense=False, is_pitch=False,
        is_opinion_seeking=False, is_confirmation_only=False, is_generic=False, is_off_topic=False, rationale="r", shown=True,
        grounding_ids=("fact:2", "fact:1", "fact:3"),
        target_hypothesis_id="h1", proposed_state="contradicted",
    )
    store.close()

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert suggestions[0]["grounding_ids"] == ["fact:2", "fact:1", "fact:3"], (
        "order must be preserved exactly as appended -- never re-sorted on read"
    )
    assert suggestions[0]["target_hypothesis_id"] == "h1"
    assert suggestions[0]["proposed_state"] == "contradicted"


def test_a_lone_primary_with_no_alternatives_is_shown_and_is_a_complete_result(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "only one good candidate", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert suggestions[0]["shown"] is True


# --- A critique failure still durably records what generate proposed -------------------


def test_a_critique_parse_failure_yields_zero_shown_suggestions_but_still_records_the_candidates(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "proposed but never vetted", "move": "advance"}]),
            tool_use_response(CRITIC_TOOL_NAME, {}),  # missing "judgments" -- unparseable
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1, "the generate stage's candidate is still recorded even though critique failed"
    assert suggestions[0]["shown"] is False
    assert suggestions[0]["band"] is None, "never vetted -- no band exists"

    failure_events = [r for r in result.records_of_kind("event") if r["event"] == EVENT_REASONING_FAILURE]
    assert len(failure_events) == 1
    assert failure_events[0]["detail"]["stage"] == "critique"


def test_pipeline_keeps_running_after_a_failed_cycle_and_the_next_cycle_succeeds(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            # Cycle 1: generate itself fails to parse.
            tool_use_response(GENERATE_TOOL_NAME, {}),
            # Cycle 2: a normal, successful cycle.
            _gen_response([{"text": "second cycle candidate", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert suggestions[0]["text"] == "second cycle candidate"
    assert suggestions[0]["shown"] is True


# --- §D9: a second trigger arriving while a cycle is in flight replaces the queued one --


def test_a_trigger_arriving_while_one_is_already_queued_replaces_it_and_is_recorded(tmp_path):
    store = _new_store(tmp_path)
    # A single queued response with a real sleep keeps the worker thread busy
    # long enough for two more submits to race the queue.
    stub = StubAnthropicClient(
        responses=[_gen_response([{"text": "irrelevant", "move": "advance"}])] * 5, sleep_s=0.3
    )
    pipeline = _make_pipeline(store, stub, queue_maxsize=1)
    pipeline.start()
    try:
        first = _trigger(store)
        pipeline.submit(first)  # picked up by the worker almost immediately
        import time

        time.sleep(0.05)  # let the worker thread pull `first` off the queue
        second = _trigger(store)
        pipeline.submit(second)  # queues cleanly (queue was empty)
        third = _trigger(store)
        pipeline.submit(third)  # queue full (second still queued) -> drop second, record event, queue third

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = read_session(store.session_dir)
            if any(r.get("event") == EVENT_CYCLE_SKIPPED for r in result.records_of_kind("event")):
                break
            time.sleep(0.02)
    finally:
        pipeline.stop()
    store.close()

    result = read_session(store.session_dir)
    skipped = [r for r in result.records_of_kind("event") if r["event"] == EVENT_CYCLE_SKIPPED]
    assert len(skipped) == 1


# --- T-04-27: no candidate/transcript text ever reaches a log line or event detail -----


def test_no_candidate_or_transcript_text_leaks_into_a_log_line_or_event_detail(tmp_path, caplog):
    marker = "audit-distinctive-marker-9f2c7e14"
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": f"a candidate mentioning {marker}", "move": "advance"}]),
            tool_use_response(CRITIC_TOOL_NAME, {}),  # forces a failure event with a detail dict
        ]
    )
    pipeline = _make_pipeline(store, stub)
    caplog.set_level(logging.DEBUG)
    trigger = store.append_segment(
        speaker="interviewee", text=marker, audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    pipeline._process(trigger)
    store.close()

    result = read_session(store.session_dir)
    # The marker DOES belong in the transcript (segment) and suggestion records --
    # full text is durable-store-only by design (§D8).
    assert any(marker in r.get("text", "") for r in result.records_of_kind("segment"))
    assert any(marker in r.get("text", "") for r in result.records_of_kind("suggestion"))
    # ...but NEVER in an event's detail, and NEVER in a log line.
    for event_record in result.records_of_kind("event"):
        assert marker not in json.dumps(event_record.get("detail") or {})
    for log_record in caplog.records:
        assert marker not in log_record.getMessage()
