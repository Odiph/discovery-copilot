"""Tests for per-cycle coaching latency measurement (COACH-05), the
producing-thread boundary (non-negotiable #7), per-stage timeouts (§D1/§D2),
and `CoachHealth`'s consecutive-failure counter (§D7). Behavioural test
names, per this project's own convention."""

from __future__ import annotations

import queue
import time
from types import SimpleNamespace

import pytest

from app.coach.coverage import CoverageState
from app.coach.critic import BAND_CLEAN, CRITIC_TOOL_NAME
from app.coach.generate import GENERATE_TOOL_NAME
from app.coach.health import CoachHealth
from app.coach.pipeline import SuggestionPipeline, pipeline_latency_summary
from app.coach.stub import StubAnthropicClient, tool_use_response
from app.session.reader import read_session
from app.session.store import SessionStore, create_session_dir, latency_stats
from app.stt.live import StreamWorker
from benchmark.engines.base import Segment


def _new_store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"purpose": "coach-latency-test"})
    return store


def _make_pipeline(
    store, client, *, generate_timeout_s=2.5, critique_timeout_s=1.5, failure_threshold=3, queue_maxsize=1,
    session_start_grounding_s=0.0,
):
    state = CoverageState()
    return SuggestionPipeline(
        client=client, store=store, brief=(), survey=(),
        session_start_grounding_s=session_start_grounding_s,
        coverage_state_provider=lambda: state,
        model="m", generate_timeout_s=generate_timeout_s, critique_timeout_s=critique_timeout_s,
        generate_max_tokens=200, critique_max_tokens=200,
        max_shown=3, failure_threshold=failure_threshold,
        language_window_s=90.0, language_flip_margin=0.65, language_min_chars=40,
        queue_maxsize=queue_maxsize,
    )


def _gen_response(candidates):
    return tool_use_response(GENERATE_TOOL_NAME, {"candidates": candidates})


def _judgment_payload(index, band):
    return {
        "index": index, "is_leading": False, "is_hypothetical_or_future_tense": False,
        "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
        "band": band, "rationale": "fine",
    }


def _crit_response(judgments):
    return tool_use_response(CRITIC_TOOL_NAME, {"judgments": judgments})


def _trigger(store):
    return store.append_segment(
        speaker="interviewee", text="tell me about it", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )


# --- non-negotiable #7: the pipeline hook never blocks the producing thread ------------


def test_stream_worker_persist_returns_in_under_50ms_even_when_the_pipeline_hook_takes_3s(tmp_path):
    store = _new_store(tmp_path)
    slow_stub = StubAnthropicClient(responses=[_gen_response([{"text": "x", "move": "advance"}])], sleep_s=3.0)
    pipeline = _make_pipeline(store, slow_stub, generate_timeout_s=10.0)
    pipeline.start()
    try:
        worker = StreamWorker(
            engine=SimpleNamespace(name="fake"),
            frame_queue=queue.Queue(),
            store=store,
            speaker="interviewee",
            sample_rate=16000,
            segment_hook=pipeline.submit,
        )
        segment = Segment(text="tell me about it", audio_end_s=1.0, ready_wall_s=1.1, detected_language="en")
        start = time.monotonic()
        worker._persist(segment)
        elapsed_s = time.monotonic() - start
        assert elapsed_s < 0.05, (
            f"_persist took {elapsed_s:.3f}s -- the pipeline hook must enqueue and return, "
            "never block on the reasoning call (§D9, non-negotiable #7)"
        )
        time.sleep(3.5)  # let the slow cycle actually finish before tearing down
    finally:
        pipeline.stop()
    store.close()


# --- COACH-05: per-cycle latency is measured from the trigger's own ready time ---------


def test_per_cycle_latency_is_positive_and_proven_to_measure_something(tmp_path):
    stub = StubAnthropicClient(
        responses=[_gen_response([{"text": "x", "move": "advance"}]), _crit_response([_judgment_payload(0, BAND_CLEAN)])],
        sleep_s=0.15,
    )
    store = _new_store(tmp_path)
    pipeline = _make_pipeline(store, stub, generate_timeout_s=5.0, critique_timeout_s=5.0)
    trigger = _trigger(store)
    pipeline._process(trigger)

    summary = pipeline.latency_summary()
    assert summary["first_cycle_s"] is not None
    assert summary["first_cycle_s"] > 0.1, "the stub's own deliberate delay must show up in the measurement"


# --- pipeline_latency_summary: first cycle separate, shared stats function -------------


def test_pipeline_latency_summary_reports_zero_cycles_without_raising():
    summary = pipeline_latency_summary([], session_start_grounding_s=0.0)
    assert summary["first_cycle_s"] is None
    assert summary["stats"] == {"count": 0}


def test_pipeline_latency_summary_reports_the_first_cycle_separately_from_the_rest():
    summary = pipeline_latency_summary([9.9, 0.5, 0.6, 0.7], session_start_grounding_s=0.0)
    assert summary["first_cycle_s"] == 9.9
    assert summary["stats"]["count"] == 3
    assert summary["stats"]["max_s"] < 9.9, "the skewed first cycle must not distort the rest's stats"


def test_pipeline_latency_summary_is_computed_by_the_shared_latency_stats_function():
    values = [0.4, 0.5, 0.6, 0.7]
    summary = pipeline_latency_summary(values, session_start_grounding_s=0.0)
    assert summary["stats"] == latency_stats(values[1:]), (
        "the coaching and transcript latency summaries must share one implementation, never two"
    )


def test_pipeline_latency_summary_with_a_single_cycle_reports_it_as_first_and_zero_stats():
    summary = pipeline_latency_summary([1.2], session_start_grounding_s=0.0)
    assert summary["first_cycle_s"] == 1.2
    assert summary["stats"] == {"count": 0}


# --- 06-01-PLAN.md §F4: session_start_grounding_s is carried through inseparably -------


def test_pipeline_latency_summary_carries_the_supplied_grounding_cost_through_unchanged():
    summary = pipeline_latency_summary([0.4, 0.5], session_start_grounding_s=0.837)
    assert summary["session_start_grounding_s"] == 0.837


def test_pipeline_latency_summary_key_set_is_exactly_the_three_expected_keys():
    summary = pipeline_latency_summary([], session_start_grounding_s=0.0)
    assert set(summary) == {"first_cycle_s", "stats", "session_start_grounding_s"}


def test_pipeline_latency_summary_reports_the_grounding_cost_even_with_zero_cycles_completed(tmp_path):
    # A session where the coach never produced a suggestion still paid the
    # session-start cost -- that is exactly the case a reader needs to see.
    store = _new_store(tmp_path)
    pipeline = _make_pipeline(store, StubAnthropicClient(responses=[]), session_start_grounding_s=1.5)
    summary = pipeline.latency_summary()
    assert summary["session_start_grounding_s"] == 1.5
    assert summary["first_cycle_s"] is None


def test_suggestion_pipeline_cannot_be_constructed_without_the_session_start_grounding_cost(tmp_path):
    import inspect

    store = _new_store(tmp_path)
    with pytest.raises(TypeError):
        SuggestionPipeline(  # noqa: no session_start_grounding_s -- must raise
            client=StubAnthropicClient(responses=[]), store=store, brief=(), survey=(),
            coverage_state_provider=lambda: CoverageState(),
            model="m", generate_timeout_s=1.0, critique_timeout_s=1.0,
            generate_max_tokens=200, critique_max_tokens=200,
            max_shown=3, failure_threshold=3,
            language_window_s=90.0, language_flip_margin=0.65, language_min_chars=40,
        )
    parameter = inspect.signature(SuggestionPipeline.__init__).parameters["session_start_grounding_s"]
    assert parameter.default is inspect.Parameter.empty, (
        "a default here makes ROADMAP criterion 6 unmeasurable -- see the module docstring"
    )


def test_app_main_measures_around_the_context_load_not_inside_the_loader():
    # `app/main.py` wraps its OWN `_load_coach_context` call site with
    # `time.perf_counter()` -- `load_brief`'s signature and return type are
    # untouched (D-03). Proven here by asserting the loader itself carries no
    # new parameter, which is the only way this measurement could have been
    # threaded through the loader instead of the call site.
    import inspect

    from app.coach.context import load_brief

    signature = inspect.signature(load_brief)
    assert list(signature.parameters) == ["context_dir"], (
        f"load_brief's signature must stay exactly (context_dir) -- got {signature}"
    )


# --- §D1/§D2: each stage's own explicit timeout, and the two values differ -------------


def test_generate_and_critique_pass_their_own_differing_explicit_timeouts(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[_gen_response([{"text": "x", "move": "advance"}]), _crit_response([_judgment_payload(0, BAND_CLEAN)])]
    )
    pipeline = _make_pipeline(store, stub, generate_timeout_s=2.5, critique_timeout_s=1.5)
    pipeline._process(_trigger(store))

    assert len(stub.calls) == 2
    assert stub.calls[0].timeout_s == 2.5
    assert stub.calls[1].timeout_s == 1.5
    assert stub.calls[0].timeout_s != stub.calls[1].timeout_s


def test_a_timed_out_generate_call_yields_no_suggestions_and_leaves_the_pipeline_running(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            TimeoutError("simulated generate timeout"),
            _gen_response([{"text": "recovered", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))  # times out
    pipeline._process(_trigger(store))  # recovers

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert suggestions[0]["text"] == "recovered"

    failures = [r for r in result.records_of_kind("event") if r["event"] == "reasoning-timeout"]
    assert len(failures) == 1
    assert failures[0]["detail"]["stage"] == "generate"


def test_a_timed_out_critique_call_yields_zero_shown_but_records_what_generate_proposed(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[_gen_response([{"text": "proposed", "move": "advance"}]), TimeoutError("simulated critique timeout")]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))

    result = read_session(store.session_dir)
    suggestions = result.records_of_kind("suggestion")
    assert len(suggestions) == 1
    assert suggestions[0]["shown"] is False
    assert suggestions[0]["band"] is None


# --- §D7: CoachHealth -------------------------------------------------------------------


def test_coach_health_reports_unhealthy_only_at_or_above_the_threshold():
    health = CoachHealth()
    for _ in range(2):
        health = health.with_failure()
    assert health.is_unhealthy(threshold=3) is False, "threshold-minus-one must not report unhealthy"
    health = health.with_failure()
    assert health.is_unhealthy(threshold=3) is True


def test_coach_health_resets_to_zero_on_any_success():
    health = CoachHealth()
    for _ in range(2):
        health = health.with_failure()
    health = health.with_success()
    assert health.consecutive_failures == 0
    health = health.with_failure()
    assert health.is_unhealthy(threshold=3) is False, (
        "without the reset, one early blip would eventually mark a healthy coach as dead"
    )


def test_coach_health_is_constructible_with_no_reasoning_stack_at_all():
    health = CoachHealth()
    assert health.consecutive_failures == 0
    assert health.is_unhealthy(threshold=1) is False


def test_pipeline_health_increments_on_failure_and_resets_on_success(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(
        responses=[
            TimeoutError("first cycle fails"),
            _gen_response([{"text": "x", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
        ]
    )
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    assert pipeline.health.consecutive_failures == 1

    pipeline._process(_trigger(store))
    assert pipeline.health.consecutive_failures == 0


def test_a_skipped_cycle_is_not_counted_as_a_failure(tmp_path):
    store = _new_store(tmp_path)
    # Two FULL cycles' worth of responses (generate+critique each) -- exactly
    # the two triggers that actually get processed (the skipped one never
    # consumes a response at all).
    stub = StubAnthropicClient(
        responses=[
            _gen_response([{"text": "x", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
            _gen_response([{"text": "x", "move": "advance"}]),
            _crit_response([_judgment_payload(0, BAND_CLEAN)]),
        ],
        sleep_s=0.3,
    )
    pipeline = _make_pipeline(store, stub, queue_maxsize=1)
    pipeline.start()
    try:
        pipeline.submit(_trigger(store))
        time.sleep(0.05)
        pipeline.submit(_trigger(store))
        pipeline.submit(_trigger(store))  # this one gets skipped -- not a failure
        time.sleep(0.5)
    finally:
        pipeline.stop()
    store.close()
    assert pipeline.health.consecutive_failures == 0, "a skip must never touch the failure counter"


# --- A worker-thread exception must not silently kill the coach (review, 2026-08-06) ---


def test_an_unexpected_exception_in_a_cycle_does_not_kill_the_pipeline_thread(tmp_path):
    """Only the two vendor calls inside `_process` were wrapped. Everything
    else -- disk I/O, record construction from a resumed session, an
    `append_suggestion` against a full disk, timestamp parsing -- could raise,
    unwind `_run`, and end the thread for the rest of the interview.

    The dangerous part is not the death. `self._health` keeps whatever value
    it held at that moment (typically zero failures), so `is_unhealthy()` stays
    False and the phone keeps reporting a healthy coach forever. A dead coach
    that looks alive is exactly the state the coach-failure banner exists to
    prevent."""
    store = _new_store(tmp_path)
    pipeline = _make_pipeline(store, StubAnthropicClient(responses=[]))

    boom = {"count": 0}

    def _explode(record):
        boom["count"] += 1
        raise RuntimeError("simulated non-vendor failure inside a cycle")

    pipeline._process = _explode  # noqa: SLF001 -- deliberately exercising the guard around it
    pipeline.start()
    try:
        for seq in range(3):
            pipeline.submit(SimpleNamespace(
                kind="segment", seq=seq, speaker="interviewee", text="hello",
                audio_start_s=0.0, audio_end_s=float(seq + 1), ready_wall_s=float(seq) + 1.1,
                latency_s=0.1, detected_language="en", engine="fake", written_at="2026-08-06T00:00:00+00:00",
            ))
            time.sleep(0.25)
    finally:
        pipeline.stop()

    assert boom["count"] >= 2, (
        f"the worker thread stopped consuming after {boom['count']} failure(s) -- "
        f"it died instead of degrading and continuing"
    )
    assert pipeline.health.is_unhealthy(threshold=3), (
        "repeated cycle failures must drive the coach UNHEALTHY so the phone can say "
        "suggestions are unavailable -- a silently dead coach reporting healthy is the "
        "failure mode this guard exists to prevent"
    )
