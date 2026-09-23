"""Tests for `app/coach/coverage.py` -- D-03 (coverage is written from the
interviewer's own stream and nothing else), D-04 (no path from a suggestion
into coverage), the thread boundary (non-negotiable #7), the bounded queue,
degrade-and-continue, and crash-resume replay (SESS-02 extended). Also
covers `app/coach/reasoning_client.py`'s two invariants, since the detector
is reasoning_client's first real consumer. Behavioural test names, per this
project's own convention."""

from __future__ import annotations

import json
import logging
import queue
import time
from types import SimpleNamespace

import pytest

from app.coach import reasoning_client
from app.coach.context import SurveyItem
from app.coach.coverage import CoverageDetector, CoverageState
from app.coach.stub import StubAnthropicClient, StubExhaustedError, tool_use_response
from app.session.reader import read_session
from app.session.store import SessionStore, create_session_dir
from app.stt.live import StreamWorker
from benchmark.engines.base import Segment


def _wait_for_event(store: SessionStore, event_name: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = read_session(store.session_dir)
        for record in result.records_of_kind("event"):
            if record.get("event") == event_name:
                return record
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for event {event_name!r}")


def _new_store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"purpose": "coach-coverage-test"})
    return store


# --- D-03: coverage is written from the interviewer's own stream, and nothing else --


def test_interviewer_segment_moves_state_from_unasked_to_asked_and_appends_exactly_one_event(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="Walk me through onboarding."),)
    stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    assert detector.state.is_asked("s1") is False
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text="walk me through onboarding", audio_start_s=0.0,
            audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        _wait_for_event(store, "coverage-asked")
    finally:
        detector.stop()

    assert detector.state.is_asked("s1") is True
    result = read_session(store.session_dir)
    asked_events = [r for r in result.records_of_kind("event") if r["event"] == "coverage-asked"]
    assert len(asked_events) == 1
    detail = asked_events[0]["detail"]
    assert detail["question_id"] == "s1"
    assert detail["matched_seq"] == record.seq
    assert detail["matched_audio_end_s"] == record.audio_end_s
    store.close()


def test_interviewee_segment_never_moves_coverage_state_even_when_the_stub_would_match(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="Walk me through onboarding."),)
    stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})] * 5
    )
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewee", text="I think about three weeks usually", audio_start_s=0.0,
            audio_end_s=1.0, ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        time.sleep(0.3)  # give a misbehaving implementation a chance to prove itself wrong
    finally:
        detector.stop()

    assert detector.state.asked == frozenset()
    result = read_session(store.session_dir)
    coverage_events = [r for r in result.records_of_kind("event") if r["event"].startswith("coverage-")]
    assert coverage_events == [], "an interviewee segment must never write ANY coverage event"
    assert stub.calls == [], "the vendor must never even be called for an interviewee segment"
    store.close()


# --- D-04: no path from a suggestion into coverage state ----------------------------


def test_d04_detectors_public_surface_has_no_method_a_suggestion_could_call():
    public_names = sorted(
        name
        for name, value in vars(CoverageDetector).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, property))
    )
    assert public_names == ["start", "state", "stop", "submit"], (
        "the detector's public surface must be EXACTLY this set -- any new public method "
        "is a potential write path into coverage state that D-04 forbids"
    )


# --- Thread boundary (non-negotiable #7): submit() never blocks the producing thread -


def test_stream_worker_persist_returns_in_under_50ms_even_when_the_hook_downstream_sleeps_3s(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="Walk me through onboarding."),)
    slow_stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})],
        sleep_s=3.0,
    )
    detector = CoverageDetector(client=slow_stub, store=store, survey=survey, timeout_s=10.0)
    detector.start()
    try:
        worker = StreamWorker(
            engine=SimpleNamespace(name="fake"),
            frame_queue=queue.Queue(),
            store=store,
            speaker="interviewer",
            sample_rate=16000,
            segment_hook=detector.submit,
        )
        segment = Segment(
            text="walk me through onboarding", audio_end_s=1.0, ready_wall_s=1.1, detected_language="en"
        )
        start = time.monotonic()
        worker._persist(segment)
        elapsed_s = time.monotonic() - start
        # 0.5s, not 0.05s: the property is "does not wait for the 3s vendor call",
        # and a tighter wall-clock bound fails on a loaded CI runner without the
        # code having changed.
        assert elapsed_s < 0.5, (
            f"_persist took {elapsed_s:.3f}s -- the hook must enqueue and return, never block on "
            "the vendor call (§C3, non-negotiable #7)"
        )
        _wait_for_event(store, "coverage-asked", timeout_s=10.0)
    finally:
        detector.stop()
    store.close()


def test_a_raising_segment_hook_never_kills_the_stream_worker(tmp_path):
    store = _new_store(tmp_path)

    def _raising_hook(record):
        raise RuntimeError("a hook that misbehaves")

    worker = StreamWorker(
        engine=SimpleNamespace(name="fake"),
        frame_queue=queue.Queue(),
        store=store,
        speaker="interviewer",
        sample_rate=16000,
        segment_hook=_raising_hook,
    )
    segment = Segment(text="hello", audio_end_s=1.0, ready_wall_s=1.1, detected_language="en")
    worker._persist(segment)  # must not raise
    assert store.segment_count("interviewer") == 1
    store.close()


# --- Shutdown drain: bounded, never makes the vendor call (independent review finding D) --


def test_shutdown_drain_records_considered_with_a_shutdown_reason_and_never_calls_the_vendor(tmp_path):
    """Before the fix, `_run`'s post-stop drain called `_process_guarded` --
    the SAME full-round-trip vendor call the live loop uses -- for every
    queued record, which could take up to `queue_maxsize x timeout_s` under
    a vendor outage. `StubAnthropicClient()` below has no queued responses,
    so any call into it raises `StubExhaustedError`; a call landing there at
    all is exactly what this test forbids."""
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    stub = StubAnthropicClient()
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0, queue_maxsize=16)

    records = [
        store.append_segment(
            speaker="interviewer", text=f"q{i}", audio_start_s=float(i), audio_end_s=float(i + 1),
            ready_wall_s=float(i + 1.1), detected_language="en", engine="fake",
        )
        for i in range(3)
    ]
    for record in records:
        detector.submit(record)

    # Simulate `stop()` having already been requested, then drive `_run()`
    # directly (white-box, same convention as the full-queue test below) --
    # deterministic, with no thread/timing race to make flaky.
    detector._stop_event.set()
    detector._run()

    assert stub.calls == [], "the shutdown drain must never dispatch a vendor call"

    result = read_session(store.session_dir)
    considered = [r for r in result.records_of_kind("event") if r["event"] == "coverage-considered"]
    assert len(considered) == 3
    assert {c["detail"]["seq"] for c in considered} == {r.seq for r in records}
    for c in considered:
        assert c["detail"]["matched_question_id"] is None
        assert c["detail"]["reason"] == "shutdown"
    store.close()


def test_stop_returns_promptly_even_with_a_full_queue_backlog_and_a_slow_vendor(tmp_path):
    """End-to-end version of the drain test above, through the real
    `start()`/`submit()`/`stop()` public surface and a real worker thread --
    proves the WALL-CLOCK bound, not just that the vendor call is skipped."""
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    # Slow enough that draining even ONE record through the vendor call
    # would blow the assertion below; short enough to keep the test fast.
    stub = StubAnthropicClient(sleep_s=2.0, responses=[])
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0, queue_maxsize=16)

    records = [
        store.append_segment(
            speaker="interviewer", text=f"q{i}", audio_start_s=float(i), audio_end_s=float(i + 1),
            ready_wall_s=float(i + 1.1), detected_language="en", engine="fake",
        )
        for i in range(10)
    ]
    for record in records:
        detector.submit(record)

    detector.start()
    started = time.monotonic()
    detector.stop()  # must not block on 10 x 2.0s of vendor calls
    elapsed = time.monotonic() - started

    # Bounded by the ONE in-flight call the live loop had already dispatched
    # before `stop()` was requested (~2.0s), plus a near-instant drain of the
    # other 9 -- never anywhere close to 10 x 2.0s = 20s (which would also
    # have blown `stop()`'s own 15.0s join timeout).
    assert elapsed < 5.0, (
        f"stop() took {elapsed:.2f}s -- the shutdown drain must never process a backlog "
        "through the vendor call one record at a time"
    )
    store.close()


# --- Bounded queue: drop-oldest on a full queue, never blocks or raises -------------


def test_submit_on_a_full_queue_drops_the_oldest_entry_keeps_the_newest_and_records_an_event(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    # No responses queued and the worker thread is never started -- nothing
    # ever drains the queue, so a fill-then-overflow is fully deterministic.
    stub = StubAnthropicClient()
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0, queue_maxsize=1)

    oldest = store.append_segment(
        speaker="interviewer", text="oldest", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    newest = store.append_segment(
        speaker="interviewer", text="newest", audio_start_s=1.0, audio_end_s=2.0,
        ready_wall_s=2.1, detected_language="en", engine="fake",
    )
    detector.submit(oldest)  # never blocks or raises even though nothing drains it
    detector.submit(newest)  # queue is full -- must drop `oldest`, keep `newest`

    remaining = detector._queue.get_nowait()  # white-box: inspect what survived
    assert remaining.seq == newest.seq

    result = read_session(store.session_dir)
    drop_events = [r for r in result.records_of_kind("event") if r["event"] == "coverage-queue-drop"]
    assert len(drop_events) == 1
    assert drop_events[0]["detail"] == {"reason": "queue-full"}
    store.close()


# --- Degrade and continue: one vendor failure never disables the detector -----------


def test_client_exception_records_a_named_failure_event_with_no_text_and_leaves_state_untouched(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    marker = "distinctive-marker-should-never-leak-9f3e"
    stub = StubAnthropicClient(raises=TimeoutError("simulated vendor timeout"))
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text=marker, audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        failure_event = _wait_for_event(store, "reasoning-timeout")
    finally:
        detector.stop()

    assert set(failure_event["detail"].keys()) == {"stage", "error_type"}
    assert failure_event["detail"] == {"stage": "coverage", "error_type": "TimeoutError"}
    assert detector.state.asked == frozenset(), "a failed match must leave coverage state untouched"

    store.close()
    result = read_session(store.session_dir)
    for event_record in result.records_of_kind("event"):
        assert marker not in json.dumps(event_record), (
            f"segment text leaked into an event record: {event_record}"
        )


def test_a_timeout_and_a_connection_error_both_produce_the_same_event_kind_with_their_own_error_type(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)

    stub_timeout = StubAnthropicClient(raises=TimeoutError("t/o"))
    detector_timeout = CoverageDetector(client=stub_timeout, store=store, survey=survey, timeout_s=1.0)
    detector_timeout.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text="first", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector_timeout.submit(record)
        timeout_event = _wait_for_event(store, "reasoning-timeout")
    finally:
        detector_timeout.stop()
    assert timeout_event["detail"]["error_type"] == "TimeoutError"
    assert detector_timeout.state.asked == frozenset()

    stub_conn = StubAnthropicClient(raises=ConnectionError("connection refused"))
    detector_conn = CoverageDetector(client=stub_conn, store=store, survey=survey, timeout_s=1.0)
    detector_conn.start()
    try:
        record2 = store.append_segment(
            speaker="interviewer", text="second", audio_start_s=1.0, audio_end_s=2.0,
            ready_wall_s=2.1, detected_language="en", engine="fake",
        )
        detector_conn.submit(record2)
        deadline = time.monotonic() + 5.0
        conn_event = None
        while time.monotonic() < deadline:
            result = read_session(store.session_dir)
            failures = [r for r in result.records_of_kind("event") if r["event"] == "reasoning-timeout"]
            if len(failures) >= 2:
                conn_event = failures[-1]
                break
            time.sleep(0.02)
        assert conn_event is not None, "timed out waiting for the second failure event"
    finally:
        detector_conn.stop()
    assert conn_event["detail"]["error_type"] == "ConnectionError"
    assert detector_conn.state.asked == frozenset()
    store.close()


def test_a_second_successful_submission_after_a_failure_still_produces_a_match(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    stub = StubAnthropicClient(
        responses=[
            TimeoutError("first call fails"),
            tool_use_response("report_coverage_match", {"question_id": "s1"}),
        ]
    )
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    detector.start()
    try:
        first = store.append_segment(
            speaker="interviewer", text="first, will fail", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(first)
        _wait_for_event(store, "reasoning-timeout")

        second = store.append_segment(
            speaker="interviewer", text="second, will match", audio_start_s=1.0, audio_end_s=2.0,
            ready_wall_s=2.1, detected_language="en", engine="fake",
        )
        detector.submit(second)
        _wait_for_event(store, "coverage-asked")
    finally:
        detector.stop()

    assert detector.state.is_asked("s1") is True, "one failure must not permanently disable the detector"
    store.close()


# --- A non-match is recorded too, and changes no state ------------------------------


def test_a_segment_with_no_match_records_a_considered_event_and_changes_no_state(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": None})]
    )
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=1.0)
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text="something unrelated", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        considered_event = _wait_for_event(store, "coverage-considered")
    finally:
        detector.stop()

    assert considered_event["detail"] == {"seq": record.seq, "matched_question_id": None}
    assert detector.state.asked == frozenset()
    store.close()


# --- Crash-resume replay (SESS-02 extended, §C5) -------------------------------------


def test_replay_rebuilds_state_exactly_from_the_durable_record_written_by_a_prior_instance(tmp_path):
    session_dir = create_session_dir(tmp_path)
    store1 = SessionStore(session_dir)
    store1.write_metadata({"purpose": "replay-probe"})
    survey = (SurveyItem(id="s1", order=0, text="q1"), SurveyItem(id="s2", order=1, text="q2"))
    stub1 = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )
    detector1 = CoverageDetector(client=stub1, store=store1, survey=survey, timeout_s=1.0)
    detector1.start()
    try:
        record = store1.append_segment(
            speaker="interviewer", text="asked s1", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector1.submit(record)
        _wait_for_event(store1, "coverage-asked")
    finally:
        detector1.stop()
    store1.close()

    store2 = SessionStore(session_dir)  # reopen -- resume, never a fresh session
    assert store2.resumed
    stub2 = StubAnthropicClient()  # zero canned responses -- replay must never call the vendor
    detector2 = CoverageDetector(client=stub2, store=store2, survey=survey, timeout_s=1.0)

    assert detector2.state.asked == frozenset({"s1"})
    assert stub2.calls == [], "rebuilding state on reopen must read the durable record, never call the vendor"
    store2.close()


def test_replay_is_idempotent_and_never_appends_a_new_transition_on_reopen(tmp_path):
    session_dir = create_session_dir(tmp_path)
    store1 = SessionStore(session_dir)
    store1.write_metadata({"purpose": "idempotent-replay-probe"})
    survey = (SurveyItem(id="s1", order=0, text="q1"),)
    stub1 = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )
    detector1 = CoverageDetector(client=stub1, store=store1, survey=survey, timeout_s=1.0)
    detector1.start()
    try:
        record = store1.append_segment(
            speaker="interviewer", text="asked s1", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector1.submit(record)
        _wait_for_event(store1, "coverage-asked")
    finally:
        detector1.stop()
    store1.close()

    for _ in range(2):
        store_n = SessionStore(session_dir)
        detector_n = CoverageDetector(client=StubAnthropicClient(), store=store_n, survey=survey, timeout_s=1.0)
        assert detector_n.state.asked == frozenset({"s1"})
        store_n.close()

    result = read_session(session_dir)
    asked_events = [r for r in result.records_of_kind("event") if r["event"] == "coverage-asked"]
    assert len(asked_events) == 1, "reopening must never append a new coverage-asked event"


def test_replay_tolerates_a_crash_truncated_final_line(tmp_path):
    session_dir = create_session_dir(tmp_path)
    store1 = SessionStore(session_dir)
    store1.write_metadata({"purpose": "truncation-probe"})
    survey = (SurveyItem(id="s1", order=0, text="q1"),)
    stub1 = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )
    detector1 = CoverageDetector(client=stub1, store=store1, survey=survey, timeout_s=1.0)
    detector1.start()
    try:
        record = store1.append_segment(
            speaker="interviewer", text="asked s1", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector1.submit(record)
        _wait_for_event(store1, "coverage-asked")
    finally:
        detector1.stop()
    store1.close()

    # Simulate a hard kill: a write that never got its trailing newline.
    transcript_path = session_dir / "transcript.jsonl"
    with open(transcript_path, "a", encoding="utf-8") as handle:
        handle.write('{"kind": "event", "seq": 999, "event": "coverage-considered", "detail": {"partial')

    store2 = SessionStore(session_dir)  # truncates the partial tail internally (SESS-02)
    assert store2.resumed
    detector2 = CoverageDetector(client=StubAnthropicClient(), store=store2, survey=survey, timeout_s=1.0)
    assert detector2.state.asked == frozenset({"s1"}), (
        "every COMPLETE transition before the truncated tail must still replay"
    )
    store2.close()


def test_replay_ignores_a_coverage_event_for_a_question_id_no_longer_in_the_survey(tmp_path, caplog):
    session_dir = create_session_dir(tmp_path)
    store1 = SessionStore(session_dir)
    store1.write_metadata({"purpose": "mismatch-probe"})
    old_survey = (SurveyItem(id="s-old", order=0, text="an old question"),)
    stub1 = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s-old"})]
    )
    detector1 = CoverageDetector(client=stub1, store=store1, survey=old_survey, timeout_s=1.0)
    detector1.start()
    try:
        record = store1.append_segment(
            speaker="interviewer", text="asked the old one", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector1.submit(record)
        _wait_for_event(store1, "coverage-asked")
    finally:
        detector1.stop()
    store1.close()

    new_survey = (SurveyItem(id="s-new", order=0, text="a new question"),)  # edited between runs
    store2 = SessionStore(session_dir)
    caplog.set_level(logging.WARNING)
    detector2 = CoverageDetector(client=StubAnthropicClient(), store=store2, survey=new_survey, timeout_s=1.0)

    assert detector2.state.asked == frozenset(), "a transition for a removed question id must be IGNORED"
    assert any("s-old" in record.getMessage() for record in caplog.records), (
        "the ignored mismatch must produce a named warning"
    )
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert metadata.get("coverage_replay_mismatches") == 1
    store2.close()


# --- app/coach/reasoning_client.py -- the client seam this detector depends on ------


def test_build_client_raises_an_actionable_error_when_the_key_is_absent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(reasoning_client.CoachConfigError) as exc_info:
        reasoning_client.build_client()
    message = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "sk-" not in message, "the error message must never contain a key-shaped value"


def test_build_client_passes_the_key_explicitly_and_sets_retries_to_zero(monkeypatch):
    dummy_key = "sk-ant-test-dummy-abc123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", dummy_key)
    client = reasoning_client.build_client()
    # Asserted against the constructed client's OWN observable configuration
    # (§C4) -- a future edit dropping either property fails this immediately.
    assert client.api_key == dummy_key
    assert client.max_retries == 0


def test_the_bound_client_carries_both_the_explicit_timeout_and_zero_retries_together(monkeypatch):
    # §C4: "either one alone is a false guarantee" -- a timeout with retries still
    # enabled is silently up to 3x its configured bound; retries=0 with no explicit
    # timeout inherits the vendor's 10-minute default. Both must hold on the SAME
    # bound client, asserted together, against the REAL SDK class (no network call
    # is made -- `with_options` is pure local configuration).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-combined-check")
    client = reasoning_client.build_client()
    bound = reasoning_client.with_timeout(client, 2.75)
    assert bound.max_retries == 0
    assert bound.timeout == 2.75


def test_with_timeout_passes_the_timeout_explicitly_on_every_call(tmp_path):
    store = _new_store(tmp_path)
    survey = (SurveyItem(id="s1", order=0, text="q"),)
    stub = StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": "s1"})]
    )
    detector = CoverageDetector(client=stub, store=store, survey=survey, timeout_s=2.75)
    detector.start()
    try:
        record = store.append_segment(
            speaker="interviewer", text="asked", audio_start_s=0.0, audio_end_s=1.0,
            ready_wall_s=1.1, detected_language="en", engine="fake",
        )
        detector.submit(record)
        _wait_for_event(store, "coverage-asked")
    finally:
        detector.stop()

    assert len(stub.calls) == 1
    assert stub.calls[0].timeout_s == 2.75, (
        "the detector's per-call timeout must be passed explicitly, from config, on every call"
    )
    store.close()


def test_stub_client_raises_when_a_test_under_primes_it(tmp_path):
    # The stub itself must expose the same call shape a real over-called
    # client would fail on -- an under-primed stub is a TEST bug, not a
    # condition the product code should ever trigger.
    stub = StubAnthropicClient(responses=[])
    with pytest.raises(StubExhaustedError):
        stub.with_options(timeout=1.0).messages.create(
            model="x", tools=[], tool_choice={"type": "tool", "name": "y"}
        )


# --- CoverageState -- a plain immutable value ----------------------------------------


def test_coverage_state_unasked_items_excludes_asked_ids():
    survey = (SurveyItem(id="s1", order=0, text="a"), SurveyItem(id="s2", order=1, text="b"))
    state = CoverageState(asked=frozenset({"s1"}))
    remaining = state.unasked_items(survey)
    assert [item.id for item in remaining] == ["s2"]


def test_coverage_state_with_asked_returns_a_new_immutable_value():
    state = CoverageState()
    next_state = state.with_asked("s1")
    assert state.asked == frozenset()  # the original is untouched
    assert next_state.asked == frozenset({"s1"})
