"""Phase C: the `survey` wire message -- the next unasked survey question,
always visible on the overlay.

What this module is really testing is Phase 4 criterion 6 ("degrades to
transcript-only, never goes blank"). Every other thing the overlay can show is
the output of a model call, so a slow or failing vendor empties the surface.
This message is computed from the authored survey plus the coverage state
`CoverageDetector` already maintains -- no client, no timeout, no failure mode
-- which is what turns that criterion from incidentally true into structurally
true. The tests below are written to hold that line: several of them assert
things that are only interesting because they must remain true while the coach
is dead.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from websockets.sync.client import connect as ws_connect

from app.coach.context import SurveyItem
from app.coach.coverage import CoverageDetector, CoverageState
from app.coach.stub import StubAnthropicClient, tool_use_response
from app.session.reader import read_session
from app.session.store import SegmentRecord, SessionStore
from app.web import server as web_server
from app.web.hub import WIRE_SURVEY_KEYS, ConnectionHub, survey_payload

# The exact real-server harness 03-01 built -- a real uvicorn server and the
# synchronous `websockets` client, never `starlette.testclient.TestClient`,
# which is not a dependency this project has (gate 4 caps it at five).
from app.tests.test_web_serving import STATIC_DIR, _running_server, _store

SURVEY = (
    SurveyItem(id="q1", order=1, text="What are you trying to get done?"),
    SurveyItem(id="q2", order=2, text="מה ניסית לפני זה?"),
    SurveyItem(id="q3", order=3, text="What made you look for something else?"),
)


# --- survey_payload: which question is "next" -----------------------------------------


def test_the_next_question_is_the_first_unasked_one_in_survey_order():
    payload = survey_payload(SURVEY, asked=frozenset({"q1"}))
    assert payload["next_question_id"] == "q2"
    assert payload["next_question_text"] == "מה ניסית לפני זה?"
    assert payload["next_question_order"] == 2


def test_a_question_asked_out_of_order_does_not_skip_the_ones_before_it():
    # The interviewer jumping ahead is normal. What must NOT happen is the
    # panel jumping with them and quietly abandoning q1 -- coverage is a set,
    # not a cursor.
    payload = survey_payload(SURVEY, asked=frozenset({"q3"}))
    assert payload["next_question_id"] == "q1"
    assert payload["asked_count"] == 1


def test_the_next_question_is_derived_from_order_not_from_the_callers_list_order():
    # `load_survey()` already sorts, but "the next question" is this message's
    # entire meaning -- it must not be changeable by handing the projector a
    # differently-ordered list (a merged multi-project read, say).
    shuffled = (SURVEY[2], SURVEY[0], SURVEY[1])
    assert survey_payload(shuffled, asked=frozenset())["next_question_id"] == "q1"


def test_an_exhausted_survey_reports_no_next_question_rather_than_the_last_one():
    payload = survey_payload(SURVEY, asked=frozenset({"q1", "q2", "q3"}))
    assert payload["next_question_id"] is None
    assert payload["next_question_text"] is None
    assert payload["next_question_order"] is None
    assert payload["asked_count"] == 3 and payload["total_count"] == 3


def test_an_empty_survey_is_a_reportable_state_not_an_error():
    payload = survey_payload((), asked=frozenset())
    assert payload["next_question_id"] is None
    assert payload["total_count"] == 0
    assert set(payload) == WIRE_SURVEY_KEYS


def test_asked_count_ignores_ids_the_survey_no_longer_contains():
    # `CoverageDetector._replay_state` deliberately tolerates a coverage event
    # naming a question a later survey edit removed. Counting `len(asked)`
    # instead of intersecting with the survey would then report more questions
    # asked than the survey has -- "4 of 3 asked" on the overlay.
    payload = survey_payload(SURVEY, asked=frozenset({"q1", "deleted-question"}))
    assert payload["asked_count"] == 1
    assert payload["total_count"] == 3


def test_the_payload_never_carries_the_remaining_questions():
    # The gate pins this by key equality; this states the property in the terms
    # that matter -- no other question's TEXT may appear anywhere in the
    # message, whatever key someone invents to carry it.
    payload = survey_payload(SURVEY, asked=frozenset())
    serialized = repr(payload)
    for item in SURVEY[1:]:
        assert item.text not in serialized, (
            f"survey item {item.id!r}'s text reached the wire -- only the NEXT question may"
        )


# --- broadcast_survey: it works with no coach, no client, nothing running -------------


class _FakeConnection:
    def __init__(self) -> None:
        self.received: "list[dict]" = []

    async def accept(self) -> None:
        pass

    async def send_json(self, payload: dict) -> None:
        self.received.append(payload)


def test_the_survey_broadcast_needs_no_client_no_pipeline_and_no_running_coach():
    # The whole point of this message. Nothing in this test constructs a
    # reasoning client, a SuggestionPipeline, or a CoverageDetector -- if this
    # ever stops being possible, the panel has acquired a model dependency and
    # criterion 6 is back to being incidental.
    async def scenario() -> None:
        hub = ConnectionHub()
        conn = _FakeConnection()
        await hub.connect(conn)  # type: ignore[arg-type]

        await hub.broadcast_survey(SURVEY, asked=frozenset({"q1"}))

        assert conn.received == [survey_payload(SURVEY, asked=frozenset({"q1"}))]
        assert conn.received[0]["type"] == "survey"

    asyncio.run(scenario())


# --- CoverageDetector.on_change -------------------------------------------------------


def _client_reporting(question_id) -> StubAnthropicClient:
    """A client that answers ONE coverage call with a fixed `question_id`
    (`None` = considered but matched nothing).

    Built on `StubAnthropicClient` rather than hand-rolled. A hand-rolled
    double here is not merely redundant, it is unsound: every call site in
    `app/coach/` goes through `reasoning_client.with_timeout()`, which is
    `client.with_options(timeout=...)`. A double missing `with_options`
    raises `AttributeError` inside `_process`'s broad except, which records a
    reasoning-failure event and returns -- so coverage never moves and the
    test fails for a reason that has nothing to do with what it asserts,
    while any *negative* assertion in the same shape passes vacuously."""
    return StubAnthropicClient(
        responses=[tool_use_response("report_coverage_match", {"question_id": question_id})]
    )


def _detector(tmp_path, client, **kwargs) -> "tuple[CoverageDetector, SessionStore]":
    store = _store(tmp_path)
    detector = CoverageDetector(
        client=client, store=store, survey=SURVEY, timeout_s=1.0, **kwargs
    )
    return detector, store


def _segment(seq: int) -> SegmentRecord:
    return SegmentRecord(
        kind="segment",
        seq=seq,
        speaker="interviewer",
        text="so what were you trying to get done?",
        audio_start_s=0.0,
        audio_end_s=1.0 * seq,
        ready_wall_s=1.0 * seq,
        latency_s=0.1,
        detected_language="en",
        engine="stub",
        written_at="2026-08-10T12:00:00+03:00",
    )


def test_on_change_fires_with_the_new_state_when_a_question_moves_to_asked(tmp_path):
    seen: "list[CoverageState]" = []
    detector, store = _detector(
        tmp_path, _client_reporting("q1"), on_change=seen.append
    )
    try:
        detector._process(_segment(1))
    finally:
        store.close()

    assert len(seen) == 1
    assert seen[0].asked == frozenset({"q1"})
    # The callback must see the state INCLUDING the new transition -- a
    # callback handed the pre-transition state would re-broadcast the same
    # question that was just asked.
    assert survey_payload(SURVEY, asked=seen[0].asked)["next_question_id"] == "q2"


def test_on_change_does_not_fire_when_the_utterance_matched_nothing(tmp_path):
    seen: "list[CoverageState]" = []
    detector, store = _detector(
        tmp_path, _client_reporting(None), on_change=seen.append
    )
    try:
        detector._process(_segment(1))
    finally:
        store.close()

    assert seen == [], "a considered-but-unmatched utterance changes no state and must not broadcast"


def test_a_raising_on_change_never_loses_the_durable_coverage_event(tmp_path):
    # A broken display concern must not cost the record that `_replay_state`
    # rebuilds coverage from on the next run.
    #
    # Mutation-tested, and the result is worth writing down because it is not
    # what this test was originally claimed to cover. Narrowing
    # `_notify_change`'s `except Exception` to `except ValueError` fails this
    # test; moving the `_notify_change` call to BEFORE `append_event` does
    # NOT. So the guard here is the SWALLOW, not the call ordering -- while
    # the swallow holds, no callback exception can reach the append at all,
    # which makes the order unobservable from outside. Do not read this test
    # as protecting the ordering; if the swallow is ever narrowed or removed,
    # the ordering becomes load-bearing and needs its own test.
    def _boom(_state):
        raise RuntimeError("broadcast exploded")

    detector, store = _detector(tmp_path, _client_reporting("q1"), on_change=_boom)
    try:
        detector._process(_segment(1))  # must not raise
        assert detector.state.asked == frozenset({"q1"})
        session_dir = store.session_dir
    finally:
        store.close()

    events = read_session(session_dir).records_of_kind("event")
    asked = [e for e in events if e.get("event") == "coverage-asked"]
    assert len(asked) == 1, "the durable coverage-asked event must survive a failing on_change"
    assert asked[0]["detail"]["question_id"] == "q1"
    failures = [e for e in events if e.get("event") == "coverage-failed"]
    assert failures == [], (
        "a failing on_change must not be recorded as a COVERAGE failure -- coverage worked; "
        "the broadcast did not"
    )


def test_on_change_is_optional_so_no_serve_runs_unchanged(tmp_path):
    detector, store = _detector(tmp_path, _client_reporting("q1"))
    try:
        detector._process(_segment(1))
        assert detector.state.asked == frozenset({"q1"})
    finally:
        store.close()


# --- the on-connect push --------------------------------------------------------------


def _connect_and_read(token: str, tmp_path, **create_app_kwargs) -> "list[dict]":
    """Opens a real websocket against a real uvicorn server and returns the
    messages pushed before any client message is sent, plus the pong that
    proves the connection is still usable afterwards."""
    app = web_server.create_app(
        token=token,
        hub=ConnectionHub(),
        static_dir=STATIC_DIR,
        store=_store(tmp_path),
        session_id="s",
        server_started_at="2026-08-10T12:00:00+03:00",
        **create_app_kwargs,
    )
    with _running_server(app) as (base_url, _loop):
        ws_url = base_url.replace("http://", "ws://") + f"/s/{token}/ws"
        with ws_connect(ws_url, open_timeout=5) as client:
            pushed = [json.loads(client.recv(timeout=5)) for _ in range(2)]
            # A third pushed message may or may not exist depending on whether
            # a provider was given, so `ping` is used as a fence: whatever
            # arrives before the pong was server-pushed on connect.
            client.send(json.dumps({"type": "ping"}))
            while True:
                message = json.loads(client.recv(timeout=5))
                pushed.append(message)
                if message["type"] == "pong":
                    return pushed


def test_a_failing_survey_provider_never_breaks_a_connect(tmp_path):
    # Guarded in `create_app` for a specific reason: this message exists to be
    # the thing that still works when other things do not, so it must not
    # become a new way for the surface to fail entirely.
    calls: "list[int]" = []

    def _boom():
        calls.append(1)
        raise RuntimeError("provider exploded")

    pushed = _connect_and_read("survey-provider-token", tmp_path, survey_provider=_boom)

    assert calls == [1], "the provider must actually have been called -- otherwise this proves nothing"
    assert [m["type"] for m in pushed] == ["hello", "server_status", "pong"], (
        "a failing provider must cost exactly the survey message and nothing else -- the connection "
        "stays open and still answers the heartbeat"
    )


def test_the_survey_is_pushed_on_connect_so_a_late_joiner_is_correct_immediately(tmp_path):
    pushed = _connect_and_read(
        "survey-connect-token",
        tmp_path,
        # Two questions already asked before this browser ever connected.
        survey_provider=lambda: survey_payload(SURVEY, asked=frozenset({"q1", "q2"})),
    )

    assert [m["type"] for m in pushed] == ["hello", "server_status", "survey", "pong"]
    message = pushed[2]
    assert message["next_question_id"] == "q3", (
        "a browser joining mid-interview must show the question that is actually next, not the "
        "first one -- otherwise the panel is wrong for however long it takes coverage to move again"
    )
    assert set(message) == WIRE_SURVEY_KEYS


def test_no_survey_provider_means_no_survey_message_and_an_otherwise_identical_connect(tmp_path):
    pushed = _connect_and_read("survey-absent-token", tmp_path)
    assert [m["type"] for m in pushed] == ["hello", "server_status", "pong"], (
        "with no provider the next message must be the pong -- a keyless run has no coach and "
        "therefore no survey to report"
    )


# --- the overlay itself ---------------------------------------------------------------


def test_the_page_ships_a_next_question_region_that_cannot_collapse():
    static = Path(__file__).resolve().parents[1] / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    css = (static / "style.css").read_text(encoding="utf-8")
    app_js = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="next-question-region"' in html
    assert ".next-question-region" in css and "min-height" in css
    assert "message.type === MESSAGE_TYPE_SURVEY" in app_js, (
        "app.js must actually handle the survey message -- and via the imported constant, "
        "never a hand-typed literal (see test_wire_vocabulary.py)"
    )


def test_a_coach_failure_never_reaches_the_next_question_region():
    # `renderServerStatus`'s existing rule (04-03 §E5) is that a coach failure
    # leaves the suggestion region alone. The next-question region is a
    # stronger case: it has nothing to do with the coach at all, so no failure
    # path may touch it.
    app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(encoding="utf-8")
    body_start = app_js.index("function renderServerStatus")
    body_end = app_js.index("function stopHeartbeatTicking")
    assert "nextQuestionRegion" not in app_js[body_start:body_end], (
        "renderServerStatus must never touch the next-question region -- a dead coach is not a "
        "reason to blank the one panel that never needed the coach"
    )
