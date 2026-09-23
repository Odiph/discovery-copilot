"""The repetition guard AT THE PIPELINE SEAM, not in isolation.

`test_coach_repetition.py` proves `RecentSuggestions` classifies text
correctly. That is INTENT. This file proves the pipeline actually withholds a
repeat from the operator and still records it durably -- which is BEHAVIOUR,
and is the half that a passing unit test would not have caught if the guard
were wired up wrongly (or not at all).

Cycles are driven through `_process` synchronously rather than through the
worker thread, so the two cycles are strictly ordered and the assertions are
not racing a background thread.
"""

from __future__ import annotations

from app.coach.coverage import CoverageState
from app.coach.critic import BAND_CLEAN, CRITIC_TOOL_NAME
from app.coach.generate import GENERATE_TOOL_NAME
from app.coach.pipeline import EVENT_SUGGESTION_REPEATED, SuggestionPipeline
from app.coach.stub import StubAnthropicClient, tool_use_response
from app.session.reader import read_session
from app.session.store import SessionStore, create_session_dir

OFFENDER = "Walk me through the last time you removed something from your gaming machine. What made you get rid of it?"
VARIANT = "Walk me through the last time you removed something from that machine. What made you get rid of it?"
OTHER = "What is on your second screen while you play, and how did it end up there?"


def _new_store(tmp_path) -> SessionStore:
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"purpose": "coach-repetition-test"})
    return store


def _gen(candidates):
    return tool_use_response(
        GENERATE_TOOL_NAME,
        {"candidates": [{"text": text, "move": "deepen"} for text in candidates]},
    )


def _crit(count):
    return tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {
            "index": index, "is_leading": False, "is_hypothetical_or_future_tense": False,
            "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
            "band": BAND_CLEAN, "rationale": "fine",
        }
        for index in range(count)
    ]})


class _RecordingClient:
    """Records the raw request bodies on their way to a real `StubAnthropicClient`.

    `StubCall` deliberately carries only model/timeout/tool_name -- its
    docstring says so, to stop tests coupling themselves to prompt WORDING.
    The two assertions below are about prompt STRUCTURE (which block the
    already-proposed list lands in), which is the contract that has to hold
    however the wording changes, so a local recorder is the honest tool rather
    than widening `StubCall` for everyone."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._options: dict = {}
        self.requests: "list[dict]" = []

    def with_options(self, **options) -> "_RecordingClient":
        # Forwarded, never swallowed: the stub REQUIRES an explicit timeout,
        # and dropping it here would make every call fail as a GenerateFailure
        # -- which the pipeline degrades past silently, so the test would go
        # green-ish while recording nothing.
        self._options = options
        return self

    @property
    def messages(self) -> "_RecordingClient":
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._inner.with_options(**self._options).messages.create(**kwargs)


def _make_pipeline(store, client, **overrides):
    state = CoverageState()
    kwargs = dict(
        client=client, store=store, brief=(), survey=(),
        session_start_grounding_s=0.0,
        coverage_state_provider=lambda: state,
        model="m", generate_timeout_s=5.0, critique_timeout_s=5.0,
        generate_max_tokens=200, critique_max_tokens=200,
        max_shown=3, failure_threshold=3,
        language_window_s=90.0, language_flip_margin=0.65, language_min_chars=40,
    )
    kwargs.update(overrides)
    return SuggestionPipeline(**kwargs)


def _trigger(store, text="tell me about it"):
    return store.append_segment(
        speaker="interviewee", text=text, audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )


def _shown_texts(store):
    result = read_session(store.session_dir)
    return [r["text"] for r in result.records_of_kind("suggestion") if r["shown"]]


def _all_texts(store):
    result = read_session(store.session_dir)
    return [r["text"] for r in result.records_of_kind("suggestion")]


def _events(store, name):
    result = read_session(store.session_dir)
    return [r for r in result.records_of_kind("event") if r.get("event") == name]


def test_the_same_question_proposed_on_a_later_cycle_is_not_shown_a_second_time(tmp_path):
    """The measured defect, reproduced: the model proposes the identical
    question on two consecutive cycles."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OFFENDER]), _crit(1),
    ])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))

    assert _shown_texts(store) == [OFFENDER], "the repeat reached the operator a second time"
    # ...but BOTH are still on the record. D-08's two-tier rule: dedup changes
    # what is SHOWN, never what is written.
    assert _all_texts(store) == [OFFENDER, OFFENDER]
    assert len(_events(store, EVENT_SUGGESTION_REPEATED)) == 1
    store.close()


def test_a_near_variant_is_suppressed_too_which_exact_matching_would_have_let_through(tmp_path):
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([VARIANT]), _crit(1),
    ])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    assert _shown_texts(store) == [OFFENDER]
    store.close()


def test_a_genuinely_new_question_still_reaches_the_operator(tmp_path):
    """The guard must not be a mute button. Without this, a threshold set to
    0.0 would pass every other test in this file."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OTHER]), _crit(1),
    ])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    assert _shown_texts(store) == [OFFENDER, OTHER]
    assert _events(store, EVENT_SUGGESTION_REPEATED) == []
    store.close()


def test_a_repeat_does_not_consume_a_shown_slot_and_starve_the_good_candidates(tmp_path):
    """Ranked, THEN de-duplicated, THEN capped. If the cap were applied first
    the repeat would eat one of the three slots and the operator would see two
    questions where three were available."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OFFENDER, "First genuinely different question about your setup",
              "Second unrelated question concerning your evening routine",
              "Third distinct question about what you tried next"]), _crit(4),
    ])
    pipeline = _make_pipeline(store, stub, max_shown=3)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    shown = _shown_texts(store)
    assert len(shown) == 4, f"expected the first plus three fresh ones, got {shown}"
    assert OFFENDER not in shown[1:]
    store.close()


def test_two_near_identical_candidates_in_the_SAME_cycle_cannot_both_be_shown(tmp_path):
    """Cross-cycle repetition is what was measured, but nothing made
    within-cycle duplication impossible, and an operator would experience it
    identically."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[_gen([OFFENDER, VARIANT]), _crit(2)])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    assert _shown_texts(store) == [OFFENDER]
    store.close()


def test_what_was_already_shown_is_fed_into_the_next_generate_prompt(tmp_path):
    """The guard is the safety net; this is the actual fix. Without it the
    model keeps generating the repeat and the guard silently eats it forever,
    which costs a full generate call every cycle to produce nothing."""
    store = _new_store(tmp_path)
    client = _RecordingClient(StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OTHER]), _crit(1),
    ]))
    pipeline = _make_pipeline(store, client)
    pipeline._process(_trigger(store))
    first_prompt = client.requests[0]["messages"][-1]["content"]
    assert OFFENDER not in first_prompt, "nothing had been shown yet on the first cycle"

    pipeline._process(_trigger(store))
    second_prompt = client.requests[2]["messages"][-1]["content"]
    assert OFFENDER in second_prompt, (
        "the second generate call was not told what the operator had already seen"
    )
    store.close()


def test_the_already_proposed_list_never_lands_in_a_cached_system_block(tmp_path):
    """It changes almost every cycle. Caching it would invalidate that
    breakpoint on every call and quietly turn the prompt cache off -- costing
    far more than the repetition it fixes."""
    store = _new_store(tmp_path)
    client = _RecordingClient(StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OTHER]), _crit(1),
    ]))
    pipeline = _make_pipeline(store, client)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    system_blocks = client.requests[2]["system"]
    for block in system_blocks:
        assert OFFENDER not in block["text"], (
            "the already-proposed list reached a system block; in a cached one it "
            "would invalidate the breakpoint every cycle"
        )
    # Every block but the volatile transcript tail is a cache breakpoint.
    assert all("cache_control" in b for b in system_blocks[:-1])
    assert "cache_control" not in system_blocks[-1]
    store.close()


def test_the_suppression_event_carries_counts_only_and_never_the_question_text(tmp_path):
    """D-08's two-tier rule (T-04-27): events carry counts, never text."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OFFENDER]), _crit(1),
    ])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    event = _events(store, EVENT_SUGGESTION_REPEATED)[0]
    assert event["detail"] == {"suppressed": 1, "shown": 0}
    assert OFFENDER not in str(event)
    store.close()


def test_suppression_is_not_counted_as_a_coach_failure(tmp_path):
    """A guard doing its job must never drive the coach-failure banner -- a
    healthy coach would start reporting itself as broken."""
    store = _new_store(tmp_path)
    stub = StubAnthropicClient(responses=[
        _gen([OFFENDER]), _crit(1),
        _gen([OFFENDER]), _crit(1),
    ])
    pipeline = _make_pipeline(store, stub)
    pipeline._process(_trigger(store))
    pipeline._process(_trigger(store))
    assert not pipeline.health.is_unhealthy(threshold=1)
    store.close()
