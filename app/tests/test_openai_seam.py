"""The OpenAI-compatible seam, driven through the REAL coach entry points.

These do not test the adapter in isolation. They call `run_generate` and
`run_critique` -- the same functions `pipeline.py` calls -- with the adapter
substituted for the Anthropic client, so what is pinned is that a candidate
model reached through this seam produces the same typed results the coach
already knows how to handle. That is the whole point of the seam; asserting
on the translated dict alone would prove intent, not behaviour.
"""

from __future__ import annotations

import json

import pytest

from app.coach.critic import BAND_CLEAN, CRITIC_TOOL_NAME, CritiqueFailure, run_critique
from app.coach.generate import (
    GENERATE_TOOL_NAME,
    Candidate,
    GenerateFailure,
    run_generate,
)
from app.coach.openai_seam import (
    OpenAICompatibleClient,
    _system_content,
    _translate_response,
)
from app.coach.reasoning_client import prompt_caching_enabled
from app.session.store import SegmentRecord

MOVE_ADVANCE = "advance"
MOVE_DEEPEN = "deepen"

# One candidate, not zero: `run_generate` treats an empty candidate list as a
# parse failure, which is correct behaviour and not what these fixtures are
# exercising.
_ONE_CANDIDATE = {"candidates": [{"text": "a question", "move": MOVE_ADVANCE}]}


def _segment(seq: int = 0, text: str = "hi") -> SegmentRecord:
    return SegmentRecord(
        kind="segment", seq=seq, speaker="interviewee", text=text, audio_start_s=0.0,
        audio_end_s=1.0, ready_wall_s=1.1, latency_s=0.1, detected_language="en",
        engine="fake", written_at="2026-08-14T00:00:00.000+00:00",
    )


def _completion(*, name: str, arguments, finish_reason: str = "tool_calls") -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
                },
            }
        ]
    }


def _client(
    payload, *, capture: "list[dict] | None" = None, prompt_caching: bool = True,
) -> OpenAICompatibleClient:
    def transport(body, timeout_s):
        if capture is not None:
            capture.append({"body": body, "timeout_s": timeout_s})
        return payload(body) if callable(payload) else payload

    return OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        transport=transport,
        prompt_caching=prompt_caching,
    )


# --- the happy path, through the real coach entry points -------------------------------


def test_a_generate_call_through_the_seam_produces_real_candidates():
    payload = _completion(
        name=GENERATE_TOOL_NAME,
        arguments={"candidates": [
            {"text": "Walk me through the last time that happened.", "move": MOVE_ADVANCE},
            {"text": "What did you do next?", "move": MOVE_DEEPEN},
        ]},
    )
    candidates = run_generate(
        _client(payload), brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
        detected_language="en", hypotheses=(), recently_shown=(), model="local-model",
        max_tokens=1000, timeout_s=5.0,
    )
    assert candidates == [
        Candidate(index=0, text="Walk me through the last time that happened.", move=MOVE_ADVANCE),
        Candidate(index=1, text="What did you do next?", move=MOVE_DEEPEN),
    ]


def test_a_critique_call_through_the_seam_produces_real_judgments():
    payload = _completion(
        name=CRITIC_TOOL_NAME,
        arguments={"judgments": [{
            "index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
            "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
            "band": BAND_CLEAN, "rationale": "fine",
        }]},
    )
    judgments = run_critique(
        _client(payload), candidates=[Candidate(index=0, text="q", move=MOVE_ADVANCE)],
        hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="local-model", max_tokens=500, timeout_s=5.0,
    )
    assert len(judgments) == 1 and judgments[0].band == BAND_CLEAN


# --- the translation that decides whether a candidate model is usable -------------------


def test_the_forced_tool_choice_survives_translation():
    """The coach forces exactly one named tool. A provider that received
    `auto` instead would be free to answer in prose, and every cycle would
    fail -- so this is gate one of choosing a model."""
    capture: "list[dict]" = []
    run_generate(
        _client(_completion(name=GENERATE_TOOL_NAME, arguments=_ONE_CANDIDATE), capture=capture),
        brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
        detected_language="en", hypotheses=(), recently_shown=(), model="m",
        max_tokens=1000, timeout_s=5.0,
    )
    body = capture[0]["body"]
    assert body["tool_choice"] == {"type": "function", "function": {"name": GENERATE_TOOL_NAME}}
    assert body["tools"][0]["function"]["name"] == GENERATE_TOOL_NAME
    # The strict schema must pass through unaltered, or the model is being
    # asked for a different shape than the parser validates.
    assert "properties" in body["tools"][0]["function"]["parameters"]


def test_the_system_blocks_reach_the_model_in_order_carrying_their_cache_markers():
    """This test used to assert the OPPOSITE -- that the markers were dropped
    -- and it passed, because the seam flattened the block list into one
    string and the markers went with it. It was pinning an accident as a
    contract. What matters is that the rubric still arrives, in order, and
    that a caller which asked for a cache breakpoint gets one sent."""
    capture: "list[dict]" = []
    run_generate(
        _client(_completion(name=GENERATE_TOOL_NAME, arguments=_ONE_CANDIDATE), capture=capture),
        brief=(), survey=(), asked={}, transcript=[], trigger=_segment(seq=0, text="the trigger"),
        detected_language="en", hypotheses=(), recently_shown=(), model="m",
        max_tokens=1000, timeout_s=5.0,
    )
    body = capture[0]["body"]
    messages = body["messages"]
    assert messages[0]["role"] == "system"
    parts = messages[0]["content"]
    assert isinstance(parts, list), "the blocks were flattened, so the markers are gone"
    assert all(part["type"] == "text" for part in parts)
    # Order is load-bearing: a prefix cache keys on the leading bytes, so a
    # reordering silently costs the cache without failing anything.
    assert "Mom Test" in parts[0]["text"], "the rubric never reached the model"
    assert any("cache_control" in part for part in parts), "no breakpoint was requested"
    # The LAST block is the per-cycle tail and must never be a breakpoint --
    # caching it would rebuild the prefix on every single cycle.
    assert "cache_control" not in parts[-1]
    assert messages[-1]["role"] == "user" and "the trigger" in messages[-1]["content"]


def test_prompt_caching_off_restores_the_flattened_string_for_a_strict_provider():
    capture: "list[dict]" = []
    run_generate(
        _client(
            _completion(name=GENERATE_TOOL_NAME, arguments=_ONE_CANDIDATE),
            capture=capture, prompt_caching=False,
        ),
        brief=(), survey=(), asked={}, transcript=[], trigger=_segment(seq=0, text="the trigger"),
        detected_language="en", hypotheses=(), recently_shown=(), model="m",
        max_tokens=1000, timeout_s=5.0,
    )
    body = capture[0]["body"]
    assert isinstance(body["messages"][0]["content"], str)
    assert "Mom Test" in body["messages"][0]["content"]
    assert "cache_control" not in json.dumps(body)


def test_a_system_that_asked_for_no_cache_is_sent_as_a_string_even_with_caching_on():
    """The array form is only sent on behalf of a caller that wanted a
    breakpoint. Everyone else keeps the byte-identical old request, so a
    strict provider is never handed an array it did not cause."""
    assert _system_content("already a string", prompt_caching=True) == "already a string"
    assert _system_content(None, prompt_caching=True) == ""
    assert _system_content(
        [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], prompt_caching=True
    ) == "a\n\nb"


def test_the_explicit_timeout_reaches_the_transport():
    capture: "list[dict]" = []
    run_generate(
        _client(_completion(name=GENERATE_TOOL_NAME, arguments=_ONE_CANDIDATE), capture=capture),
        brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
        detected_language="en", hypotheses=(), recently_shown=(), model="m",
        max_tokens=1000, timeout_s=3.5,
    )
    assert capture[0]["timeout_s"] == 3.5


# --- the failures a cheap model will actually produce ------------------------------------


def test_a_truncated_tool_call_is_refused_rather_than_parsed():
    """OpenAI reports truncation as `length`; the coach refuses to trust
    `max_tokens`. If the mapping were wrong, a half-written candidate list
    would be parsed as though it were complete -- silently dropping whatever
    the model had not finished saying."""
    payload = _completion(
        name=GENERATE_TOOL_NAME,
        arguments={"candidates": [{"text": "a question", "move": MOVE_ADVANCE}]},
        finish_reason="length",
    )
    with pytest.raises(GenerateFailure) as excinfo:
        run_generate(
            _client(payload), brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
            detected_language="en", hypotheses=(), recently_shown=(), model="m",
            max_tokens=10, timeout_s=5.0,
        )
    assert excinfo.value.stop_reason == "max_tokens"


def test_a_content_filter_stop_is_refused_too():
    payload = _completion(
        name=CRITIC_TOOL_NAME, arguments={"judgments": []}, finish_reason="content_filter"
    )
    with pytest.raises(CritiqueFailure) as excinfo:
        run_critique(
            _client(payload), candidates=[Candidate(index=0, text="q", move=MOVE_ADVANCE)],
            hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=500, timeout_s=5.0,
        )
    assert excinfo.value.stop_reason == "refusal"


def test_malformed_tool_arguments_fail_loudly_instead_of_becoming_zero_candidates():
    """The failure mode that matters most for a small local model. Emitting a
    tool_use block with an empty input would look identical to a model that
    simply had nothing to suggest, and the coach would report itself healthy
    while producing nothing all interview."""
    payload = _completion(name=GENERATE_TOOL_NAME, arguments="{not valid json at all")
    with pytest.raises(GenerateFailure):
        run_generate(
            _client(payload), brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
            detected_language="en", hypotheses=(), recently_shown=(), model="m",
            max_tokens=1000, timeout_s=5.0,
        )


def test_prose_instead_of_a_tool_call_fails_loudly():
    """Small models frequently answer in text despite a forced tool choice."""
    payload = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "Sure! Here are some questions you could ask...",
                        "tool_calls": []},
        }]
    }
    with pytest.raises(GenerateFailure):
        run_generate(
            _client(payload), brief=(), survey=(), asked={}, transcript=[], trigger=_segment(),
            detected_language="en", hypotheses=(), recently_shown=(), model="m",
            max_tokens=1000, timeout_s=5.0,
        )


def test_an_empty_choices_array_does_not_raise_inside_the_adapter():
    """Degrade at the coach's own typed boundary, not with an IndexError from
    inside the translation layer."""
    adapted = _translate_response({"choices": []})
    assert adapted.content == [] and adapted.stop_reason is None


@pytest.mark.parametrize(
    "finish,expected",
    [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use"),
     ("content_filter", "refusal")],
)
def test_every_finish_reason_maps_to_the_stop_reason_the_coach_reads(finish, expected):
    adapted = _translate_response({"choices": [{"finish_reason": finish, "message": {}}]})
    assert adapted.stop_reason == expected


def test_an_unknown_finish_reason_is_passed_through_not_silently_called_end_turn():
    """A provider inventing its own reason must not be laundered into 'this
    completed normally'."""
    adapted = _translate_response({"choices": [{"finish_reason": "wibble", "message": {}}]})
    assert adapted.stop_reason == "wibble"


# --- did the cache actually hit? ---------------------------------------------------------


def test_the_cache_hit_count_is_carried_off_the_response():
    """Emitting `cache_control` proves a cache was REQUESTED. Only this number
    says one was read, which is why the seam surfaces it at all."""
    adapted = _translate_response(
        {
            "choices": [{"finish_reason": "stop", "message": {}}],
            "usage": {"prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": 4096}},
        }
    )
    assert adapted.cached_tokens == 4096


def test_a_reported_zero_is_a_measured_miss_and_survives_as_zero():
    adapted = _translate_response(
        {
            "choices": [{"finish_reason": "stop", "message": {}}],
            "usage": {"prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": 0}},
        }
    )
    assert adapted.cached_tokens == 0


@pytest.mark.parametrize(
    "usage",
    [None, {}, {"prompt_tokens": 5000}, {"prompt_tokens_details": {}},
     {"prompt_tokens_details": {"cached_tokens": "4096"}},
     {"prompt_tokens_details": {"cached_tokens": True}},
     {"prompt_tokens_details": None}],
)
def test_a_provider_that_says_nothing_about_caching_reads_as_unknown_not_as_a_miss(usage):
    """None and 0 must not share a channel. "The provider did not report it"
    and "the cache was read for zero tokens" would otherwise both render as a
    miss, and the first would send someone hunting a cache bug that is really
    a provider that never answered the question."""
    payload = {"choices": [{"finish_reason": "stop", "message": {}}]}
    if usage is not None:
        payload["usage"] = usage
    assert _translate_response(payload).cached_tokens is None


def test_an_empty_choices_array_still_reports_the_cache_count():
    """A response the coach will refuse to parse still cost prefill, so the
    cost question is still answerable for it."""
    adapted = _translate_response(
        {"choices": [], "usage": {"prompt_tokens_details": {"cached_tokens": 1024}}}
    )
    assert adapted.cached_tokens == 1024


# --- the kill switch ---------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " Off "])
def test_an_explicit_opt_out_turns_the_markers_off(monkeypatch, value):
    monkeypatch.setenv("COACH_PROMPT_CACHING", value)
    assert prompt_caching_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "", "   ", "wibble", "00"])
def test_anything_that_is_not_an_explicit_opt_out_leaves_caching_on(monkeypatch, value):
    """A typo must not silently disable the cache. The bill for that is an
    ~8x prefill rise per cycle with nothing failing to point at it, so the
    default has to survive every value that is not a deliberate 'no'."""
    monkeypatch.setenv("COACH_PROMPT_CACHING", value)
    assert prompt_caching_enabled() is True


def test_caching_is_on_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("COACH_PROMPT_CACHING", raising=False)
    assert prompt_caching_enabled() is True


def test_a_cache_read_reaches_a_log_line(caplog):
    """The value was carried on the response and recorded NOWHERE, so a real
    session still could not answer "did caching work". That gap survived a live
    interview before it was noticed."""
    import logging
    with caplog.at_level(logging.INFO, logger="app.coach.openai_seam"):
        _client(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5000,
                          "prompt_tokens_details": {"cached_tokens": 4096}},
            }
        ).with_options(timeout=5).messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "x"}]
        )
    assert "cached_tokens=4096" in caplog.text
    assert "prompt_tokens=5000" in caplog.text


def test_a_provider_that_never_reports_caching_stays_silent(caplog):
    """A line reading `cached=None` on every call would train the reader to
    ignore the one place the answer appears."""
    import logging
    with caplog.at_level(logging.INFO, logger="app.coach.openai_seam"):
        _client(
            {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
        ).with_options(timeout=5).messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "x"}]
        )
    assert "prefix cache" not in caplog.text


def test_a_measured_miss_is_logged_rather_than_hidden(caplog):
    """Zero is a MEASUREMENT -- the cache was asked for and not read, which is
    exactly the thing worth seeing. Only `None` is silence."""
    import logging
    with caplog.at_level(logging.INFO, logger="app.coach.openai_seam"):
        _client(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 5000,
                          "prompt_tokens_details": {"cached_tokens": 0}},
            }
        ).with_options(timeout=5).messages.create(
            model="m", max_tokens=10, messages=[{"role": "user", "content": "x"}]
        )
    assert "cached_tokens=0" in caplog.text


def test_the_cache_line_never_carries_prompt_or_candidate_text(caplog):
    """This module sits under app/coach/, where gate 12 forbids putting
    transcript-derived text into a log line."""
    import logging
    with caplog.at_level(logging.INFO, logger="app.coach.openai_seam"):
        _client(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "a secret answer"}}],
                "usage": {"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 4}},
            }
        ).with_options(timeout=5).messages.create(
            model="m", max_tokens=10,
            system=[{"type": "text", "text": "a secret system block"}],
            messages=[{"role": "user", "content": "a secret utterance"}],
        )
    for secret in ("secret answer", "secret system block", "secret utterance"):
        assert secret not in caplog.text
