"""Task 4 contract tests: every engine satisfies the Engine protocol and is
constructible without network access or credentials; only `transcribe()`
touches the network. Cloud adapters' message handlers are proven against
recorded/hand-authored response payloads with NO live call. Missing/empty API
keys raise typed configuration errors that never leak key material. The
frozen-comparison guard and the CUDA fallback ladder are exercised directly.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from benchmark import config
from benchmark.engines import assemblyai, local_whisper, soniox
from benchmark.engines.base import Engine

_SONIOX_KEY_PRESENT = bool(os.environ.get("SONIOX_API_KEY"))
_ASSEMBLYAI_KEY_PRESENT = bool(os.environ.get("ASSEMBLYAI_API_KEY"))


# --- Every engine satisfies the Engine protocol, constructible with no network -------


def test_soniox_engine_constructible_without_network_or_credentials(monkeypatch):
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    engine: Engine = soniox.SonioxEngine()
    assert engine.name == "soniox"
    assert hasattr(engine, "transcribe")


def test_assemblyai_engine_constructible_without_network_or_credentials(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    engine: Engine = assemblyai.AssemblyAIEngine()
    assert engine.name == "assemblyai"
    assert hasattr(engine, "transcribe")


def test_local_finetune_and_base_engines_constructible_without_network():
    finetune: Engine = local_whisper.make_finetune_engine(device="cpu")
    base: Engine = local_whisper.make_base_engine(device="cpu")
    assert finetune.name == "local-finetune"
    assert base.name == "local-base"


# --- Soniox: replay a recorded response payload through the message handler ----------


def test_soniox_process_response_extracts_finalized_tokens_offline():
    recorded_response = {
        "tokens": [
            {"text": "hello ", "start_ms": 0, "end_ms": 500, "is_final": True, "language": "en"},
            {"text": "world", "start_ms": 500, "end_ms": 1500, "is_final": True, "language": "en"},
            {
                "text": " there",
                "start_ms": 1500,
                "end_ms": 2200,
                "is_final": False,  # interim, must be dropped
                "language": "en",
            },
        ],
        "finished": False,
    }
    finalized, finished, error = soniox.process_response(recorded_response)
    assert error is None
    assert finished is False
    assert [t["text"] for t in finalized] == ["hello ", "world"]

    # fallback_audio_end_s is deliberately a decoy: real tokens carry end_ms, so
    # the fallback must never be used here. See the "loud regression" test below
    # for the case that actually exercises the fallback path.
    segment = soniox.tokens_to_segment(finalized, fallback_audio_end_s=999.0, ready_wall_s=1.7)
    assert segment is not None
    assert segment.text == "hello  world".strip() or segment.text == "hello world"
    assert segment.audio_end_s == 1.5  # from the tokens' own end_ms (max = 1500ms), not the fallback
    assert segment.ready_wall_s == 1.7
    assert segment.detected_language == "en"


# --- Soniox: audio_end_s comes from the tokens' own end_ms, never the push counter ----


def test_soniox_audio_end_s_comes_from_token_end_ms_not_fallback():
    """A silent regression to the push counter must be loud: fallback_audio_end_s
    is set to a wildly different value (999.0) than any token's end_ms, so if the
    implementation ever reads the fallback when it shouldn't, this test screams."""
    tokens = [
        {"text": "hi", "start_ms": 0, "end_ms": 800, "is_final": True, "language": "en"},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.audio_end_s == pytest.approx(0.8)


def test_soniox_endpoint_marker_never_appears_in_segment_text():
    """Verbatim real payload (recorded 2026-08-10): the <end> marker token was
    previously concatenated straight into segment text -- a real shipped bug
    (265 and 434 lines of the real call transcripts contain literal '<end>')."""
    recorded_response = {
        "tokens": [
            {
                "text": " כ",
                "start_ms": 23580,
                "end_ms": 23640,
                "confidence": 0.996,
                "is_final": True,
                "language": "he",
            },
            {
                "text": "ן.",
                "start_ms": 23640,
                "end_ms": 23640,
                "confidence": 0.999,
                "is_final": True,
                "language": "he",
            },
            {
                "text": "<end>",
                "start_ms": 0,
                "end_ms": 0,
                "confidence": 1,
                "is_final": True,
            },
        ],
        "finished": False,
    }
    finalized, _, _ = soniox.process_response(recorded_response)
    segment = soniox.tokens_to_segment(finalized, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is not None
    assert soniox.ENDPOINT_MARKER_TEXT not in segment.text
    assert "<end>" not in segment.text


def test_soniox_endpoint_marker_end_ms_zero_never_becomes_audio_end_s():
    """THE KEY TEST. A naive 'use the LAST token's end_ms' implementation would
    read the trailing <end> marker's end_ms: 0 and report audio_end_s == 0.0 --
    which is wrong but would still pass the 'comes from tokens, not fallback'
    test above, since 0.0 != 999.0 too. Only `max` over SPEECH tokens (with the
    marker excluded) gets this right."""
    recorded_response = {
        "tokens": [
            {
                "text": " כ",
                "start_ms": 23580,
                "end_ms": 23640,
                "confidence": 0.996,
                "is_final": True,
                "language": "he",
            },
            {
                "text": "ן.",
                "start_ms": 23640,
                "end_ms": 23640,
                "confidence": 0.999,
                "is_final": True,
                "language": "he",
            },
            {
                "text": "<end>",
                "start_ms": 0,
                "end_ms": 0,
                "confidence": 1,
                "is_final": True,
            },
        ],
        "finished": False,
    }
    finalized, _, _ = soniox.process_response(recorded_response)
    segment = soniox.tokens_to_segment(finalized, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.audio_end_s == pytest.approx(23.64)


def test_soniox_out_of_order_tokens_use_max_end_ms_not_last():
    """A response whose tokens are not in chronological order must still
    report the LATEST speech end, not whichever token happened to be last in
    the list -- `max`, not `tokens[-1]`."""
    tokens = [
        {"text": "second ", "start_ms": 1000, "end_ms": 5000, "is_final": True, "language": "en"},
        {"text": "first", "start_ms": 0, "end_ms": 500, "is_final": True, "language": "en"},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.audio_end_s == pytest.approx(5.0)


def test_soniox_no_usable_end_ms_falls_back_and_warns(caplog):
    """When no token carries a usable end_ms (never observed live, but the code
    must not crash a live interview over a metric), the fallback is used AND a
    warning is logged -- a session log with this warning invalidates that
    session's latency figures, so it must never be silent."""
    tokens = [{"text": "hello", "is_final": True, "language": "en"}]  # no end_ms at all
    with caplog.at_level("WARNING", logger=soniox.logger.name):
        segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=2.5, ready_wall_s=3.0)
    assert segment is not None
    assert segment.audio_end_s == 2.5
    assert any("end_ms" in record.message for record in caplog.records)


def test_soniox_endpoint_marker_end_ms_zero_not_picked_over_fallback_when_real_end_ms_missing():
    """A weaker version of the KEY TEST above that the verbatim-payload version
    cannot expose: there, `max` finds the real tokens' end_ms (23640ms) is
    already bigger than the marker's 0, so the marker being included or
    excluded from the `max` doesn't change the outcome for that particular
    payload -- 23.64 either way. This test isolates the marker: the only real
    speech token here has NO end_ms at all, so if `speech_tokens` ever failed
    to filter the marker, `token_audio_end_s` would compute `max([0]) == 0.0`
    (the marker's own end_ms) and silently mask what should be a warned
    fallback to `fallback_audio_end_s`."""
    tokens = [
        {"text": "hi", "is_final": True, "language": "en"},  # no end_ms
        {"text": "<end>", "start_ms": 0, "end_ms": 0, "is_final": True},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=42.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.audio_end_s == 42.0  # NOT 0.0 (the marker's end_ms)


def test_soniox_response_with_only_endpoint_marker_produces_no_segment():
    """A response carrying ONLY the endpoint marker (no real speech) must
    return None -- not an empty-text Segment. An empty-but-present segment
    would still get appended to hyp_parts and segments upstream."""
    tokens = [{"text": "<end>", "start_ms": 0, "end_ms": 0, "is_final": True}]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is None


def test_soniox_detected_language_unaffected_by_endpoint_marker():
    """The marker carries no `language` key at all. It must not shift, blank,
    or otherwise corrupt the detected language of the real speech tokens."""
    tokens = [
        {"text": "hola", "start_ms": 0, "end_ms": 500, "is_final": True, "language": "es"},
        {"text": "<end>", "start_ms": 0, "end_ms": 0, "is_final": True},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=999.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.detected_language == "es"


def test_soniox_speech_tokens_strips_only_the_marker():
    """Direct unit test of the helper: exactly the marker token is removed,
    everything else (including a real token that happens to also carry no
    language) survives untouched."""
    tokens = [
        {"text": "real", "start_ms": 0, "end_ms": 100, "is_final": True, "language": "en"},
        {"text": "<end>", "start_ms": 0, "end_ms": 0, "is_final": True},
    ]
    result = soniox.speech_tokens(tokens)
    assert [t["text"] for t in result] == ["real"]


def test_soniox_token_audio_end_s_returns_none_when_no_usable_end_ms():
    """Direct unit test of the helper in isolation, independent of the
    fallback/warning behaviour tested via tokens_to_segment above."""
    assert soniox.token_audio_end_s([{"text": "hi", "is_final": True}]) is None
    assert soniox.token_audio_end_s([]) is None


def test_soniox_token_audio_end_s_ignores_boolean_end_ms():
    """`isinstance(x, (int, float))` is True for Python bools -- a token that
    somehow carried `end_ms: True/False` must not be silently treated as a
    numeric timestamp of 1000ms/0ms."""
    tokens = [{"text": "weird", "end_ms": True, "is_final": True}]
    assert soniox.token_audio_end_s(tokens) is None


def test_soniox_process_response_recognizes_finished_and_error_frames():
    finished_response = {"tokens": [], "finished": True}
    _, finished, error = soniox.process_response(finished_response)
    assert finished is True
    assert error is None

    error_response = {"error_code": 401, "error_message": "Unauthorized"}
    finalized, finished, error = soniox.process_response(error_response)
    assert finalized == []
    assert error == (401, "Unauthorized")


def test_soniox_missing_api_key_raises_typed_error_without_leaking_key(monkeypatch):
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    engine = soniox.SonioxEngine()
    with pytest.raises(soniox.SonioxConfigError) as exc_info:
        engine._get_api_key()
    assert "SONIOX_API_KEY" in str(exc_info.value)


def test_soniox_empty_api_key_raises_typed_error_without_leaking_key(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "")
    engine = soniox.SonioxEngine()
    with pytest.raises(soniox.SonioxConfigError):
        engine._get_api_key()


def test_soniox_start_message_never_logs_key_in_a_readable_error_but_does_include_it_in_payload():
    # The start message payload legitimately carries the key (it's the auth
    # mechanism) -- what must never happen is the key leaking into an
    # exception message. Confirm the payload shape is correct offline.
    message = soniox.build_start_message("sk-secret-value", soniox.SonioxConfig())
    import json

    payload = json.loads(message)
    assert payload["api_key"] == "sk-secret-value"
    assert "language_hints" not in payload  # no pre-declared language, deliberately


# --- Soniox: the live session's wall clock is anchored to the FIRST AUDIO SENT -------
#
# Soniox's token `end_ms` counts from the first audio the server received, so
# `ready_wall_s` must share that origin. Anchoring it to session-open instead
# bills the connect handshake -- and any gap before the capture thread's first
# frame -- as model latency. Nothing else in this suite exercises
# `_audio_clock_origin`: every other test passes `ready_wall_s` in as a
# literal, so reverting the anchor left the whole suite green (verified by
# mutation 2026-08-10). These two tests are that gap's only guard.


class _RecordingWebSocket:
    """Minimal stand-in for the `websockets` sync client: records what was
    sent and never touches a network."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, payload) -> None:
        self.sent.append(payload)


def _session_opened_seconds_ago(seconds: float) -> soniox._SonioxStreamingSession:
    """A session whose `open_session()` returned `seconds` ago -- i.e. a slow
    handshake, or simply a session opened before the capture thread produced
    its first frame."""
    return soniox._SonioxStreamingSession(
        ws=_RecordingWebSocket(),
        config=soniox.SonioxConfig(),
        session_start=time.monotonic() - seconds,
    )


def test_soniox_ready_wall_s_is_measured_from_first_audio_not_from_session_open():
    session = _session_opened_seconds_ago(10.0)
    # The handshake and the wait for frame 0 are NOT latency, so they must not
    # appear in the measurement.
    session.send_audio(b"\x00\x00" * 100)

    segment = session._handle_message(
        json.dumps({"tokens": [{"text": "hi", "start_ms": 0, "end_ms": 0, "is_final": True}]})
    )[0]

    # ~0s since the first frame, NOT the ~10s since the session opened. The
    # bound is deliberately far below 10.0 so this cannot pass by luck.
    assert segment.ready_wall_s < 1.0


def test_soniox_audio_clock_origin_is_set_once_and_never_moves():
    session = _session_opened_seconds_ago(0.0)
    session.send_audio(b"\x00\x00")
    first_origin = session._audio_clock_origin
    assert first_origin is not None

    time.sleep(0.02)
    session.send_audio(b"\x00\x00")

    # Re-anchoring on every frame would make `ready_wall_s` measure "time since
    # the most recent frame", which is near-zero forever -- a metric that always
    # looks excellent. That is the same shape of failure as the push-counter bug.
    assert session._audio_clock_origin == first_origin


def test_soniox_empty_audio_frame_does_not_start_the_clock():
    session = _session_opened_seconds_ago(0.0)
    session.send_audio(b"")
    assert session._audio_clock_origin is None


# --- AssemblyAI: replay a recorded event payload through the message handler ---------


def test_assemblyai_turn_event_to_segment_offline_via_sdk_parser():
    from assemblyai.streaming.v3.client import StreamingClient

    recorded_payload = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": True,
        "end_of_turn": True,
        "transcript": "hello world",
        "end_of_turn_confidence": 0.95,
        "words": [
            {"start": 0, "end": 500, "confidence": 0.9, "text": "hello", "word_is_final": True},
            {"start": 500, "end": 1000, "confidence": 0.9, "text": "world", "word_is_final": True},
        ],
        "language_code": "en",
    }

    event = StreamingClient._parse_message(recorded_payload)
    assert event is not None

    session_start = time.perf_counter()
    now = session_start + 0.3
    segment = assemblyai.turn_event_to_segment(event, session_start, now)

    assert segment is not None
    assert segment.text == "hello world"
    assert segment.audio_end_s == pytest.approx(1.0)  # last word ends at 1000ms
    assert segment.ready_wall_s == pytest.approx(0.3)
    assert segment.detected_language == "en"


def test_assemblyai_non_final_turn_produces_no_segment():
    from assemblyai.streaming.v3.client import StreamingClient

    recorded_payload = {
        "type": "Turn",
        "turn_order": 1,
        "turn_is_formatted": False,
        "end_of_turn": False,
        "transcript": "hello wor",
        "end_of_turn_confidence": 0.2,
        "words": [],
        "language_code": None,
    }
    event = StreamingClient._parse_message(recorded_payload)
    assert assemblyai.turn_event_to_segment(event, 0.0, 0.1) is None


def test_assemblyai_missing_api_key_raises_typed_error_without_leaking_key(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    engine = assemblyai.AssemblyAIEngine()
    with pytest.raises(assemblyai.AssemblyAIConfigError) as exc_info:
        engine._get_api_key()
    assert "ASSEMBLYAI_API_KEY" in str(exc_info.value)


def test_assemblyai_empty_api_key_raises_typed_error_without_leaking_key(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    engine = assemblyai.AssemblyAIEngine()
    with pytest.raises(assemblyai.AssemblyAIConfigError):
        engine._get_api_key()


# --- Frozen local-pair comparison guard ------------------------------------------------


def test_frozen_comparison_passes_when_only_model_id_differs():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", device="cpu", compute_type="int8")
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", device="cpu", compute_type="int8")
    local_whisper.assert_frozen_comparison(a, b)  # must not raise


def test_frozen_comparison_raises_on_device_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", device="cpu")
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", device="cuda")
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_compute_type_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", compute_type="int8")
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", compute_type="int8_float16")
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_beam_size_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", beam_size=1)
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", beam_size=5)
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_temperature_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", temperature=0.0)
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", temperature=0.5)
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_chunk_length_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", chunk_max_s=3.0)
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", chunk_max_s=5.0)
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_vad_settings_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", vad_options={})
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", vad_options={"threshold": 0.7})
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_condition_on_previous_text_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", condition_on_previous_text=False)
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", condition_on_previous_text=True)
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_frozen_comparison_raises_on_base_model_architecture_mismatch():
    a = local_whisper.LocalWhisperConfig(model_id="repo-a", base_model_architecture="large-v3-turbo")
    b = local_whisper.LocalWhisperConfig(model_id="repo-b", base_model_architecture="large-v3")
    with pytest.raises(local_whisper.FrozenComparisonViolation):
        local_whisper.assert_frozen_comparison(a, b)


def test_finetune_and_base_factories_produce_a_valid_frozen_pair():
    finetune = local_whisper.make_finetune_engine(device="cpu")
    base = local_whisper.make_base_engine(device="cpu")
    local_whisper.assert_frozen_comparison(finetune._config, base._config)  # must not raise
    assert finetune._config.model_id != base._config.model_id
    assert finetune._config.base_model_architecture == base._config.base_model_architecture


# --- CUDA fallback ladder ----------------------------------------------------------------


def test_cuda_ladder_resolves_to_cpu_floor_without_raising_when_no_gpu():
    rung = config.resolve_cuda_device(gpu_available=False)
    assert rung["device"] == "cpu"
    assert rung["compute_type"] == "int8"
    assert "reason" in rung and rung["reason"]


def test_cuda_ladder_resolves_to_top_rung_when_gpu_available():
    rung = config.resolve_cuda_device(gpu_available=True)
    assert rung["device"] == "cuda"
    assert "reason" in rung and rung["reason"]


def test_cuda_ladder_auto_detects_on_this_machine_without_raising():
    # No override -- exercises the real detection path. This dev machine has a
    # driver that predates the CUDA ctranslate2 wants, so this must resolve to
    # the CPU floor rather than raise (matches the plan's non-blocking claim).
    rung = config.resolve_cuda_device()
    assert rung["device"] in ("cpu", "cuda")
    assert "reason" in rung and rung["reason"]


# --- Live smoke tests: skip cleanly when credentials are absent (this plan's -----------
# --- constraint is that it stays runnable with no credentials and no GPU) --------------


@pytest.mark.live
@pytest.mark.skipif(not _SONIOX_KEY_PRESENT, reason="SONIOX_API_KEY not set")
def test_soniox_live_smoke():
    from benchmark.fixtures import make_synthetic

    wav_path, _ = make_synthetic.generate()
    engine = soniox.SonioxEngine()
    result = engine.transcribe(str(wav_path))
    assert result.hypothesis_text


@pytest.mark.live
@pytest.mark.skipif(not _ASSEMBLYAI_KEY_PRESENT, reason="ASSEMBLYAI_API_KEY not set")
def test_assemblyai_live_smoke():
    from benchmark.fixtures import make_synthetic

    wav_path, _ = make_synthetic.generate()
    engine = assemblyai.AssemblyAIEngine()
    result = engine.transcribe(str(wav_path))
    assert result.hypothesis_text


# --- which language a MIXED segment is reported as ------------------------------------------
#
# Added 2026-08-12 after a real Hebrew/English recording reported segments as
# ITALIAN. `detected_language` was `languages[0]` -- the FIRST token's tag
# labelling the whole segment -- so a two-character leading token outvoted every
# Hebrew character after it, by position alone.
#
# Code-switching is this project's whole subject, so "the segment has more than
# one language in it" is the NORMAL case here, not an edge one.


def test_a_leading_foreign_token_does_not_relabel_a_mixed_segment():
    """The real 2026-08-12 segment's shape, with synthetic text. Two Latin characters ahead of
    nineteen Hebrew ones used to make the whole thing Italian."""
    tokens = [
        {"text": "fa, ", "start_ms": 0, "end_ms": 200, "is_final": True, "language": "it"},
        {"text": "מעולה, אתה יודע, כן", "start_ms": 200, "end_ms": 1500,
         "is_final": True, "language": "he"},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=9.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.detected_language == "he"


def test_a_genuinely_foreign_segment_is_still_reported_as_foreign():
    """The control, and it matters: the OTHER Italian-tagged segment in that
    same recording really was Italian text (a short Italian phrase), hallucinated by
    the engine from bleed-degraded audio. Reporting it as Hebrew because Hebrew
    is expected would be a different lie, and would hide the transcription
    problem that produced it."""
    tokens = [
        {"text": "Io sono qui", "start_ms": 0, "end_ms": 900,
         "is_final": True, "language": "it"},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=9.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.detected_language == "it"


def test_the_language_is_weighted_by_characters_not_by_token_count():
    """Three short tokens must not outvote one long one. Token count is an
    artefact of how the engine chose to split the audio; characters are what
    was actually said -- the same reasoning `app/coach/language.py` gives for
    weighting its own window by character count."""
    tokens = [
        {"text": "a", "start_ms": 0, "end_ms": 100, "is_final": True, "language": "en"},
        {"text": "b", "start_ms": 100, "end_ms": 200, "is_final": True, "language": "en"},
        {"text": "c", "start_ms": 200, "end_ms": 300, "is_final": True, "language": "en"},
        {"text": "מילה ארוכה מאוד כאן", "start_ms": 300, "end_ms": 1500,
         "is_final": True, "language": "he"},
    ]
    segment = soniox.tokens_to_segment(tokens, fallback_audio_end_s=9.0, ready_wall_s=1.0)
    assert segment is not None
    assert segment.detected_language == "he"


def test_a_tie_is_broken_deterministically_by_first_appearance():
    """A genuinely balanced code-switch has no right answer, so it must at
    least have a STABLE one -- otherwise the same audio reports differently
    between runs and nobody can reproduce a complaint about it."""
    tokens = [
        {"text": "abcd", "start_ms": 0, "end_ms": 400, "is_final": True, "language": "en"},
        {"text": "אבגד", "start_ms": 400, "end_ms": 800, "is_final": True, "language": "he"},
    ]
    first = soniox.dominant_language(tokens)
    assert first == "en"
    assert soniox.dominant_language(list(reversed(tokens))) == "he", (
        "the tie must follow the ORDER SPOKEN, not a fixed language preference"
    )


def test_a_language_tag_on_an_empty_token_carries_no_weight():
    """The endpoint marker is stripped before this runs, but a zero-length
    token with a language tag is not evidence about what was said -- only about
    what the vendor guessed. Counting it would let a phantom outvote speech."""
    assert soniox.dominant_language([
        {"text": "", "language": "it"},
        {"text": "שלום עולם", "language": "he"},
    ]) == "he"


def test_no_language_tags_at_all_stays_none():
    """Unchanged behaviour, asserted so the rewrite cannot have invented a
    default. A missing tag means the engine did not say, and guessing here
    would put a fabricated language into the permanent record."""
    assert soniox.dominant_language([{"text": "hello"}, {"text": "world"}]) is None
    assert soniox.dominant_language([]) is None
