"""Task 3: the engine is a config value, and this is what makes shipping on an
unmeasured engine (STT-03; PROJECT.md's ⚠️ Revisit) defensible instead of a
gamble. Written FIRST (TDD RED) -- these fail until app/stt/replay.py exists
and the config-resolution/logging discipline described in phase_specification
Task 3 is finished.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import time
from pathlib import Path

import pytest

from app import config as config_mod
from app import main as app_main
from app.capture.stream import CaptureFrame
from app.session.store import SessionStore, create_session_dir
from app.stt.live import (
    EngineNotReadyError,
    StreamWorker,
    UnknownEngineError,
    available_engines,
    get_engine,
    preflight_engines,
)

MARKER = "ZQXJ7731-TRANSCRIPT-MARKER"
REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Registry: at least 3 names, each structurally satisfies StreamingEngine -------

# Engines whose `open_session()` needs no credentials or network -- safe to
# actually open/close a session against, offline, in this structural test.
# The registry may also hold credential/network-requiring engines (soniox);
# those are checked structurally (hasattr) only here and exercised via
# `open_session()` failure-mode tests further down and the live-marked tests
# in test_tracer_capture_to_disk.py.
_OFFLINE_SAFE_ENGINES = {"fake", "replay"}


def test_registry_exposes_at_least_three_engines_satisfying_the_protocol():
    names = available_engines()
    assert len(names) >= 3, names
    for name in names:
        engine = get_engine(name)
        assert hasattr(engine, "name")
        assert hasattr(engine, "open_session")
        if name not in _OFFLINE_SAFE_ENGINES:
            continue
        session = engine.open_session(sample_rate=16000, label="interviewee")
        assert hasattr(session, "send_audio")
        assert hasattr(session, "poll")
        assert hasattr(session, "close")
        session.close()


def test_every_registered_engine_constructs_with_no_credentials(monkeypatch):
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    for name in available_engines():
        engine = get_engine(name)  # must not raise -- construction never touches the network
        assert engine.name


# --- Config surface: resolution order, unknown name fails before device/dir --------


def test_env_var_overrides_default_engine(monkeypatch):
    monkeypatch.delenv("STT_ENGINE", raising=False)
    monkeypatch.setenv("STT_ENGINE", "fake")
    cfg = config_mod.load_config()
    assert cfg.engine_name == "fake"


def test_cli_override_beats_environment(monkeypatch):
    monkeypatch.setenv("STT_ENGINE", "fake")
    cfg = config_mod.load_config(engine_override="replay")
    assert cfg.engine_name == "replay"


def test_unknown_engine_name_fails_before_any_device_or_session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(tmp_path))
    args = app_main.build_arg_parser().parse_args(["start", "--engine", "not-a-real-engine"])
    code = app_main.run_start(args)
    assert code == 1
    assert list(tmp_path.iterdir()) == [], "an unknown engine name must not create a session directory"


# --- The credential pre-flight: a missing key fails before ANYTHING is opened -----
#
# The defect this pins: `SONIOX_API_KEY` was read inside `open_session`, on the
# STT worker thread, which runs long after `start` has opened both audio
# devices and written `sessions/<id>/`. A forgotten environment variable
# therefore produced a traceback AND a session directory with no `closed_at` --
# on disk, indistinguishable from an interview that crashed halfway through.


def test_missing_soniox_key_fails_before_any_device_or_session_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(tmp_path))
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)

    # The ordering assertion, and the reason this test is not just an exit-code
    # check. `_resolve_devices` ALSO returns 1 on a machine with no loopback
    # device, so "exit 1 and no session directory" is a false pass waiting to
    # happen on any CI runner without audio -- it would stay green if the
    # pre-flight were deleted outright. Blowing up in `construct_pyaudio` proves
    # the pre-flight ran FIRST: if it did its job, this is never called at all.
    def _must_not_be_reached():
        raise AssertionError(
            "construct_pyaudio was called -- the credential pre-flight must run "
            "before any audio device is opened"
        )

    monkeypatch.setattr(app_main, "construct_pyaudio", _must_not_be_reached)

    args = app_main.build_arg_parser().parse_args(
        ["start", "--engine", "soniox", "--no-serve", "--no-coach"]
    )
    code = app_main.run_start(args)

    assert code == 1
    # Names the variable, so this cannot pass on a device error that happens to
    # share the exit code.
    assert "SONIOX_API_KEY" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [], "a missing API key must not create a session directory"


def test_preflight_wraps_the_vendor_error_in_the_type_main_catches(monkeypatch):
    """`app/main.py` catches `EngineNotReadyError` and nothing else.

    If `preflight_engines` let the vendor's own `SonioxConfigError` escape, the
    `except` clause in `_async_main` would not match it and the operator would
    get back the exact traceback this whole guard exists to remove -- while
    every "it exits 1" test above still passed, because the process does exit
    non-zero on an uncaught exception. Hence the type, asserted directly."""
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    with pytest.raises(EngineNotReadyError) as exc_info:
        preflight_engines([get_engine("soniox")])
    assert "SONIOX_API_KEY" in str(exc_info.value)


# Engines that legitimately need NO credentials, pinned BY EQUALITY below.
# Deliberately a second constant rather than a reuse of `_OFFLINE_SAFE_ENGINES`:
# that one means "safe to open a session against in an offline test", which is a
# different question that happens to have the same answer today. Collapsing them
# would make a future engine that is credential-free but network-using silently
# change the meaning of both.
_CREDENTIAL_FREE_ENGINES = {"fake", "replay"}


def test_every_registered_engine_either_preflights_or_is_pinned_credential_free():
    """**The hole this closes, found by audit and proved by mutation.**

    `preflight_engines` skips any engine that does not define `preflight` --
    that optionality is what keeps `fake`/`replay` from needing two empty
    overrides. But nothing asserted the opt-IN, so an engine that requires a key
    and merely forgot the method was skipped in silence.

    The audit registered a fourth engine -- an AssemblyAI adapter needing
    `ASSEMBLYAI_API_KEY`, raising inside `open_session`, defining no
    `preflight` -- and the ENTIRE suite stayed green while that engine did
    exactly what this whole change exists to prevent: open both audio devices,
    write `sessions/<id>/`, then fail in the worker thread.

    Not hypothetical. AssemblyAI is this project's own named STT alternative and
    is already benchmarked beside Soniox in `benchmark/engines/`.

    Pinned by EQUALITY, the idiom gates 18/19/20 already use, because a
    subset check would pass for a new credentialed engine. Adding any engine
    now forces a decision here: implement `preflight`, or say in this set that
    it needs no credentials."""
    no_preflight = {
        name for name in available_engines() if not hasattr(get_engine(name), "preflight")
    }
    assert no_preflight == _CREDENTIAL_FREE_ENGINES, (
        "a registered engine defines no `preflight`, so `preflight_engines` skips it. If it needs "
        "credentials, give it one -- otherwise a missing key is discovered in the STT worker "
        "thread, after the devices are open and the session directory is written. If it genuinely "
        "needs none, add it to _CREDENTIAL_FREE_ENGINES and say why."
    )


def test_a_whitespace_only_key_is_refused(monkeypatch):
    """`"   "` is truthy, so `if not api_key` accepted it.

    Reproduced end to end by the audit: pre-flight passed and the run went on to
    open devices. `app/config.py`'s .env loader strips whitespace and matched
    quotes already, so this only ever reached an exported shell variable -- and
    `set SONIOX_API_KEY= ` is exactly the shape of a half-finished setup."""
    monkeypatch.setenv("SONIOX_API_KEY", "   ")
    with pytest.raises(EngineNotReadyError) as exc_info:
        preflight_engines([get_engine("soniox")])
    assert "SONIOX_API_KEY" in str(exc_info.value)


def test_preflight_is_silent_for_engines_that_declare_none(monkeypatch):
    """`fake` and `replay` need no credentials and define no `preflight`.

    Pinned because the obvious implementation -- making `preflight` a required
    protocol method -- would have forced two empty overrides and, if either were
    forgotten, turned every offline test run into an `AttributeError`."""
    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    for name in sorted(_OFFLINE_SAFE_ENGINES):
        assert not hasattr(get_engine(name), "preflight"), name
    preflight_engines([get_engine(name) for name in sorted(_OFFLINE_SAFE_ENGINES)])


def test_preflight_checks_a_repeated_engine_once(monkeypatch):
    """A two-sided run resolves one engine instance PER SPEAKER.

    Both read the same process environment, so checking twice can only ever
    produce two copies of one answer -- and if the check ever becomes a network
    call, doing it per speaker would double the cost of starting an interview.
    Pinned by counting."""
    calls: list[int] = []

    class _CountingEngine:
        name = "counting"

        def preflight(self) -> None:
            calls.append(1)

        def open_session(self, *, sample_rate: int, label: str = ""):  # pragma: no cover
            raise AssertionError("preflight must never open a session")

    preflight_engines([_CountingEngine(), _CountingEngine()])
    assert calls == [1]


def test_preflight_checks_every_DISTINCT_engine_even_after_the_first():
    """Deduplication must key on the engine TYPE, never on "have we checked
    anything yet".

    Found by mutation, and it is the sneakier of the two: rewriting the skip
    condition to `if preflight is None or bool(seen)` passes
    `test_preflight_checks_a_repeated_engine_once` perfectly, because that test
    uses two instances of ONE class -- the two are indistinguishable until the
    list holds two DIFFERENT classes. Under that mutation the second engine's
    credentials are never checked at all, and a two-engine run would fail at
    `open_session` inside a worker thread, which is the precise failure this
    whole guard was built to remove.

    Nothing today resolves two different engine types in one run. That is a
    property of the current call site, not of this function, and a test that
    only pins today's call site would not have caught the mutation either."""
    checked: "list[str]" = []

    def _engine(label: str):
        class _Engine:
            name = label

            def preflight(self) -> None:
                checked.append(label)

        return _Engine()

    preflight_engines([_engine("first"), _engine("second")])
    assert checked == ["first", "second"]


def test_preflight_does_not_open_a_session(monkeypatch):
    """It answers "is this configured", never "is the vendor up".

    A reachability check here would spend a connection on every single start
    and would turn a transient vendor blip into a refusal to record. The stub's
    `open_session` raises, so any implementation that reached for one fails."""
    monkeypatch.setenv("SONIOX_API_KEY", "not-a-real-key-and-never-used")
    engine = get_engine("soniox")
    opened: list[int] = []

    def _tracked_open(**_kwargs):
        opened.append(1)
        raise AssertionError("preflight opened a session")

    monkeypatch.setattr(engine, "open_session", _tracked_open)
    engine.preflight()
    assert opened == []


def test_unknown_engine_error_names_available_engines():
    with pytest.raises(UnknownEngineError) as exc_info:
        get_engine("not-a-real-engine")
    message = str(exc_info.value)
    for name in available_engines():
        assert name in message


# --- Swap proof: two offline engines through the SAME wiring, no hardware ----------


def _run_pipeline_with_engine(tmp_path: Path, engine_name: str) -> Path:
    """Drives one engine through the real StreamWorker + SessionStore wiring
    with a synthetic, hardware-free frame queue -- no PortAudio device, no
    network. Sends enough deterministic silence to cross both offline
    engines' default cadence (32000 bytes) so exactly one segment emits."""
    engine = get_engine(engine_name)
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": engine.name})

    frame_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    pcm16 = b"\x00" * 16000  # 0.5s of 16 kHz mono PCM16 per frame
    for i in range(3):  # 3 * 16000 bytes = 48000 > the 32000-byte default cadence
        frame_queue.put(CaptureFrame(seq=i, pcm16=pcm16, kind="synthesized", captured_monotonic=0.0))

    worker = StreamWorker(
        engine=engine, frame_queue=frame_queue, store=store, speaker="interviewee", sample_rate=16000
    )
    worker.start()
    deadline = time.monotonic() + 10.0
    while not frame_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)  # one more drain cycle after the queue empties
    worker.stop()
    store.close()
    return store.transcript_path


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_swapping_the_offline_engine_produces_schema_identical_sessions(tmp_path):
    fake_path = _run_pipeline_with_engine(tmp_path / "fake_run", "fake")
    replay_path = _run_pipeline_with_engine(tmp_path / "replay_run", "replay")

    fake_records = [r for r in _read_records(fake_path) if r["kind"] == "segment"]
    replay_records = [r for r in _read_records(replay_path) if r["kind"] == "segment"]
    assert fake_records, "fake engine produced no segments"
    assert replay_records, "replay engine produced no segments"

    fake_seg = fake_records[0]
    replay_seg = replay_records[0]

    # Same schema: identical key sets.
    assert set(fake_seg.keys()) == set(replay_seg.keys())

    # Same speaker, same ordering guarantees -- neither depends on the engine.
    assert fake_seg["speaker"] == replay_seg["speaker"] == "interviewee"
    assert fake_seg["seq"] == replay_seg["seq"] == 0

    # Differ ONLY in text, detected_language and engine.
    assert fake_seg["text"] != replay_seg["text"]
    assert fake_seg["engine"] != replay_seg["engine"]
    assert fake_seg["engine"] == "fake"
    assert replay_seg["engine"] == "replay"

    # Every other field is present with the right type/invariants on both --
    # exact numeric equality is not expected (each engine's recorded/synthetic
    # timing content genuinely differs), only structural equivalence is.
    for seg in (fake_seg, replay_seg):
        assert isinstance(seg["audio_start_s"], (int, float))
        assert isinstance(seg["audio_end_s"], (int, float))
        assert seg["audio_end_s"] > seg["audio_start_s"]
        assert isinstance(seg["ready_wall_s"], (int, float))
        assert isinstance(seg["latency_s"], (int, float))
        assert seg["latency_s"] >= 0.0
        assert isinstance(seg["written_at"], str)


# --- Replay engine reuses AssemblyAI's own parsing path, not Soniox's --------------


def test_replay_engine_uses_assemblyai_parsing_path_not_soniox():
    from app.stt.replay import RECORDED_TURN_PAYLOAD, ReplayStreamingEngine
    from benchmark.engines.assemblyai import turn_event_to_segment as assemblyai_handler

    engine = ReplayStreamingEngine(cadence_bytes=1)  # emit on the very first byte
    session = engine.open_session(sample_rate=16000)
    session.send_audio(b"\x00\x00")
    segments = session.poll()
    assert segments
    seg = segments[0]
    assert seg.text == RECORDED_TURN_PAYLOAD["transcript"]
    assert seg.detected_language == RECORDED_TURN_PAYLOAD["language_code"]

    # Cross-check against AssemblyAI's own handler directly -- proves replay
    # goes through AssemblyAI's parsing path, not Soniox's.
    from assemblyai.streaming.v3.client import StreamingClient

    event = StreamingClient._parse_message(RECORDED_TURN_PAYLOAD)
    direct = assemblyai_handler(event, 0.0, 0.0)
    assert direct is not None
    assert direct.text == seg.text


# --- No vendor identifier under app/capture/ or app/session/ (executable lines) ----


def test_no_vendor_identifier_in_capture_or_session_executable_code():
    # Same semantics as this task's grep-based verify command
    # (`^[^#]*\b(vendor)\b`): a vendor word appearing before any `#` on a
    # line is a violation; a word appearing only after a `#` (an inline
    # comment explaining the rule) is not.
    vendor_pattern = re.compile(r"^[^#]*\b(soniox|assemblyai|whisper|deepgram)\b", re.IGNORECASE)
    violations = []
    for package in ("app/capture", "app/session"):
        for py_file in sorted((REPO_ROOT / package).rglob("*.py")):
            for lineno, raw_line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), start=1):
                if vendor_pattern.search(raw_line):
                    violations.append(f"{py_file}:{lineno}: {raw_line}")
    assert not violations, "\n".join(violations)


# --- STT-02: the start frame declares no language of any kind ----------------------


def test_soniox_start_frame_declares_no_language_of_any_kind():
    from benchmark.engines.soniox import SonioxConfig, build_start_message

    message = build_start_message("dummy-key-for-structural-test-only", SonioxConfig())
    payload = json.loads(message)
    for forbidden_key in ("language_hints", "language", "language_code", "locale"):
        assert forbidden_key not in payload, f"start frame declares a language via {forbidden_key!r}"
    # Auto-detection itself stays ON -- only a PRE-DECLARED language is forbidden.
    assert payload.get("enable_language_identification") is True


# --- S7/T-02-05: never a leaked key, never leaked segment text ---------------------


def test_soniox_open_session_with_missing_key_raises_typed_error_without_leaking_key(monkeypatch):
    from benchmark.engines.soniox import SonioxConfigError

    monkeypatch.delenv("SONIOX_API_KEY", raising=False)
    engine = get_engine("soniox")  # construction succeeds -- no credentials touched yet
    with pytest.raises(SonioxConfigError) as exc_info:
        engine.open_session(sample_rate=16000)
    assert "SONIOX_API_KEY" in str(exc_info.value)


def test_soniox_open_session_with_empty_key_raises_typed_error(monkeypatch):
    from benchmark.engines.soniox import SonioxConfigError

    monkeypatch.setenv("SONIOX_API_KEY", "")
    engine = get_engine("soniox")
    with pytest.raises(SonioxConfigError):
        engine.open_session(sample_rate=16000)


def test_marker_text_never_appears_in_logs_but_does_appear_in_the_session_file(tmp_path, caplog):
    from app.stt.fake import FakeStreamingEngine

    caplog.set_level(logging.DEBUG)
    engine = FakeStreamingEngine(text=MARKER, cadence_bytes=1)  # emit on the first byte

    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": engine.name})

    frame_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    frame_queue.put(CaptureFrame(seq=0, pcm16=b"\x00\x00", kind="synthesized", captured_monotonic=0.0))

    worker = StreamWorker(
        engine=engine, frame_queue=frame_queue, store=store, speaker="interviewee", sample_rate=16000
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not frame_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    worker.stop()
    store.close()

    transcript_text = store.transcript_path.read_text(encoding="utf-8")
    assert MARKER in transcript_text, "the session file must retain the segment's real text"

    assert MARKER not in caplog.text, "the marker text leaked into log output"
    logger_names = {record.name for record in caplog.records}
    assert "app.stt.live" in logger_names, "the StreamWorker's own logger produced no DEBUG records"
    assert "app.session.store" in logger_names, "the SessionStore's own logger produced no DEBUG records"
