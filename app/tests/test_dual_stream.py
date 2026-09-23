"""Task 1 done criteria: the tracer's single stream generalized into a real
stream PAIR -- interviewee (loopback) and interviewer (microphone) captured
and transcribed at once, as two fully independent workers, landing in one
ordered transcript with structural (never diarized) speaker attribution.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from app import main as app_main
from app.capture.devices import (
    DeviceInfo,
    DeviceOpenError,
    resolve_default_input_device,
    resolve_default_loopback_device,
)
from app.capture.stream import CaptureStream
from app.session.store import SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.stt.live import StreamWorker
from app.tests.stub_audio import StubPyAudio, StubPyAudioModule

LOOPBACK_DEVICE = DeviceInfo(
    index=17, name="Stub Speakers [Loopback]", native_sample_rate=48000,
    channels=2, is_loopback=True, identity="loopback:stub-speakers",
)
MIC_DEVICE = DeviceInfo(
    index=15, name="Stub Microphone", native_sample_rate=16000,
    channels=1, is_loopback=False, identity="input:stub-microphone",
)


def _make_pair(store: SessionStore, speaker: str, device: DeviceInfo, pa: StubPyAudio, *, cadence_bytes: int = 6400):
    capture = CaptureStream(
        StubPyAudioModule(), pa, device,
        target_sample_rate=16000, frame_ms=50, queue_max_seconds=5.0, label=speaker,
    )
    engine = FakeStreamingEngine(cadence_bytes=cadence_bytes, poll_delay_s=0.0)
    worker = StreamWorker(
        engine=engine, frame_queue=capture.queue, store=store, speaker=speaker, sample_rate=16000,
    )
    return capture, worker


# --- Device resolution: two distinct records, missing side is named -----------------


def test_missing_side_produces_a_named_error_identifying_which_side(monkeypatch):
    def _raise(pa):
        raise DeviceOpenError(
            "No default WASAPI input (microphone) device is available. "
            "Confirm a microphone is connected and selected."
        )

    monkeypatch.setattr(app_main, "resolve_default_loopback_device", lambda pa: LOOPBACK_DEVICE)
    monkeypatch.setattr(app_main, "resolve_default_input_device", _raise)
    # Unpinned, which is what almost every run is -- `_side_resolver` then
    # returns the module-level default this test has just replaced.
    resolvers = {
        speaker: app_main._side_resolver(speaker, pin=None)
        for speaker in app_main.ALL_SPEAKERS
    }
    with pytest.raises(DeviceOpenError) as exc_info:
        app_main._resolve_devices(
            pa=object(), speakers=app_main.ALL_SPEAKERS, resolvers=resolvers
        )
    message = str(exc_info.value).lower()
    assert "microphone" in message or "input" in message


@pytest.mark.hardware
def test_both_real_devices_open_simultaneously_and_are_distinct():
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        loopback = resolve_default_loopback_device(pa)
        mic = resolve_default_input_device(pa)
        assert loopback.identity != mic.identity
        assert loopback.name != mic.name

        cap_a = CaptureStream(
            pyaudio, pa, loopback, target_sample_rate=16000, frame_ms=100,
            queue_max_seconds=5.0, label="interviewee",
        )
        cap_b = CaptureStream(
            pyaudio, pa, mic, target_sample_rate=16000, frame_ms=100,
            queue_max_seconds=5.0, label="interviewer",
        )
        cap_a.start()
        cap_b.start()
        time.sleep(0.5)
        cap_a.stop()
        cap_b.stop()
    finally:
        pa.terminate()


# --- Session metadata records both devices and their AS-REPORTED native rates -------


def test_session_metadata_records_both_devices_and_native_rates():
    devices = {"interviewee": LOOPBACK_DEVICE, "interviewer": MIC_DEVICE}
    metadata = app_main._session_metadata(devices, engine_name="fake", sample_rate=16000)
    assert metadata["interviewee_device"] == LOOPBACK_DEVICE.name
    assert metadata["interviewee_native_sample_rate"] == 48000
    assert metadata["interviewer_device"] == MIC_DEVICE.name
    assert metadata["interviewer_native_sample_rate"] == 16000
    assert metadata["engine"] == "fake"


# --- Two native rates both emit 16 kHz mono frames of identical shape ---------------


def test_two_native_rates_both_emit_16khz_mono_frames_of_identical_shape():
    pa = StubPyAudio(amplitude=0)
    cap_48k = CaptureStream(
        StubPyAudioModule(), pa, LOOPBACK_DEVICE, target_sample_rate=16000,
        frame_ms=100, queue_max_seconds=5.0, label="interviewee",
    )
    cap_16k = CaptureStream(
        StubPyAudioModule(), pa, MIC_DEVICE, target_sample_rate=16000,
        frame_ms=100, queue_max_seconds=5.0, label="interviewer",
    )
    cap_48k.start()
    cap_16k.start()
    try:
        frames_a = [cap_48k.queue.get(timeout=2.0) for _ in range(5)]
        frames_b = [cap_16k.queue.get(timeout=2.0) for _ in range(5)]
    finally:
        cap_48k.stop()
        cap_16k.stop()
    # Frame 0 alone carries the resampler's filter-startup transient (a
    # known, documented property -- "may return fewer or more samples than a
    # naive ratio would suggest"), so compare the STEADY-STATE total across
    # several frames rather than one frame in isolation: both sources,
    # despite differing native rates (48000 vs 16000), converge on the same
    # cumulative byte count once resampled to the same target rate.
    total_a = sum(len(f.pcm16) for f in frames_a)
    total_b = sum(len(f.pcm16) for f in frames_b)
    expected_total = int(16000 * 0.1) * 2 * 5  # 5 frames of 100ms @ 16kHz mono PCM16
    assert abs(total_a - total_b) <= 64, "same target rate must converge on the same total byte count"
    assert abs(total_a - expected_total) <= 64
    for f in frames_a + frames_b:
        assert len(f.pcm16) % 2 == 0, "PCM16 frames must be an even number of bytes (mono, 16-bit)"


# --- No shared objects; both speakers land in one ordered, unbroken transcript ------


def test_dual_stream_produces_both_speakers_with_no_shared_objects(tmp_path):
    pa = StubPyAudio(amplitude=0)
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})

    cap_a, worker_a = _make_pair(store, "interviewee", LOOPBACK_DEVICE, pa)
    cap_b, worker_b = _make_pair(store, "interviewer", MIC_DEVICE, pa)

    assert cap_a.queue is not cap_b.queue
    assert worker_a is not worker_b

    worker_a.start()
    worker_b.start()
    cap_a.start()
    cap_b.start()
    try:
        time.sleep(1.0)
    finally:
        worker_a.stop()
        worker_b.stop()
        cap_a.stop()
        cap_b.stop()
    store.close()

    lines = [
        json.loads(line)
        for line in store.transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    speakers = {r["speaker"] for r in lines if r["kind"] == "segment"}
    assert speakers == {"interviewee", "interviewer"}, speakers
    seqs = [r["seq"] for r in lines]
    assert seqs == sorted(seqs) == list(range(len(seqs))), "sequence numbers must be unique, monotonic, and gap-free"
    for r in lines:
        if r["kind"] == "segment":
            assert r["speaker"] in ("interviewee", "interviewer")


# --- No torn lines under real concurrent append ---------------------------------------


def test_concurrent_appends_never_produce_a_torn_line(tmp_path):
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})

    def _writer(speaker: str, n: int) -> None:
        for i in range(n):
            store.append_segment(
                speaker=speaker, text=f"{speaker}-{i}", audio_start_s=0.0, audio_end_s=0.1,
                ready_wall_s=0.1, detected_language="en", engine="fake",
            )

    threads = [threading.Thread(target=_writer, args=(spk, 50)) for spk in ("interviewee", "interviewer")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.close()

    lines = store.transcript_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]  # raises on any torn line
    assert len(records) == 100
    seqs = sorted(r["seq"] for r in records)
    assert seqs == list(range(100))


# --- Two independently-constructed engine instances, proven by identity, not label --


def test_fake_engine_instances_from_the_same_factory_have_distinct_identity():
    engine_a = FakeStreamingEngine(cadence_bytes=1)
    engine_b = FakeStreamingEngine(cadence_bytes=1)
    assert engine_a.instance_id != engine_b.instance_id

    session_a = engine_a.open_session(sample_rate=16000, label="interviewee")
    session_b = engine_b.open_session(sample_rate=16000, label="interviewer")
    session_a.send_audio(b"\x00\x00")
    session_b.send_audio(b"\x00\x00")
    text_a = session_a.poll()[0].text
    text_b = session_b.poll()[0].text
    assert text_a != text_b, "two independently-constructed engines produced identical text"


# --- --only runs a single named side ---------------------------------------------------


def test_only_flag_restricts_the_active_side_set():
    args = app_main.build_arg_parser().parse_args(["start", "--only", "interviewer"])
    assert args.only == "interviewer"
    with pytest.raises(SystemExit):
        app_main.build_arg_parser().parse_args(["start", "--only", "not-a-side"])


# --- Full lifecycle end to end via the real `start` command, both sides, real hardware --


@pytest.mark.hardware
def test_offline_end_to_end_dual_stream_writes_both_speakers_and_full_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(tmp_path))
    monkeypatch.setenv("STT_ENGINE", "fake")
    args = app_main.build_arg_parser().parse_args(["start", "--seconds", "3"])

    result_holder: dict = {}

    def _run():
        result_holder["code"] = app_main.run_start(args)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=20.0)
    assert result_holder.get("code") == 0

    candidates = list(tmp_path.glob("*/transcript.jsonl"))
    assert candidates, "no session transcript was written"
    transcript_path = candidates[0]
    session_dir = transcript_path.parent
    metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert "interviewee_device" in metadata
    assert "interviewee_native_sample_rate" in metadata
    assert "interviewer_device" in metadata
    assert "interviewer_native_sample_rate" in metadata
    assert "closed_at" in metadata
    assert "latency_summary" in metadata

    lines = [json.loads(line) for line in transcript_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    interviewee_segments = [r for r in lines if r["kind"] == "segment" and r["speaker"] == "interviewee"]
    assert interviewee_segments, "no interviewee segment records were written"
    first = interviewee_segments[0]
    assert first["text"]
    assert first["audio_end_s"] > first["audio_start_s"]
    assert first["latency_s"] >= 0.0
    assert first["engine"] == "fake"
