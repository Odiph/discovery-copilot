"""Task 2 done criteria: one named test per documented Windows audio trap
(phase_specification T1), plus the S13 standing gate. A test that merely
asserts the mitigation code exists is not a test of the trap -- every test
here drives real behaviour: a stub or a real device, never a mock of the
assertion itself.
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest

from app.capture.devices import DeviceInfo, DeviceOpenError, open_capture_stream, resolve_default_loopback_device
from app.capture.keepalive import KeepAliveStream
from app.capture.stream import CaptureFrame, CaptureStream
from app.capture.supervisor import StreamSupervisor, SupervisorConfig
from app.session.store import SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.stt.live import StreamWorker
from app.tests.stub_audio import StubPyAudio, StubPyAudioModule, StubStream
from app import config as config_mod

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_events(store: SessionStore) -> list[dict]:
    return [
        json.loads(line)
        for line in store.transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- Idle stream, deterministic ------------------------------------------------------


def test_idle_stream_deterministic_records_one_idle_and_one_recovery(tmp_path):
    stub_stream = StubStream(frames_per_buffer=160, channels=1, amplitude=0, available_frames=0)
    pa = StubPyAudio(stream_factory=lambda **kw: stub_stream)
    device = DeviceInfo(
        index=1, name="Stub Loopback", native_sample_rate=16000, channels=1,
        is_loopback=True, identity="loopback:stub",
    )
    capture = CaptureStream(
        StubPyAudioModule(), pa, device, target_sample_rate=16000, frame_ms=10,
        queue_max_seconds=5.0, label="interviewee",
    )
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    supervisor = StreamSupervisor(
        speaker="interviewee", capture=capture, store=store,
        config=SupervisorConfig(idle_threshold_s=0.05, poll_interval_s=0.02),
    )

    capture.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and capture.health.consecutive_synthesized_frames < 10:
            time.sleep(0.01)
        assert time.monotonic() < deadline, "capture frame counter never advanced -- pacer appears stuck"
        supervisor.poll_once()
        supervisor.poll_once()  # debounced: must NOT emit a second idle event

        stub_stream.available_frames = 999  # "recovery": device now reports real samples
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and capture.health.consecutive_synthesized_frames != 0:
            time.sleep(0.01)
        assert time.monotonic() < deadline, "capture never saw the recovered device as real"
        supervisor.poll_once()
    finally:
        capture.stop()
    store.close()

    events = _read_events(store)
    idle_events = [e for e in events if e.get("event") == "idle"]
    recovered_events = [e for e in events if e.get("event") == "recovered"]
    assert len(idle_events) == 1, idle_events
    assert len(recovered_events) == 1, recovered_events


# --- Idle stream, hardware -------------------------------------------------------------


@pytest.mark.hardware
def test_idle_stream_hardware_advances_and_records_idle_then_recovery(tmp_path):
    import pyaudiowpatch as pyaudio

    from app.tests.playback import play_tone

    pa = pyaudio.PyAudio()
    try:
        device = resolve_default_loopback_device(pa)
        capture = CaptureStream(
            pyaudio, pa, device, target_sample_rate=16000, frame_ms=100,
            queue_max_seconds=10.0, label="interviewee",
        )
        store = SessionStore(create_session_dir(tmp_path))
        store.write_metadata({"engine": "fake"})
        supervisor = StreamSupervisor(
            speaker="interviewee", capture=capture, store=store,
            config=SupervisorConfig(idle_threshold_s=1.0, poll_interval_s=0.2),
        )

        capture.start()
        try:
            # Hard deadline, standard library only: a blocked/stalled read
            # must FAIL this test, never hang it. A generous margin (8s,
            # >8 consecutive frames) absorbs a brief tail of residual OS
            # audio-session teardown from a prior test's render stream --
            # observed once during a full-suite run, never in isolation.
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline and capture.health.consecutive_synthesized_frames <= 8:
                supervisor.poll_once()
                time.sleep(0.1)
            assert time.monotonic() < deadline, "capture never advanced into idle -- pacer appears blocked"
            supervisor.poll_once()

            player = play_tone(440.0, 1.5, sample_rate=48000)
            player.join(timeout=5.0)
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline and capture.health.consecutive_synthesized_frames != 0:
                supervisor.poll_once()
                time.sleep(0.1)
            assert time.monotonic() < deadline, "capture never saw real audio again after playback"
            supervisor.poll_once()
        finally:
            capture.stop()
    finally:
        pa.terminate()
    store.close()

    events = _read_events(store)
    assert any(e.get("event") == "idle" for e in events)
    assert any(e.get("event") == "recovered" for e in events)


# --- Keep-alive ------------------------------------------------------------------------


def test_keepalive_enabled_opens_and_closes_render_stream():
    device = DeviceInfo(
        index=5, name="Stub Speakers", native_sample_rate=48000, channels=2,
        is_loopback=False, identity="render:stub",
    )
    pa = StubPyAudio()
    keepalive = KeepAliveStream(StubPyAudioModule(), pa, device)
    keepalive.start()
    try:
        assert keepalive.opened
        assert len(pa.opened_streams) == 1
    finally:
        keepalive.stop()
    assert pa.opened_streams[0].closed


def test_keepalive_open_failure_is_a_recorded_warning_never_fatal():
    device = DeviceInfo(
        index=5, name="Stub Speakers", native_sample_rate=48000, channels=2,
        is_loopback=False, identity="render:stub",
    )
    pa = StubPyAudio(open_error=OSError("device busy"))
    keepalive = KeepAliveStream(StubPyAudioModule(), pa, device)
    keepalive.start()  # must not raise
    assert not keepalive.opened
    assert keepalive.warning is not None
    keepalive.stop()  # must not raise even though nothing ever opened


def test_keepalive_enabled_config_parses_env_flag(monkeypatch):
    monkeypatch.setenv("APP_KEEPALIVE_ENABLED", "false")
    assert config_mod.load_config().keepalive_enabled is False
    monkeypatch.setenv("APP_KEEPALIVE_ENABLED", "true")
    assert config_mod.load_config().keepalive_enabled is True


# --- Device change: stubbed resolver, success and failure ------------------------------


class _FakeHealth:
    def __init__(self) -> None:
        self.last_real_audio_monotonic = time.monotonic()
        self.consecutive_synthesized_frames = 0
        self.dropped_frame_count = 0
        self.queue_depth = 0


class _FakeQueue:
    def __init__(self, maxsize: int = 10) -> None:
        self.maxsize = maxsize


class _FakeCapture:
    """Satisfies `SupervisedCapture` without any real PortAudio object --
    exactly the deterministic double the plan calls for."""

    def __init__(self, device: DeviceInfo, *, reopen_should_fail: bool = False) -> None:
        self.health = _FakeHealth()
        self._device = device
        self.queue = _FakeQueue()
        self._dead = False
        self.reopen_calls: list[DeviceInfo] = []
        self._reopen_should_fail = reopen_should_fail

    @property
    def device(self) -> DeviceInfo:
        return self._device

    @property
    def is_dead(self) -> bool:
        return self._dead

    def reopen(self, new_device: DeviceInfo) -> None:
        self.reopen_calls.append(new_device)
        if self._reopen_should_fail:
            raise DeviceOpenError(f"could not reopen '{new_device.name}'")
        self._device = new_device

    def mark_dead(self) -> None:
        self._dead = True


def test_device_change_success_emits_changed_and_reopened_exactly_once(tmp_path):
    old_device = DeviceInfo(
        index=1, name="Old Speakers", native_sample_rate=48000, channels=2,
        is_loopback=True, identity="loopback:old",
    )
    new_device = DeviceInfo(
        index=2, name="New Headphones", native_sample_rate=48000, channels=2,
        is_loopback=True, identity="loopback:new",
    )
    capture = _FakeCapture(old_device)
    calls = {"n": 0}

    def resolver() -> DeviceInfo:
        calls["n"] += 1
        return old_device if calls["n"] == 1 else new_device

    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    supervisor = StreamSupervisor(
        speaker="interviewee", capture=capture, store=store,
        device_resolver=resolver, config=SupervisorConfig(device_poll_interval_s=0.0),
    )

    supervisor.poll_once()  # resolver -> old_device: no change
    supervisor.poll_once()  # resolver -> new_device: change detected, one reopen
    supervisor.poll_once()  # resolver -> new_device again: already current, no further action
    store.close()

    assert len(capture.reopen_calls) == 1
    events = _read_events(store)
    changed = [e for e in events if e.get("event") == "device-changed"]
    reopened = [e for e in events if e.get("event") == "reopened"]
    assert len(changed) == 1
    assert changed[0]["detail"] == {"old_device": "Old Speakers", "new_device": "New Headphones"}
    assert len(reopened) == 1


def test_device_change_failure_emits_dead_once_and_never_retry_loops(tmp_path):
    old_device = DeviceInfo(
        index=1, name="Old Speakers", native_sample_rate=48000, channels=2,
        is_loopback=True, identity="loopback:old",
    )
    ghost_device = DeviceInfo(
        index=2, name="Ghost Device", native_sample_rate=48000, channels=2,
        is_loopback=True, identity="loopback:ghost",
    )
    capture = _FakeCapture(old_device, reopen_should_fail=True)

    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    supervisor = StreamSupervisor(
        speaker="interviewee", capture=capture, store=store,
        device_resolver=lambda: ghost_device, config=SupervisorConfig(device_poll_interval_s=0.0),
    )

    supervisor.poll_once()
    assert capture.is_dead
    supervisor.poll_once()  # the top-level is_dead guard must stop the watch entirely
    supervisor.poll_once()
    store.close()

    assert len(capture.reopen_calls) == 1, "must attempt exactly ONE reopen, never a retry loop"
    events = _read_events(store)
    dead = [e for e in events if e.get("event") == "dead"]
    assert len(dead) == 1


# --- Sample rate: two stubbed sources, 48000 and 44100 ----------------------------------


def test_two_stubbed_sample_rates_yield_identical_16khz_mono_shape_and_duration():
    device_a = DeviceInfo(
        index=1, name="A", native_sample_rate=48000, channels=1, is_loopback=True, identity="loopback:a",
    )
    device_b = DeviceInfo(
        index=2, name="B", native_sample_rate=44100, channels=1, is_loopback=False, identity="input:b",
    )
    pa = StubPyAudio(amplitude=0)
    cap_a = CaptureStream(
        StubPyAudioModule(), pa, device_a, target_sample_rate=16000, frame_ms=100,
        queue_max_seconds=5.0, label="a",
    )
    cap_b = CaptureStream(
        StubPyAudioModule(), pa, device_b, target_sample_rate=16000, frame_ms=100,
        queue_max_seconds=5.0, label="b",
    )
    cap_a.start()
    cap_b.start()
    try:
        frames_a = [cap_a.queue.get(timeout=2.0) for _ in range(5)]
        frames_b = [cap_b.queue.get(timeout=2.0) for _ in range(5)]
    finally:
        cap_a.stop()
        cap_b.stop()

    total_a = sum(len(f.pcm16) for f in frames_a)
    total_b = sum(len(f.pcm16) for f in frames_b)
    assert abs(total_a - total_b) <= 64, "48000 Hz and 44100 Hz sources must converge on the same total byte count"
    duration_a = total_a / 2 / 16000
    duration_b = total_b / 2 / 16000
    assert abs(duration_a - duration_b) < 0.01, "must yield identical frame duration once resampled"


# --- Exclusive mode / open failure -------------------------------------------------------


def test_exclusive_mode_open_failure_names_device_and_setting_preserves_cause():
    device = DeviceInfo(
        index=9, name="Exclusive Speakers", native_sample_rate=48000, channels=2,
        is_loopback=True, identity="loopback:exclusive",
    )
    original = OSError("[Errno -9997] Invalid sample rate")
    pa = StubPyAudio(open_error=original)
    with pytest.raises(DeviceOpenError) as exc_info:
        open_capture_stream(StubPyAudioModule(), pa, device, frames_per_buffer=100)
    message = str(exc_info.value)
    assert "Exclusive Speakers" in message
    assert "exclusive control" in message.lower()
    assert exc_info.value.__cause__ is original, "the raw driver error must be preserved as the cause, never swallowed"


# --- Lagging engine: lag + drop events, capture keeps advancing -------------------------


def test_lagging_engine_causes_lag_and_drop_events_while_capture_keeps_advancing(tmp_path):
    device = DeviceInfo(
        index=1, name="Loud Loopback", native_sample_rate=16000, channels=1,
        is_loopback=True, identity="loopback:loud",
    )
    pa = StubPyAudio(amplitude=0)
    capture = CaptureStream(
        StubPyAudioModule(), pa, device, target_sample_rate=16000, frame_ms=10,
        queue_max_seconds=0.2, label="interviewee",  # tiny queue -- overflow triggers fast, deterministically
    )
    # Never finalizes (huge cadence), but every poll() sleeps -- a consume
    # rate far slower than the 10ms frame period, exactly what backs up the
    # queue behind it (fake.py's own documented lag-testing shape).
    engine = FakeStreamingEngine(cadence_bytes=10_000_000, poll_delay_s=0.2)
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "fake"})
    worker = StreamWorker(
        engine=engine, frame_queue=capture.queue, store=store, speaker="interviewee",
        sample_rate=16000, queue_get_timeout_s=0.05,
    )
    supervisor = StreamSupervisor(
        speaker="interviewee", capture=capture, store=store,
        config=SupervisorConfig(lag_queue_fraction=0.5, poll_interval_s=0.05),
    )

    worker.start()
    capture.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and capture.health.dropped_frame_count == 0:
            supervisor.poll_once()
            time.sleep(0.05)
        assert time.monotonic() < deadline, "queue never overflowed -- test setup did not create real lag"
        dropped_at_first_overflow = capture.health.dropped_frame_count
        supervisor.poll_once()
        time.sleep(0.2)
        assert capture.health.dropped_frame_count >= dropped_at_first_overflow, (
            "capture must keep advancing (and therefore keep dropping) after the engine falls behind"
        )
    finally:
        worker.stop()
        capture.stop()
    store.close()

    events = _read_events(store)
    assert any(e.get("event") == "lagging" for e in events)
    dropped_events = [e for e in events if e.get("event") == "dropped"]
    assert dropped_events
    assert dropped_events[-1]["detail"]["total_dropped"] >= 1


# --- Worker exception: confined to its own stream ----------------------------------------


class _RaisingSession:
    def __init__(self, fail_after: int = 1) -> None:
        self._sent = 0
        self._fail_after = fail_after

    def send_audio(self, pcm16: bytes) -> None:
        self._sent += 1
        if self._sent > self._fail_after:
            raise RuntimeError("simulated engine failure mid-session")

    def poll(self, timeout_s: float = 0.0) -> tuple:
        return ()

    def close(self) -> tuple:
        return ()


class _RaisingEngine:
    name = "raising-stub"

    def open_session(self, *, sample_rate: int, label: str = ""):
        return _RaisingSession()


def test_worker_exception_produces_stream_failure_and_leaves_other_side_running(tmp_path):
    store = SessionStore(create_session_dir(tmp_path))
    store.write_metadata({"engine": "mixed"})

    failing_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    for i in range(5):
        failing_queue.put(CaptureFrame(seq=i, pcm16=b"\x00\x00", kind="synthesized", captured_monotonic=0.0))
    failing_worker = StreamWorker(
        engine=_RaisingEngine(), frame_queue=failing_queue, store=store,
        speaker="interviewee", sample_rate=16000,
    )

    healthy_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    for i in range(3):
        healthy_queue.put(CaptureFrame(seq=i, pcm16=b"\x00\x00", kind="synthesized", captured_monotonic=0.0))
    healthy_worker = StreamWorker(
        engine=FakeStreamingEngine(cadence_bytes=1), frame_queue=healthy_queue, store=store,
        speaker="interviewer", sample_rate=16000,
    )

    failing_worker.start()
    healthy_worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not failing_worker.failed:
            time.sleep(0.02)
        assert failing_worker.failed, "the failing worker's own thread must record its failure"
    finally:
        failing_worker.stop()
        healthy_worker.stop()
    store.close()

    events = _read_events(store)
    failures = [e for e in events if e.get("event") == "stream-failure"]
    assert len(failures) == 1
    assert failures[0]["speaker"] == "interviewee"
    assert failures[0]["detail"]["error_type"] == "RuntimeError"

    healthy_segments = [e for e in events if e.get("kind") == "segment" and e.get("speaker") == "interviewer"]
    assert healthy_segments, "the other stream must keep producing segments despite the first one's failure"


# --- S13: exactly one PyAudio() construction site under app/ outside tests ---------------


def test_pyaudio_constructed_in_exactly_one_place_under_app_outside_tests():
    # AST-based, not regex: a docstring PROSE mention of "PyAudio()" (this
    # module's own explanatory comments included) must never count as a
    # construction site -- only an actual `ast.Call` naming `PyAudio` does.
    import ast

    hits: list[tuple[Path, int]] = []
    for py_file in sorted((REPO_ROOT / "app").rglob("*.py")):
        relative = py_file.relative_to(REPO_ROOT).as_posix()
        if relative.startswith("app/tests/"):
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "PyAudio":
                hits.append((py_file, node.lineno))
    assert len(hits) == 1, f"expected exactly ONE PyAudio() construction site under app/ (outside tests), found: {hits}"

    file_path, _lineno = hits[0]
    assert file_path.name == "main.py", (
        f"PyAudio() must be constructed only in app/main.py (the main thread, before any worker "
        f"starts -- phase_specification S13), found in {file_path} instead"
    )
    # ...and never inside a module that owns a worker-thread entry point.
    worker_thread_modules = {"stream.py", "live.py", "supervisor.py", "keepalive.py"}
    assert file_path.name not in worker_thread_modules
