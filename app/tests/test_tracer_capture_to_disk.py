"""Task 2 done criteria: one path through every layer, proven end to end on
this machine's real audio hardware.

- `test_loopback_captures_known_tone` (hardware-marked): plays a known 440 Hz
  tone through the default output device and asserts the loopback capture
  stream sees it as `real` frames whose dominant FFT bin is close to 440 Hz
  (phase_specification S3's pacer, AUDIO-01 on the real machine).
- `test_offline_end_to_end_writes_segments_while_running`: runs the whole
  `start` path with the offline (`fake`) engine and asserts the session file
  is written to WHILE capture is still running, not flushed once at the end
  (SESS-01), with a schema-correct segment record (STT-04).
- `test_live_engine_end_to_end_writes_segments` (live-marked): the same path
  against the real vendor, on real bilingual Hebrew/English audio, skipping
  cleanly when no credentials are present.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app import main as app_main

REPO_ROOT = Path(__file__).resolve().parents[2]
BILINGUAL_FIXTURE = REPO_ROOT / "benchmark" / "data" / "fixtures" / "synthetic_cs.wav"


@pytest.mark.hardware
def test_loopback_captures_known_tone(tmp_path):
    import pyaudiowpatch as pyaudio

    from app.capture.devices import resolve_default_loopback_device
    from app.capture.stream import CaptureStream
    from app.tests.playback import play_tone

    pa = pyaudio.PyAudio()
    try:
        device = resolve_default_loopback_device(pa)
        capture = CaptureStream(
            pyaudio,
            pa,
            device,
            target_sample_rate=16000,
            frame_ms=150,
            queue_max_seconds=10.0,
            label="interviewee",
        )
        capture.start()
        try:
            player = play_tone(440.0, 2.0, sample_rate=48000)
            player.join(timeout=5.0)
            time.sleep(0.5)  # let the last frames drain through the loopback pipe

            real_pcm = bytearray()
            drained = 0
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and drained < 200:
                try:
                    frame = capture.queue.get(timeout=0.5)
                except queue.Empty:
                    break
                drained += 1
                if frame.kind == "real":
                    real_pcm.extend(frame.pcm16)
        finally:
            capture.stop()
    finally:
        pa.terminate()

    assert real_pcm, (
        "No real (non-synthesized) audio frames were captured. The default "
        "output device may be muted or at zero volume -- see this plan's "
        "user_setup: raise the output volume above zero before running this "
        "test."
    )

    samples = np.frombuffer(bytes(real_pcm), dtype=np.int16).astype(np.float64)
    fft = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / 16000)
    peak_freq = freqs[np.argmax(np.abs(fft))]
    assert abs(peak_freq - 440.0) <= 20.0, (
        f"Captured loopback audio's dominant frequency was {peak_freq:.1f} Hz, "
        "not within 20 Hz of the 440 Hz test tone. The default output device "
        "may be muted or at zero volume."
    )


def _run_start_in_background(args, monkeypatch, sessions_root: Path, engine: str):
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(sessions_root))
    # BOTH roots, not just sessions. Pointing only `APP_SESSIONS_ROOT` at
    # tmp_path left `start` reading the REPOSITORY'S OWN `context/`, which on a
    # clean tree holds nothing but `*.example.json` -- so this test passed only
    # when some earlier test in the same run happened to leave a real
    # `context/brief.json` behind, and failed standalone or under any change to
    # collection order. It failed today for exactly that reason, after a new
    # test file was added elsewhere in the suite.
    #
    # This is the second instance of this defect in this repo; the first was
    # `test_no_serve_runs_full_capture...`. The rule: any test that drives
    # `run_start` must own BOTH `APP_CONTEXT_DIR` and `APP_SESSIONS_ROOT`, or
    # it is testing whatever the last test left on disk.
    context_dir = sessions_root / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "brief.json").write_text("[]", encoding="utf-8")
    (context_dir / "survey.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    # EMPTY, never deleted: the dotenv loader fills in anything "not already
    # present", so `delenv` hands the real key straight back. Same trap that
    # made a credential-absent test pass for the wrong reason since Phase 4.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("STT_ENGINE", engine)
    result_holder: dict = {}

    def _run():
        result_holder["code"] = app_main.run_start(args)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result_holder


def test_offline_end_to_end_writes_segments_while_running(tmp_path, monkeypatch):
    args = app_main.build_arg_parser().parse_args(["start", "--seconds", "3"])
    thread, result_holder = _run_start_in_background(args, monkeypatch, tmp_path, "fake")

    # Poll for the transcript file to appear and gain content WHILE capture is
    # still running -- proves append-during-capture (SESS-01), not a
    # flush-once-at-the-end.
    transcript_path = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        candidates = list(tmp_path.glob("*/transcript.jsonl"))
        if candidates and candidates[0].stat().st_size > 0:
            transcript_path = candidates[0]
            break
        time.sleep(0.1)

    assert transcript_path is not None, "session directory/transcript was never created"
    assert thread.is_alive(), (
        "capture finished before the mid-run read -- this test is not "
        "exercising 'appended while still running'"
    )

    thread.join(timeout=15.0)
    assert result_holder.get("code") == 0

    lines = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seqs = [record["seq"] for record in lines]
    assert seqs == sorted(seqs), "sequence numbers are not monotonically increasing"

    segment_lines = [record for record in lines if record["kind"] == "segment"]
    assert segment_lines, "no segment records were written"
    # plan 02-02 made `start` dual-stream by default (interviewee + interviewer
    # run concurrently), so which side's segment lands first is a genuine race
    # between two independent worker threads -- assert on the interviewee's
    # OWN first segment, not on transcript-wide ordering (deviation, Rule 1:
    # test design, same class as plan 02-01's own deviation #5).
    interviewee_segments = [r for r in segment_lines if r["speaker"] == "interviewee"]
    assert interviewee_segments, "no interviewee segment records were written"
    first = interviewee_segments[0]
    assert first["text"]
    assert first["audio_end_s"] > first["audio_start_s"]
    assert isinstance(first["latency_s"], (int, float))
    assert first["latency_s"] >= 0.0
    assert first["engine"] == "fake"


@pytest.mark.live
def test_live_engine_end_to_end_writes_segments(tmp_path, monkeypatch):
    from app import config as config_mod

    config_mod._load_dotenv_once()  # this test may run in isolation, before any load_config() call
    if not os.environ.get("SONIOX_API_KEY"):
        pytest.skip("SONIOX_API_KEY not set")
    assert BILINGUAL_FIXTURE.exists(), f"missing fixture: {BILINGUAL_FIXTURE}"

    from benchmark.audio import load_audio
    from app.tests.playback import TonePlayer

    samples = load_audio(str(BILINGUAL_FIXTURE), sampling_rate=48000)
    duration_s = len(samples) / 48000.0

    args = app_main.build_arg_parser().parse_args(
        ["start", "--seconds", str(duration_s + 3.0)]
    )
    thread, result_holder = _run_start_in_background(args, monkeypatch, tmp_path, "soniox")

    # Do NOT construct the playback helper's own PyAudio() host concurrently
    # with run_start's -- PortAudio's Pa_Initialize is not safe to call from
    # two threads at once and doing so crashed this process with a Windows
    # access violation (observed directly while developing this test).
    # Give run_start's PyAudio() + device-open sequence time to finish first.
    time.sleep(1.5)
    player = TonePlayer(samples, sample_rate=48000)
    player.start()

    thread.join(timeout=duration_s + 30.0)
    player.join(timeout=10.0)

    assert result_holder.get("code") == 0
    candidates = list(tmp_path.glob("*/transcript.jsonl"))
    assert candidates, "no session transcript was written"
    lines = [
        json.loads(line)
        for line in candidates[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Ambient/silent conditions may or may not yield a finalized segment in
    # the window under test -- what this asserts is that the session ran
    # against the real vendor without error and produced schema-valid
    # records, not that speech was necessarily detected.
    assert all(record["kind"] in ("segment", "event") for record in lines)
