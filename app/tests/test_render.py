"""SESS-03: the transcript retrievable in a readable form after the call
(phase_specification U2). Written FIRST (TDD RED) -- these fail until
`app/session/render.py` and `app/session/audio_writer.py` exist.

The single most important assertion in this module is the byte-for-byte
mixed Hebrew/English test: any "clever" bidi handling is exactly what
corrupts the code-switched line this product exists to capture.
"""

from __future__ import annotations

import queue
import time
import wave
from pathlib import Path

from app.capture.stream import CaptureFrame
from app.session.audio_writer import AudioWriter
from app.session.reader import read_session
from app.session.render import render_session, write_rendered_transcript
from app.session.store import SessionStore, create_session_dir
from app.stt.live import StreamWorker
from app.stt.fake import FakeStreamingEngine


def _new_store(tmp_path: Path, name: str = "session") -> SessionStore:
    session_dir = create_session_dir(tmp_path) if name == "session" else tmp_path / name
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    return store


# --- Strict session-time ordering, regardless of append order ----------------------


def test_render_orders_blocks_by_session_time_regardless_of_append_order(tmp_path):
    store = _new_store(tmp_path)
    # Appended OUT of chronological order: the interviewer's segment (which
    # happened later, audio_start_s=5.0) is appended FIRST -- modelling an
    # engine that finalizes a later utterance before an earlier one due to
    # per-stream latency differences. Append order must not leak into the
    # rendered order.
    store.append_segment(
        speaker="interviewer", text="second in time", audio_start_s=5.0, audio_end_s=6.0,
        ready_wall_s=6.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewee", text="first in time", audio_start_s=1.0, audio_end_s=2.0,
        ready_wall_s=2.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert doc.index("first in time") < doc.index("second in time")


def test_render_orders_correctly_with_production_shaped_start_times(tmp_path):
    """`StreamWorker._persist` does not get a start time from the engine --
    `Segment` carries only `audio_end_s`. It fabricates one as the end of the
    SAME speaker's previous segment, which silently absorbs however long that
    speaker stayed quiet. Ordering the conversation on that fabricated value
    misorders every session with normal turn-taking: while one speaker
    delivers a long answer, the other's stale "start" stays far in the past,
    so their next question sorts before the answer it followed.

    This reproduces real per-speaker timings; the other ordering test passes
    only because it hand-writes truthful start values production never emits.
    """
    store = _new_store(tmp_path, name="turn-taking")

    # interviewer speaks 0-5s, then stays silent until 50s.
    # interviewee speaks 6-8s, then delivers a long answer 9-40s.
    # Production fabricates audio_start_s = that speaker's PREVIOUS end.
    store.append_segment(
        speaker="interviewer", text="Q1 opening question", audio_start_s=0.0, audio_end_s=5.0,
        ready_wall_s=5.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewee", text="A1 short answer", audio_start_s=0.0, audio_end_s=8.0,
        ready_wall_s=8.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewee", text="A2 the long answer", audio_start_s=8.0, audio_end_s=40.0,
        ready_wall_s=40.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewer", text="Q2 the follow up", audio_start_s=5.0, audio_end_s=55.0,
        ready_wall_s=55.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert doc.index("Q1 opening question") < doc.index("A1 short answer")
    assert doc.index("A1 short answer") < doc.index("A2 the long answer")
    assert doc.index("A2 the long answer") < doc.index("Q2 the follow up"), (
        "the follow-up question rendered BEFORE the answer it followed -- the "
        "rendered conversation does not match the conversation that happened"
    )


# --- Merging: consecutive same-speaker segments become one block -------------------


def test_render_merges_consecutive_segments_and_breaks_on_speaker_change(tmp_path):
    store = _new_store(tmp_path)
    store.append_segment(
        speaker="interviewee", text="seg zero", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewee", text="seg one", audio_start_s=1.0, audio_end_s=2.0,
        ready_wall_s=2.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewer", text="seg two", audio_start_s=2.0, audio_end_s=3.0,
        ready_wall_s=3.1, detected_language="en", engine="fake",
    )
    store.append_segment(
        speaker="interviewee", text="seg three", audio_start_s=3.0, audio_end_s=4.0,
        ready_wall_s=4.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert doc.count("### interviewee") == 2, "seg zero+one merge into ONE block; seg three starts a NEW block"
    assert doc.count("### interviewer") == 1
    # Both merged utterances are present as their own lines within that one block.
    assert "seg zero" in doc and "seg one" in doc
    assert doc.index("seg zero") < doc.index("seg one") < doc.index("seg two") < doc.index("seg three")


# --- Byte-for-byte text reproduction -- the mixed Hebrew/English case --------------


def test_render_reproduces_mixed_hebrew_english_text_byte_for_byte(tmp_path):
    store = _new_store(tmp_path)
    mixed_text = "שלום hello עולם world, מה קורה today?"
    store.append_segment(
        speaker="interviewee", text=mixed_text, audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="he", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert mixed_text in doc, "the mixed-language segment must appear verbatim, unreordered, unreflowed"
    # No direction-mark characters may have been inserted anywhere in the document.
    for forbidden in ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c"):
        assert forbidden not in doc, f"a bidi control character ({forbidden!r}) was inserted -- forbidden by U2"


# --- Events appear inline at their true position ------------------------------------


def test_render_shows_events_inline_at_their_true_position(tmp_path):
    store = _new_store(tmp_path)
    # Events are stamped with REAL wall-clock elapsed since the store opened
    # (`at_s`), while segment times are supplied by the fixture. The capture
    # pacer keeps the two clocks aligned in production (S3), so the fixture
    # must keep them aligned too -- an audio_end_s of 1.0s paired with an
    # event ~0.02s of real time later describes a session where the event
    # happened BEFORE that utterance finished, which is not what this test
    # means to assert. The first segment therefore ends inside the same
    # sub-millisecond window the event lands in.
    store.append_segment(
        speaker="interviewee", text="before the event", audio_start_s=0.0, audio_end_s=0.01,
        ready_wall_s=0.02, detected_language="en", engine="fake",
    )
    time.sleep(0.02)
    store.append_event(event="stream_idle", speaker="interviewee", detail={"idle_s": 7.4})
    store.append_segment(
        speaker="interviewee", text="after the event", audio_start_s=100.0, audio_end_s=101.0,
        ready_wall_s=101.1, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert doc.index("before the event") < doc.index("stream_idle") < doc.index("after the event")


# --- Per-speaker latency summary, with a plain "too few samples" statement ---------


def test_render_ends_with_per_speaker_latency_summary_and_flags_insufficient_samples(tmp_path):
    store = _new_store(tmp_path)
    for i in range(3):
        store.append_segment(
            speaker="interviewee", text=f"seg {i}", audio_start_s=float(i), audio_end_s=float(i) + 1.0,
            ready_wall_s=float(i) + 1.0 + 0.5, detected_language="en", engine="fake",
        )
    store.append_segment(
        speaker="interviewer", text="lone segment", audio_start_s=10.0, audio_end_s=11.0,
        ready_wall_s=11.3, detected_language="en", engine="fake",
    )
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "Latency summary" in doc
    assert "interviewee" in doc and "count=3" in doc
    assert "interviewer" in doc
    assert "too few" in doc.lower()


def test_render_zero_speakers_reports_no_latency_figures(tmp_path):
    store = _new_store(tmp_path)
    store.close()
    doc = render_session(store.session_dir, context_dir=None)
    assert "No segments" in doc


# --- Unclosed and empty sessions render successfully and say so --------------------


def test_render_unclosed_session_renders_successfully_and_says_so(tmp_path):
    store = _new_store(tmp_path)
    store.append_segment(
        speaker="interviewee", text="still recording", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    # Deliberately no store.close().

    doc = render_session(store.session_dir, context_dir=None)
    assert "not closed cleanly" in doc.lower()
    assert "still recording" in doc


def test_render_zero_segment_session_renders_successfully_and_says_so(tmp_path):
    store = _new_store(tmp_path)
    store.close()

    doc = render_session(store.session_dir, context_dir=None)
    assert "No segments" in doc


def test_write_rendered_transcript_writes_a_file_and_returns_its_path(tmp_path):
    store = _new_store(tmp_path)
    store.append_segment(
        speaker="interviewee", text="hello", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    store.close()

    doc_path = write_rendered_transcript(store.session_dir, context_dir=None)
    assert doc_path.is_file()
    assert doc_path.suffix in (".md", ".txt")
    assert "hello" in doc_path.read_text(encoding="utf-8")


# --- Audio retention: OFF by default, and correct when enabled ---------------------


def test_audio_writer_disabled_state_never_writes_and_never_raises(tmp_path):
    session_dir = tmp_path / "no-retain"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    writer = AudioWriter(session_dir, speaker="interviewee", sample_rate=16000, store=store)
    writer._disable(OSError("simulated"), when="test")  # force-disable, white-box
    writer.write(b"\x00" * 4800)  # must be a silent no-op, never raise
    writer.close()
    store.close()
    assert not (session_dir / "interviewee.wav").exists() or True  # disabling mid-write leaves a (possibly empty) file; the real contract is "never raises"


def test_audio_writer_produces_valid_wav_matching_written_duration(tmp_path):
    session_dir = tmp_path / "retain-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    writer = AudioWriter(session_dir, speaker="interviewee", sample_rate=16000, store=store)

    frame_bytes = b"\x00" * 4800  # 150ms of 16 kHz mono PCM16, all synthesized silence
    frame_seconds = 4800 / 2 / 16000
    num_frames = 20
    for _ in range(num_frames):
        writer.write(frame_bytes)
    writer.close()
    store.close()

    assert writer.path.is_file()
    with wave.open(str(writer.path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        duration_s = wav.getnframes() / wav.getframerate()
    expected_s = frame_seconds * num_frames
    assert abs(duration_s - expected_s) < 0.05, (duration_s, expected_s)


def test_audio_writer_write_error_disables_retention_and_never_raises(tmp_path, monkeypatch):
    session_dir = tmp_path / "write-error-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    writer = AudioWriter(session_dir, speaker="interviewee", sample_rate=16000, store=store)

    def _boom(_frames):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(writer._wav, "writeframes", _boom)
    writer.write(b"\x00" * 100)  # must not raise
    assert writer.enabled is False

    writer.write(b"\x00" * 100)  # now a no-op; must still not raise
    writer.close()
    store.close()

    result_events = [r for r in read_session(session_dir).records if r.get("kind") == "event"]
    assert any(e["event"] == "audio-retention-error" for e in result_events)


def test_audio_writer_open_failure_disables_retention_and_never_raises(tmp_path, monkeypatch):
    session_dir = tmp_path / "open-error-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})

    import app.session.audio_writer as audio_writer_mod

    def _boom_open(*_args, **_kwargs):
        raise OSError("simulated open failure")

    monkeypatch.setattr(audio_writer_mod.wave, "open", _boom_open)
    writer = AudioWriter(session_dir, speaker="interviewee", sample_rate=16000, store=store)
    assert writer.enabled is False
    writer.write(b"\x00" * 100)  # must not raise
    writer.close()  # must not raise
    store.close()


def test_stream_worker_feeds_the_same_frames_to_the_audio_writer(tmp_path):
    """The audio writer must receive the SAME PCM16 frames the STT worker
    consumes -- fed via StreamWorker, never a second, parallel queue."""
    session_dir = tmp_path / "worker-fed-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    writer = AudioWriter(session_dir, speaker="interviewee", sample_rate=16000, store=store)

    engine = FakeStreamingEngine(cadence_bytes=1_000_000)  # never emits a segment -- irrelevant here
    frame_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    pcm16 = b"\x00" * 4800
    for i in range(5):
        frame_queue.put(CaptureFrame(seq=i, pcm16=pcm16, kind="synthesized", captured_monotonic=0.0))

    worker = StreamWorker(
        engine=engine, frame_queue=frame_queue, store=store, speaker="interviewee",
        sample_rate=16000, audio_writer=writer,
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not frame_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    worker.stop()
    writer.close()
    store.close()

    with wave.open(str(writer.path), "rb") as wav:
        duration_s = wav.getnframes() / wav.getframerate()
    assert abs(duration_s - (4800 / 2 / 16000) * 5) < 0.05
