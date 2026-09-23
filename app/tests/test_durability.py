"""SESS-01/SESS-02: the durability contract stated in phase_specification U1
--"when `append()` returns, that record is on disk and will still be there
after the process is terminated without warning" -- proven as an observable,
not reasoned about from fsync being present in the source.

Written FIRST (TDD RED): these fail until `app/session/reader.py` exists,
`SessionStore` exposes `fsync_count`/`resumed` and supports reopening an
existing session directory, and `app/tests/crash_writer.py` exists as a real
subprocess entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.session.reader import MalformedRecordError, read_session
from app.session.store import SessionStore, create_session_dir

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- The fsync-per-record contract, asserted by count -------------------------------


def test_fsync_count_equals_number_of_records_appended(tmp_path):
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    for i in range(5):
        store.append_segment(
            speaker="interviewee",
            text=f"segment {i}",
            audio_start_s=float(i),
            audio_end_s=float(i) + 1.0,
            ready_wall_s=float(i) + 1.1,
            detected_language="en",
            engine="fake",
        )
    assert store.fsync_count == 5, "one fsync per append, asserted by count, not by reasoning"

    store.append_event(event="stream_idle", speaker="interviewee", detail={"idle_s": 1.0})
    assert store.fsync_count == 6, "append_event fsyncs too -- every record, not only segments"


# --- The real kill: a subprocess, not a simulation -----------------------------------


def _spawn_crash_writer(session_dir: Path, count: int) -> "subprocess.Popen[str]":
    return subprocess.Popen(
        [sys.executable, "-m", "app.tests.crash_writer", str(session_dir), str(count)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _read_n_reported_lines(proc: "subprocess.Popen[str]", n: int, deadline_s: float) -> list[int]:
    reported: list[int] = []
    deadline = time.monotonic() + deadline_s
    while len(reported) < n and time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        stripped = line.strip()
        if stripped:
            reported.append(int(stripped))
    return reported


def test_kill_a_real_child_process_mid_append_leaves_every_reported_record_on_disk(tmp_path):
    """The SESS-02 test. This kills a REAL process -- no simulation, no mock,
    no reasoning about fsync. `crash_writer.py` reports a sequence number
    only after `append_segment` has returned (i.e. flushed and fsync'd); this
    test reads a handful of those reports, then kills the process without
    warning, and asserts every record it saw reported is actually present and
    parseable on disk afterwards."""
    session_dir = tmp_path / "crash-session"
    proc = _spawn_crash_writer(session_dir, count=2000)
    try:
        reported = _read_n_reported_lines(proc, n=20, deadline_s=20.0)
        assert len(reported) >= 20, (
            f"child process did not report 20 appends within the deadline "
            f"(got {reported}); stderr={proc.stderr.read() if proc.stderr else ''}"
        )
        # Kill without warning -- Popen.kill() is TerminateProcess() on
        # Windows and SIGKILL on POSIX: immediate, uncatchable, no chance for
        # the child to run any cleanup or finish an in-flight write.
        proc.kill()
        proc.wait(timeout=10.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10.0)

    result = read_session(session_dir)
    segments_by_seq = {r["seq"]: r for r in result.records_of_kind("segment")}

    missing = [i for i in reported if i not in segments_by_seq]
    assert not missing, (
        f"records reported as appended before the kill are missing on disk: {missing}. "
        f"This is the durability claim SESS-02 makes, and it just failed."
    )
    for i in reported:
        record = segments_by_seq[i]
        assert record["text"] == f"crash-writer segment {i}"
        assert record["seq"] == i


# --- Tolerant reading: truncated tail named, interior corruption refused ------------


def _write_raw_transcript(session_dir: Path, lines: list[str], *, trailing_newline: bool) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if trailing_newline:
        content += "\n"
    (session_dir / "transcript.jsonl").write_text(content, encoding="utf-8")


def test_reader_recovers_truncated_final_line_and_reports_it(tmp_path):
    session_dir = tmp_path / "truncated-session"
    valid_lines = [
        '{"kind":"segment","seq":0,"speaker":"interviewee","text":"hello","audio_start_s":0.0,'
        '"audio_end_s":1.0,"ready_wall_s":1.1,"latency_s":0.1,"detected_language":"en",'
        '"engine":"fake","written_at":"2026-08-05T00:00:00+00:00"}',
        '{"kind":"segment","seq":1,"speaker":"interviewer","text":"world","audio_start_s":1.0,'
        '"audio_end_s":2.0,"ready_wall_s":2.1,"latency_s":0.1,"detected_language":"en",'
        '"engine":"fake","written_at":"2026-08-05T00:00:01+00:00"}',
    ]
    truncated_tail = '{"kind":"segment","seq":2,"speaker":"interviewee","text":"cut off mid-w'
    _write_raw_transcript(session_dir, valid_lines + [truncated_tail], trailing_newline=False)
    # No session.json at all -- an unclosed session that never got the chance
    # to write metadata is still expected to read cleanly (closed defaults False).

    result = read_session(session_dir)

    assert len(result.records) == 2
    assert result.records[0]["text"] == "hello"
    assert result.records[1]["text"] == "world"
    assert result.truncated_tail == truncated_tail
    assert result.closed is False


def test_reader_rejects_malformed_interior_line_loudly(tmp_path):
    session_dir = tmp_path / "corrupt-interior-session"
    valid_line_1 = '{"kind":"segment","seq":0,"speaker":"interviewee","text":"a"}'
    corrupt_interior = '{"kind":"segment","seq":1,"speaker":"interviewee","text":"b"'  # missing closing brace, NOT last
    valid_line_3 = '{"kind":"segment","seq":2,"speaker":"interviewee","text":"c"}'
    _write_raw_transcript(
        session_dir, [valid_line_1, corrupt_interior, valid_line_3], trailing_newline=True
    )

    with pytest.raises(MalformedRecordError):
        read_session(session_dir)


# --- Reopening: continues the sequence, never truncates, never reuses --------------


def test_reopening_session_continues_sequence_without_reuse_or_truncation(tmp_path):
    session_dir = tmp_path / "resume-session"

    store1 = SessionStore(session_dir)
    store1.write_metadata({"engine": "fake"})
    for i in range(3):
        store1.append_segment(
            speaker="interviewee",
            text=f"before-crash {i}",
            audio_start_s=float(i),
            audio_end_s=float(i) + 1.0,
            ready_wall_s=float(i) + 1.1,
            detected_language="en",
            engine="fake",
        )
    # No store1.close() -- a real crash never gets an orderly close either.
    # The transcript file descriptor is deliberately left open, exactly like
    # a killed process would leave it.

    original_bytes = store1.transcript_path.read_bytes()

    store2 = SessionStore(session_dir)
    assert store2.resumed is True

    new_record = store2.append_segment(
        speaker="interviewer",
        text="after-resume 0",
        audio_start_s=10.0,
        audio_end_s=11.0,
        ready_wall_s=11.1,
        detected_language="en",
        engine="fake",
    )
    store2.close()

    # Everything store1 wrote is untouched, byte for byte -- reopening never
    # truncates or rewrites what was already on disk.
    on_disk = store2.transcript_path.read_bytes()
    assert on_disk.startswith(original_bytes)

    result = read_session(session_dir)
    seqs = [r["seq"] for r in result.records]
    assert seqs == sorted(seqs), "sequence numbers must never be reused or go backwards"
    assert len(seqs) == len(set(seqs)), "no sequence number may be reused"
    assert seqs == list(range(len(seqs))), "sequence must continue with no gaps"

    resumed_events = result.records_of_kind("event")
    assert any(e["event"] == "session-resumed" for e in resumed_events), (
        "reopening an existing session must record a session-resumed event, "
        "so a restarted interview is visible in the record rather than inferred"
    )
    assert new_record.seq == seqs[-1]


# --- An unclosed session is obviously unclosed, and readable while open ------------


def test_reading_unclosed_session_reports_unclosed_and_yields_all_records(tmp_path):
    session_dir = tmp_path / "unclosed-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    store.append_segment(
        speaker="interviewee", text="still recording", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )
    # Deliberately never call store.close().

    result = read_session(session_dir)
    assert result.closed is False
    assert len(result.records) == 1


def test_records_readable_while_session_still_being_written(tmp_path):
    session_dir = tmp_path / "open-writer-session"
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake"})
    store.append_segment(
        speaker="interviewee", text="live", audio_start_s=0.0, audio_end_s=1.0,
        ready_wall_s=1.1, detected_language="en", engine="fake",
    )

    # The store's own file handle is still open, in append mode, right now --
    # a reader must not be blocked by that (no exclusive lock is ever taken).
    result = read_session(session_dir)
    assert len(result.records) == 1
    assert result.records[0]["text"] == "live"

    store.append_segment(
        speaker="interviewer", text="also live", audio_start_s=1.0, audio_end_s=2.0,
        ready_wall_s=2.1, detected_language="en", engine="fake",
    )
    result_after_second = read_session(session_dir)
    assert len(result_after_second.records) == 2


# --- The Hebrew crash tail: a kill truncates BYTES, not characters -----------------


def _cut_inside_a_hebrew_codepoint(blob: bytes, *, after: int) -> bytes:
    """Truncates `blob` so the final byte is the LEAD byte of a two-byte
    Hebrew codepoint whose continuation byte is gone -- the byte-level shape a
    hard kill actually produces on this product's transcripts."""
    for i in range(len(blob) - 1, after, -1):
        if 0xD0 <= blob[i - 1] <= 0xD7:
            return blob[:i]
    raise AssertionError("fixture contains no two-byte Hebrew codepoint to cut inside")


def test_reader_tolerates_a_final_line_cut_inside_a_hebrew_codepoint(tmp_path):
    """A hard kill truncates at a BYTE offset. Almost every character in this
    product's transcripts is a two-byte Hebrew codepoint, so the single most
    likely crash tail is one cut mid-codepoint -- invalid UTF-8, not merely
    invalid JSON. Decoding the whole file up front raises before any
    line-level tolerance can run, which would make `render` and resume both
    die on exactly the interviews SESS-02 exists to protect."""
    session_dir = tmp_path / "hebrew-crash-tail"
    session_dir.mkdir()

    complete = json.dumps(
        {
            "kind": "segment", "seq": 0, "speaker": "interviewee",
            "text": "שלום, we need to check the API first", "audio_start_s": 0.0,
            "audio_end_s": 1.0, "ready_wall_s": 1.1, "latency_s": 0.1,
            "detected_language": "he", "engine": "fake",
            "written_at": "2026-08-05T00:00:00+00:00",
        },
        ensure_ascii=False,
    )
    interrupted = json.dumps(
        {"kind": "segment", "seq": 1, "speaker": "interviewee", "text": "אז בעצם הבעיה היא"},
        ensure_ascii=False,
    )
    blob = (complete + "\n" + interrupted).encode("utf-8")
    (session_dir / "transcript.jsonl").write_bytes(
        _cut_inside_a_hebrew_codepoint(blob, after=len(complete.encode("utf-8")))
    )

    result = read_session(session_dir)

    assert len(result.records) == 1, "the complete record before the cut must survive"
    assert result.records[0]["text"] == "שלום, we need to check the API first"
    assert result.truncated_tail is not None, "the undecodable tail must be NAMED, not raised"


def test_reader_refuses_an_interior_line_that_is_invalid_utf8(tmp_path):
    """A mid-codepoint cut is expected at the TAIL only. The same byte damage
    in the middle of the file is real corruption and must be refused, exactly
    as a malformed interior JSON line is."""
    session_dir = tmp_path / "hebrew-corrupt-interior"
    session_dir.mkdir()

    first = json.dumps({"kind": "segment", "seq": 0, "text": "a"}, ensure_ascii=False)
    third = json.dumps({"kind": "segment", "seq": 2, "text": "c"}, ensure_ascii=False)
    broken_middle = b'{"kind":"segment","seq":1,"text":"\xd7'  # lone Hebrew lead byte
    blob = first.encode("utf-8") + b"\n" + broken_middle + b"\n" + third.encode("utf-8") + b"\n"
    (session_dir / "transcript.jsonl").write_bytes(blob)

    with pytest.raises(MalformedRecordError):
        read_session(session_dir)


# --- Resume after a REAL crash, not after a clean append --------------------------


def test_resume_after_a_crash_tail_leaves_the_session_permanently_readable(tmp_path):
    """The two must-have truths interact: "reopening appends rather than
    truncates" and "a half-written final line is tolerated". A crash leaves a
    partial tail BY DEFINITION, so appending the session-resumed event
    straight onto it welds two records into one malformed line -- which, once
    any further record lands, becomes an INTERIOR line the reader refuses.
    The session then reads as corrupt forever, including the transcript the
    interview actually captured."""
    session_dir = tmp_path / "resume-after-crash"

    store1 = SessionStore(session_dir)
    store1.write_metadata({"engine": "fake"})
    store1.append_segment(
        speaker="interviewee", text="זה מה שהקלטנו before the crash", audio_start_s=0.0,
        audio_end_s=1.0, ready_wall_s=1.1, detected_language="he", engine="fake",
    )
    durable_records = len(read_session(session_dir).records)

    # A hard kill mid-flush: a partial record, no trailing newline. This is
    # the shape the reader is explicitly built to tolerate.
    with open(store1.transcript_path, "ab") as handle:
        handle.write(b'{"kind":"segment","seq":1,"speaker":"interviewee","text":"cut mi')

    store2 = SessionStore(session_dir)
    assert store2.resumed is True
    store2.append_segment(
        speaker="interviewer", text="and this is after the resume", audio_start_s=2.0,
        audio_end_s=3.0, ready_wall_s=3.1, detected_language="en", engine="fake",
    )
    store2.close()

    # Must not raise -- and must not have lost the durable record.
    result = read_session(session_dir)
    texts = [r.get("text") for r in result.records_of_kind("segment")]
    assert "זה מה שהקלטנו before the crash" in texts, (
        "the record that was durable before the crash was lost by the resume"
    )
    assert "and this is after the resume" in texts
    assert len(result.records_of_kind("segment")) >= durable_records + 1

    seqs = [r["seq"] for r in result.records]
    assert len(seqs) == len(set(seqs)), "no sequence number may be reused across a resume"
