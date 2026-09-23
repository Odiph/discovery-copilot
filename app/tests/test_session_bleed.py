"""Detecting one track hearing the other -- without crying wolf at a good
interviewer.

**The failure, measured.** 2026-08-12, the first real overlay recording anyone
made: a podcast played through SPEAKERS reached both the loopback and the webcam
microphone, and 67% of the interviewer track was the podcast, attributed to
someone who had not spoken. `--only interviewee` now prevents that for one-sided
recordings; a REAL interview cannot use it, because the microphone must stay
open. There, detection is the only defence available.

**The failure this file is written against is the SECOND one -- a false alarm.**
Reflecting is a real interviewing technique: "so you went back to paper?" after
"we went back to paper". A detector that flags that teaches the operator to
ignore it, and a warning that is ignored is worse than no warning, because it
also costs the credibility of every other warning on the page.

So the negative cases here are the point, and two of them are REAL RECORDINGS
rather than fixtures -- the two genuine 2026-08-09 calls in this repo, 494 and
405 comparable interviewer segments of real bidirectional Hebrew/English
conversation. A heuristic calibrated only against its own fixtures proves
nothing about speech.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.session.bleed import (
    MIN_SEGMENTS_FOR_A_VERDICT,
    BleedReport,
    detect_bleed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _seg(speaker: str, text: str, end_s: float) -> dict:
    # `latency_s` and `audio_start_s` are not read by the detector -- they are
    # here because `render_session` requires them, and a fixture that could not
    # be rendered would quietly cost this file its two end-to-end tests.
    return {
        "kind": "segment", "speaker": speaker, "text": text,
        "audio_start_s": max(0.0, end_s - 2.0), "audio_end_s": end_s,
        "latency_s": 1.2, "ready_wall_s": end_s + 1.2,
    }


def _bled(n: int = 12) -> "list[dict]":
    """`n` sentences, each landing on both tracks a moment apart."""
    out: "list[dict]" = []
    for i in range(n):
        text = f"the export pipeline broke again in the {i} th week of the quarter"
        out.append(_seg("interviewee", text, 10.0 * i))
        out.append(_seg("interviewer", text, 10.0 * i + 0.3))
    return out


# --- it fires on the real thing ------------------------------------------------------------


def test_the_real_podcast_recording_is_flagged():
    """The recording that started all of this, checked as itself rather than
    as a fixture derived from it."""
    path = REPO_ROOT / "projects" / "_unfiled" / "sessions" / "20260812T215237" / "transcript.jsonl"
    if not path.is_file():
        pytest.skip("the 2026-08-12 podcast recording is not in this checkout")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    report = detect_bleed([r for r in records if r.get("kind") == "segment"])
    assert report.suspected is True
    assert report.percent >= 50, f"only {report.percent}% -- the measured figure was ~67%"


def test_a_synthetic_bleed_is_flagged():
    report = detect_bleed(_bled())
    assert report.suspected is True
    assert report.duplicate_pairs == 12
    assert report.fraction == 1.0
    assert len(report.examples) == 3, "an operator needs to SEE a pair to judge the claim"


# --- and stays quiet on real two-sided conversation -----------------------------------------


@pytest.mark.parametrize("session_id", ["20260809T140347", "20260809T145730"])
def test_a_real_two_sided_call_is_not_flagged(session_id):
    """THE test. Two genuine recorded interviews, hundreds of interviewer
    segments each, real reflections and real shared vocabulary between two
    people discussing one subject. A detector that fires here is useless."""
    path = REPO_ROOT / "projects" / "_eval" / "sessions" / session_id / "transcript.jsonl"
    if not path.is_file():
        pytest.skip(f"the real call {session_id} is not in this checkout")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    segments = [r for r in records if r.get("kind") == "segment"]
    report = detect_bleed(segments)
    assert report.comparable_interviewer_segments > 100, "not enough real speech to prove anything"
    assert report.suspected is False, (
        f"a real interview was reported as bleed: {report.percent}% "
        f"({report.duplicate_pairs}/{report.comparable_interviewer_segments}) -- examples "
        f"{report.examples}"
    )


def test_reflecting_a_question_back_is_not_bleed():
    """The interviewing technique, written out. Same subject, same nouns,
    rephrased, and a conversational turn apart -- which is what separates it."""
    segments = []
    for i in range(12):
        segments.append(_seg("interviewee", f"we went back to paper in week {i}", 20.0 * i))
        segments.append(
            _seg("interviewer", f"so you went back to paper, in week {i}?", 20.0 * i + 5.0)
        )
    assert detect_bleed(segments).suspected is False


def test_the_same_sentence_a_turn_later_is_not_bleed():
    """Simultaneity is the strongest discriminator, isolated here: identical
    text, one conversational turn apart. Bleed is the same sound reaching two
    devices at once; this is two people saying the same words."""
    segments = []
    for i in range(12):
        text = f"the export pipeline broke again in week {i} of the quarter"
        segments.append(_seg("interviewee", text, 30.0 * i))
        segments.append(_seg("interviewer", text, 30.0 * i + 8.0))
    assert detect_bleed(segments).suspected is False


def test_short_agreements_never_trigger_it():
    """"yeah", "OK", "right" collide across any two tracks in any conversation.
    Counting them would produce a confident bleed report for a healthy
    recording."""
    segments = []
    for i in range(20):
        segments.append(_seg("interviewee", "yeah", 5.0 * i))
        segments.append(_seg("interviewer", "yeah", 5.0 * i + 0.2))
    report = detect_bleed(segments)
    assert report.comparable_interviewer_segments == 0
    assert report.suspected is False


# --- and refuses to answer when it cannot ---------------------------------------------------


def test_a_handful_of_segments_is_not_enough_to_accuse():
    """Three of four is 75% and means nothing. A rate needs enough to rate --
    the same refusal `is_scorable` makes about a hypothesis with no rule."""
    segments = _bled(n=MIN_SEGMENTS_FOR_A_VERDICT - 1)
    report = detect_bleed(segments)
    assert report.fraction == 1.0, "every pair still matched"
    assert report.suspected is False, "but there were too few to call it"


def test_a_one_sided_recording_is_never_flagged():
    """`--only interviewee` means there is no interviewer track at all. A
    detector that divided by zero, or that read 'no matches' as a clean bill of
    health for a track that does not exist, would be answering a question
    nobody asked."""
    segments = [_seg("interviewee", f"a long enough sentence number {i}", i) for i in range(30)]
    report = detect_bleed(segments)
    assert report.comparable_interviewer_segments == 0
    assert report.suspected is False
    assert report.fraction == 0.0


def test_an_empty_transcript_is_not_a_crash_or_an_accusation():
    assert detect_bleed([]) == BleedReport(
        suspected=False, duplicate_pairs=0, comparable_interviewer_segments=0,
        comparable_interviewee_segments=0, fraction=0.0, examples=(),
    )


# --- the counting rule ----------------------------------------------------------------------


def test_one_microphone_segment_counts_once_however_it_was_chopped():
    """The two streams segment at different boundaries, so one sentence on the
    microphone can resemble four on the loopback. Counting each would report
    400% -- the per-fact-instead-of-per-interview failure, one module over."""
    text = "the export pipeline broke again in the third week of the quarter"
    segments = [
        _seg("interviewer", text, 10.0),
        _seg("interviewee", text[:30], 9.8),
        _seg("interviewee", text[10:50], 10.1),
        _seg("interviewee", text[20:], 10.3),
    ]
    report = detect_bleed(segments)
    assert report.duplicate_pairs == 1
    assert report.fraction <= 1.0


def test_segments_with_no_timestamp_are_skipped_not_matched_at_zero():
    """A missing `audio_end_s` defaulting to 0.0 would make every untimed
    segment simultaneous with every other one, which is the shape of a
    confident false positive."""
    segments = _bled()
    segments.append({"kind": "segment", "speaker": "interviewer", "text": "no timestamp at all"})
    before = detect_bleed(_bled())
    after = detect_bleed(segments)
    assert after.comparable_interviewer_segments == before.comparable_interviewer_segments


def test_the_report_publishes_enough_to_argue_with():
    """Every threshold here is a judgement call. A reader who disagrees must be
    able to see the working without re-running anything -- the reason
    `interview_summary` puts interviews and facts side by side."""
    report = detect_bleed(_bled())
    assert report.duplicate_pairs and report.comparable_interviewer_segments
    assert report.comparable_interviewee_segments
    assert 0.0 <= report.fraction <= 1.0
    assert report.percent == round(report.fraction * 100)


# --- and it reaches the operator ------------------------------------------------------------


def test_the_rendered_transcript_says_so_above_the_transcript(tmp_path):
    """A footer is read after the transcript has already been believed."""
    from app.session.render import render_session

    session_dir = tmp_path / "20260812T215237"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps({"started_at": "2026-08-12T21:52:37+03:00"}), encoding="utf-8"
    )
    lines = [json.dumps({**s, "seq": i}) for i, s in enumerate(_bled())]
    (session_dir / "transcript.jsonl").write_text("\n".join(lines), encoding="utf-8")

    document = render_session(session_dir, context_dir=None)
    assert "appear to be hearing each other" in document
    assert "SPEAKERS" in document, "name the cause -- the operator can act on that"
    assert "survey coverage" in document, "name the consequence that is not recoverable"
    body = document.index("appear to be hearing each other")
    assert body < len(document) // 2, "the warning must sit above the transcript, not under it"


def test_a_clean_render_carries_no_warning(tmp_path):
    """The vacuous-pass guard: a renderer that always printed the banner would
    satisfy the test above."""
    from app.session.render import render_session

    session_dir = tmp_path / "20260812T220000"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(
        json.dumps({"started_at": "2026-08-12T22:00:00+03:00"}), encoding="utf-8"
    )
    segments = [
        _seg("interviewee", f"they rebuilt the export by hand in week {i}", 20.0 * i)
        for i in range(12)
    ]
    lines = [json.dumps({**s, "seq": i}) for i, s in enumerate(segments)]
    (session_dir / "transcript.jsonl").write_text("\n".join(lines), encoding="utf-8")

    assert "hearing each other" not in render_session(session_dir, context_dir=None)
