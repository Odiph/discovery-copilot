"""Tests for `python -m app critique-questions` (`app/main.py`).

The ordering is the contract: everything that can fail for free must fail
BEFORE anything reaches the vendor. A user whose session has no interviewer
speech, or whose path is wrong, should not pay for a call whose result would be
discarded -- the same rule `run_extract` follows, and for the same reason.
"""

from __future__ import annotations

import json

import pytest

from app import main as main_mod
from app.coach.critic import BAND_CLEAN, BAND_CLEAR_VIOLATION, Judgment


def _segment(seq, speaker, text, end_s):
    return {"kind": "segment", "seq": seq, "speaker": speaker, "text": text,
            "audio_end_s": end_s, "audio_start_s": end_s, "ready_wall_s": end_s,
            "latency_s": 0.1, "detected_language": "en", "engine": "test",
            "written_at": "2026-08-14T00:00:00Z"}


@pytest.fixture
def session(tmp_path):
    """A closed session with two interviewer turns and one answer."""
    session_dir = tmp_path / "sessions" / "20260813T213559"
    session_dir.mkdir(parents=True)
    records = [
        _segment(0, "interviewer", "Walk me through the last time you sat", 26.64),
        _segment(1, "interviewer", "down to play, from the machine on.", 30.96),
        _segment(2, "interviewee", "I turn it on and then I wait a while", 60.0),
        _segment(3, "interviewer", "How long is that wait, usually?", 71.4),
    ]
    (session_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    (session_dir / "session.json").write_text(
        json.dumps({"closed_at": "2026-08-13T22:00:00Z"}), encoding="utf-8"
    )
    return session_dir


def _judgment(index, band=BAND_CLEAN, leading=False):
    return Judgment(index=index, is_leading=leading,
                    is_hypothetical_or_future_tense=False, is_pitch=False,
                    is_opinion_seeking=False, is_confirmation_only=False, is_generic=False, is_off_topic=False,
                    band=band, rationale="because")


@pytest.fixture
def no_vendor(monkeypatch):
    """Fails loudly if anything reaches a client. Every test that should not
    make a call installs this and asserts nothing was built."""
    calls = []

    def _build_client():
        calls.append("built")
        raise AssertionError("a client was built when none should have been")

    monkeypatch.setattr(main_mod.reasoning_client, "build_client", _build_client)
    return calls


@pytest.fixture
def stub_vendor(monkeypatch):
    """A client that is never really used, and a critique that answers clean."""
    monkeypatch.setattr(main_mod.reasoning_client, "build_client", lambda: object())
    monkeypatch.setattr(main_mod.reasoning_client, "resolve_model", lambda: "test-model")

    def _run_critique(client, *, candidates, hypotheses, trigger_text, model, max_tokens, timeout_s):
        return [_judgment(c.index) for c in candidates]

    monkeypatch.setattr(
        main_mod.coach_asked_critique.critic_mod, "run_critique", _run_critique
    )
    return _run_critique


# --- nothing reaches the vendor that does not have to --------------------------


def test_a_missing_session_never_builds_a_client(tmp_path, no_vendor, capsys):
    code = main_mod.main(["critique-questions", str(tmp_path / "nope")])
    assert code == 1
    assert no_vendor == []
    assert "no session directory found" in capsys.readouterr().err


def test_a_session_with_no_interviewer_speech_never_builds_a_client(
    tmp_path, no_vendor, capsys
):
    """An interview where only the interviewee was captured is a real and
    common failure -- five of the real recordings so far were misconfigured
    one way or another. It must cost nothing."""
    session_dir = tmp_path / "sessions" / "s"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text(
        json.dumps(_segment(0, "interviewee", "I turn it on and wait", 10.0)) + "\n",
        encoding="utf-8",
    )
    code = main_mod.main(["critique-questions", str(session_dir)])
    assert code == 1
    assert no_vendor == []
    assert "nothing written, nothing called" in capsys.readouterr().err


# --- the happy path -------------------------------------------------------------


def test_it_writes_the_artifact_next_to_the_transcript(session, stub_vendor, capsys):
    code = main_mod.main(["critique-questions", str(session)])
    assert code == 0
    written = session / main_mod.CRITIQUE_DOC_FILENAME
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["session_id"] == "20260813T213559"
    assert payload["summary"]["turns"] == 2
    assert payload["summary"]["judged"] == 2


def test_the_split_question_is_judged_as_one_turn(session, stub_vendor):
    """End to end: two fragments in, one turn judged."""
    main_mod.main(["critique-questions", str(session)])
    payload = json.loads(
        (session / main_mod.CRITIQUE_DOC_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["turns"][0]["seqs"] == [0, 1]
    assert payload["turns"][0]["text"].startswith("Walk me through the last time you sat down")


def test_the_calibration_caveat_is_printed_on_every_run(session, stub_vendor, capsys):
    """A percentage is exactly the kind of number that gets quoted without its
    caveat, so the caveat rides with it rather than living in a docstring."""
    main_mod.main(["critique-questions", str(session)])
    out = capsys.readouterr().out
    assert "calibrated on PROPOSED questions" in out


def test_an_unjudged_turn_makes_the_command_exit_nonzero(session, monkeypatch, capsys):
    """Partial results are written, but the exit code says they are partial --
    a script that treats this as success would file an unreachable critic
    under 'the interviewer did fine'."""
    monkeypatch.setattr(main_mod.reasoning_client, "build_client", lambda: object())
    monkeypatch.setattr(main_mod.reasoning_client, "resolve_model", lambda: "test-model")

    def _failing(client, *, candidates, hypotheses, trigger_text, model, max_tokens, timeout_s):
        raise main_mod.coach_asked_critique.CritiqueFailure(reason="timeout")

    monkeypatch.setattr(
        main_mod.coach_asked_critique.critic_mod, "run_critique", _failing
    )
    code = main_mod.main(["critique-questions", str(session)])
    assert code == 1
    payload = json.loads(
        (session / main_mod.CRITIQUE_DOC_FILENAME).read_text(encoding="utf-8")
    )
    assert payload["summary"]["unjudged"] == 2
    assert payload["summary"]["clean_rate"] is None
    assert "clean_rate=n/a" in capsys.readouterr().out


def test_min_words_is_actually_plumbed_through(session, stub_vendor):
    main_mod.main(["critique-questions", str(session), "--min-words", "99"])
    payload = json.loads(
        (session / main_mod.CRITIQUE_DOC_FILENAME).read_text(encoding="utf-8")
    )
    # Both real turns end in '.' or '?'; only the one with '?' survives a
    # floor nothing else can clear.
    assert payload["summary"]["turns"] == 1
    assert payload["turns"][0]["text"] == "How long is that wait, usually?"
