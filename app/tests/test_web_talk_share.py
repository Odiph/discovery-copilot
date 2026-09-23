"""The talk-share meter as the hub actually serves it.

`test_talk_share.py` covers the arithmetic. This file covers the two things
that can only go wrong at the boundary: that the meter's speaker keys are the
SAME strings the transcript below it is labelled with, and that an interview
with nothing to measure still arrives as a present, explicit absence rather
than a missing key.

The second one is the common path, exactly as it is for the participant
label: the meter is new, and every interview already on disk was recorded
before it existed. It reads them fine -- the words were always there -- but a
project with no readable transcript is still the shape this has to survive.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import STATIC_REVIEW_DIR
from app.web.review import SPEAKER_LABELS, create_review_app


def _project(tmp_path, *, session="S1", lines=()):
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    session_dir = root / "sessions" / session
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    return root


def _segment(seq, speaker, text):
    return {"kind": "segment", "seq": seq, "speaker": speaker, "text": text}


def _review(tmp_path, session="S1"):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    client = TestClient(app, base_url="http://127.0.0.1")
    response = client.get("/api/review")
    assert response.status_code == 200
    return response.json()


def test_the_payload_carries_the_meter(tmp_path):
    _project(tmp_path, lines=[
        _segment(0, "interviewer", "one two three four five six"),
        _segment(1, "interviewee", "seven eight nine ten"),
    ])
    meter = _review(tmp_path)["talk_share"]
    assert meter["measurable"] is True
    assert meter["total_words"] == 10


def test_the_meters_speaker_keys_match_the_transcript_labels(tmp_path):
    """A share attributed to `interviewer` sitting above turns labelled `you`
    invites the reader to wonder whether they are the same person. The meter is
    computed from the projected transcript precisely so they cannot diverge."""
    _project(tmp_path, lines=[
        _segment(0, "interviewer", "a b c"),
        _segment(1, "interviewee", "d"),
    ])
    review = _review(tmp_path)
    transcript_speakers = {turn["speaker"] for turn in review["transcript"]}
    assert set(review["talk_share"]["speakers"]) == transcript_speakers


def test_the_client_is_told_which_row_is_the_interviewer(tmp_path):
    """Sent rather than guessed. `SPEAKER_LABELS` belongs to the server, and a
    client that had hardcoded `you` would silently stop emphasising anything
    the day that label changed."""
    _project(tmp_path, lines=[_segment(0, "interviewer", "a b")])
    meter = _review(tmp_path)["talk_share"]
    assert meter["interviewer_key"] == SPEAKER_LABELS["interviewer"]
    assert meter["interviewer_key"] in meter["speakers"]


def test_the_interviewers_share_is_the_number_this_was_built_for(tmp_path):
    """~65% in both real calls, and nothing computed it. Six of ten words."""
    _project(tmp_path, lines=[
        _segment(0, "interviewer", "one two three four five six"),
        _segment(1, "interviewee", "seven eight nine ten"),
    ])
    meter = _review(tmp_path)["talk_share"]
    assert meter["speakers"][meter["interviewer_key"]]["share"] == 0.6


def test_an_interview_with_no_speech_is_a_present_absence(tmp_path):
    """The key is never omitted. A missing key arrives in the client as
    `undefined`, which is how a page ends up rendering a meter with no bars and
    no explanation -- indistinguishable from a meter that failed to load."""
    _project(tmp_path, lines=[])
    review = _review(tmp_path)
    assert "talk_share" in review
    assert review["talk_share"]["measurable"] is False
    assert review["talk_share"]["speakers"] == {}


def test_no_interview_selected_still_carries_the_key(tmp_path):
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    (root / "sessions").mkdir(parents=True)
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=None, repo_root=tmp_path
    )
    review = TestClient(app, base_url="http://127.0.0.1").get("/api/review").json()
    assert review["talk_share"]["measurable"] is False


def test_an_unreadable_transcript_does_not_take_the_page_down(tmp_path):
    """`_transcript` answers `[]` for an unreadable session, so the meter
    inherits that degrade rather than needing its own. Asserted here so the
    inheritance is a tested property and not an accident of ordering."""
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    session_dir = root / "sessions" / "S1"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("{not json at all\n", encoding="utf-8")
    review = _review(tmp_path)
    assert review["talk_share"]["measurable"] is False
