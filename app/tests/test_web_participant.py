"""WHO was interviewed -- the label, its absence, and the two things a save
must never cost.

The absence is tested FIRST and most, deliberately. Every real recording this
product has made carries no participant label, so `None` is not an edge case
here: it is the state the page will be in almost every time anyone opens it,
and a feature that renders it as a failure would be worse than no feature.

The two costs a save must never impose are the reason this file is longer than
the feature:

- **A save must not touch `transcript.jsonl`.** The obvious implementation
  reaches for `SessionStore.update_metadata`, and `SessionStore`'s CONSTRUCTOR
  appends a `session-resumed` event to the transcript when the directory
  already holds one. A label typed on the hub three days later would then claim
  the interview restarted. The test asserts the transcript is byte-identical,
  not merely that it still parses.
- **A save must not cost provenance.** `session.json` holds the engine, both
  device names, both native sample rates, the stored `context_survey` the
  coverage report replays against, and the latency summary -- none of it
  recomputable. A merge that dropped any of it in exchange for a name would be
  a bad trade made silently.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import STATIC_REVIEW_DIR, _session_metadata, build_arg_parser
from app.session.participant import (
    MAX_PARTICIPANT_LENGTH,
    normalize_participant,
    read_participant,
)
from app.web.participant import PARTICIPANT_ROUTE, save_participant
from app.web.review import create_review_app, sessions_at

# Everything `start` writes into `session.json` that a participant save must
# leave standing. Held as a fixture rather than asserted field by field so a
# future key added to the metadata is covered by the same test.
_PROVENANCE = {
    "engine": "an-engine",
    "target_sample_rate": 16000,
    "interviewee_device": "Loopback (Realtek)",
    "interviewee_native_sample_rate": 48000,
    "interviewer_device": "Microphone (Realtek)",
    "interviewer_native_sample_rate": 44100,
    "started_at": "2026-08-13T19:25:00.000+03:00",
    "closed_at": "2026-08-13T20:10:00.000+03:00",
    "context_survey": [{"id": "q1", "order": 0, "text": "what did you try first?"}],
    "latency_summary": {"interviewee": {"count": 3, "median_s": 0.9}},
}

_TRANSCRIPT_LINE = json.dumps(
    {"kind": "segment", "seq": 0, "speaker": "interviewee", "text": "we went back to paper"}
) + "\n"


def _project(tmp_path, *, sessions=("S1",), metadata=None):
    """A project node with real session directories. `metadata=None` means NO
    `session.json` at all, which is a shape that exists on disk today."""
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    for session_id in sessions:
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text(_TRANSCRIPT_LINE, encoding="utf-8")
        if metadata is not None:
            (session_dir / "session.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
    return root


def _client(tmp_path, session=None):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=session, repo_root=tmp_path
    )
    # Loopback base_url, because the app's own middleware refuses anything else
    # outright -- see `test_web_review.py::test_a_non_loopback_host_header_is_refused`.
    return TestClient(app, base_url="http://127.0.0.1")


# --- the absent label, which is the common path ------------------------------------------------


def test_a_session_with_no_metadata_file_at_all_reads_as_nobody_named(tmp_path):
    # Not every session directory has a `session.json`. Raising here would make
    # the picker unable to list an interview that is otherwise perfectly
    # readable.
    root = _project(tmp_path, metadata=None)
    assert read_participant(root / "sessions" / "S1") is None


def test_a_session_recorded_before_this_field_existed_reads_as_nobody_named(tmp_path):
    # The state of every real recording on disk today.
    root = _project(tmp_path, metadata=_PROVENANCE)
    assert read_participant(root / "sessions" / "S1") is None


def test_an_unreadable_session_json_reads_as_nobody_named_rather_than_raising(tmp_path):
    # One hand-edited metadata file must not take the picker down for every
    # other interview in the project.
    root = _project(tmp_path, metadata=_PROVENANCE)
    (root / "sessions" / "S1" / "session.json").write_text("{not json", encoding="utf-8")
    assert read_participant(root / "sessions" / "S1") is None


def test_a_participant_key_holding_something_other_than_a_string_reads_as_nobody_named(tmp_path):
    # A hand-edited `session.json` with a number, an object or a list under the
    # key. The honest answer to "who was interviewed" is still "nothing on disk
    # says", and rendering `{'name': 'Dana'}` onto the page would be worse.
    for stored in (7, {"name": "Dana"}, ["Dana"], None, True):
        root = _project(tmp_path / str(id(stored)), metadata={**_PROVENANCE, "participant": stored})
        assert read_participant(root / "sessions" / "S1") is None, stored


def test_a_whitespace_only_stored_label_reads_as_nobody_named(tmp_path):
    root = _project(tmp_path, metadata={**_PROVENANCE, "participant": "   "})
    assert read_participant(root / "sessions" / "S1") is None


def test_a_hand_typed_label_is_stripped_on_the_way_out(tmp_path):
    # It never passed `normalize_participant`, and " Dana " on screen reads as
    # a rendering bug rather than as what someone typed.
    root = _project(tmp_path, metadata={**_PROVENANCE, "participant": "  Dana (ops lead)  "})
    assert read_participant(root / "sessions" / "S1") == "Dana (ops lead)"


# --- validation ---------------------------------------------------------------------------------


def test_an_absent_or_blank_label_is_not_an_error(tmp_path):
    # `(None, None)` is a legitimate answer, and a caller that branches on the
    # label instead of the error refuses the normal case.
    for value in (None, "", "   ", "\t\n "):
        assert normalize_participant(value) == (None, None), value


def test_a_label_is_stripped_but_its_interior_is_left_alone():
    label, error = normalize_participant("  Dana Levi — ops lead, mid-size clinic  ")
    assert error is None
    assert label == "Dana Levi — ops lead, mid-size clinic"


def test_a_label_longer_than_the_cap_is_refused_and_says_both_numbers():
    label, error = normalize_participant("x" * (MAX_PARTICIPANT_LENGTH + 1))
    assert label is None
    assert str(MAX_PARTICIPANT_LENGTH) in error and str(MAX_PARTICIPANT_LENGTH + 1) in error


def test_a_label_exactly_at_the_cap_is_accepted():
    # The boundary in the direction that matters: an off-by-one here refuses a
    # value the message just said was legal.
    label, error = normalize_participant("x" * MAX_PARTICIPANT_LENGTH)
    assert error is None and label == "x" * MAX_PARTICIPANT_LENGTH


def test_a_pasted_multi_line_value_is_refused_rather_than_silently_truncated():
    # Quietly keeping the first line would store a value the operator never
    # read back.
    label, error = normalize_participant("Dana\nops lead")
    assert label is None and "single line" in error


def test_a_non_string_label_is_refused_by_naming_its_type():
    label, error = normalize_participant(17)
    assert label is None and "int" in error


# --- saving --------------------------------------------------------------------------------------


def test_a_saved_label_reads_back(tmp_path):
    root = _project(tmp_path, metadata=_PROVENANCE)
    stored, error = save_participant(root / "sessions", "S1", "  Dana (ops lead) ")
    assert error is None and stored == "Dana (ops lead)"
    assert read_participant(root / "sessions" / "S1") == "Dana (ops lead)"


def test_a_save_leaves_every_other_metadata_key_exactly_as_it_was(tmp_path):
    # The engine, both devices, both native rates, the stored survey the
    # coverage report replays against, and the latency summary -- none of it
    # recomputable once the interview is over.
    root = _project(tmp_path, metadata=_PROVENANCE)
    save_participant(root / "sessions", "S1", "Dana")
    after = json.loads((root / "sessions" / "S1" / "session.json").read_text(encoding="utf-8"))
    assert {key: after[key] for key in _PROVENANCE} == _PROVENANCE


def test_a_save_writes_nothing_into_the_transcript(tmp_path):
    # The trap this feature is one wrong import away from. `SessionStore`'s
    # CONSTRUCTOR appends a `session-resumed` event when the directory already
    # holds a transcript, so `SessionStore(dir).update_metadata(...)` -- the
    # method whose name reads like the right tool -- would put a record into
    # the transcript claiming the interview restarted, days after it ended.
    #
    # Byte-identical, not merely "still parses": an appended event parses fine.
    root = _project(tmp_path, metadata=_PROVENANCE)
    before = (root / "sessions" / "S1" / "transcript.jsonl").read_bytes()
    save_participant(root / "sessions", "S1", "Dana")
    assert (root / "sessions" / "S1" / "transcript.jsonl").read_bytes() == before


def test_a_session_with_no_metadata_file_gets_one_holding_just_the_label(tmp_path):
    root = _project(tmp_path, metadata=None)
    stored, error = save_participant(root / "sessions", "S1", "Dana")
    assert error is None and stored == "Dana"
    assert json.loads(
        (root / "sessions" / "S1" / "session.json").read_text(encoding="utf-8")
    ) == {"participant": "Dana"}


def test_clearing_removes_the_key_rather_than_storing_a_null(tmp_path):
    # A session recorded before this field existed and a session someone
    # deliberately blanked are the same situation -- nobody knows who it was --
    # and a stored `null` would assert a distinction no byte on disk supports.
    root = _project(tmp_path, metadata={**_PROVENANCE, "participant": "Dana"})
    stored, error = save_participant(root / "sessions", "S1", "   ")
    assert error is None and stored is None
    after = json.loads((root / "sessions" / "S1" / "session.json").read_text(encoding="utf-8"))
    assert "participant" not in after
    assert {key: after[key] for key in _PROVENANCE} == _PROVENANCE


def test_a_save_onto_an_unreadable_session_json_is_refused_and_changes_nothing(tmp_path):
    # A file we cannot read is a file we refuse to replace -- `accept.py`'s rule
    # applied to a different artifact. The tolerant alternative would replace
    # the engine, the devices, the survey and the latency summary with one key.
    root = _project(tmp_path, metadata=_PROVENANCE)
    path = root / "sessions" / "S1" / "session.json"
    path.write_text('{"engine": "an-engine",,,', encoding="utf-8")
    before = path.read_bytes()
    stored, error = save_participant(root / "sessions", "S1", "Dana")
    assert stored is None and "could not be read" in error
    assert path.read_bytes() == before


def test_a_save_onto_metadata_that_is_not_an_object_is_refused(tmp_path):
    root = _project(tmp_path, metadata=_PROVENANCE)
    path = root / "sessions" / "S1" / "session.json"
    path.write_text('["not", "an", "object"]', encoding="utf-8")
    stored, error = save_participant(root / "sessions", "S1", "Dana")
    assert stored is None and "JSON object" in error
    assert json.loads(path.read_text(encoding="utf-8")) == ["not", "an", "object"]


def test_a_session_id_that_is_not_one_of_this_projects_interviews_is_refused(tmp_path):
    # MEMBERSHIP of the real listing, never a pattern: `sessions_root / ".."`
    # resolves outside the project entirely.
    root = _project(tmp_path, metadata=_PROVENANCE)
    for requested in ("../../../etc", "s1", "S2", ""):
        stored, error = save_participant(root / "sessions", requested, "Dana")
        assert stored is None and "not one of this project's interviews" in error, requested


def test_the_refusal_does_not_echo_the_requested_session_back(tmp_path):
    # It is attacker-controlled text heading for a page that renders it.
    root = _project(tmp_path, metadata=_PROVENANCE)
    _, error = save_participant(root / "sessions", "<script>alert(1)</script>", "Dana")
    assert "<script>" not in error


def test_saving_with_no_interview_selected_is_refused_without_writing(tmp_path):
    root = _project(tmp_path, metadata=_PROVENANCE)
    stored, error = save_participant(root / "sessions", None, "Dana")
    assert stored is None and "no interview is selected" in error


def test_an_invalid_label_is_refused_before_anything_is_written(tmp_path):
    # Validated, then written, never the other order.
    root = _project(tmp_path, metadata=_PROVENANCE)
    path = root / "sessions" / "S1" / "session.json"
    before = path.read_bytes()
    stored, error = save_participant(root / "sessions", "S1", "x" * 500)
    assert stored is None and error is not None
    assert path.read_bytes() == before


# --- the routes ------------------------------------------------------------------------------------


def test_the_route_answers_null_for_an_interview_with_no_label(tmp_path):
    # A successful read, not a 404 and not an error. "Nobody is named" is the
    # ordinary reply for every recording made before this field existed.
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        response = client.get(PARTICIPANT_ROUTE)
    assert response.status_code == 200
    assert response.json() == {"session_id": "S1", "participant": None}


def test_the_route_round_trips_a_label(tmp_path):
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        saved = client.post(PARTICIPANT_ROUTE, json={"participant": "Dana (ops lead)"})
        assert saved.status_code == 200 and saved.json()["participant"] == "Dana (ops lead)"
        assert client.get(PARTICIPANT_ROUTE).json()["participant"] == "Dana (ops lead)"


def test_the_route_follows_the_pickers_session_rather_than_the_command_lines(tmp_path):
    # The hub's picker chooses the interview; a save that ignored it would
    # attach a name to a different call than the one on screen.
    _project(tmp_path, sessions=("S1", "S2"), metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        client.post(PARTICIPANT_ROUTE, json={"session": "S2", "participant": "Dana"})
        assert client.get(PARTICIPANT_ROUTE + "?session=S2").json()["participant"] == "Dana"
        assert client.get(PARTICIPANT_ROUTE + "?session=S1").json()["participant"] is None


def test_the_route_refuses_a_session_outside_the_projects_listing(tmp_path):
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        response = client.post(
            PARTICIPANT_ROUTE, json={"session": "../../secrets", "participant": "Dana"}
        )
    assert response.status_code == 400
    assert "not one of this project's interviews" in response.json()["error"]


def test_the_route_refuses_a_non_loopback_host_header(tmp_path):
    # The route's OWN guard, not the mounting app's -- a write must not depend
    # on a caller having remembered.
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        response = client.post(
            PARTICIPANT_ROUTE,
            json={"participant": "Dana"},
            headers={"host": "example.com"},
        )
    assert response.status_code == 403
    assert read_participant(tmp_path / "projects" / "acme" / "sessions" / "S1") is None


def test_the_route_refuses_a_body_that_is_not_an_object(tmp_path):
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        response = client.post(PARTICIPANT_ROUTE, json=["Dana"])
    assert response.status_code == 400


def test_the_label_never_reaches_a_log_line(tmp_path, caplog):
    # It is the name of a real person who sat for an interview. This repo keeps
    # transcript text out of every log line for the same reason (S7, T-02-05);
    # a name is not a softer case of that rule.
    _project(tmp_path, metadata=_PROVENANCE)
    with caplog.at_level("DEBUG"):
        with _client(tmp_path, session="S1") as client:
            client.post(PARTICIPANT_ROUTE, json={"participant": "Dana Levi"})
    assert "Dana" not in caplog.text


# --- what the hub shows ------------------------------------------------------------------------------


def test_the_review_payload_carries_the_label(tmp_path):
    _project(tmp_path, metadata={**_PROVENANCE, "participant": "Dana (ops lead)"})
    with _client(tmp_path, session="S1") as client:
        assert client.get("/api/review").json()["participant"] == "Dana (ops lead)"


def test_the_review_payload_carries_the_key_even_when_nobody_is_named(tmp_path):
    # Present-and-null, never absent. A missing key reads as `undefined` in the
    # client, which is how a page ends up rendering the empty string into the
    # place a name belongs and calling it a name.
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path, session="S1") as client:
        body = client.get("/api/review").json()
    assert "participant" in body and body["participant"] is None


def test_the_review_payload_carries_a_null_label_with_no_session_selected(tmp_path):
    _project(tmp_path, metadata=_PROVENANCE)
    with _client(tmp_path) as client:
        body = client.get("/api/review").json()
    assert body["session_id"] is None and body["participant"] is None


def test_the_picker_listing_carries_each_interviews_label(tmp_path):
    # Which interview is which, in the one list of them. The interviews with no
    # label carry `None` rather than being omitted or blanked out.
    root = _project(tmp_path, sessions=("S1", "S2"), metadata=_PROVENANCE)
    save_participant(root / "sessions", "S2", "Dana")
    assert {row["id"]: row["participant"] for row in sessions_at(root / "sessions")} == {
        "S1": None, "S2": "Dana",
    }


# --- recording it at `start` -------------------------------------------------------------------------


def _devices():
    class _Device:
        def __init__(self, name, rate):
            self.name, self.native_sample_rate = name, rate

    return {"interviewee": _Device("Loopback", 48000)}


def test_start_records_the_label_into_the_sessions_metadata():
    metadata = _session_metadata(
        _devices(), engine_name="an-engine", sample_rate=16000, participant="Dana (ops lead)"
    )
    assert metadata["participant"] == "Dana (ops lead)"


def test_start_omits_the_key_entirely_when_no_participant_was_given():
    # The common path, and the key is ABSENT rather than null for the same
    # reason clearing removes it: a stored `null` asserts a distinction no byte
    # on disk supports.
    metadata = _session_metadata(_devices(), engine_name="an-engine", sample_rate=16000)
    assert "participant" not in metadata


def test_the_start_flag_normalizes_what_was_typed():
    args = build_arg_parser().parse_args(["start", "--participant", "  Dana (ops lead) "])
    assert args.participant == "Dana (ops lead)"


def test_the_start_flag_defaults_to_nobody_named():
    assert build_arg_parser().parse_args(["start"]).participant is None


def test_the_start_flag_refuses_an_invalid_label_before_a_device_is_opened():
    # At PARSE time, so nothing has been opened, no session directory has been
    # created, and the operator finds out before the interview rather than
    # after it.
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["start", "--participant", "x" * 500])
