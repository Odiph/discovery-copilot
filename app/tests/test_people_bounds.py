"""How many PEOPLE the interviews represent, as against how many interviews.

The counting rule in `interview_summary.py` is keyed on `session_id` and is
right to be. What it cannot answer is the question the criteria are actually
written about: h1 in the demo project says "At least 3 of 6 interviewees", and
three sessions satisfy that only if they are three different people.

These tests are mostly about the RANGE refusing to collapse into a single
number while any interview is unlabelled. That is the whole feature; a version
that quietly reported `fewest` as "the" answer would be more convenient and
would let an entirely unlabelled corpus satisfy any criterion it liked.
"""

from __future__ import annotations

import pytest

from app.coach.interview_summary import PeopleBounds, people_bounds


def test_no_interviews_is_no_people():
    bounds = people_bounds({})
    assert bounds.sessions == 0
    assert bounds.fewest == 0
    assert bounds.most == 0
    assert bounds.is_certain is True


def test_every_interview_labelled_gives_one_certain_number():
    bounds = people_bounds({"S1": "Dana", "S2": "Roi", "S3": "Mor"})
    assert bounds.fewest == 3
    assert bounds.most == 3
    assert bounds.is_certain is True


def test_the_same_person_twice_is_one_person():
    """The case that makes sessions and people disagree, and the reason this
    exists at all. Three interviews, two people -- a criterion reading "at
    least 3 of 6 interviewees" is NOT satisfied by this."""
    bounds = people_bounds({"S1": "Dana", "S2": "Roi", "S3": "Dana"})
    assert bounds.sessions == 3
    assert bounds.named == 2
    assert bounds.fewest == 2
    assert bounds.most == 2
    assert bounds.is_certain is True


def test_unlabelled_interviews_widen_the_range_rather_than_being_guessed():
    bounds = people_bounds({"S1": "Dana", "S2": None, "S3": None})
    assert bounds.fewest == 1, "an unlabelled call may have been Dana again"
    assert bounds.most == 3, "or it may have been two people nobody named"
    assert bounds.is_certain is False


def test_an_entirely_unlabelled_corpus_is_at_least_one_person():
    """NOT zero. Zero people would be a claim that the interviews did not
    happen. This is the state every real recording on disk is in."""
    bounds = people_bounds({"S1": None, "S2": None, "S3": None})
    assert bounds.fewest == 1
    assert bounds.most == 3
    assert bounds.is_certain is False


def test_certainty_requires_every_interview_to_be_labelled():
    assert people_bounds({"S1": "Dana", "S2": None}).is_certain is False
    assert people_bounds({"S1": "Dana", "S2": "Roi"}).is_certain is True


@pytest.mark.parametrize("blank", [None, "", "   ", "\t", 7, [], {"name": "x"}])
def test_anything_that_is_not_a_name_counts_as_unlabelled(blank):
    """`read_participant` already answers `None` for every failure it can meet.
    A second interpretation of "not a name" here would be a place for the two
    to disagree, and the disagreement would show up as a person appearing or
    vanishing from a count."""
    bounds = people_bounds({"S1": "Dana", "S2": blank})
    assert bounds.named == 1
    assert bounds.unlabelled == 1


def test_a_trailing_space_does_not_manufacture_a_second_person():
    bounds = people_bounds({"S1": "Dana", "S2": "Dana "})
    assert bounds.named == 1
    assert bounds.is_certain is True


def test_case_is_deliberately_not_folded():
    """"dana" and "Dana" may be one person, and may not. Silently merging two
    names on a guess is a worse failure than a range one wider than it needed
    to be -- the operator can see both rows in the picker and fix it; nothing
    in here can. This test pins the decision so a later "tidy-up" has to argue
    with it rather than quietly change an answer."""
    bounds = people_bounds({"S1": "dana", "S2": "Dana"})
    assert bounds.named == 2


def test_no_label_is_ever_returned_or_stored():
    """A participant label is a real person's name. This module sits outside
    gate 20's pinned set for verbatim speech and a name deserves the same
    treatment. Every field must be an integer."""
    bounds = people_bounds({"S1": "Dana Levy", "S2": None})
    for value in vars(bounds).values():
        assert isinstance(value, int), f"a non-integer escaped: {value!r}"
    assert "Dana" not in repr(bounds)


def test_the_bounds_are_ordered_and_never_exceed_the_interviews():
    """A person cannot appear without an interview, so `most` can never exceed
    `sessions`, and `fewest` can never exceed `most`. Property-checked over a
    spread of shapes rather than asserted on one."""
    shapes = [
        {}, {"S1": None}, {"S1": "a"},
        {"S1": "a", "S2": None}, {"S1": "a", "S2": "a", "S3": None},
        {"S1": "a", "S2": "b", "S3": "c"}, {"S1": None, "S2": None},
    ]
    for shape in shapes:
        bounds = people_bounds(shape)
        assert bounds.fewest <= bounds.most, shape
        assert bounds.most <= bounds.sessions, shape
        # `<=`, NOT `==`. `named` counts DISTINCT labels, so two interviews
        # with the same person contribute one. Equality held for every shape I
        # wrote first and is still false in general -- it fails exactly on the
        # repeat case, which is the one this whole feature exists for. Pinned
        # as the inequality it actually is, with the equality case asserted
        # separately below where it is genuinely guaranteed.
        assert bounds.named + bounds.unlabelled <= bounds.sessions, shape
        if bounds.named == len(shape) - bounds.unlabelled:
            assert bounds.named + bounds.unlabelled == bounds.sessions, shape


def test_the_dataclass_is_frozen():
    bounds = PeopleBounds(sessions=1, named=1, unlabelled=0)
    with pytest.raises(Exception):
        bounds.sessions = 5


# --- as the hub serves it -----------------------------------------------------------------------

import json  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import STATIC_REVIEW_DIR  # noqa: E402
from app.web.review import create_review_app  # noqa: E402


def _project(tmp_path, participants):
    """`participants` maps session id to a label, or to None for no
    `session.json` at all -- which is the shape every real recording is in."""
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    for session_id, label in participants.items():
        session_dir = root / "sessions" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "transcript.jsonl").write_text("", encoding="utf-8")
        if label is not None:
            (session_dir / "session.json").write_text(
                json.dumps({"participant": label}), encoding="utf-8"
            )
    return root


def _people(tmp_path):
    app = create_review_app(
        static_dir=STATIC_REVIEW_DIR, project="acme", session_id=None, repo_root=tmp_path
    )
    return TestClient(app, base_url="http://127.0.0.1").get("/api/review").json()["people"]


def test_the_payload_carries_the_range(tmp_path):
    _project(tmp_path, {"S1": "Dana", "S2": None, "S3": None})
    people = _people(tmp_path)
    assert people["sessions"] == 3
    assert people["fewest"] == 1
    assert people["most"] == 3
    assert people["certain"] is False


def test_a_fully_labelled_project_reports_one_certain_number(tmp_path):
    _project(tmp_path, {"S1": "Dana", "S2": "Roi"})
    people = _people(tmp_path)
    assert (people["fewest"], people["most"]) == (2, 2)
    assert people["certain"] is True


def test_two_interviews_with_one_person_do_not_read_as_two_people(tmp_path):
    """The whole point. A criterion reading "at least 2 of 2 interviewees" is
    NOT satisfied by interviewing the same person twice, and until this landed
    nothing on the page could have said so."""
    _project(tmp_path, {"S1": "Dana", "S2": "Dana"})
    people = _people(tmp_path)
    assert people["sessions"] == 2
    assert people["most"] == 1


def test_no_labels_anywhere_is_the_common_path_and_still_answers(tmp_path):
    _project(tmp_path, {"S1": None, "S2": None, "S3": None, "S4": None})
    people = _people(tmp_path)
    assert people["fewest"] == 1
    assert people["most"] == 4
    assert people["certain"] is False


def test_a_project_with_no_sessions_directory_is_an_absence_not_a_crash(tmp_path):
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    people = _people(tmp_path)
    assert people["sessions"] == 0
    assert people["most"] == 0


def test_no_participant_name_ever_reaches_the_people_block(tmp_path):
    """A name is a real person's. The block is counts only, and this asserts it
    over the serialized JSON rather than over the dataclass, because the wire is
    where a leak would actually matter."""
    _project(tmp_path, {"S1": "Dana Levy", "S2": "Roi Ben-Ami"})
    assert "Dana" not in json.dumps(_people(tmp_path))
    assert "Roi" not in json.dumps(_people(tmp_path))
