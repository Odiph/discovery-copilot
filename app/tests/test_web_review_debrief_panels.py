"""The hub's three debrief panels: how you interviewed, and what was refused.

Three artifacts already existed on disk and none of them reached a screen:
`questions_critique.json` (written per session by `python -m app
critique-questions`), `context/rejections.json` (written by every extraction
since `rejection_store` existed), and the evidence/inference split, which was
already on the review payload and was drawn only as a line of text.

Every test here is about ONE property, because it is the property all three
panels can silently break:

    "not run yet" and "ran and found nothing" are different claims, and the
    page must never render them alike.

That is not a stylistic preference. Both of the new artifacts are produced by a
step the operator has to take deliberately -- one of them a paid model call --
so ABSENCE is the common state and the state most readers will see. A missing
critique drawn as a score is an interview nobody measured reported as the worst
one on disk; a missing rejection log drawn as a zero manufactures the very
finding (`~30% unlocatable-quote`) the log was written to make checkable.

The mirror-image trap is guarded here too, and it is the subtler one: a
rejection log that EXISTS holding no rows is a real, meaningful zero --
`replace_session_rejections` is called unconditionally, including with an empty
sequence, precisely so that "extracted and refused nothing" leaves a mark.
Reading that as "nothing is known" would throw away the answer the record was
built to keep.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.coach.asked_critique import AskedCritique, to_json_dict
from app.coach.asked_store import CRITIQUE_DOC_FILENAME
from app.coach.extract import Rejection
from app.coach import rejection_store
from app.web import review as review_mod
from app.web.review import build_review

_HYPOTHESES = [{"id": "h1", "statement": "onboarding is the blocker", "falsifier": "setup was easy"}]


def _node(tmp_path: Path, *, sessions=("s1",)) -> Path:
    context = tmp_path / "projects" / "acme" / "context"
    context.mkdir(parents=True, exist_ok=True)
    (context / "hypotheses.json").write_text(json.dumps(_HYPOTHESES), encoding="utf-8")
    for name in sessions:
        (tmp_path / "projects" / "acme" / "sessions" / name).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _context_dir(root: Path) -> Path:
    return root / "projects" / "acme" / "context"


def _write_critique(root: Path, session_id: str, payload) -> None:
    path = root / "projects" / "acme" / "sessions" / session_id / CRITIQUE_DOC_FILENAME
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


def _summary(**overrides) -> dict:
    base = {
        "turns": 20,
        "judged": 17,
        "unjudged": 3,
        "clean": 11,
        "clean_rate": 0.63,
        "marked_questions": 10,
        "marked_question_clean_rate": 0.71,
        "unmarked_clean_rate": 0.52,
        "leading": 9,
        "hypothetical_or_future_tense": 2,
        "pitch": 0,
        "opinion_seeking": 3,
        "confirmation_only": 5,
        "generic": 0,
    }
    base.update(overrides)
    return base


# --- the question critique: absence is not a score --------------------------------------


def test_a_critique_that_was_never_run_is_named_as_absent(tmp_path):
    root = _node(tmp_path)
    critique = build_review("acme", "s1", repo_root=root)["question_critique"]
    assert critique["status"] == "not-run"
    # `None`, never a zeroed summary object. A payload of zeroes here is the
    # shape a page draws a 0%-clean bar from.
    assert critique["summary"] is None


def test_a_critique_that_ran_and_one_that_never_ran_are_different_answers(tmp_path):
    # THE test for this panel, in the shape `test_web_review_extraction_status`
    # already uses for the corpus: both render "no flags to show", and they are
    # opposite situations.
    never = build_review("acme", "s1", repo_root=_node(tmp_path / "a"))["question_critique"]

    ran_root = _node(tmp_path / "b")
    _write_critique(ran_root, "s1", {"session_id": "s1", "summary": _summary(
        leading=0, hypothetical_or_future_tense=0, pitch=0,
        opinion_seeking=0, confirmation_only=0, clean=17, clean_rate=1.0,
    )})
    ran = build_review("acme", "s1", repo_root=ran_root)["question_critique"]

    assert never["status"] != ran["status"]
    assert ran["status"] == "read"
    assert ran["summary"]["clean_rate"] == 1.0
    assert ran["summary"]["flags"]["leading"] == 0


def test_the_summary_counts_reach_the_payload(tmp_path):
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary()})
    summary = build_review("acme", "s1", repo_root=root)["question_critique"]["summary"]
    assert summary["turns"] == 20
    assert summary["judged"] == 17 and summary["unjudged"] == 3
    assert summary["clean"] == 11
    assert summary["marked_questions"] == 10
    assert summary["marked_question_clean_rate"] == 0.71
    assert summary["unmarked_clean_rate"] == 0.52
    assert summary["flags"] == {
        "leading": 9, "hypothetical_or_future_tense": 2, "pitch": 0,
        "opinion_seeking": 3, "confirmation_only": 5, "generic": 0,
    }


def test_a_null_clean_rate_survives_as_null_rather_than_becoming_zero(tmp_path):
    """`AskedCritique.clean_rate` answers `None` when nothing could be judged,
    and that `None` is the whole reason the property exists: an interview the
    critic could not be reached for is neither a perfect score nor a zero one.
    Coercing it anywhere on the way to the page would file an unreachable
    critic with the worst interview on disk."""
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary(
        judged=0, unjudged=20, clean=0, clean_rate=None,
        marked_questions=0, marked_question_clean_rate=None, unmarked_clean_rate=None,
    )})
    summary = build_review("acme", "s1", repo_root=root)["question_critique"]["summary"]
    assert summary["clean_rate"] is None
    assert summary["marked_question_clean_rate"] is None
    assert summary["unmarked_clean_rate"] is None
    # And the reachability counts still arrive, because they are what the page
    # says instead of a rate.
    assert summary["judged"] == 0 and summary["unjudged"] == 20


def test_a_rate_outside_zero_to_one_is_refused_rather_than_drawn(tmp_path):
    # A hand-edited artifact. A rate of 5 becomes a bar five times the width of
    # its own track, which reads as an emphatic result rather than as garbage.
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary(clean_rate=5)})
    summary = build_review("acme", "s1", repo_root=root)["question_critique"]["summary"]
    assert summary["clean_rate"] is None


def test_a_boolean_is_not_read_as_a_count(tmp_path):
    # `True` is an `int` in Python and would arrive on the wire as the count 1.
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary(leading=True)})
    summary = build_review("acme", "s1", repo_root=root)["question_critique"]["summary"]
    assert summary["flags"]["leading"] is None


def test_a_critique_that_will_not_parse_is_never_reported_as_a_clean_interview(tmp_path):
    root = _node(tmp_path)
    _write_critique(root, "s1", "{ not json")
    critique = build_review("acme", "s1", repo_root=root)["question_critique"]
    assert critique["status"] == "unreadable"
    assert critique["error"] is not None
    assert critique["summary"] is None


def test_a_critique_file_with_no_summary_block_is_unreadable_not_empty(tmp_path):
    # A broken artifact, not an interview the critic found nothing in. Saying
    # the second is a confident sentence about something nothing on disk
    # supports.
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "turns": []})
    critique = build_review("acme", "s1", repo_root=root)["question_critique"]
    assert critique["status"] == "unreadable"


def test_no_interview_selected_is_its_own_state(tmp_path):
    root = _node(tmp_path)
    critique = build_review("acme", None, repo_root=root)["question_critique"]
    assert critique["status"] == "no-session"


def test_the_critique_panel_carries_no_verbatim_interviewer_speech(tmp_path):
    """The artifact holds every judged turn with its text and the critic's
    rationale. The panel draws counts, so none of that crosses onto the wire --
    the same rule the summary block keeps, and the reason only `summary` is
    read out of the file."""
    sentinel = "so you would say the export is the part that breaks, right?"
    root = _node(tmp_path)
    _write_critique(root, "s1", {
        "session_id": "s1",
        "summary": _summary(),
        "turns": [{"index": 0, "seqs": [4], "at_s": 1.0, "text": sentinel,
                   "failure_reason": None, "judgment": None}],
    })
    critique = build_review("acme", "s1", repo_root=root)["question_critique"]
    assert sentinel not in json.dumps(critique)


# --- the flag vocabulary, held to the module that writes it -----------------------------


def test_the_page_reads_every_flag_the_critique_actually_writes():
    """THE CROSSING TEST for this panel, and the reason it is written against
    `to_json_dict` rather than against a hand-typed list.

    `review.py` names six flags, `review.js` keys a label table on the same six
    by hand, and `asked_critique.to_json_dict` is what actually decides which
    keys exist in the file. Two lists that can agree with each other while
    disagreeing with the writer is exactly the shape the 2026-08-12 audit found
    in the extraction banner."""
    written = to_json_dict(AskedCritique(judged=()), session_id="s1")["summary"]
    missing = [name for name in review_mod.QUESTION_FLAGS if name not in written]
    assert not missing, (
        f"the hub reads flags the critique does not write: {missing} -- every one of them "
        "renders as an unreadable row on a page that otherwise looks fine"
    )
    # And the other direction, for the counts the panel is drawn from.
    for field in (
        "turns", "judged", "unjudged", "clean", "clean_rate",
        "marked_questions", "marked_question_clean_rate", "unmarked_clean_rate",
    ):
        assert field in written, f"the hub reads {field!r}, which the critique does not write"


def test_the_flags_that_cannot_be_judged_here_are_named_on_the_wire():
    """`generic` is 0 in this artifact BY CONSTRUCTION: `critique_turns` sends
    a BATCH of turns from different points in the interview and `is_generic` is
    judged against the single preceding utterance, so there is no trigger that
    could serve them and the prompt is told to answer "no" rather than guess.

    It travels on the wire rather than being hardcoded in the page, for the
    reason `talk_share.interviewer_key` does: which flags are unjudgeable is a
    property of the pipeline, and a client that guessed would go on printing
    "0" as a pass the day a trigger became available."""
    assert set(review_mod.QUESTION_FLAGS_NOT_JUDGED) <= set(review_mod.QUESTION_FLAGS)
    assert set(review_mod.QUESTION_FLAGS_NOT_JUDGED) == {"generic"}


def test_the_unjudgeable_flags_reach_the_client(tmp_path):
    root = _node(tmp_path)
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary()})
    summary = build_review("acme", "s1", repo_root=root)["question_critique"]["summary"]
    assert summary["flags_not_judged"] == ["generic"]


def test_the_critique_status_vocabulary_is_pinned_by_equality():
    # Equality, not membership, following the extraction banner: a new status is
    # a decision about how the panel reads, never a drift.
    assert set(review_mod.ALL_CRITIQUE_STATUSES) == {
        "no-session", "not-run", "unreadable", "read",
    }


def test_the_page_words_every_critique_status_the_server_can_emit():
    source = (Path(review_mod.__file__).parent.parent / "static" / "review" / "review.js").read_text(
        encoding="utf-8"
    )
    for status in review_mod.ALL_CRITIQUE_STATUSES:
        assert f'"{status}"' in source, (
            f"review.py can emit {status!r} and review.js has no sentence for it -- the panel "
            "would render nothing where a caveat belongs"
        )


def test_the_page_names_every_flag_the_server_sends():
    # The label table is hand-keyed; an unlabelled row is a bar with no name.
    source = (Path(review_mod.__file__).parent.parent / "static" / "review" / "review.js").read_text(
        encoding="utf-8"
    )
    for flag in review_mod.QUESTION_FLAGS:
        assert flag in source, f"{flag!r} reaches the page with no label"


# --- the rejection log: silence is not zero ---------------------------------------------


def test_a_project_with_no_rejection_log_is_silence_rather_than_a_zero(tmp_path):
    root = _node(tmp_path)
    rejections = build_review("acme", "s1", repo_root=root)["rejections"]
    assert rejections["status"] == "not-kept"
    assert rejections["reasons"] == {}
    # No per-interview block either: there is no log to have a slice of.
    assert rejections["this_interview"] is None


def test_a_log_that_exists_holding_nothing_is_a_real_zero(tmp_path):
    """The mirror image, and the reason `rejection_store` writes an empty slice
    at all: a run that accepted everything is the case that most needs to leave
    a mark, because it is otherwise indistinguishable from a run that never
    happened."""
    root = _node(tmp_path)
    rejection_store.replace_session_rejections(_context_dir(root), "s1", [])
    rejections = build_review("acme", "s1", repo_root=root)["rejections"]
    assert rejections["status"] == "kept"
    assert rejections["total"] == 0
    assert rejections["reasons"] == {}


def test_silence_and_a_real_zero_are_different_answers(tmp_path):
    silent = build_review("acme", "s1", repo_root=_node(tmp_path / "a"))["rejections"]

    zero_root = _node(tmp_path / "b")
    rejection_store.replace_session_rejections(_context_dir(zero_root), "s1", [])
    zero = build_review("acme", "s1", repo_root=zero_root)["rejections"]

    assert silent["status"] != zero["status"]
    assert (silent["total"], zero["total"]) == (0, 0), (
        "both are zero, which is precisely why the status has to carry the difference"
    )


def test_rejections_are_counted_by_reason(tmp_path):
    root = _node(tmp_path)
    rejection_store.replace_session_rejections(_context_dir(root), "s1", [
        Rejection(claim="a", reason="unlocatable-quote"),
        Rejection(claim="b", reason="unlocatable-quote"),
        Rejection(claim="c", reason="span-too-long"),
    ])
    rejections = build_review("acme", "s1", repo_root=root)["rejections"]
    assert rejections["total"] == 3
    # Descending, as `count_by_reason` orders them -- the panel draws them in
    # payload order, so the ordering is part of the answer rather than a detail
    # of the page.
    assert list(rejections["reasons"]) == ["unlocatable-quote", "span-too-long"]
    assert rejections["reasons"]["unlocatable-quote"] == 2


def test_the_project_total_and_this_interview_are_reported_separately(tmp_path):
    # The question the log was written for ("is it really ~30%?") is a
    # PROJECT-wide one; the interview on screen is a different number, and one
    # standing in for the other is the flattering failure this page is built
    # against.
    root = _node(tmp_path, sessions=("s1", "s2"))
    context = _context_dir(root)
    rejection_store.replace_session_rejections(context, "s1", [
        Rejection(claim="a", reason="unlocatable-quote"),
    ])
    rejection_store.replace_session_rejections(context, "s2", [
        Rejection(claim="b", reason="unlocatable-quote"),
        Rejection(claim="c", reason="mixed-speaker-span"),
    ])
    rejections = build_review("acme", "s2", repo_root=root)["rejections"]
    assert rejections["total"] == 3
    assert rejections["sessions_recorded"] == 2
    assert rejections["this_interview"]["total"] == 2
    assert rejections["this_interview"]["reasons"] == {
        "mixed-speaker-span": 1, "unlocatable-quote": 1,
    }


def test_an_interview_with_no_rows_still_gets_a_block_reporting_zero(tmp_path):
    """A zero here is genuinely AMBIGUOUS and the payload must let the page say
    so: an extraction that refused nothing writes no row naming this session,
    and neither does an extraction that never ran on it. The block is present
    with a total of zero rather than `None`, because `None` would be read as
    "there is no log" -- which is a third, different situation already carried
    by the status above."""
    root = _node(tmp_path, sessions=("s1", "s2"))
    rejection_store.replace_session_rejections(
        _context_dir(root), "s1", [Rejection(claim="a", reason="unlocatable-quote")]
    )
    rejections = build_review("acme", "s2", repo_root=root)["rejections"]
    assert rejections["status"] == "kept"
    assert rejections["this_interview"] == {"total": 0, "reasons": {}}


def test_a_corrupt_log_is_unreadable_rather_than_a_project_that_refused_nothing(tmp_path):
    """`load_rejections` is deliberately TOLERANT -- a corrupt diagnostic log
    must never block extraction itself -- so it answers `()` for a broken file
    exactly as it does for an empty one. The hub parses the file on its own
    account for precisely this reason: without that, a corrupt log would render
    as the most reassuring sentence on the panel."""
    root = _node(tmp_path)
    (_context_dir(root) / rejection_store.REJECTIONS_FILENAME).write_text(
        "{ not json", encoding="utf-8"
    )
    rejections = build_review("acme", "s1", repo_root=root)["rejections"]
    assert rejections["status"] == "unreadable"
    assert rejections["error"] is not None


def test_a_log_that_is_not_a_list_is_unreadable_too(tmp_path):
    root = _node(tmp_path)
    (_context_dir(root) / rejection_store.REJECTIONS_FILENAME).write_text(
        '{"rows": []}', encoding="utf-8"
    )
    assert build_review("acme", "s1", repo_root=root)["rejections"]["status"] == "unreadable"


def test_the_rejection_status_vocabulary_is_pinned_by_equality():
    assert set(review_mod.ALL_REJECTION_STATUSES) == {"not-kept", "unreadable", "kept"}


def test_the_page_words_every_rejection_status_the_server_can_emit():
    source = (Path(review_mod.__file__).parent.parent / "static" / "review" / "review.js").read_text(
        encoding="utf-8"
    )
    for status in review_mod.ALL_REJECTION_STATUSES:
        assert f'"{status}"' in source, (
            f"review.py can emit {status!r} and review.js has no sentence for it"
        )


# --- the panels degrade in their own section --------------------------------------------


def test_an_unreadable_corpus_does_not_take_the_two_new_panels_down(tmp_path):
    """Each section of this page degrades alone, and these two are read from
    files the fact corpus has nothing to do with. Folding them into `summary`
    -- which is `null` when `facts.json` will not parse -- would make one
    broken file delete three unrelated answers."""
    root = _node(tmp_path)
    (_context_dir(root) / "facts.json").write_text("{ not json", encoding="utf-8")
    _write_critique(root, "s1", {"session_id": "s1", "summary": _summary()})
    rejection_store.replace_session_rejections(
        _context_dir(root), "s1", [Rejection(claim="a", reason="unlocatable-quote")]
    )
    review = build_review("acme", "s1", repo_root=root)
    assert review["summary"] is None
    assert review["question_critique"]["status"] == "read"
    assert review["rejections"]["total"] == 1


def test_a_session_directory_that_does_not_exist_reads_as_not_run(tmp_path):
    # The picker refuses unknown sessions, but `build_review` is called directly
    # by tests and by the CLI, and an absent directory must not raise here.
    root = _node(tmp_path)
    critique = build_review("acme", "never-recorded", repo_root=root)["question_critique"]
    assert critique["status"] == "not-run"


def test_the_evidence_and_inference_counts_the_bar_is_drawn_from_are_still_present(tmp_path):
    """The third panel needs no new server field -- it is drawn from numbers
    `_summary_payload` has always carried. Pinned here anyway, because a panel
    with no field of its own is the one nobody thinks to check before deleting
    the numbers under it."""
    root = _node(tmp_path)
    (_context_dir(root) / "facts.json").write_text(json.dumps([{
        "id": "f1", "claim": "they rebuilt the export by hand", "provenance": "evidence",
        "session_id": "s1", "seqs": [1], "quoted_text": "they rebuilt the export by hand",
        "bearings": [],
    }]), encoding="utf-8")
    summary = build_review("acme", "s1", repo_root=root)["summary"]
    assert summary["evidence_facts"] == 1
    assert summary["inference_facts"] == 0
    assert summary["fact_count"] == 1
