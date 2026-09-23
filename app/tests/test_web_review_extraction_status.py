"""The hub must not report "nothing moved" when it simply does not know.

`build_review` treated an absent `facts.json` as SUCCESS: `facts` stayed `()`,
`facts_error` stayed `None`, and every standing came back with zeroes. The page
then rendered "Nothing in the corpus supports this yet" and "No disconfirming
evidence found -- was any sought?" -- prose written to provoke a reaction, and
therefore read as a FINDING.

Four different situations produced that identical page:

  1. `python -m app extract <session>` was never run. Nothing in `run_hub`
     triggers, checks or mentions extraction.
  2. Extraction ran against a context dir with no hypotheses, so a bearing was
     structurally impossible.
  3. Extraction ran, hypotheses were loaded, and the model found no bearing.
  4. Nothing in the world bears on this belief.

Only (4) is a research finding. This cost real diagnostic time on 2026-08-12:
the only real corpus in the repo showed 23 facts with zero bearings, and the
page's answer was indistinguishable from a broken pipeline -- so a correct
result (a career-history interview against note-taking hypotheses) and a
plausible defect looked exactly alike.

Every number asserted here comes from reads `build_review` ALREADY performs.
This is a reporting fix, not new machinery.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.web.review import build_review


def _node(tmp_path: Path, *, facts=None, hypotheses=None, raw_facts=None) -> Path:
    context = tmp_path / "projects" / "acme" / "context"
    context.mkdir(parents=True, exist_ok=True)
    (tmp_path / "projects" / "acme" / "sessions").mkdir(parents=True, exist_ok=True)
    if hypotheses is not None:
        (context / "hypotheses.json").write_text(json.dumps(hypotheses), encoding="utf-8")
    if raw_facts is not None:
        (context / "facts.json").write_text(raw_facts, encoding="utf-8")
    elif facts is not None:
        (context / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    return tmp_path


_HYPOTHESES = [{"id": "h1", "statement": "onboarding is the blocker", "falsifier": "setup was easy"}]


def _fact(fact_id: str, session_id: str, bearings=()) -> dict:
    return {
        "id": fact_id,
        "claim": "they rebuilt the export by hand",
        "provenance": "evidence",
        "session_id": session_id,
        "seqs": [1],
        "quoted_text": "they rebuilt the export by hand",
        "bearings": list(bearings),
    }


def test_a_session_that_was_never_extracted_says_so(tmp_path):
    root = _node(tmp_path, hypotheses=_HYPOTHESES)
    review = build_review("acme", "20260812T090000", repo_root=root)
    assert review["extraction"]["status"] == "not-extracted"


def test_an_extracted_session_with_no_bearings_is_not_the_same_as_never_extracted(tmp_path):
    # THE test. Both render zero counts; they are opposite situations. One
    # means "you have not run the step yet", the other means "the step ran and
    # found nothing", and only the second is evidence about the world.
    never = build_review(
        "acme", "s1", repo_root=_node(tmp_path / "a", hypotheses=_HYPOTHESES)
    )["extraction"]
    ran = build_review(
        "acme",
        "s1",
        repo_root=_node(tmp_path / "b", hypotheses=_HYPOTHESES, facts=[_fact("f1", "s1")]),
    )["extraction"]

    assert never != ran
    assert ran["status"] == "extracted"
    assert ran["facts_this_interview"] == 1
    assert ran["facts_without_bearings"] == 1


def test_extraction_with_no_hypotheses_loaded_is_named_as_its_own_state(tmp_path):
    # A bearing is structurally impossible with no hypotheses to bear on, so
    # zero bearings here says nothing whatever about the interview. Reporting
    # it as "the model found nothing" would be an outright false negative.
    root = _node(tmp_path, hypotheses=[], facts=[_fact("f1", "s1")])
    extraction = build_review("acme", "s1", repo_root=root)["extraction"]
    assert extraction["hypotheses_loaded"] == 0
    assert extraction["status"] == "no-hypotheses"


def test_facts_are_counted_for_THIS_interview_not_the_whole_corpus(tmp_path):
    # The sneaky version. A corpus rich from earlier interviews makes a session
    # that extracted nothing look thoroughly examined -- the page would report
    # a healthy count for an interview that contributed none of it.
    root = _node(
        tmp_path,
        hypotheses=_HYPOTHESES,
        facts=[_fact("f1", "older"), _fact("f2", "older"), _fact("f3", "older")],
    )
    extraction = build_review("acme", "s-new", repo_root=root)["extraction"]
    assert extraction["facts_total"] == 3
    assert extraction["facts_this_interview"] == 0
    # CHANGED after the 2026-08-12 audit, and the old assertion is worth
    # quoting because it encoded the defect: it read `status == "not-extracted"`
    # with the comment "a session that contributed no facts has not been
    # extracted, however full the corpus is". That is not knowable. A corpus
    # this full proves extraction has RUN in this project; what it cannot say
    # is whether this session was one of the ones extracted.
    assert extraction["status"] == "no-facts-from-this-interview"


def test_a_corpus_that_will_not_parse_is_never_reported_as_extracted(tmp_path):
    # The failure that must not be silent: `load_facts` raising leaves `facts`
    # empty, which is byte-identical to a clean corpus with no bearings.
    root = _node(tmp_path, hypotheses=_HYPOTHESES, raw_facts="{ not json")
    review = build_review("acme", "s1", repo_root=root)
    assert review["facts_error"] is not None
    assert review["extraction"]["status"] == "unreadable"


def test_bearings_are_counted_only_against_hypotheses_that_exist(tmp_path):
    # A fact can carry a bearing on a hypothesis id that has since been renamed
    # or deleted. Counting it as "bearing on something" would report the corpus
    # as connected to beliefs the operator can no longer see on the page.
    root = _node(
        tmp_path,
        hypotheses=_HYPOTHESES,
        facts=[_fact("f1", "s1", bearings=[{"hypothesis_id": "ghost", "stance": "supports"}])],
    )
    extraction = build_review("acme", "s1", repo_root=root)["extraction"]
    assert extraction["facts_without_bearings"] == 1, (
        "a bearing on a hypothesis that no longer exists must not count as a connection"
    )


# --- the split: "you have not run it" versus "it holds nothing from this call" ----------


def test_an_absent_corpus_and_an_unrelated_corpus_are_different_answers(tmp_path):
    """The collapse the 2026-08-12 audit found. Both used to say
    "not-extracted" and print a remedy underneath -- so an operator whose
    corpus already held twenty claims from earlier interviews was told to run
    a command that may well have just run and found nothing.

    What separates them is knowable and cheap: whether `facts.json` EXISTS.
    What is NOT knowable is which of two causes produced an empty slice for
    this session, and the page says so rather than picking one."""
    absent = build_review(
        "acme", "s-new", repo_root=_node(tmp_path / "a", hypotheses=_HYPOTHESES)
    )["extraction"]
    unrelated = build_review(
        "acme",
        "s-new",
        repo_root=_node(tmp_path / "b", hypotheses=_HYPOTHESES, facts=[_fact("f1", "older")]),
    )["extraction"]

    assert absent["status"] == "not-extracted"
    assert unrelated["status"] == "no-facts-from-this-interview"
    assert absent["status"] != unrelated["status"]


def test_an_empty_corpus_file_is_not_reported_as_never_extracted(tmp_path):
    """`facts.json` present and holding `[]` means extraction ran and filed
    nothing. It is the same answer as the case above -- the corpus exists and
    has nothing from this interview -- and specifically NOT the absent case,
    which is the one whose remedy is unconditional."""
    root = _node(tmp_path, hypotheses=_HYPOTHESES, facts=[])
    extraction = build_review("acme", "s1", repo_root=root)["extraction"]
    assert extraction["status"] == "no-facts-from-this-interview"
    assert extraction["facts_total"] == 0


def test_no_hypotheses_still_outranks_the_new_status(tmp_path):
    """The ladder's order is load-bearing and unchanged. With nothing to bear
    on, an empty slice is a statement about the context dir, not about the
    interview -- so `no-hypotheses` must still be decided first, even now that
    the case below it has been split in two."""
    root = _node(tmp_path, hypotheses=[], facts=[_fact("f1", "older")])
    assert build_review("acme", "s1", repo_root=root)["extraction"]["status"] == "no-hypotheses"


# --- the two hand-typed vocabularies, held together --------------------------------------


def test_the_page_words_every_status_the_server_can_emit(tmp_path):
    """THE CROSSING TEST, and the reason this one is written against source
    text rather than behaviour.

    The audit's finding: `review.py` emitted four status literals, `review.js`
    keyed a `said` lookup on the same four by hand, and NOTHING rendered the
    banner -- `extraction` was in the JS harness's PAGE_IDS and every fixture
    passed `extraction: null`. So a rename on either side produced
    `headline === undefined`, an empty banner, and a green suite. Two lists
    that can agree with each other while disagreeing with the code is the same
    shape as `message["candidates"]` against a wire pinned to `primary`.

    The JS half now renders all five (see `review_render.test.mjs`). This half
    is the direction that scan cannot cover: a status added to PYTHON with no
    sentence in the page."""
    from app.web import review as review_mod

    source = (Path(review_mod.__file__).parent.parent / "static" / "review" / "review.js").read_text(
        encoding="utf-8"
    )
    for status in review_mod.ALL_EXTRACTION_STATUSES:
        if status == review_mod.STATUS_EXTRACTED:
            # The one status with no fixed sentence: its headline is built from
            # the counts, so there is deliberately no literal to find.
            continue
        assert f'"{status}"' in source, (
            f"review.py can emit {status!r} and review.js has no sentence for it -- the page "
            "would render an empty banner above a screen full of zeroes"
        )


def test_the_status_vocabulary_is_pinned_by_equality():
    """Equality, not membership, following gates 8/18/19. A sixth status is a
    decision about how the banner reads and what colour it is, never a drift."""
    from app.web import review as review_mod

    assert set(review_mod.ALL_EXTRACTION_STATUSES) == {
        "extracted",
        "not-extracted",
        "no-facts-from-this-interview",
        "no-hypotheses",
        "unreadable",
    }


def test_every_status_has_a_colour_so_none_renders_unstyled():
    """The status string becomes a CSS class verbatim
    (`box.className = "extraction " + status`), so a status with no rule
    renders in the default border colour -- which on this page means
    "extracted, nothing to see". A missing rule is therefore not cosmetic."""
    from app.web import review as review_mod

    css = (Path(review_mod.__file__).parent.parent / "static" / "review" / "review.css").read_text(
        encoding="utf-8"
    )
    for status in review_mod.ALL_EXTRACTION_STATUSES:
        assert f".extraction.{status}" in css, (
            f"{status!r} has no border colour, so it renders as if nothing were wrong"
        )
