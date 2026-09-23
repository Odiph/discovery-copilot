"""Tests for `app/coach/context.py` -- the grounding seam (04-CONTEXT.md D-01,
04-01-PLAN.md §C6; CTX-01/02/03). Behavioural names, per this project's own
test-naming convention (see `app/tests/test_phase_gates.py`)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import main as main_mod
from app.coach import language
from app.coach.context import (
    BRIEF_FILENAME,
    FACT_ID_PREFIX,
    SURVEY_FILENAME,
    ContextItem,
    ContextValidationError,
    SurveyItem,
    load_brief,
    load_survey,
)
from app.coach.prompts import static_block

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONTEXT_DIR = REPO_ROOT / "context"


def _write_brief(tmp_path: Path, items: list) -> Path:
    (tmp_path / "brief.json").write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


def _write_survey(tmp_path: Path, items: list) -> Path:
    (tmp_path / "survey.json").write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


def _write_facts(tmp_path: Path, items: list) -> Path:
    (tmp_path / "facts.json").write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


def _write_hypotheses(tmp_path: Path, items: list) -> Path:
    (tmp_path / "hypotheses.json").write_text(json.dumps(items), encoding="utf-8")
    return tmp_path


def _fact(
    *, id: str = "f1", session_id: str = "sess-1", seqs: "tuple[int, ...]" = (1,),
    claim: str = "onboarding takes about three weeks and happens mostly over email",
    provenance: str = "evidence", quoted_text: str = "we onboard in three weeks, mostly by email",
    related_hypothesis_ids: "list[str]" = (),
) -> dict:
    return {
        "id": id, "session_id": session_id, "seqs": list(seqs), "claim": claim,
        "provenance": provenance, "quoted_text": quoted_text,
        "bearings": [{"hypothesis_id": h, "stance": "supports"} for h in related_hypothesis_ids],
    }


def _hypothesis(*, id: str = "h1", statement: str = "onboarding delay is the top churn reason",
                 falsifier: str = "churned customers cite a different top reason more often") -> dict:
    return {"id": id, "statement": statement, "falsifier": falsifier}


def _example_context_dir(tmp_path: Path) -> Path:
    """Copies the committed `context/*.example.json` samples into a real
    `brief.json`/`survey.json` pair so `load_brief`/`load_survey` -- which
    read the non-`.example` filenames -- can be exercised against the exact
    content this repository ships as its format sample."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "brief.json").write_text(
        (REPO_CONTEXT_DIR / "brief.example.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "survey.json").write_text(
        (REPO_CONTEXT_DIR / "survey.example.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return tmp_path


# --- load_brief() -------------------------------------------------------------------


def test_load_brief_parses_the_example_file_with_provenance_preserved_exactly(tmp_path):
    items = load_brief(_example_context_dir(tmp_path))
    assert items, "context/brief.example.json must contain at least one item"
    assert all(isinstance(item, ContextItem) for item in items)
    provenances = {item.provenance for item in items}
    assert provenances <= {"evidence", "inference"}
    # The example is deliberately authored to exercise BOTH values.
    assert provenances == {"evidence", "inference"}


def test_load_brief_missing_file_raises_context_validation_error(tmp_path):
    with pytest.raises(ContextValidationError):
        load_brief(tmp_path)


@pytest.mark.parametrize(
    "provenance_value",
    ["misspelled", "", "Evidence", "EVIDENCE", None],
    ids=["misspelled", "empty", "wrong-case", "upper", "absent-value"],
)
def test_load_brief_rejects_a_malformed_provenance_naming_the_item_id(tmp_path, provenance_value):
    entry = {"id": "b-bad", "text": "some claim", "source": "s", "citation": None}
    if provenance_value is not None:
        entry["provenance"] = provenance_value
    _write_brief(tmp_path, [entry])
    with pytest.raises(ContextValidationError) as exc_info:
        load_brief(tmp_path)
    assert "b-bad" in str(exc_info.value), (
        "the error must name the offending item's id, not just say 'invalid'"
    )


def test_load_brief_never_defaults_a_malformed_provenance_to_evidence(tmp_path):
    # Directly asserts the CTX-02 property: a malformed entry never silently
    # becomes a validated ContextItem with provenance=="evidence" -- it is a
    # raised error every time, not a coerced value we could inspect.
    entry = {"id": "b-bad", "text": "some claim", "provenance": "maybe"}
    _write_brief(tmp_path, [entry])
    with pytest.raises(ContextValidationError):
        load_brief(tmp_path)


def test_load_brief_absent_provenance_field_entirely_raises(tmp_path):
    entry = {"id": "b-no-prov", "text": "some claim"}
    _write_brief(tmp_path, [entry])
    with pytest.raises(ContextValidationError) as exc_info:
        load_brief(tmp_path)
    assert "b-no-prov" in str(exc_info.value)


# --- load_survey() -------------------------------------------------------------------


def test_load_survey_parses_the_example_file_in_order_with_ids_preserved(tmp_path):
    items = load_survey(_example_context_dir(tmp_path))
    assert items, "context/survey.example.json must contain at least one item"
    assert all(isinstance(item, SurveyItem) for item in items)
    assert [item.order for item in items] == sorted(item.order for item in items)
    assert len({item.id for item in items}) == len(items), "ids must be unique"


def test_load_survey_returns_items_sorted_by_order_even_when_the_file_is_out_of_order(tmp_path):
    _write_survey(
        tmp_path,
        [
            {"id": "s2", "order": 1, "text": "second"},
            {"id": "s1", "order": 0, "text": "first"},
        ],
    )
    items = load_survey(tmp_path)
    assert [item.id for item in items] == ["s1", "s2"]


def test_load_survey_duplicate_id_raises_context_validation_error(tmp_path):
    _write_survey(
        tmp_path,
        [
            {"id": "s1", "order": 0, "text": "first"},
            {"id": "s1", "order": 1, "text": "duplicate id"},
        ],
    )
    with pytest.raises(ContextValidationError) as exc_info:
        load_survey(tmp_path)
    assert "s1" in str(exc_info.value)


def test_load_survey_missing_file_raises_context_validation_error(tmp_path):
    with pytest.raises(ContextValidationError):
        load_survey(tmp_path)


# --- 06-01-PLAN.md §F2: the import direction, proven in fresh interpreters -----------


def test_context_module_names_no_module_scope_import_of_corpus_in_its_own_ast():
    # The AST check alone: a module-scope `from app.coach import corpus` (or
    # `import app.coach.corpus`) anywhere in `t.body` (top-level statements
    # only) would be circular via `corpus.py:35`'s own `from app.coach.context
    # import PROVENANCE_VALUES`. This does NOT prove the import is absent from
    # inside a function body -- it only proves it is absent at MODULE scope,
    # which is exactly the property that matters for the circular-import risk.
    import ast

    source = (REPO_ROOT / "app" / "coach" / "context.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "corpus" in ast.dump(node)
    ]
    assert not bad, "a module-scope import of app.coach.corpus in context.py would be circular"


def test_context_first_import_succeeds_in_a_fresh_interpreter():
    # A FRESH subprocess, not this pytest process -- asserting only in-process
    # would prove nothing, since some other test module in this same pytest
    # run has typically already put `app.coach.corpus` into `sys.modules`,
    # which would silently hide a module-scope circular import.
    result = subprocess.run(
        [sys.executable, "-c", "import app.coach.context; print('ok')"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_corpus_first_import_succeeds_in_a_fresh_interpreter():
    # The mirror order: `app.coach.corpus` imported before `app.coach.context`
    # has ever been touched in this interpreter.
    result = subprocess.run(
        [sys.executable, "-c", "import app.coach.corpus; print('ok')"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --- 06-01-PLAN.md §F2/§F3: v2 retrieval over the fact corpus -------------------------


def _grounded_context_dir(tmp_path: Path) -> Path:
    """A brief + survey + one hypothesis + one fact whose claim shares
    vocabulary with the survey question -- the minimal shape needed to
    exercise retrieval end to end."""
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "walk me through onboarding a new customer"}])
    _write_hypotheses(tmp_path, [_hypothesis()])
    _write_facts(tmp_path, [_fact(related_hypothesis_ids=["h1"])])
    return tmp_path


def test_a_fact_reaches_the_prompt_alongside_the_hand_written_brief_item(tmp_path):
    items = load_brief(_grounded_context_dir(tmp_path))
    assert all(isinstance(item, ContextItem) for item in items)
    ids = {item.id for item in items}
    assert "b1" in ids, "the hand-written brief item must still be present"
    fact_ids = [i for i in ids if i.startswith(FACT_ID_PREFIX)]
    assert len(fact_ids) == 1, f"expected exactly one fact-derived item, got {fact_ids}"
    assert fact_ids[0] == f"{FACT_ID_PREFIX}f1", "the fact-derived id must carry the namespace prefix"


def test_the_fact_derived_mapping_is_exact(tmp_path):
    items = load_brief(_grounded_context_dir(tmp_path))
    fact_item = next(i for i in items if i.id.startswith(FACT_ID_PREFIX))
    assert fact_item.text == _fact()["claim"], "text must be byte-identical to Fact.claim, never a superstring"
    assert fact_item.provenance == "evidence", "provenance must pass through with no translation"
    assert "sess-1" in fact_item.citation and "1" in fact_item.citation, (
        "citation must resolve back to the originating session id and seq"
    )


def test_empty_corpus_with_no_facts_file_is_byte_identical_to_phase_four(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "some survey question"}])
    # No facts.json at all.
    items = load_brief(tmp_path)
    assert all(not item.id.startswith(FACT_ID_PREFIX) for item in items)

    hand_written_only = (ContextItem(id="b1", text="hand-written brief item", provenance="evidence"),)
    assert static_block(brief=items, hypotheses=()) == static_block(brief=hand_written_only, hypotheses=()), (
        "an absent facts.json must produce a static_block byte-identical to the hand-written brief alone"
    )


def test_empty_corpus_with_an_empty_facts_array_is_byte_identical_to_phase_four(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "some survey question"}])
    _write_facts(tmp_path, [])
    items = load_brief(tmp_path)

    hand_written_only = (ContextItem(id="b1", text="hand-written brief item", provenance="evidence"),)
    assert static_block(brief=items, hypotheses=()) == static_block(brief=hand_written_only, hypotheses=()), (
        "an empty facts.json array must degrade identically to an absent one"
    )


def test_determinism_across_two_interpreters_with_different_hash_seeds(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding process for a new customer"}])
    _write_hypotheses(tmp_path, [_hypothesis()])
    # Several facts that score EQUALLY (identical claim text) so the tie-break
    # sort key -- not score alone -- is what determinism depends on here.
    _write_facts(tmp_path, [
        _fact(id=f"f{i}", session_id=f"sess-{i % 3}", seqs=(i,), claim="onboarding a new customer takes a long time")
        for i in range(6)
    ])

    script = (
        "import hashlib, os, sys; sys.path.insert(0, '.'); "
        "from app.coach.context import load_brief; from app.coach.prompts import static_block; "
        "print(hashlib.sha256(static_block(brief=load_brief(os.environ['CTXDIR']), hypotheses=()).encode()).hexdigest())"
    )
    hashes = []
    for seed in ("0", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONHASHSEED": seed, "CTXDIR": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        hashes.append(result.stdout.strip())
    assert hashes[0] == hashes[1], (
        "two interpreters with different PYTHONHASHSEED values produced different static_block bytes -- "
        "retrieval ordering is not deterministic"
    )


def test_a_fact_sharing_no_vocabulary_with_the_survey_or_hypotheses_is_not_selected(tmp_path):
    # No stopword filtering exists (or is claimed) anywhere in this design --
    # deliberately no shared token at all, not even "the"/"a", between the
    # fact's claim and the survey question, so the assertion is unambiguous.
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding process for new customer"}])
    _write_facts(tmp_path, [_fact(id="irrelevant", claim="weather unusually warm that week")])
    items = load_brief(tmp_path)
    assert not any(item.id.startswith(FACT_ID_PREFIX) for item in items), (
        "a fact with zero token overlap against the survey/hypotheses must not be selected"
    )


def test_selection_is_capped_at_the_configured_maximum(tmp_path):
    from app import config as config_mod

    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer takes time"}])
    _write_facts(tmp_path, [
        _fact(id=f"f{i}", session_id=f"sess-{i}", seqs=(i,), claim="onboarding a new customer takes time")
        for i in range(config_mod.DEFAULT_COACH_MAX_RETRIEVED_FACTS + 5)
    ])
    items = load_brief(tmp_path)
    fact_items = [item for item in items if item.id.startswith(FACT_ID_PREFIX)]
    assert len(fact_items) == config_mod.DEFAULT_COACH_MAX_RETRIEVED_FACTS


def test_two_contradicting_facts_that_both_score_relevant_both_appear_with_distinct_citations(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer takes time today"}])
    _write_facts(tmp_path, [
        _fact(id="f-yes", session_id="sess-a", seqs=(1,), claim="onboarding a new customer today takes three weeks"),
        _fact(id="f-no", session_id="sess-b", seqs=(2,), claim="onboarding a new customer today takes only two days"),
    ])
    items = load_brief(tmp_path)
    fact_items = [item for item in items if item.id.startswith(FACT_ID_PREFIX)]
    assert len(fact_items) == 2, "both contradicting facts must be present, never deduplicated to one 'winner'"
    citations = {item.citation for item in fact_items}
    assert len(citations) == 2, "each contradicting fact must keep its own distinct citation"


def test_absent_facts_file_returns_the_hand_written_brief_with_no_error(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "some survey question"}])
    items = load_brief(tmp_path)  # must not raise
    assert [item.id for item in items] == ["b1"]


def test_a_malformed_fact_raises_naming_the_offending_facts_id(tmp_path):
    # Raised by `corpus.load_facts` itself (`CorpusValidationError`), the
    # SAME four-way error ladder `context.py`'s own loaders use -- `load_brief`
    # never wraps or re-raises it, per "never a second parser" (§F2).
    from app.coach.corpus import CorpusValidationError

    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer"}])
    _write_facts(tmp_path, [_fact(id="bad-fact", provenance="misspelled")])
    with pytest.raises(CorpusValidationError) as exc_info:
        load_brief(tmp_path)
    assert "bad-fact" in str(exc_info.value)


def test_absent_hypotheses_file_is_tolerated_when_facts_are_present(tmp_path):
    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer takes time"}])
    _write_facts(tmp_path, [_fact(claim="onboarding a new customer takes time")])
    items = load_brief(tmp_path)  # must not raise even with no hypotheses.json
    assert any(item.id.startswith(FACT_ID_PREFIX) for item in items)


def test_a_malformed_hypotheses_file_raises(tmp_path):
    from app.coach.corpus import CorpusValidationError

    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer"}])
    _write_facts(tmp_path, [_fact()])
    _write_hypotheses(tmp_path, [{"id": "h1", "statement": "no falsifier here"}])  # missing required falsifier
    with pytest.raises(CorpusValidationError):
        load_brief(tmp_path)


def test_a_retrieved_hebrew_fact_does_not_disturb_english_language_selection(tmp_path):
    # 06-CONTEXT.md D-06: language.detect_language reads only the pipeline's
    # own accumulated INTERVIEWEE segments and never the brief -- structurally
    # unaffected by what load_brief returns. This is the assertion that turns
    # that structural claim into a test rather than a comment.
    from app.session.store import SegmentRecord

    _write_brief(tmp_path, [{"id": "b1", "text": "hand-written brief item", "provenance": "evidence"}])
    _write_survey(tmp_path, [{"id": "s1", "order": 0, "text": "onboarding a new customer takes time today"}])
    # The claim shares no vocabulary with the (English) survey question, so
    # scoring alone would never select it -- `related_hypothesis_ids` is
    # irrelevant here too (no hypotheses.json). This test's point is not
    # whether the Hebrew fact gets retrieved by score; it is that BUILDING a
    # brief containing Hebrew script (however it got there) never disturbs
    # language selection, which reads only the interviewee's own transcript.
    _write_facts(tmp_path, [_fact(claim="לקוח חדש מתחיל תהליך קליטה שלוקח הרבה זמן")])
    items = load_brief(tmp_path)

    english_segments = [
        SegmentRecord(
            kind="segment", seq=0, speaker="interviewee", text="we onboard customers over several weeks",
            audio_start_s=0.0, audio_end_s=5.0, ready_wall_s=5.1, latency_s=0.1,
            detected_language="en", engine="fake", written_at="2026-08-06T00:00:00.000+00:00",
        )
    ]
    detected = language.detect_language(
        english_segments, window_s=90.0, flip_margin=0.65, min_chars=1,
        previous_language=language.LANGUAGE_ENGLISH,
    )
    assert detected == language.LANGUAGE_ENGLISH, (
        "a Hebrew-claim fact in the brief must never disturb script-ratio language "
        "selection, which reads only the interviewee's own transcript segments"
    )
    # `items` (which may or may not include the Hebrew fact, depending on scoring)
    # is passed through static_block only to prove it does not raise or otherwise
    # misbehave when a retrieved item carries non-Latin script.
    assert isinstance(static_block(brief=items, hypotheses=()), str)


# --- CTX-03: load_brief() is a substitutable seam ------------------------------------


def _read_grounding(context_dir: Path, *, loader=load_brief):
    """Stands in for the shape a real caller (04-02's generate stage) will
    take: a loader parameter defaulting to `load_brief`, never touching the
    filesystem itself. This is the property CTX-03 requires -- a substituted
    loader behaves identically, with no change to this caller and no change
    to the return type."""
    return loader(context_dir)


def test_ctx03_a_substituted_brief_loader_behaves_identically_through_the_same_caller(tmp_path):
    real_context_dir = _example_context_dir(tmp_path / "real")
    real_items = _read_grounding(real_context_dir)

    def _v2_style_loader(_context_dir: Path) -> "tuple[ContextItem, ...]":
        # A stand-in for "v2 retrieval" -- reads from somewhere else entirely
        # (here, a hardcoded literal) but returns the exact same type.
        return (
            ContextItem(id="r1", text="retrieved evidence", source="retrieval", citation=None, provenance="evidence"),
        )

    substituted_items = _read_grounding(tmp_path / "unused", loader=_v2_style_loader)

    assert isinstance(real_items, tuple) and isinstance(substituted_items, tuple)
    assert all(isinstance(item, ContextItem) for item in real_items)
    assert all(isinstance(item, ContextItem) for item in substituted_items)
    assert real_items != substituted_items, "sanity: the two loaders must actually differ in content"


# --- app/main.py CLI wiring (04-01-PLAN.md §C6) --------------------------------------


def test_previous_session_started_at_returns_none_with_no_prior_sessions(tmp_path):
    assert main_mod._previous_session_started_at(tmp_path / "sessions-does-not-exist") is None


def test_previous_session_started_at_returns_the_latest_started_at_among_existing_sessions(tmp_path):
    sessions_root = tmp_path / "sessions"
    older = sessions_root / "20260101T000000"
    newer = sessions_root / "20260102T000000"
    for session_dir, started_at in (
        (older, "2026-01-01T00:00:00.000+00:00"),
        (newer, "2026-01-02T00:00:00.000+00:00"),
    ):
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text(
            json.dumps({"started_at": started_at}), encoding="utf-8"
        )
    result = main_mod._previous_session_started_at(sessions_root)
    assert result == datetime.fromisoformat("2026-01-02T00:00:00.000+00:00")


def test_warn_if_context_stale_warns_when_the_context_files_predate_the_previous_session(tmp_path, capsys):
    context_dir = _example_context_dir(tmp_path)
    previous_started_at = datetime.now(timezone.utc) + timedelta(days=1)  # forces "older than this"
    main_mod._warn_if_context_stale(context_dir, previous_started_at)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert BRIEF_FILENAME in captured.err or SURVEY_FILENAME in captured.err


def test_warn_if_context_stale_is_silent_when_there_is_no_previous_session(tmp_path, capsys):
    context_dir = _example_context_dir(tmp_path)
    main_mod._warn_if_context_stale(context_dir, None)
    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_if_context_stale_is_silent_when_the_context_is_newer_than_the_previous_session(tmp_path, capsys):
    context_dir = _example_context_dir(tmp_path)
    long_ago = datetime.now(timezone.utc) - timedelta(days=365)
    main_mod._warn_if_context_stale(context_dir, long_ago)
    captured = capsys.readouterr()
    assert captured.err == ""


@pytest.mark.audio_device
def test_start_proceeds_with_coach_disabled_and_a_printed_explanation_when_the_key_is_absent(tmp_path):
    env = dict(os.environ)
    # Set it EMPTY rather than popping it. The subprocess runs with cwd=REPO_ROOT,
    # so `config._load_dotenv_once` reads the developer's real `.env` -- and a
    # popped variable is "not already present", which is exactly the case that
    # loader fills in. Popping therefore RE-INTRODUCED a real key and this test
    # silently stopped exercising the degrade path the moment a key existed
    # (observed 2026-08-09, the day one first did). An empty value is present, so
    # the loader's own "the environment always wins" rule keeps `.env` out, and
    # `reasoning_client.build_client`'s `if not api_key` still raises.
    env["ANTHROPIC_API_KEY"] = ""
    # And the SECOND way to get a client, added in `ce1f5bb`. `build_client()`
    # checks `COACH_BASE_URL` FIRST and returns an OpenAI-compatible client
    # without consulting the Anthropic key at all, so blanking that key alone
    # stopped disabling the coach the day a `COACH_BASE_URL` appeared in the
    # developer's `.env` (observed 2026-08-14, the day one first did). Same
    # empty-not-popped rule as above, and for the same dotenv reason.
    #
    # This is the SECOND time this test has been silently disarmed by a real
    # credential appearing on the machine, which is the argument for blanking
    # every route to a client here rather than the one that happens to exist:
    # a degrade path is only under test while EVERY way of not degrading is
    # closed.
    env["COACH_BASE_URL"] = ""
    env["COACH_API_KEY"] = ""
    env["APP_SESSIONS_ROOT"] = str(tmp_path / "sessions")

    result = subprocess.run(
        [sys.executable, "-m", "app", "start", "--engine", "fake", "--no-serve", "--seconds", "1"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Coach disabled" in result.stdout
    assert "ANTHROPIC_API_KEY" in result.stdout


def test_start_fails_loudly_before_any_device_or_session_when_context_is_missing(tmp_path):
    # A dummy, obviously-fake key -- this path returns 1 (ContextValidationError)
    # BEFORE any engine is resolved, any device opened, or the vendor client ever
    # makes a call, so this is fully offline and never touches the network.
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = "sk-ant-test-dummy-never-billed-0000"
    env["APP_CONTEXT_DIR"] = str(tmp_path / "nonexistent-context")
    env["APP_SESSIONS_ROOT"] = str(tmp_path / "sessions")

    result = subprocess.run(
        [sys.executable, "-m", "app", "start", "--engine", "fake", "--no-serve", "--seconds", "1"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert "does not exist" in result.stderr
    sessions_root = tmp_path / "sessions"
    assert not sessions_root.exists() or not any(sessions_root.iterdir()), (
        "no session directory may be created when the brief/survey fails to load"
    )
