"""Tests for `app/coach/extract.py`, `app/coach/corpus.py` (the extraction
half), and `app/main.py`'s `extract` subcommand -- 05-01-PLAN.md Task 1. One
finished session becomes one cited fact on disk, end to end, through the real
CLI entry point, against the offline stub, with `ANTHROPIC_API_KEY` stripped
from the environment for the whole run. Behavioural test names, per this
project's own convention."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app import main as main_mod
from app.coach import corpus, reasoning_client
from app.coach.corpus import Fact, Hypothesis
from app.coach.context import PROVENANCE_EVIDENCE, PROVENANCE_INFERENCE
from app.coach.extract import (
    EXTRACT_TOOL_NAME,
    REASON_CONTENTLESS_QUOTE,
    REASON_MIXED_SPEAKER_SPAN,
    REASON_NONCONTIGUOUS_SPAN,
    REASON_SPAN_TOO_LONG,
    REASON_UNLOCATABLE_QUOTE,
    REASON_UNRESOLVED_CITATION,
    ExtractFailure,
    assemble_extract_request,
    extract_facts,
    quote_states_nothing,
)
from app.coach.stub import StubAnthropicClient, ToolUseBlock, ToolUseResponse, tool_use_response
from app.session.reader import read_session
from app.session.store import SegmentRecord, SessionStore, create_session_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_REASONING_MARKER_PATTERN = re.compile(r"@pytest\.mark\.live_reasoning\b")


def _segment(seq: int, speaker: str, text: str, audio_end_s: float) -> SegmentRecord:
    return SegmentRecord(
        kind="segment", seq=seq, speaker=speaker, text=text, audio_start_s=audio_end_s - 1.0,
        audio_end_s=audio_end_s, ready_wall_s=audio_end_s + 0.1, latency_s=0.1,
        detected_language="en", engine="fake", written_at="2026-08-08T00:00:00.000+00:00",
    )


def _new_session(tmp_path: Path, turns: "list[tuple[str, str]]") -> "tuple[Path, tuple[SegmentRecord, ...]]":
    """Builds a REAL session directory via `SessionStore`, so `read_session`
    and the citation resolver see genuine `seq` values -- never a hand-rolled
    transcript file."""
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "extract-test"})
    records = []
    for speaker, text in turns:
        records.append(
            store.append_segment(
                speaker=speaker, text=text, audio_start_s=0.0, audio_end_s=1.0,
                ready_wall_s=1.1, detected_language="en", engine="fake",
            )
        )
    store.close()
    return session_dir, tuple(records)


def _new_open_session(tmp_path: Path, turns: "list[tuple[str, str]]") -> "tuple[Path, tuple[SegmentRecord, ...]]":
    """The sibling of `_new_session` for §K7b's unclosed-session case: builds
    a REAL session directory but deliberately never calls `store.close()`,
    so no `closed_at` marker is ever written -- exactly what `read_session`
    reports as `closed=False`."""
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "extract-unclosed-test"})
    records = []
    for speaker, text in turns:
        records.append(
            store.append_segment(
                speaker=speaker, text=text, audio_start_s=0.0, audio_end_s=1.0,
                ready_wall_s=1.1, detected_language="en", engine="fake",
            )
        )
    # Deliberately no store.close() -- see docstring.
    return session_dir, tuple(records)


class _FakeUsage:
    """Duck-types `anthropic.types.Usage`'s two token-count fields --
    `StubAnthropicClient`'s own `ToolUseResponse` never carries `.usage` at
    all, so a test that needs one builds its own response object rather than
    extending the shared stub shape."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponseWithUsage:
    """A minimal, duck-typed response carrying `.content`/`.stop_reason`
    (the same shape `ToolUseResponse` provides) PLUS `.usage` -- 05-02-PLAN.md
    §K8's token-accounting tests are the only place this repository needs a
    response with usage attached, so it lives here rather than in
    `app/coach/stub.py`."""

    def __init__(self, *, tool_name: str, input_payload: dict, input_tokens: int, output_tokens: int) -> None:
        self.content = (ToolUseBlock(name=tool_name, input=input_payload),)
        self.stop_reason = "tool_use"
        self.usage = _FakeUsage(input_tokens, output_tokens)


def _fact_payload(*, claim: str, seq: int, provenance: str, quoted_text: str, related=()) -> dict:
    """`seq` stays a single int here for every existing call site's
    convenience -- STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1 only
    requires the WIRE key to be `seqs` (a non-empty array); a single-element
    span is the common case this whole file exercises."""
    return {
        "claim": claim, "seqs": [seq], "provenance": provenance,
        "quoted_text": quoted_text,
        "bearings": [{"hypothesis_id": h, "stance": "supports"} for h in related],
    }


# --- assemble_extract_request: pure, no client -----------------------------------------


def test_assemble_extract_request_carries_every_segments_own_seq():
    transcript = [
        _segment(0, "interviewer", "hello", 1.0),
        _segment(1, "interviewee", "we onboard fast", 2.0),
    ]
    request = assemble_extract_request(transcript=transcript, hypotheses=(), model="m", max_tokens=100)
    message_text = request["messages"][-1]["content"]
    assert "seq=0" in message_text
    assert "seq=1" in message_text


def test_assemble_extract_request_tool_choice_forces_exactly_one_named_tool():
    request = assemble_extract_request(transcript=[], hypotheses=(), model="m", max_tokens=100)
    assert request["tool_choice"] == {"type": "tool", "name": EXTRACT_TOOL_NAME}
    assert len(request["tools"]) == 1
    assert request["tools"][0]["name"] == EXTRACT_TOOL_NAME


# --- extract_facts: parsing, timeout, deterministic validation -------------------------


def test_extract_facts_binds_its_own_explicit_timeout_differing_from_a_default():
    stub = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": []})])
    extract_facts(
        stub, session_id="s1", transcript=[], hypotheses=(), model="m", max_tokens=100, timeout_s=42.0
    )
    assert stub.calls[0].timeout_s == 42.0


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_extract_facts_treats_refusal_and_truncation_as_untrusted(stop_reason):
    stub = StubAnthropicClient(responses=[ToolUseResponse(content=(), stop_reason=stop_reason)])
    with pytest.raises(ExtractFailure) as excinfo:
        extract_facts(
            stub, session_id="s1", transcript=[], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
        )
    assert excinfo.value.stop_reason == stop_reason


def test_extract_facts_wraps_any_exception_from_the_call_itself():
    stub = StubAnthropicClient(responses=[TimeoutError("simulated")])
    with pytest.raises(ExtractFailure) as excinfo:
        extract_facts(
            stub, session_id="s1", transcript=[], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
        )
    assert excinfo.value.stop_reason is None


def test_extract_facts_accepts_a_fact_whose_citation_resolves():
    trigger = _segment(5, "interviewee", "we onboard a customer in three weeks", 3.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                EXTRACT_TOOL_NAME,
                {"facts": [_fact_payload(
                    claim="Onboarding takes three weeks.", seq=5, provenance="evidence",
                    quoted_text="we onboard a customer in three weeks",
                )]},
            )
        ]
    )
    result = extract_facts(
        stub, session_id="sess-1", transcript=[trigger], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1
    assert not result.rejections
    fact = result.accepted[0]
    assert fact.session_id == "sess-1"
    assert fact.seqs == (5,)
    assert fact.provenance == "evidence"


def test_extract_facts_rejects_a_fact_whose_citation_does_not_resolve():
    trigger = _segment(5, "interviewee", "we onboard a customer in three weeks", 3.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                EXTRACT_TOOL_NAME,
                {"facts": [_fact_payload(
                    claim="a claim citing a fake segment", seq=999, provenance="evidence",
                    quoted_text="we onboard a customer in three weeks",
                )]},
            )
        ]
    )
    result = extract_facts(
        stub, session_id="sess-1", transcript=[trigger], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.seqs == (999,)
    assert "a claim citing a fake segment" in rejection.claim


def test_extract_facts_twice_from_identical_response_differs_only_by_id():
    trigger = _segment(3, "interviewee", "same claim text here", 3.0)
    payload = {"facts": [_fact_payload(
        claim="the same claim", seq=3, provenance="evidence", quoted_text="same claim text here",
    )]}
    stub1 = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, payload)])
    stub2 = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, payload)])

    result1 = extract_facts(
        stub1, session_id="sess-1", transcript=[trigger], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    result2 = extract_facts(
        stub2, session_id="sess-1", transcript=[trigger], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    fact1, fact2 = result1.accepted[0], result2.accepted[0]
    assert fact1.id != fact2.id

    dump1 = fact1.model_dump()
    dump2 = fact2.model_dump()
    del dump1["id"]
    del dump2["id"]
    assert dump1 == dump2, "every field except id must be identical across identical extractions"


# --- Fact schema: exactly seven fields, no time-valued field ---------------------------


def test_fact_field_set_is_pinned_by_exact_equality():
    assert set(Fact.model_fields.keys()) == {
        "id", "session_id", "seqs", "claim", "provenance", "quoted_text", "bearings",
    }, "a field nobody thought to add must also fail this assertion"


def test_fact_field_set_holds_no_time_valued_field_of_any_kind():
    suspicious = {
        name for name in Fact.model_fields
        if "time" in name.lower() or name.endswith("_at") or name.endswith("_s") or "timestamp" in name.lower()
    }
    assert not suspicious, f"a Fact field looks time-valued -- §K2 forbids this entirely: {suspicious}"


# --- app.coach.corpus.load_facts: the LOUD, validating loader --------------------------


def test_load_facts_raises_naming_the_fact_id_for_a_misspelled_provenance(tmp_path):
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    bad_fact = {
        "id": "f-bad", "session_id": "s1", "seqs": [0], "claim": "x",
        "provenance": "maybe", "quoted_text": "", "bearings": [],
    }
    (context_dir / "facts.json").write_text(json.dumps([bad_fact]), encoding="utf-8")
    with pytest.raises(corpus.CorpusValidationError) as excinfo:
        corpus.load_facts(context_dir, sessions_root=None)
    assert "f-bad" in str(excinfo.value)


# --- app.coach.extract: no direct transcript read (§K7) ---------------------------------


def test_extract_module_performs_no_direct_file_read_by_ast_walk():
    path = REPO_ROOT / "app" / "coach" / "extract.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                violations.append("open(...)")
            elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "read_bytes", "open"):
                violations.append(f".{func.attr}(...)")
    assert not violations, f"app/coach/extract.py performs a direct file read: {violations}"


def test_extraction_reaches_the_transcript_only_through_read_session(tmp_path, monkeypatch):
    session_dir, records = _new_session(tmp_path, [("interviewer", "hi"), ("interviewee", "we onboard fast")])
    interviewee_seq = records[1].seq
    context_dir = tmp_path / "context"

    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                EXTRACT_TOOL_NAME,
                {"facts": [_fact_payload(
                    claim="onboarding is fast", seq=interviewee_seq, provenance="evidence",
                    quoted_text="we onboard fast",
                )]},
            )
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    real_read_session = main_mod.read_session
    calls: "list[Path]" = []

    def _wrapped(path):
        calls.append(path)
        return real_read_session(path)

    monkeypatch.setattr(main_mod, "read_session", _wrapped)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0
    assert calls, "read_session was never called by the extraction CLI path"


# --- CLI: the real end-to-end path, driven through main_mod.main(), against the stub ----


def test_cli_extract_end_to_end_writes_one_fact_naming_a_real_session_and_seq_and_exits_zero(tmp_path, monkeypatch):
    session_dir, records = _new_session(
        tmp_path, [("interviewer", "walk me through onboarding"), ("interviewee", "we onboard a customer in three weeks")]
    )
    interviewee_seq = records[1].seq
    context_dir = tmp_path / "context"

    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                EXTRACT_TOOL_NAME,
                {"facts": [_fact_payload(
                    claim="Onboarding takes three weeks.", seq=interviewee_seq, provenance="evidence",
                    quoted_text="we onboard a customer in three weeks",
                )]},
            )
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0

    facts_path = context_dir / "facts.json"
    assert facts_path.is_file()
    raw = json.loads(facts_path.read_text(encoding="utf-8"))
    assert len(raw) == 1
    assert raw[0]["session_id"] == session_dir.name
    assert raw[0]["seqs"] == [interviewee_seq]

    # The written file must itself load through the module's own loader.
    loaded = corpus.load_facts(context_dir, sessions_root=None)
    assert len(loaded) == 1
    assert loaded[0].session_id == session_dir.name


def test_cli_extract_unresolvable_citation_writes_nothing_names_the_fact_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    session_dir, records = _new_session(tmp_path, [("interviewee", "some real claim text")])
    context_dir = tmp_path / "context"
    bogus_seq = 999999

    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                EXTRACT_TOOL_NAME,
                {"facts": [_fact_payload(
                    claim="a claim citing a fake segment", seq=bogus_seq, provenance="evidence",
                    quoted_text="some real claim text",
                )]},
            )
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code != 0
    assert not (context_dir / "facts.json").exists(), "a rejected fact must never be written"

    captured = capsys.readouterr()
    assert "a claim citing a fake segment" in captured.out
    assert str(bogus_seq) in captured.out


def test_cli_extract_missing_api_key_exits_nonzero_names_the_env_var_and_writes_nothing(tmp_path, monkeypatch):
    session_dir, _records = _new_session(tmp_path, [("interviewee", "some claim")])
    context_dir = tmp_path / "context"

    # EMPTY, never popped. `_load_dotenv_once` fills in any variable "not
    # already present", so a deleted key is handed straight back from the
    # developer's real `.env` and this test passes for the wrong reason -- the
    # defect `test_coach_context.py`'s docstring records from Phase 4. The
    # OpenAI-compatible seam, the other route to a client, is blanked for every
    # test by `conftest.py::_no_ambient_coach_seam`.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code != 0
    assert not (context_dir / "facts.json").exists()


def test_extract_help_succeeds_and_existing_subcommands_are_unchanged():
    result = subprocess.run(
        [sys.executable, "-m", "app", "extract", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    result_top = subprocess.run(
        [sys.executable, "-m", "app", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result_top.returncode == 0
    for command in ("start", "render", "precheck", "author", "extract"):
        assert command in result_top.stdout


def test_run_extract_never_touches_audio_device_or_creates_a_session_directory():
    # The sibling of test_web_author.py:419 -- proven here by source scan rather
    # than a timed subprocess, since run_extract runs to completion quickly
    # rather than blocking a server forever the way `author` does.
    source = inspect.getsource(main_mod.run_extract)
    assert "construct_pyaudio" not in source
    assert "create_session_dir" not in source


# --- 05-01-PLAN.md Task 3, §K4: the evidence backstop -----------------------------------


def test_extract_facts_accepts_an_evidence_claim_whose_quote_is_verbatim_in_the_cited_segment():
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="Ships happen every Tuesday.", seq=1, provenance="evidence",
                quoted_text="we ship every Tuesday without fail",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1
    assert not result.rejections


def test_extract_facts_refuses_an_evidence_claim_whose_quote_is_nowhere_in_the_cited_segment():
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim with an unlocatable quote", seq=1, provenance="evidence",
                quoted_text="this text never appeared anywhere",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.reason == REASON_UNLOCATABLE_QUOTE
    assert "a claim with an unlocatable quote" in rejection.claim
    assert rejection.quoted_text == "this text never appeared anywhere"


def test_extract_facts_refuses_an_evidence_claim_whose_quote_exists_only_in_a_different_segment():
    # The case that separates a real check from a check that only looks like
    # one: the quote is real, it just came from the WRONG segment.
    cited_segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    other_segment = _segment(2, "interviewee", "our pricing model changed last quarter", 3.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim citing the wrong segment", seq=1, provenance="evidence",
                quoted_text="our pricing model changed last quarter",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[cited_segment, other_segment], hypotheses=(),
        model="m", max_tokens=100, timeout_s=1.0,
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_UNLOCATABLE_QUOTE


def test_extract_facts_refuses_an_evidence_claim_with_an_empty_quote():
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim with no quote at all", seq=1, provenance="evidence", quoted_text="",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert result.rejections[0].reason == REASON_UNLOCATABLE_QUOTE


def test_extract_facts_accepts_an_inference_claim_regardless_of_an_unlocatable_quote():
    # The pair to the "refuses an unlocatable evidence quote" test above --
    # this is what keeps the backstop from quietly banning inferences.
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="an inference the interviewee never said outright", seq=1, provenance="inference",
                quoted_text="this text never appeared anywhere",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1
    assert not result.rejections


def test_extract_facts_accepts_an_inference_claim_with_an_empty_quote():
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="an inference with no quote at all", seq=1, provenance="inference", quoted_text="",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1
    assert not result.rejections


# --- normalization: real mixed Hebrew and English, case by case ------------------------


_MIXED_SEGMENT_TEXT = "אנחנו עושים onboarding תוך שלושה שבועות בערך"


def _accept_evidence(quoted_text: str, *, segment_text: str = _MIXED_SEGMENT_TEXT):
    segment = _segment(1, "interviewee", segment_text, 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim citing mixed Hebrew/English text", seq=1, provenance="evidence",
                quoted_text=quoted_text,
            )]})
        ]
    )
    return extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )


def test_normalization_matches_through_leading_and_trailing_whitespace():
    result = _accept_evidence("   " + _MIXED_SEGMENT_TEXT + "  \t")
    assert len(result.accepted) == 1


def test_normalization_matches_through_a_run_of_internal_whitespace_collapsed_to_one_space():
    noisy = "אנחנו   עושים    onboarding תוך שלושה   שבועות בערך"
    result = _accept_evidence(noisy)
    assert len(result.accepted) == 1


def test_normalization_matches_through_a_letter_case_difference():
    upper_english = _MIXED_SEGMENT_TEXT.replace("onboarding", "ONBOARDING")
    result = _accept_evidence(upper_english)
    assert len(result.accepted) == 1


def test_normalization_matches_through_a_unicode_composition_difference_in_hebrew_text():
    # "café" as a single precomposed é (NFC) vs. e + combining acute accent
    # (NFD) -- a real composition difference, embedded in a mixed
    # Hebrew/English sentence rather than an ASCII-only one.
    composed_segment = "אני אוהב café בבוקר, לפני ה-onboarding call"
    decomposed_quote = "אני אוהב café בבוקר, לפני ה-onboarding call"
    assert composed_segment != decomposed_quote, "sanity: the two strings must differ in raw form"
    result = _accept_evidence(decomposed_quote, segment_text=composed_segment)
    assert len(result.accepted) == 1


def test_normalization_matches_through_an_invisible_bidi_control_mark():
    segment_with_mark = "אנחנו עושים \u200eonboarding תוך שלושה שבועות בערך"
    quote_without_mark = "אנחנו עושים onboarding תוך שלושה שבועות בערך"
    assert segment_with_mark != quote_without_mark, "sanity: the two strings must differ in raw form"
    result = _accept_evidence(quote_without_mark, segment_text=segment_with_mark)
    assert len(result.accepted) == 1


def test_normalization_does_not_match_across_a_wording_or_punctuation_difference():
    # A later well-meaning loosening of the check must fail THIS test, not
    # pass it silently -- the comma below is not in _MIXED_SEGMENT_TEXT.
    result = _accept_evidence("אנחנו עושים onboarding, תוך שלושה שבועות בערך")
    assert not result.accepted
    assert result.rejections[0].reason == REASON_UNLOCATABLE_QUOTE


# =========================================================================================
# STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: citation SPANS, not a single seq
# =========================================================================================
#
# Two real bilingual calls measured a ~5-word median segment with a ~20-word ceiling --
# against that, a single-segment citation cannot carry a claim. `Fact.seqs` (a span of
# 1-6 segments) replaces `Fact.seq`. `app.coach.corpus.validate_citation_span` checks
# three conditions, each with its own reason code: every cited seq resolves
# (REASON_UNRESOLVED_CITATION), every cited seq shares ONE speaker
# (REASON_MIXED_SPEAKER_SPAN), and the cited seqs form a CONSECUTIVE RUN within that
# speaker's own ordered segments (REASON_NONCONTIGUOUS_SPAN) -- other speakers'
# segments interleaving between the cited ones is fine (that is just the other person
# talking); skipping one of the SAME speaker's own segments is not (05-01-PLAN.md §K2's
# original "join that never existed" concern). `_validate_and_build` adds a fourth,
# structural check: the span may not exceed `app.coach.corpus.MAX_SPAN_SEGMENTS`
# (REASON_SPAN_TOO_LONG).


def _fact_span_payload(*, claim: str, seqs, provenance: str, quoted_text: str, related=()) -> dict:
    return {
        "claim": claim, "seqs": list(seqs), "provenance": provenance,
        "quoted_text": quoted_text,
        "bearings": [{"hypothesis_id": h, "stance": "supports"} for h in related],
    }


def test_extract_facts_accepts_a_two_segment_span_from_the_same_speaker_said_back_to_back():
    # The fix itself: the transcription engine split one continuous
    # utterance into two short segments from the SAME speaker, said back to
    # back with nobody else's turn between them.
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    segment_b = _segment(1, "interviewee", "in about three weeks total", 2.0)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="Onboarding takes about three weeks.", seqs=[0, 1], provenance="evidence",
            quoted_text="we onboard a new customer in about three weeks total",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a, segment_b], hypotheses=(),
        model="m", max_tokens=100, timeout_s=1.0,
    )
    assert len(result.accepted) == 1, result.rejections
    assert not result.rejections
    assert result.accepted[0].seqs == (0, 1)


def test_extract_facts_accepts_a_span_with_another_speakers_segment_interleaved_in_the_raw_gap():
    # The POSITIVE control paired with the rejection test right below it --
    # deliberately the IDENTICAL raw-seq shape (one record sitting at seq=1,
    # between the two cited seqs 0 and 2), differing ONLY in who speaks at
    # seq=1. Nothing of the CITED speaker's own is skipped here (the other
    # person's turn is what sits in the gap), so this must be ACCEPTED --
    # proving the guard does not over-reject legitimate interleaving.
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    segment_interviewer = _segment(1, "interviewer", "how long does that take", 1.5)
    segment_b = _segment(2, "interviewee", "in about three weeks total", 2.0)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="Onboarding takes about three weeks.", seqs=[0, 2], provenance="evidence",
            quoted_text="we onboard a new customer in about three weeks total",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a, segment_interviewer, segment_b],
        hypotheses=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert len(result.accepted) == 1, result.rejections
    assert not result.rejections


def test_extract_facts_rejects_a_span_that_skips_one_of_the_same_speakers_own_segments():
    # The "join that never existed" case (05-01-PLAN.md §K2's original
    # concern), and the mutation shape that matters most: the cited speaker
    # has an UNCITED segment of their OWN sitting between the two cited
    # ones -- the identical raw-seq shape as the ACCEPT test right above,
    # so a naive check of the cited seqs' own raw numeric spread (`max(seqs)
    # - min(seqs) + 1 == len(seqs)`) computes the SAME number for both and
    # cannot tell them apart. Only resolving WHO occupies the gap (this
    # speaker's own segment, here, vs. the other speaker, there) does.
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    segment_skipped = _segment(1, "interviewee", "and it is honestly a bit painful", 1.5)
    segment_b = _segment(2, "interviewee", "in about three weeks total", 2.0)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim skipping the interviewee's own middle segment", seqs=[0, 2],
            provenance="evidence", quoted_text="we onboard a new customer in about three weeks total",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a, segment_skipped, segment_b],
        hypotheses=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_NONCONTIGUOUS_SPAN
    assert result.rejections[0].seqs == (0, 2)


def test_extract_facts_rejects_a_span_mixing_two_speakers():
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    segment_b = _segment(1, "interviewer", "how long does that take", 1.5)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim citing across two speakers", seqs=[0, 1], provenance="evidence",
            quoted_text="we onboard a new customer how long does that take",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a, segment_b], hypotheses=(),
        model="m", max_tokens=100, timeout_s=1.0,
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_MIXED_SPEAKER_SPAN


def test_extract_facts_rejects_a_span_longer_than_the_segment_cap():
    too_many = corpus.MAX_SPAN_SEGMENTS + 1
    segments = [_segment(i, "interviewee", f"word{i}", float(i + 1)) for i in range(too_many)]
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim citing too many segments", seqs=list(range(too_many)), provenance="evidence",
            quoted_text=" ".join(f"word{i}" for i in range(too_many)),
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=segments, hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_SPAN_TOO_LONG


def test_extract_facts_accepts_a_span_exactly_at_the_segment_cap():
    # The boundary companion to the rejection test above -- MAX_SPAN_SEGMENTS
    # itself must still be accepted; only ONE MORE than it is refused.
    at_cap = corpus.MAX_SPAN_SEGMENTS
    segments = [_segment(i, "interviewee", f"word{i}", float(i + 1)) for i in range(at_cap)]
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim citing exactly the cap", seqs=list(range(at_cap)), provenance="evidence",
            quoted_text=" ".join(f"word{i}" for i in range(at_cap)),
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=segments, hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1, result.rejections
    assert not result.rejections


def test_extract_facts_refuses_a_span_whose_quote_exists_only_outside_the_cited_span():
    # The span version of "the quote is real, it just came from the WRONG
    # place" -- here the wrong place is a THIRD, uncited segment, not merely
    # a different single segment.
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    segment_b = _segment(1, "interviewee", "in about three weeks total", 2.0)
    segment_c = _segment(2, "interviewee", "pricing changed last quarter", 3.0)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim citing the wrong span for its quote", seqs=[0, 1], provenance="evidence",
            quoted_text="pricing changed last quarter",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a, segment_b, segment_c],
        hypotheses=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_UNLOCATABLE_QUOTE


def test_extract_facts_rejects_a_span_where_one_cited_seq_does_not_resolve():
    segment_a = _segment(0, "interviewee", "we onboard a new customer", 1.0)
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim half-citing a real segment and half a fake one", seqs=[0, 999],
            provenance="evidence", quoted_text="we onboard a new customer",
        )]})]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment_a], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert len(result.rejections) == 1
    assert result.rejections[0].reason == REASON_UNRESOLVED_CITATION


def test_extract_facts_treats_an_empty_seqs_array_as_a_whole_response_parse_failure():
    # A shape violation (like a missing/non-integer `seq` before this fix),
    # not a per-fact quality judgment -- a fact citing NOTHING is malformed,
    # so it fails the WHOLE response rather than becoming a Rejection.
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_span_payload(
            claim="a claim citing nothing", seqs=[], provenance="evidence", quoted_text="x",
        )]})]
    )
    with pytest.raises(ExtractFailure) as excinfo:
        extract_facts(
            stub, session_id="s1", transcript=[_segment(0, "interviewee", "hi", 1.0)],
            hypotheses=(), model="m", max_tokens=100, timeout_s=1.0,
        )
    assert excinfo.value.reason == "parse-error"


def test_extract_tool_schemas_seqs_array_is_bounded_by_the_imported_span_cap():
    # Mirrors gate 18's own provenance-enum/required-set pins (test_phase_gates.py)
    # -- the maxItems bound is IMPORTED from app.coach.corpus.MAX_SPAN_SEGMENTS,
    # never a second literal `6`, so the two can never drift apart.
    from app.coach.extract import _extract_tool_schema

    schema = _extract_tool_schema()
    seqs_schema = schema["input_schema"]["properties"]["facts"]["items"]["properties"]["seqs"]
    assert seqs_schema["type"] == "array"
    assert seqs_schema["items"] == {"type": "integer"}
    assert seqs_schema["minItems"] == 1
    assert seqs_schema["maxItems"] == corpus.MAX_SPAN_SEGMENTS


# --- §K8: an unmatched related_hypothesis_id is a warning, never a rejection -----------


def test_extract_facts_drops_an_unmatched_related_hypothesis_id_as_a_warning_but_still_accepts_the_fact():
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="ships every Tuesday", seq=1, provenance="evidence",
                quoted_text="we ship every Tuesday without fail", related=["nonexistent-hypothesis-id"],
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1, "an unmatched related id must be a warning, never a rejection"
    assert result.accepted[0].related_hypothesis_ids == ()
    assert len(result.dropped_hypothesis_warnings) == 1
    assert "nonexistent-hypothesis-id" in result.dropped_hypothesis_warnings[0]
    assert not result.rejections


def test_extract_facts_keeps_a_related_hypothesis_id_that_matches_a_current_hypothesis():
    hypothesis = Hypothesis(id="h1", statement="onboarding speed drives retention", falsifier="a customer with slow onboarding stays anyway")
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="ships every Tuesday", seq=1, provenance="evidence",
                quoted_text="we ship every Tuesday without fail", related=["h1"],
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(hypothesis,), model="m", max_tokens=100, timeout_s=1.0
    )
    assert result.accepted[0].related_hypothesis_ids == ("h1",)
    assert not result.dropped_hypothesis_warnings


# --- §K5: a mixed run writes only the accepted facts, names every refusal, exits non-zero -


def test_cli_extract_mixed_run_writes_only_accepted_facts_prints_every_refusal_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    session_dir, records = _new_session(tmp_path, [("interviewee", "we ship every Tuesday without fail")])
    good_seq = records[0].seq
    context_dir = tmp_path / "context"

    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [
                _fact_payload(
                    claim="a good, cited fact", seq=good_seq, provenance="evidence",
                    quoted_text="we ship every Tuesday without fail",
                ),
                _fact_payload(
                    claim="a bad fact citing nowhere", seq=999999, provenance="evidence",
                    quoted_text="we ship every Tuesday without fail",
                ),
            ]})
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code != 0, "any rejection must make the whole run exit non-zero"

    raw = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw) == 1, "only the accepted fact may be written -- writing nothing at all would be wrong too"
    assert raw[0]["claim"] == "a good, cited fact"

    captured = capsys.readouterr()
    assert "a bad fact citing nowhere" in captured.out


# --- §K5/gate-12 shape: no claim, quote or segment text ever reaches a log record -------


def test_no_claim_quote_or_segment_text_reaches_a_log_record_from_extraction(caplog):
    marker = "extract-gate-marker-9d4c7b1a"
    segment = _segment(1, "interviewee", marker, 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim=f"a claim mentioning {marker}", seq=1, provenance="evidence", quoted_text=marker,
            )]})
        ]
    )
    caplog.set_level(logging.DEBUG)
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1, "sanity: the marker fact must actually have been processed"
    for record in caplog.records:
        assert marker not in record.getMessage(), (
            f"claim/quote/segment text leaked into a log record: {record.getMessage()!r}"
        )


# =========================================================================================
# 05-02-PLAN.md Task 1: D-08 re-runnability, end to end through the real CLI path
# =========================================================================================


def test_cli_extract_run_twice_with_different_facts_leaves_only_the_second_runs_facts_for_that_session(
    tmp_path, monkeypatch
):
    session_dir, records = _new_session(
        tmp_path, [("interviewee", "we onboard fast"), ("interviewee", "pricing changed last quarter")]
    )
    first_seq, second_seq = records[0].seq, records[1].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))

    stub1 = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="onboarding is fast", seq=first_seq, provenance="evidence", quoted_text="we onboard fast",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub1)
    assert main_mod.main(["extract", str(session_dir)]) == 0
    raw1 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw1) == 1 and raw1[0]["claim"] == "onboarding is fast"

    stub2 = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="pricing changed recently", seq=second_seq, provenance="evidence",
            quoted_text="pricing changed last quarter",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub2)
    assert main_mod.main(["extract", str(session_dir)]) == 0

    raw2 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw2) == 1, "the corpus must hold ONLY the second run's facts for this session -- not both"
    assert raw2[0]["claim"] == "pricing changed recently"
    assert raw2[0]["seqs"] == [second_seq]


def test_cli_extract_run_twice_identically_differs_only_by_id(tmp_path, monkeypatch):
    session_dir, records = _new_session(tmp_path, [("interviewee", "we onboard fast")])
    seq = records[0].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))

    def _payload():
        return {"facts": [_fact_payload(
            claim="onboarding is fast", seq=seq, provenance="evidence", quoted_text="we onboard fast",
        )]}

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, _payload())]),
    )
    assert main_mod.main(["extract", str(session_dir)]) == 0
    raw1 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, _payload())]),
    )
    assert main_mod.main(["extract", str(session_dir)]) == 0
    raw2 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))

    assert len(raw1) == len(raw2) == 1
    assert raw1[0]["id"] != raw2[0]["id"], "a re-run assigns a FRESH id -- identity is never stable across runs (§K2)"
    entry1, entry2 = dict(raw1[0]), dict(raw2[0])
    del entry1["id"]
    del entry2["id"]
    assert entry1 == entry2, "every field except id must be identical across two identical extractions"


def test_cli_extract_a_then_b_then_a_again_leaves_bs_facts_intact(tmp_path, monkeypatch):
    session_a, records_a = _new_session(tmp_path, [("interviewee", "claim about a")])
    session_b, records_b = _new_session(tmp_path, [("interviewee", "claim about b")])
    seq_a, seq_b = records_a[0].seq, records_b[0].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="fact about a", seq=seq_a, provenance="evidence", quoted_text="claim about a",
        )]})]),
    )
    assert main_mod.main(["extract", str(session_a)]) == 0

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="fact about b", seq=seq_b, provenance="evidence", quoted_text="claim about b",
        )]})]),
    )
    assert main_mod.main(["extract", str(session_b)]) == 0

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="fact about a v2", seq=seq_a, provenance="evidence", quoted_text="claim about a",
        )]})]),
    )
    assert main_mod.main(["extract", str(session_a)]) == 0

    raw = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw) == 2, "session B's fact must be neither dropped nor duplicated by re-running session A"
    by_session = {entry["session_id"]: entry for entry in raw}
    assert by_session[session_b.name]["claim"] == "fact about b"
    assert by_session[session_a.name]["claim"] == "fact about a v2"


def test_cli_extract_run_twice_with_fewer_facts_the_second_time_leaves_only_the_fewer(tmp_path, monkeypatch):
    session_dir, records = _new_session(tmp_path, [("interviewee", "claim one"), ("interviewee", "claim two")])
    seq1, seq2 = records[0].seq, records[1].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [
            _fact_payload(claim="fact one", seq=seq1, provenance="evidence", quoted_text="claim one"),
            _fact_payload(claim="fact two", seq=seq2, provenance="evidence", quoted_text="claim two"),
        ]})]),
    )
    assert main_mod.main(["extract", str(session_dir)]) == 0
    raw1 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw1) == 2

    monkeypatch.setattr(
        reasoning_client, "build_client",
        lambda: StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [
            _fact_payload(claim="fact one only", seq=seq1, provenance="evidence", quoted_text="claim one"),
        ]})]),
    )
    assert main_mod.main(["extract", str(session_dir)]) == 0
    raw2 = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw2) == 1, "the shrinking (prompt-fix) case: no fact from the first, larger run may survive"
    assert raw2[0]["claim"] == "fact one only"


def test_cli_extract_aborts_before_the_vendor_call_when_the_existing_corpus_is_not_valid_json(
    tmp_path, monkeypatch, capsys
):
    session_dir, _records = _new_session(tmp_path, [("interviewee", "some claim")])
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "facts.json").write_text("not valid json{{{", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": []})])
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code != 0
    assert stub.calls == [], "a corpus that cannot be parsed must abort BEFORE the vendor is ever called"
    assert (context_dir / "facts.json").read_text(encoding="utf-8") == "not valid json{{{"

    captured = capsys.readouterr()
    assert "facts.json" in captured.err


def test_cli_extract_aborts_when_the_existing_corpus_fails_the_fact_validator(tmp_path, monkeypatch, capsys):
    session_dir, _records = _new_session(tmp_path, [("interviewee", "some claim")])
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    bad_entry = {
        "id": "bad-fact-id", "session_id": "other-session", "seqs": [0], "claim": "x",
        "provenance": "maybe", "quoted_text": "", "bearings": [],
    }
    original_text = json.dumps([bad_entry])
    (context_dir / "facts.json").write_text(original_text, encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": []})])
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code != 0
    assert stub.calls == [], "a corpus that fails the fact validator must also abort before any vendor call"
    captured = capsys.readouterr()
    assert "bad-fact-id" in captured.err, "the message must name the offending item"
    assert (context_dir / "facts.json").read_bytes() == original_text.encode("utf-8"), (
        "the file on disk must be unchanged byte for byte"
    )


def test_cli_extract_against_an_unclosed_session_warns_and_still_completes(tmp_path, monkeypatch, capsys):
    session_dir, records = _new_open_session(tmp_path, [("interviewee", "we onboard fast")])
    seq = records[0].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="onboarding is fast", seq=seq, provenance="evidence", quoted_text="we onboard fast",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0, "an unclosed session must NOT be refused -- only warned about (§K7b)"
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert session_dir.name in captured.err

    raw = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw) == 1, "the warning must not have blocked the run -- a warning that blocks is a refusal"


def test_cli_extract_against_a_truncated_transcript_tail_reports_it_and_still_runs(tmp_path, monkeypatch, capsys):
    session_dir, records = _new_session(tmp_path, [("interviewee", "we onboard fast")])
    seq = records[0].seq
    # Simulate a crash tail directly: an unterminated, invalid-JSON final line.
    transcript_path = session_dir / "transcript.jsonl"
    with open(transcript_path, "ab") as handle:
        handle.write(b'{"kind": "segment", "seq": 1, "speaker": "interviewee", "text": "cut off mid-w')
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="onboarding is fast", seq=seq, provenance="evidence", quoted_text="we onboard fast",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "truncated" in captured.err.lower()
    assert session_dir.name in captured.err

    raw = json.loads((context_dir / "facts.json").read_text(encoding="utf-8"))
    assert len(raw) == 1, "extraction must still run over the records that DID parse"


def test_cli_extract_first_extraction_ever_against_no_corpus_at_all_succeeds_and_creates_it(tmp_path, monkeypatch):
    # D-01: the normal first case, asserted by name rather than left incidental.
    session_dir, records = _new_session(tmp_path, [("interviewee", "we onboard fast")])
    seq = records[0].seq
    context_dir = tmp_path / "context"
    assert not context_dir.exists(), "sanity: no corpus directory exists yet"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="onboarding is fast", seq=seq, provenance="evidence", quoted_text="we onboard fast",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0
    assert (context_dir / "facts.json").is_file(), (
        "D-01: the very first extraction, against no corpus at all, is the normal case, not an edge case"
    )


# =========================================================================================
# 05-02-PLAN.md Task 2: gate 18's schema/validator pin, real cost accounting, the A2 smoke test
# =========================================================================================


def test_extract_facts_reads_usage_off_the_response_when_present():
    response = _FakeResponseWithUsage(
        tool_name=EXTRACT_TOOL_NAME, input_payload={"facts": []}, input_tokens=321, output_tokens=44,
    )
    stub = StubAnthropicClient(responses=[response])
    result = extract_facts(
        stub, session_id="s1", transcript=[], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert result.usage_input_tokens == 321
    assert result.usage_output_tokens == 44


def test_extract_facts_usage_is_none_when_the_response_carries_none_at_all():
    # The offline stub's own ToolUseResponse never carries `.usage` -- every
    # offline test in this suite drives exactly this path; it must never crash.
    stub = StubAnthropicClient(responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": []})])
    result = extract_facts(
        stub, session_id="s1", transcript=[], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert result.usage_input_tokens is None
    assert result.usage_output_tokens is None


def test_cli_extract_prints_accepted_and_rejected_counts_even_when_usage_is_not_reported(
    tmp_path, monkeypatch, capsys
):
    session_dir, records = _new_session(tmp_path, [("interviewee", "we onboard fast")])
    seq = records[0].seq
    context_dir = tmp_path / "context"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context_dir))
    stub = StubAnthropicClient(
        responses=[tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
            claim="onboarding is fast", seq=seq, provenance="evidence", quoted_text="we onboard fast",
        )]})]
    )
    monkeypatch.setattr(reasoning_client, "build_client", lambda: stub)

    exit_code = main_mod.main(["extract", str(session_dir)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "accepted=1" in captured.out
    assert "rejected=0" in captured.out


def test_exactly_four_live_reasoning_marked_tests_exist():
    # 06-03-PLAN.md §H5: updated deliberately, in the SAME commit that adds
    # the THIRD live_reasoning-marked test (the A7 smoke test,
    # test_coach_critic.py) -- including this test's own NAME, which encodes
    # the count, per 05-02's own precedent of asserting the marker count
    # programmatically rather than by prose. The two pre-existing ones:
    # test_coach_generate.py::test_static_block_clears_the_models_minimum_cacheable_prefix
    # (Phase 4, the real-token-count cache-floor check) and
    # test_coach_extract.py::test_live_extraction_distinguishes_a_plainly_stated_claim_from_a_merely_implied_one
    # (05-02, A2). Asserted programmatically over the SOURCE, never by
    # reading pytest's own collection/deselection output, so this stays true
    # as tests are added.
    hits: "list[str]" = []
    for test_dir in (REPO_ROOT / "app" / "tests", REPO_ROOT / "benchmark" / "tests"):
        if not test_dir.is_dir():
            continue
        for py_file in sorted(test_dir.rglob("test_*.py")):
            count = len(_LIVE_REASONING_MARKER_PATTERN.findall(py_file.read_text(encoding="utf-8")))
            hits.extend([str(py_file.relative_to(REPO_ROOT))] * count)
    # Four now. The fourth is the stance probe below -- added in the same
    # commit as this line, which is what this pin exists to force.
    assert len(hits) == 4, f"expected exactly four live_reasoning-marked tests in the repository, found: {hits}"


# --- 05-RESEARCH.md A2: the phase's single highest-priority unverified claim -----------


@pytest.mark.live_reasoning
def test_live_extraction_distinguishes_a_plainly_stated_claim_from_a_merely_implied_one(tmp_path):
    """05-02-PLAN.md §K8/§K9: made genuinely runnable the moment a real
    `ANTHROPIC_API_KEY` exists -- deselected offline by the `live_reasoning`
    marker, and proves NOTHING about model judgment until it actually runs
    against a real key (every other test in this file drives the offline
    stub, which proves only the deterministic pipeline). Asserts the
    DISTINCTION between the two provenances, never a specific label on a
    specific sentence -- an exact-wording assertion is a prompt-wording test
    that will churn every time the prompt changes, and a test that churns is
    a test someone eventually deletes."""
    session_dir, records = _new_session(
        tmp_path,
        [
            ("interviewer", "how many paying customers do you have today, and how does the team feel about growth?"),
            ("interviewee", "We have exactly 42 paying customers today."),
            ("interviewee", "Honestly, it feels like we could double that by next year if the new feature lands."),
        ],
    )
    stated_seq, implied_seq = records[1].seq, records[2].seq
    read_result = read_session(session_dir)
    segments = tuple(SegmentRecord(**r) for r in read_result.records_of_kind("segment"))

    # Built through the ONE construction site (§K8's own instruction) -- a
    # missing key produces reasoning_client.CoachConfigError, that site's own
    # named error, rather than an obscure failure somewhere downstream.
    client = reasoning_client.build_client()
    result = extract_facts(
        client, session_id=session_dir.name, transcript=segments, hypotheses=(),
        model=reasoning_client.DEFAULT_MODEL, max_tokens=1000, timeout_s=60.0,
    )
    assert result.accepted, "the real model must extract at least the plainly-stated fact"
    provenance_by_seq = {seq: fact.provenance for fact in result.accepted for seq in fact.seqs}
    assert provenance_by_seq.get(stated_seq) == "evidence", (
        "the plainly-stated customer count must be labelled evidence"
    )
    assert provenance_by_seq.get(implied_seq) == "inference", (
        "the merely-implied growth guess must be labelled inference, not evidence"
    )


# --- does the extractor judge DIRECTION, or only ever link? -----------------------------

# The A7 smoke test's shape, applied to `bearings`. Hand-authored utterances,
# each written to bear unambiguously one way on a REAL hypothesis from the
# eval project -- two of them meeting a falsifier almost verbatim, so a model
# that reads only surface positivity gets them wrong. Module-level inspectable
# data on purpose: a future reader can read one and disagree with the label,
# which is the only way this test stays honest.
#
# WHY IT EXISTS. `bearings` replaced a field that carried no direction, and
# the first real extraction against it returned ZERO bearings across 23 facts
# -- which was ambiguous in exactly the way that bites this project: the
# transcript was career biography with genuinely nothing bearing on the
# hypotheses, so "correctly declined" and "field is decorative" looked
# identical. This is the run that separates them.
#
# NO PASS THRESHOLD IS ASSERTED. The counts print for a human to read, like
# the A7 test. What IS asserted is only that the model returned bearings at
# all and used more than one stance -- the two failures that would make the
# field decorative.

_STANCE_PROBES = (
    (
        "interviewee",
        "Honestly the notepad has nothing to do with the recorder. I'd keep it even with a "
        "recording I trusted completely. It's how I mark a moment to come back to, and it "
        "lets me break eye contact so they keep talking.",
        "undermines h1 -- this is h1's falsifier almost verbatim",
    ),
    (
        "interviewee",
        "We used my reconstructed notes in the analysis exactly like the transcribed ones. "
        "Honestly it changed nothing about how that interview was used.",
        "undermines h3 -- meets h3's falsifier",
    ),
    (
        "interviewee",
        "It was the deadline, plainly. Three months, back to back, and the analysis got "
        "whatever was left. The software was fine, I just never had the weeks it needed.",
        "supports h2",
    ),
    (
        "interviewee",
        "Yes -- after the recorder died on me that first time I never trusted it again, so "
        "I wrote everything down as backup.",
        "supports h1",
    ),
)


@pytest.mark.live_reasoning
def test_live_extraction_judges_the_direction_a_claim_bears_not_merely_the_link(tmp_path, capsys):

    from app.coach.corpus import Hypothesis

    hypotheses = (
        Hypothesis(
            id="h1",
            statement=(
                "Researchers keep a manual note-taking fallback during interviews primarily "
                "because they do not trust their recording equipment."
            ),
            falsifier=(
                "If she says the notepad is kept for reasons unrelated to recording failure "
                "-- steering the conversation, marking moments, managing eye contact -- and "
                "would keep it even with a recording she fully trusted, this is false."
            ),
        ),
        Hypothesis(
            id="h2",
            statement=(
                "The binding constraint on a contract researcher's analysis is time pressure, "
                "not the analysis tooling available to them."
            ),
            falsifier="If she attributes weak analysis mainly to the tools, this is false.",
        ),
        Hypothesis(
            id="h3",
            statement=(
                "When an interview's audio is lost, the reconstructed notes are treated as "
                "substantially less usable evidence, so the interview is effectively wasted."
            ),
            falsifier=(
                "If she reports the reconstructed notes were used on the same footing as "
                "transcribed interviews, this is false."
            ),
        ),
    )

    transcript = tuple(
        _segment(seq=index, speaker=speaker, text=text, audio_end_s=float(index) + 1.0)
        for index, (speaker, text, _label) in enumerate(_STANCE_PROBES)
    )
    result = extract_facts(
        reasoning_client.build_client(),
        session_id="stance-probe",
        transcript=transcript,
        hypotheses=hypotheses,
        model=reasoning_client.DEFAULT_MODEL,
        max_tokens=2000,
        timeout_s=90.0,
    )

    stances: "dict[str, int]" = {}
    lines: "list[str]" = []
    for fact in result.accepted:
        for bearing in fact.bearings:
            stances[bearing.stance] = stances.get(bearing.stance, 0) + 1
            lines.append(f"  [{bearing.stance:10}] {bearing.hypothesis_id}  <- {fact.claim[:70]}")

    with capsys.disabled():
        print("\n--- what a careful human would say ---")
        for _s, _t, label in _STANCE_PROBES:
            print(f"  {label}")
        print("--- what the model said ---")
        print("\n".join(lines) or "  (no bearings at all)")
        print(f"--- stance tally: {stances or '(none)'} ---")
        print("Read the two lists against each other. No threshold is asserted here.")

    assert stances, (
        "the model returned no bearings at all on utterances written to bear unambiguously "
        "on these hypotheses -- `bearings` would be a decorative field"
    )
    assert len(stances) > 1, (
        f"every bearing got the same stance ({stances}) -- a model that only ever says "
        "'supports' is the monotonic count this field replaced"
    )


# --- the A2 yes/no guard: a quote that states nothing cannot BE evidence ---------------
#
# STATE 2026-08-16: A2 ("Haiku 4.5 reliably distinguishes evidence from
# inference") is falsified, and the DOMINANT failure mode is a claim whose
# whole support is the interviewee agreeing with the interviewer's question --
# claim "I keep a spare laptop beside the console." quoted to "Yep." The quote is genuinely
# verbatim, so every gate this pipeline had passes it. The claim is usually
# TRUE; what is wrong is the LABEL, which is why this guard DEMOTES and never
# rejects -- rejecting would delete a true fact.


def test_extract_facts_demotes_an_evidence_claim_whose_quote_is_only_an_affirmation():
    segment = _segment(1, "interviewee", "Yep.", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="I keep a spare laptop beside the console.", seq=1, provenance="evidence", quoted_text="Yep.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.accepted) == 1
    assert result.accepted[0].provenance == PROVENANCE_INFERENCE
    assert result.accepted[0].claim == "I keep a spare laptop beside the console."


def test_extract_facts_demoting_a_contentless_quote_never_rejects_the_fact():
    # The whole point of the design: the claim is usually true, so deleting it
    # is a WORSE error than mislabelling it.
    segment = _segment(1, "interviewee", "Not really. Uh, no.", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="I have not changed my setup this year.",
                seq=1, provenance="evidence", quoted_text="Not really. Uh, no.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.rejections
    assert len(result.accepted) == 1
    assert result.accepted[0].provenance == PROVENANCE_INFERENCE


def test_a_demotion_is_reported_by_name_and_kept_separate_from_rejections():
    segment = _segment(1, "interviewee", "Yeah, exactly.", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim resting on agreement alone", seq=1, provenance="evidence",
                quoted_text="Yeah, exactly.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert len(result.demotions) == 1
    assert "a claim resting on agreement alone" in result.demotions[0].claim
    assert result.demotions[0].quoted_text == "Yeah, exactly."
    assert result.demotions[0].reason == REASON_CONTENTLESS_QUOTE


def test_extract_facts_leaves_an_evidence_quote_carrying_one_content_word_alone():
    # The near-miss boundary, taken from the real corpus: "That can be
    # annoying." is FOUR particles and one content word, and it is a real
    # statement. One content token is enough.
    segment = _segment(1, "interviewee", "That can be annoying.", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="He finds it annoying.", seq=1, provenance="evidence",
                quoted_text="That can be annoying.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.demotions
    assert result.accepted[0].provenance == PROVENANCE_EVIDENCE


@pytest.mark.parametrize(
    "quoted_text",
    [
        "Uh-huh.",
        "For sure.",
        "Pretty much.",
        "Not at all.",
        "No way.",
        "Sure thing.",
        "More or less.",
        "A little bit.",
    ],
)
def test_the_guard_flags_the_yes_no_answers_that_the_first_token_set_let_through(quoted_text):
    """The measured gaps in the SHIPPED token set (2026-08-16 audit, finding
    4). Every one of these is a pure yes/no answer that states nothing on its
    own, and every one slipped past the first version of `_CONTENTLESS_TOKENS`
    on ONE missing token -- "huh", "for", "pretty", "at", "all", "way",
    "thing", "more", "less", "little". A quote that says only "For sure."
    cannot support a claim, and mode 1 is exactly the failure of accepting one
    that does.

    Parametrized one phrase per case on purpose: a single test asserting all
    eight would go green the moment ANY of them flagged, which is how a
    partial widening gets mistaken for a complete one."""
    assert quote_states_nothing(quoted_text)


@pytest.mark.parametrize("quoted_text", ["That can be annoying.", "I think so."])
def test_widening_the_token_set_did_not_swallow_the_two_quotes_that_must_stay_content(quoted_text):
    """The regression pins for the widening, and the two are pinned for
    DIFFERENT reasons.

    "That can be annoying." is the near-miss from the real corpus -- four
    particles and one content word. One content token is enough, by design,
    and every token added above is a fresh chance to swallow that last word.

    "I think so." is mode 2, the hedged answer, and it is DELIBERATELY out of
    this guard's scope. The lexical hedge scan that would catch it measured
    100% false positives on the real corpora (see the module docstring) and
    was thrown away. If a future widening makes this one flag, mode 2 has
    leaked into mode 1's guard and the measurement that justifies the set no
    longer describes it."""
    assert not quote_states_nothing(quoted_text)


def test_a_contentless_quote_that_is_not_in_the_cited_span_is_still_rejected_not_demoted():
    # Order matters, and this test is what pins it. The locatability check
    # runs on the model's OWN label, before any demotion -- otherwise
    # demotion would launder a FABRICATED quote into an accepted inference.
    segment = _segment(1, "interviewee", "we ship every Tuesday without fail", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="a claim quoting a yes that was never said", seq=1, provenance="evidence",
                quoted_text="Yep.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.accepted
    assert not result.demotions
    assert result.rejections[0].reason == REASON_UNLOCATABLE_QUOTE


def test_an_inference_claim_is_never_demoted_and_never_flagged():
    segment = _segment(1, "interviewee", "Yep.", 2.0)
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(EXTRACT_TOOL_NAME, {"facts": [_fact_payload(
                claim="an inference already labelled correctly", seq=1, provenance="inference",
                quoted_text="Yep.",
            )]})
        ]
    )
    result = extract_facts(
        stub, session_id="s1", transcript=[segment], hypotheses=(), model="m", max_tokens=100, timeout_s=1.0
    )
    assert not result.demotions
    assert result.accepted[0].provenance == PROVENANCE_INFERENCE


def test_the_guard_flags_only_the_two_known_facts_across_every_real_corpus_on_this_machine():
    """The standing rule: run a new heuristic over the REAL corpus before
    believing green tests. Measured 2026-08-16 over BOTH shipped fact corpora
    -- 2 flagged out of 46, 0 false positives, and the 2 are exactly the pair
    hand-adjudicated as wrong.

    ⚠️ Scans EVERY fact, not only the `evidence` ones. The first version of
    this test scanned `evidence` alone and broke the moment the guard did its
    job: the Opus 5 re-extraction demoted both offenders, they were stored as
    `inference`, and a filter on `evidence` then found nothing and read that
    as "the guard stopped working". A measurement that disappears when the fix
    lands measures the wrong thing.

    So the provenance assertion below is the interesting half -- both flagged
    facts are `inference` in the corpus ON DISK, which is the demotion having
    survived a REAL end-to-end extraction, not a stub.

    If this fails because the corpora grew, RE-MEASURE by hand; do not relax
    the assertion."""
    repo_root = Path(__file__).resolve().parents[2]
    corpora = sorted(repo_root.glob("projects/*/context/facts.json"))
    if not corpora:
        pytest.skip("no local fact corpus under projects/ -- this test measures the operator's own data")

    flagged: "list[tuple[str, str]]" = []
    fact_count = 0
    for path in corpora:
        payload = json.loads(path.read_text(encoding="utf-8"))
        facts = payload["facts"] if isinstance(payload, dict) else payload
        for fact in facts:
            fact_count += 1
            if quote_states_nothing(fact["quoted_text"]):
                flagged.append((fact["provenance"], fact["quoted_text"]))

    assert fact_count >= 45, f"corpus shrank to {fact_count} facts -- re-measure"
    assert flagged == [
        (PROVENANCE_INFERENCE, "Yep."),
        (PROVENANCE_INFERENCE, "Not really. Uh, no."),
    ], f"the guard's behaviour over real data changed: {flagged}"
