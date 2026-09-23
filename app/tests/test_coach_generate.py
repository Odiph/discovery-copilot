"""Tests for `app/coach/generate.py` and `app/coach/prompts.py` -- D3/D4's
prompt-assembly rules (the whole survey including asked items, the whole
untruncated transcript, the trigger explicitly distinguished, the pinned
move set, cache-breakpoint byte-stability, and the cache-floor sanity check).
Behavioural test names, per this project's own convention."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.coach import prompts
from app.coach.context import ContextItem
from app.coach.generate import (
    ALL_MOVES,
    GENERATE_TOOL_NAME,
    MOVE_ADVANCE,
    MOVE_DEEPEN,
    MOVE_FILL_GAP,
    MOVE_NO_QUESTION,
    MOVE_TEST_HYPOTHESIS,
    SHOWABLE_MOVES,
    AskedAt,
    Candidate,
    GenerateFailure,
    _generate_tool_schema,
    assemble_generate_request,
    run_generate,
)
from app.coach.context import SurveyItem
from app.coach.stub import StubAnthropicClient, ToolUseResponse, tool_use_response
from app.session.store import SegmentRecord


def _hypothesis(*, id: str = "h1", statement: str = "onboarding delay is the top churn reason",
                 falsifier: str = "churned customers cite a different top reason more often") -> SimpleNamespace:
    return SimpleNamespace(id=id, statement=statement, falsifier=falsifier)


def _segment(seq: int, speaker: str, text: str, audio_end_s: float) -> SegmentRecord:
    return SegmentRecord(
        kind="segment", seq=seq, speaker=speaker, text=text, audio_start_s=audio_end_s - 1.0,
        audio_end_s=audio_end_s, ready_wall_s=audio_end_s + 0.1, latency_s=0.1,
        detected_language="en", engine="fake", written_at="2026-08-06T00:00:00.000+00:00",
    )


def _run_generate(client):
    """One `run_generate` call with the minimum context the stage needs. The
    stub answers whatever it was primed with, so what is under test here is
    the PARSE step, not the prompt."""
    return run_generate(
        client, brief=(), survey=(), asked={}, transcript=[],
        trigger=_segment(1, "interviewee", "something they just said", 10.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m",
        max_tokens=100, timeout_s=1.0,
    )


def _basic_request(**overrides):
    survey = (
        SurveyItem(id="s1", order=0, text="Walk me through onboarding"),
        SurveyItem(id="s2", order=1, text="How do you track KPIs today"),
    )
    asked = {"s1": AskedAt(seq=3, audio_end_s=12.5)}
    transcript = [
        _segment(0, "interviewer", "Let's get started", 1.0),
        _segment(1, "interviewee", "Sure, sounds good", 2.0),
    ]
    trigger = _segment(2, "interviewee", "trigger text here", 3.0)
    kwargs = dict(
        brief=(), survey=survey, asked=asked, transcript=transcript, trigger=trigger,
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100,
    )
    kwargs.update(overrides)
    return assemble_generate_request(**kwargs)


# --- D3/D4: prompt assembly -----------------------------------------------------------


def test_generate_prompt_contains_every_survey_item_both_asked_and_unasked_with_status():
    request = _basic_request()
    survey_text = request["system"][1]["text"]
    assert "s1" in survey_text and "ASKED" in survey_text
    assert "s2" in survey_text and "UNASKED" in survey_text


def test_generate_prompt_asked_items_carry_the_matched_seq_and_time():
    request = _basic_request()
    survey_text = request["system"][1]["text"]
    assert "seq=3" in survey_text
    assert "12.5" in survey_text


def test_generate_prompt_contains_the_whole_untruncated_transcript():
    long_transcript = [_segment(i, "interviewer" if i % 2 == 0 else "interviewee", f"segment number {i}", float(i)) for i in range(200)]
    request = _basic_request(transcript=long_transcript)
    transcript_text = request["system"][2]["text"]
    assert "segment number 0" in transcript_text, "a naive tail-slice would have dropped the earliest segment"
    assert "segment number 199" in transcript_text


def test_generate_prompt_trigger_is_present_and_explicitly_distinguished_from_the_transcript():
    request = _basic_request()
    trigger_message = request["messages"][-1]["content"]
    assert "trigger text here" in trigger_message
    assert "just said" in trigger_message.lower()


def test_generate_allowed_moves_are_the_complete_set_pinned_against_the_schema():
    # 06-02-PLAN.md §G1: deliberately, consciously updated here in the SAME
    # commit that grows ALL_MOVES -- the docstring's own predicted failure,
    # fixed on purpose, matching how SERVER_MESSAGE_TYPES was grown in 04-03.
    # Grown again to five on 2026-08-15 by `no_question`, and this assertion
    # is once again what failed first.
    schema_moves = set(_generate_tool_schema()["input_schema"]["properties"]["candidates"]["items"]["properties"]["move"]["enum"])
    assert schema_moves == ALL_MOVES == {
        MOVE_ADVANCE, MOVE_DEEPEN, MOVE_FILL_GAP, MOVE_TEST_HYPOTHESIS, MOVE_NO_QUESTION,
    }


def test_an_empty_candidate_array_is_still_a_parse_failure():
    """The gap this fills: nothing anywhere asserted what `{"candidates": []}`
    did, so a silent change to it would have been invisible.

    It stays a FAILURE on purpose, now that `no_question` exists. "The model
    returned nothing" and "the model judged nothing should be asked" are
    different events and must not arrive through the same channel -- the first
    is a broken response, the second is a considered answer that gets recorded
    as one. Collapsing them would make a malfunctioning vendor look exactly
    like a coach exercising judgement."""
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": []})]
    )
    with pytest.raises(GenerateFailure) as excinfo:
        _run_generate(stub)
    assert excinfo.value.reason == "parse-error"


def test_declining_is_expressed_as_a_candidate_not_as_an_empty_array():
    """The other half of the pair above: the supported way to say nothing is a
    single `no_question` candidate, and it parses normally."""
    stub = StubAnthropicClient(responses=[tool_use_response(GENERATE_TOOL_NAME, {
        "candidates": [{"text": "", "move": MOVE_NO_QUESTION}],
    })])
    candidates = _run_generate(stub)
    assert len(candidates) == 1
    assert candidates[0].move == MOVE_NO_QUESTION


def test_the_showable_moves_are_every_move_except_the_declining_one():
    """`SHOWABLE_MOVES` is derived, so a future sixth move joins it by default.
    That is the safe direction: a move nobody remembered to add to a hand-typed
    list would otherwise be generated and then silently never shown."""
    assert SHOWABLE_MOVES == ALL_MOVES - {MOVE_NO_QUESTION}
    assert MOVE_NO_QUESTION not in SHOWABLE_MOVES


def test_every_move_name_appears_in_the_assembled_generate_prompt_text():
    # 06-02-PLAN.md §G1/§G6: the schema enum is DERIVED from ALL_MOVES and can
    # never drift by itself -- the hand-typed prose in _SYSTEM_PREAMBLE and
    # decision_instruction is what the model actually reads, and is what this
    # asserts over, not the schema.
    request = _basic_request()
    combined = "\n".join(block["text"] for block in request["system"]) + "\n" + request["messages"][-1]["content"]
    for move in ALL_MOVES:
        assert move in combined, f"move {move!r} never appears in the assembled generate prompt text"


def test_static_block_assembles_to_identical_bytes_built_twice_a_second_apart():
    brief = (ContextItem(id="b1", text="the customer is a mid-market SaaS company", provenance="evidence"),)
    first = prompts.static_block(brief=brief, hypotheses=())
    time.sleep(1.1)
    second = prompts.static_block(brief=brief, hypotheses=())
    assert first == second, "static prompt block is not byte-stable -- every cache breakpoint after it is dead"


def test_static_block_offline_length_floor_guards_against_a_trimmed_rubric():
    # The REAL 4,096-token floor can only be confirmed against the live API
    # (see the live_reasoning-marked test below); this is a cheap, offline
    # regression guard so the rubric/few-shot content cannot silently shrink
    # back down without a test noticing.
    assert len(prompts.static_block(brief=(), hypotheses=())) > prompts.MIN_STATIC_BLOCK_CHARS


@pytest.mark.live_reasoning
def test_static_block_clears_the_models_minimum_cacheable_prefix():
    # Needs a real endpoint -- deselected from the offline suite. Confirms the
    # ACTUAL token count, not the character-length proxy above.
    from app.coach import reasoning_client

    client = reasoning_client.build_client()
    brief = (ContextItem(id="b1", text="short example brief", provenance="evidence"),)
    result = client.messages.count_tokens(
        model=reasoning_client.DEFAULT_MODEL,
        system=prompts.static_block(brief=brief, hypotheses=()),
        messages=[{"role": "user", "content": "placeholder"}],
    )
    assert result.input_tokens >= 4096


# --- 06-01-PLAN.md §F3: the retrieved-facts block partition inside static_block ------


def test_static_block_with_a_fact_derived_item_carries_its_id_claim_source_and_disclosure():
    from app.coach.context import FACT_ID_PREFIX

    fact_item = ContextItem(
        id=f"{FACT_ID_PREFIX}f1", text="onboarding takes three weeks", source="past interview sess-1",
        citation="session:sess-1#seq:4", provenance="evidence",
    )
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    block = prompts.static_block(brief=(hand_written, fact_item), hypotheses=())

    assert "hand-written brief item" in block
    assert f"{FACT_ID_PREFIX}f1" in block, "a fact-derived item's id must be rendered so a later self-report can cite it"
    assert "onboarding takes three weeks" in block
    assert "past interview sess-1" in block
    assert "not settled truth" in block, "the per-source-claim disclosure sentence must be present"


def test_static_block_with_no_fact_derived_items_omits_the_retrieved_facts_block_entirely():
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    with_only_hand_written = prompts.static_block(brief=(hand_written,), hypotheses=())
    assert "Facts from past interviews" not in with_only_hand_written
    assert "not settled truth" not in with_only_hand_written


# --- 06-02-PLAN.md §G3: the hypotheses block partition inside static_block -----------


def test_static_block_with_a_hypothesis_carries_its_id_statement_and_falsifier():
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    block = prompts.static_block(brief=(hand_written,), hypotheses=(_hypothesis(),))
    assert "h1" in block
    assert "onboarding delay is the top churn reason" in block
    assert "churned customers cite a different top reason more often" in block


def test_static_block_with_no_hypotheses_omits_the_hypotheses_block_entirely():
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    block = prompts.static_block(brief=(hand_written,), hypotheses=())
    assert "Hypotheses the interviewer has written down" not in block


def test_static_block_with_hypotheses_present_and_no_facts_has_hypotheses_but_no_facts_block():
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    block = prompts.static_block(brief=(hand_written,), hypotheses=(_hypothesis(),))
    assert "Hypotheses the interviewer has written down" in block
    assert "Facts from past interviews" not in block


def test_static_block_with_neither_facts_nor_hypotheses_is_byte_identical_to_phase_four():
    # ROADMAP criterion 7/GC-07, precisely: the nothing-authored case must
    # stay byte-identical, not merely equivalent, to what Phase 4 produced --
    # both new blocks (retrieved facts, hypotheses) must be absent together.
    hand_written = (ContextItem(id="b1", text="hand-written brief item", provenance="evidence"),)
    with_both_empty = prompts.static_block(brief=hand_written, hypotheses=())
    twice_more = prompts.static_block(brief=hand_written, hypotheses=())
    assert with_both_empty == twice_more
    assert "Facts from past interviews" not in with_both_empty
    assert "Hypotheses the interviewer has written down" not in with_both_empty


def test_generate_prompt_instructs_the_detected_language_reaches_the_prompt():
    # Assert the MECHANISM (the detected language reaches the request), not
    # the prose, which will churn.
    en_request = _basic_request(detected_language="en")
    he_request = _basic_request(detected_language="he")
    en_message = en_request["messages"][-1]["content"]
    he_message = he_request["messages"][-1]["content"]
    assert en_message != he_message
    assert "English" in en_message
    assert "Hebrew" in he_message


# --- The Hebrew suggestion in an English interview (2026-08-15) ----------------------
#
# One suggestion of 29 in a real coached, English-only interview came back in
# Hebrew and was SHOWN (session 20260815T222827, suggestion seq=57). Detection
# was ruled out first and is pinned in `test_coach_language.py`: replaying that
# session's own interviewee segments returns "en" at that cycle and at every
# other one, from either prior. So the request below is the one the model
# actually got, and the English instruction was in it.
#
# What it was competing with is measurable and is what these tests pin:
# `static_block` ships three worked HEBREW examples on every call, English ones
# included, and the transcript block carried the interviewer's own
# Hebrew-rendered asides. The instruction named neither, so the model was left
# to resolve the conflict and resolved it the wrong way.
#
# These assertions are about the SUBSTANCE of the instruction, not its wording:
# each one names a specific thing the model must be told, and every one of them
# was absent before. A rewrite that keeps the substance keeps them passing.


def test_the_english_instruction_says_the_transcripts_own_language_does_not_decide():
    message = _basic_request(detected_language="en")["messages"][-1]["content"]
    assert "transcript" in message.lower(), (
        "the instruction must name the transcript's language as something to "
        "disregard -- an unqualified 'write in English' is what shipped, and it lost"
    )


def test_the_english_instruction_disarms_the_hebrew_code_switch_examples_by_name():
    """`static_block` appends `_CODE_SWITCH_EXAMPLES` unconditionally -- three
    well-formed Hebrew questions sitting in the cached prefix of an
    English-only call. Dropping them for English would make the cached prefix
    language-dependent and cost D4's byte-stability contract (see the comment
    above `prompts._LANGUAGE_OVERRIDE_NOTE`), so the instruction must instead
    say plainly that they do not apply here."""
    system = "".join(block["text"] for block in _basic_request(detected_language="en")["system"])
    assert "code-switch" in system.lower(), (
        "premise of this test: the Hebrew register examples really are in the "
        "English call's prompt"
    )
    message = _basic_request(detected_language="en")["messages"][-1]["content"]
    assert "code-switch" in message.lower() and "DO NOT APPLY" in message


def test_the_hebrew_instruction_gets_the_same_override_note_not_only_the_english_one():
    """Symmetry, and not for tidiness. The failure was the model following the
    transcript's surface language instead of the instruction; a Hebrew
    interview transcribed with English-rendered stretches is the same failure
    mirrored, and fixing only the branch that happened to break first leaves
    the other one to be discovered in production."""
    he_message = _basic_request(detected_language="he")["messages"][-1]["content"]
    assert prompts._LANGUAGE_OVERRIDE_NOTE in he_message
    assert prompts._LANGUAGE_OVERRIDE_NOTE in _basic_request(detected_language="en")["messages"][-1]["content"]


def test_the_code_switch_block_no_longer_claims_to_be_gated_by_the_final_instruction():
    """It said "see the final instruction in your prompt, which names the
    language". The language instruction is NOT final -- `decision_instruction`
    follows it (generate.py's user content). A model looking for its gate at
    the end of the prompt finds a decision instruction that names no language
    at all, which is a bad place to leave a block of Hebrew examples."""
    system = "".join(block["text"] for block in _basic_request(detected_language="en")["system"])
    assert "final instruction in your prompt" not in system


def test_generate_prompt_caches_every_block_except_the_volatile_transcript_tail():
    """D4's breakpoints, after the transcript split.

    The old shape put a breakpoint at the END of a transcript block that grew
    every cycle, so the entry written by one call could never be matched by
    the next: a 1.25x write premium on the whole prompt, paid every cycle,
    never read back. The tail is now deliberately UNCACHED, and the stable
    head sits behind the third breakpoint."""
    transcript = [_segment(i, "interviewee", f"segment number {i}", float(i)) for i in range(60)]
    request = _basic_request(transcript=transcript)
    system_blocks = request["system"]
    assert len(system_blocks) == 4
    for block in system_blocks[:-1]:
        assert block["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_blocks[-1], (
        "the transcript tail changes every cycle -- caching it recreates the defect"
    )


def test_the_cached_prefix_holds_still_across_cycles_so_it_can_be_read(tmp_path=None):
    """The whole point. Bytes identical across a run of cycles == a cache HIT;
    bytes that differ every cycle == a write that is never read."""
    base = [_segment(i, "interviewee", f"segment number {i}", float(i)) for i in range(50)]
    prefixes = []
    for extra in range(6):  # transcript grows by one segment per cycle
        transcript = base + [
            _segment(50 + j, "interviewee", f"later segment {j}", float(50 + j))
            for j in range(extra)
        ]
        request = _basic_request(transcript=transcript)
        cached = [b["text"] for b in request["system"] if "cache_control" in b]
        prefixes.append(tuple(cached))
    assert len(set(prefixes)) == 1, (
        "the cached prefix changed while the transcript grew -- every cycle would "
        "write a fresh entry and none would ever be read"
    )


def test_the_cached_prefix_advances_once_the_chunk_boundary_is_crossed():
    """It must not be frozen either -- a prefix that never grows would push the
    whole interview into the uncached tail."""
    short = [_segment(i, "interviewee", f"segment {i}", float(i)) for i in range(24)]
    long = [_segment(i, "interviewee", f"segment {i}", float(i)) for i in range(60)]
    cached_short = [b["text"] for b in _basic_request(transcript=short)["system"] if "cache_control" in b]
    cached_long = [b["text"] for b in _basic_request(transcript=long)["system"] if "cache_control" in b]
    assert len(cached_short) == 2, "below one chunk there is no stable transcript yet"
    assert len(cached_long) == 3
    assert "segment 49" in cached_long[-1]


def test_the_whole_transcript_still_reaches_the_model_across_the_split():
    """The split is a caching concern and must be invisible to the model:
    every segment present, exactly one heading."""
    transcript = [_segment(i, "interviewee", f"unique text {i}", float(i)) for i in range(60)]
    request = _basic_request(transcript=transcript)
    system_text = "\n".join(b["text"] for b in request["system"])
    for i in range(60):
        assert f"unique text {i}" in system_text
    # Counted as a whole LINE, not a substring: the rubric prose in
    # static_block also says "transcript so far" while instructing the model,
    # and a substring count picks that up and fails for the wrong reason.
    heading_lines = [line for line in system_text.splitlines() if line == prompts._TRANSCRIPT_HEADER]
    assert len(heading_lines) == 1, "two headings read as two transcripts"


def test_a_zero_chunk_disables_the_split_and_keeps_one_transcript_block():
    transcript = [_segment(i, "interviewee", f"segment {i}", float(i)) for i in range(60)]
    request = _basic_request(transcript=transcript, transcript_cache_chunk=0)
    assert len(request["system"]) == 3
    assert "cache_control" not in request["system"][-1]


# --- run_generate: parsing and failure discipline --------------------------------------


def _candidate_payload(text: str, move: str) -> dict:
    return {"text": text, "move": move}


def test_run_generate_parses_a_normal_response_into_candidates():
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                GENERATE_TOOL_NAME,
                {"candidates": [_candidate_payload("first candidate", MOVE_ADVANCE), _candidate_payload("second", MOVE_DEEPEN)]},
            )
        ]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates == [Candidate(index=0, text="first candidate", move=MOVE_ADVANCE), Candidate(index=1, text="second", move=MOVE_DEEPEN)]


def test_run_generate_passes_its_own_explicit_timeout_and_it_differs_from_a_default():
    stub = StubAnthropicClient(responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [_candidate_payload("x", MOVE_ADVANCE)]})])
    run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=2.5,
    )
    assert stub.calls[0].timeout_s == 2.5


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_run_generate_treats_refusal_and_truncation_as_untrusted(stop_reason):
    stub = StubAnthropicClient(responses=[ToolUseResponse(content=(), stop_reason=stop_reason)])
    with pytest.raises(GenerateFailure) as excinfo:
        run_generate(
            stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
            detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
        )
    assert excinfo.value.stop_reason == stop_reason


def test_run_generate_raises_on_an_unparseable_body():
    stub = StubAnthropicClient(responses=[tool_use_response(GENERATE_TOOL_NAME, {})])  # missing "candidates"
    with pytest.raises(GenerateFailure):
        run_generate(
            stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
            detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
        )


def test_run_generate_wraps_any_exception_from_the_call_itself():
    stub = StubAnthropicClient(responses=[TimeoutError("simulated")])
    with pytest.raises(GenerateFailure) as excinfo:
        run_generate(
            stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
            detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
        )
    assert excinfo.value.stop_reason is None


# --- 06-02-PLAN.md §G3/§G5: a test_hypothesis candidate carries its target -------------


def _candidate_payload_with_target(text: str, move: str, target_hypothesis_id) -> dict:
    return {"text": text, "move": move, "target_hypothesis_id": target_hypothesis_id}


def test_run_generate_parses_a_test_hypothesis_candidate_with_its_target_id():
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                GENERATE_TOOL_NAME,
                {"candidates": [_candidate_payload_with_target("does X still happen?", MOVE_TEST_HYPOTHESIS, "h1")]},
            )
        ]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates == [
        Candidate(index=0, text="does X still happen?", move=MOVE_TEST_HYPOTHESIS, target_hypothesis_id="h1")
    ]


def test_run_generate_drops_an_unrecognized_target_hypothesis_id_but_keeps_the_candidate():
    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                GENERATE_TOOL_NAME,
                {"candidates": [_candidate_payload_with_target("does X still happen?", MOVE_TEST_HYPOTHESIS, "h-invented")]},
            )
        ]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert len(candidates) == 1, "an invented target_hypothesis_id must never cost the interviewer a candidate"
    assert candidates[0].target_hypothesis_id is None
    assert candidates[0].text == "does X still happen?"


def test_run_generate_drops_an_unrecognized_target_hypothesis_id_and_logs_only_a_count(caplog):
    import logging

    stub = StubAnthropicClient(
        responses=[
            tool_use_response(
                GENERATE_TOOL_NAME,
                {"candidates": [_candidate_payload_with_target("x", MOVE_TEST_HYPOTHESIS, "h-secret-user-typed-this")]},
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        run_generate(
            stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
            detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
        )
    formatted = "\n".join(record.getMessage() for record in caplog.records)
    assert "h-secret-user-typed-this" not in formatted, "gate 12: the dropped id's own value must never reach a log line"
    assert "1" in formatted, "the drop must still be counted, just never by value"


def test_run_generate_with_a_candidate_targeting_no_hypothesis_carries_none():
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [_candidate_payload("x", MOVE_ADVANCE)]})]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates == [Candidate(index=0, text="x", move=MOVE_ADVANCE, target_hypothesis_id=None)]


# --- 06-02-PLAN.md §G5: a candidate self-reports grounding and a proposed state --------


def _fact_item(id_suffix: str = "f1", text: str = "onboarding takes three weeks") -> ContextItem:
    from app.coach.context import FACT_ID_PREFIX

    return ContextItem(
        id=f"{FACT_ID_PREFIX}{id_suffix}", text=text, source="past interview sess-1",
        citation="session:sess-1#seq:4", provenance="evidence",
    )


def test_generate_tool_schema_carries_both_new_fields_and_neither_is_required():
    item_schema = _generate_tool_schema()["input_schema"]["properties"]["candidates"]["items"]
    assert "grounding_ids" in item_schema["properties"]
    assert "proposed_state" in item_schema["properties"]
    assert "grounding_ids" not in item_schema["required"]
    assert "proposed_state" not in item_schema["required"]


def test_run_generate_keeps_grounding_ids_that_all_match_fact_derived_brief_items_in_order():
    fact1, fact2 = _fact_item("f1"), _fact_item("f2", text="a different claim")
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {"text": "x", "move": MOVE_ADVANCE, "grounding_ids": [fact1.id, fact2.id]},
        ]})]
    )
    candidates = run_generate(
        stub, brief=(fact1, fact2), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates[0].grounding_ids == (fact1.id, fact2.id)


def test_run_generate_drops_an_invented_grounding_id_but_keeps_the_valid_ones_and_the_candidate():
    fact1 = _fact_item("f1")
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {"text": "x", "move": MOVE_ADVANCE, "grounding_ids": [fact1.id, "fact:invented-by-the-model"]},
        ]})]
    )
    candidates = run_generate(
        stub, brief=(fact1,), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert len(candidates) == 1, "a hallucinated grounding id must never cost the interviewer a suggestion"
    assert candidates[0].grounding_ids == (fact1.id,)


def test_run_generate_drops_grounding_ids_reported_against_a_hand_written_brief_item():
    # Hand-written brief items render no id in the prompt (`_brief_block`) --
    # the mechanism is scoped to retrieved facts by construction, so a
    # reported hand-written id is dropped exactly like any other non-match.
    hand_written = ContextItem(id="b1", text="hand-written brief item", provenance="evidence")
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {"text": "x", "move": MOVE_ADVANCE, "grounding_ids": ["b1"]},
        ]})]
    )
    candidates = run_generate(
        stub, brief=(hand_written,), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates[0].grounding_ids == ()


def test_run_generate_keeps_a_recognized_proposed_state():
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {"text": "x", "move": MOVE_TEST_HYPOTHESIS, "target_hypothesis_id": "h1", "proposed_state": "contradicted"},
        ]})]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates[0].proposed_state == "contradicted"


def test_run_generate_drops_an_unrecognized_proposed_state_but_keeps_the_candidate():
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {"text": "x", "move": MOVE_TEST_HYPOTHESIS, "target_hypothesis_id": "h1", "proposed_state": "definitely-true"},
        ]})]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert len(candidates) == 1
    assert candidates[0].proposed_state is None


def test_run_generate_a_candidate_grounded_in_nothing_and_proposing_nothing_is_complete():
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [_candidate_payload("x", MOVE_ADVANCE)]})]
    )
    candidates = run_generate(
        stub, brief=(), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
        detected_language="en", hypotheses=(), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
    )
    assert candidates == [Candidate(index=0, text="x", move=MOVE_ADVANCE)]


def test_dropped_grounding_and_state_values_never_reach_a_log_line_only_their_counts_do(caplog):
    import logging

    fact1 = _fact_item("f1")
    stub = StubAnthropicClient(
        responses=[tool_use_response(GENERATE_TOOL_NAME, {"candidates": [
            {
                "text": "x", "move": MOVE_TEST_HYPOTHESIS, "target_hypothesis_id": "h1",
                "grounding_ids": [fact1.id, "fact:a-user-typed-secret-value"],
                "proposed_state": "a-user-typed-secret-state",
            },
        ]})]
    )
    with caplog.at_level(logging.WARNING):
        run_generate(
            stub, brief=(fact1,), survey=(), asked={}, transcript=[], trigger=_segment(0, "interviewee", "hi", 1.0),
            detected_language="en", hypotheses=(_hypothesis(id="h1"),), recently_shown=(), model="m", max_tokens=100, timeout_s=1.0,
        )
    formatted = "\n".join(record.getMessage() for record in caplog.records)
    assert "a-user-typed-secret-value" not in formatted, "gate 12: a dropped grounding id must never be logged by value"
    assert "a-user-typed-secret-state" not in formatted, "gate 12: a dropped state value must never be logged by value"
    assert "1" in formatted, "both drops must still be counted"
