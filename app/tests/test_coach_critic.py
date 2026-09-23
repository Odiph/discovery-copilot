"""Tests for `app/coach/critic.py` -- D-05's ranking rules (the model
classifies, code ranks), the critique call's own parse-failure/refusal/
truncation discipline, and the tool-use request assembly. Behavioural test
names, per this project's own convention."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.coach.critic import (
    BAND_CLEAN,
    BAND_CLEAR_VIOLATION,
    BAND_UNCERTAIN,
    CRITIC_TOOL_NAME,
    CritiqueFailure,
    Judgment,
    assemble_critique_request,
    rank_candidates,
    run_critique,
)
from app.coach.generate import MOVE_ADVANCE, MOVE_DEEPEN, MOVE_TEST_HYPOTHESIS, Candidate
from app.coach.stub import StubAnthropicClient, ToolUseResponse, tool_use_response


def _candidate(index: int, text: str = "text", move: str = MOVE_ADVANCE) -> Candidate:
    return Candidate(index=index, text=text, move=move)


def _judgment(
    index: int,
    band: str,
    rationale: str = "r",
    is_confirmation_only: bool = False,
    is_generic: bool = False,
    is_off_topic: bool = False,
) -> Judgment:
    return Judgment(
        index=index,
        is_leading=False,
        is_hypothetical_or_future_tense=False,
        is_pitch=False,
        is_opinion_seeking=False,
        is_confirmation_only=is_confirmation_only,
        is_generic=is_generic,
        is_off_topic=is_off_topic,
        band=band,
        rationale=rationale,
    )


# --- the sixth flag, is_generic (2026-08-15) ------------------------------------------


def test_the_trigger_utterance_reaches_the_user_message_and_never_the_cached_system_block():
    """The whole cost argument for this change. The system prefix is
    byte-identical for the life of a session and that is what makes it
    cacheable; the trigger changes every cycle. If it ever drifts upward into
    the system block the caching silently stops -- no error, just
    `cache_creation_input_tokens: 0` and a third of the coach bill back."""
    trigger = "a very distinctive thing the interviewee said about their tuesday"
    request = assemble_critique_request(
        candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text=trigger,
        model="m", max_tokens=100,
    )
    system_text = "\n".join(block["text"] for block in request["system"])
    user_text = request["messages"][0]["content"]

    assert trigger in user_text, "the critic cannot judge is_generic without it"
    assert trigger not in system_text, (
        "per-cycle text in the cached prefix un-caches every call"
    )


def test_the_cached_prefix_is_identical_across_two_different_triggers():
    """The sharper version of the test above: not merely absent, but provably
    unable to move the cached bytes."""
    first = assemble_critique_request(
        candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="one utterance",
        model="m", max_tokens=100,
    )
    second = assemble_critique_request(
        candidates=[_candidate(0)], hypotheses=(), survey=(),
        trigger_text="an entirely different utterance", model="m", max_tokens=100,
    )
    assert first["system"][0]["text"] == second["system"][0]["text"]
    assert first["messages"][0]["content"] != second["messages"][0]["content"]


def test_an_empty_trigger_is_rendered_honestly_rather_than_omitted():
    """A missing heading would leave the model to infer that no utterance
    existed. The post-call asked-question critique passes an empty trigger by
    design, so this is a real path, not a defensive one."""
    request = assemble_critique_request(
        candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="",
        model="m", max_tokens=100,
    )
    user_text = request["messages"][0]["content"]
    assert "What the interviewee just said" in user_text
    assert "nothing" in user_text.lower()


def test_a_critique_response_missing_is_generic_makes_the_whole_response_untrusted():
    """No silent False default. A defaulted-False `is_generic` would read as
    "the model judged this engaged with what was said" when it never answered
    at all -- a fabricated verdict is worse than a failed cycle."""
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {"index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False,
         # is_generic deliberately omitted
         "band": BAND_CLEAN, "rationale": "fine"},
    ]})])
    with pytest.raises(CritiqueFailure) as excinfo:
        run_critique(
            stub, candidates=[_candidate(0)], hypotheses=(), survey=(),
            trigger_text="something they said", model="m", max_tokens=100, timeout_s=1.0,
        )
    assert excinfo.value.reason == "parse-error"


def test_judgment_cannot_be_constructed_without_is_generic():
    with pytest.raises(TypeError):
        Judgment(  # noqa: no is_generic -- must raise, no default
            index=0,
            is_leading=False,
            is_hypothetical_or_future_tense=False,
            is_pitch=False,
            is_opinion_seeking=False,
            is_confirmation_only=False,
            band=BAND_CLEAN,
            rationale="r",
        )


@pytest.mark.parametrize("band", [BAND_CLEAN, BAND_UNCERTAIN])
def test_a_generic_candidate_is_dropped_only_through_the_band_never_directly(band):
    """`is_generic` is the flag most likely to tempt a direct drop, because it
    is the one that most nearly means "not worth showing". It must still act
    only through the band -- a special case here would be a SECOND ordering
    path, which `critic.py`'s own contract forbids."""
    judgments = [_judgment(0, band, is_generic=True)]
    result = rank_candidates([_candidate(0)], judgments)
    assert len(result) == 1, (
        f"is_generic=True banded {band!r} must NOT be dropped -- the drop is "
        "band-mediated, exactly like every other flag"
    )


def test_is_generic_appears_in_the_assembled_critique_prompt_text():
    """A flag pinned in every schema but never described to the model would be
    legal and never meaningfully judged."""
    request = assemble_critique_request(
        candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="something",
        model="m", max_tokens=100,
    )
    system_text = "\n".join(block["text"] for block in request["system"])
    assert "is_generic" in system_text


# --- D-05: rank_candidates -- the model classifies, code ranks -----------------------


def test_rank_candidates_drops_every_clear_violation_at_any_rank():
    candidates = [_candidate(0, "a"), _candidate(1, "b"), _candidate(2, "c")]
    judgments = [
        _judgment(0, BAND_CLEAR_VIOLATION),
        _judgment(1, BAND_CLEAN),
        _judgment(2, BAND_CLEAR_VIOLATION),
    ]
    result = rank_candidates(candidates, judgments)
    assert [c.index for c in result] == [1]
    assert all(c.index not in (0, 2) for c in result)


def test_rank_candidates_keeps_uncertain_candidates_ranked_down_not_removed():
    candidates = [_candidate(0, "a"), _candidate(1, "b")]
    judgments = [_judgment(0, BAND_UNCERTAIN), _judgment(1, BAND_UNCERTAIN)]
    result = rank_candidates(candidates, judgments)
    assert len(result) == 2, "a suggestion set built entirely from the uncertain band is valid"


def test_rank_candidates_orders_clean_before_uncertain_and_preserves_emission_order_within_a_band():
    # Deliberately construct a case where "clean before uncertain" and "later
    # emission index" disagree, so precedence is pinned: index 0 is uncertain
    # (would come first by emission order alone) but index 1 is clean (must
    # still come first because band beats emission order).
    candidates = [_candidate(0, "uncertain-one"), _candidate(1, "clean-one"), _candidate(2, "clean-two")]
    judgments = [
        _judgment(0, BAND_UNCERTAIN),
        _judgment(1, BAND_CLEAN),
        _judgment(2, BAND_CLEAN),
    ]
    result = rank_candidates(candidates, judgments)
    assert [c.index for c in result] == [1, 2, 0], (
        "clean (1, 2) must precede uncertain (0) even though 0 was emitted first; "
        "within the clean band, 1 precedes 2 because that is generate's own emission order"
    )


def test_rank_candidates_returns_at_most_three_and_a_single_survivor_is_complete():
    candidates = [_candidate(i) for i in range(5)]
    judgments = [_judgment(i, BAND_CLEAN) for i in range(5)]
    result = rank_candidates(candidates, judgments, max_shown=3)
    assert len(result) == 3

    one_candidate = [_candidate(0)]
    one_judgment = [_judgment(0, BAND_CLEAN)]
    single = rank_candidates(one_candidate, one_judgment)
    assert len(single) == 1, "a lone survivor is a complete, successful result -- never partial"


def test_rank_candidates_is_deterministic_across_repeated_calls():
    candidates = [_candidate(0), _candidate(1), _candidate(2)]
    judgments = [
        _judgment(0, BAND_UNCERTAIN),
        _judgment(1, BAND_CLEAN),
        _judgment(2, BAND_UNCERTAIN),
    ]
    results = [tuple(c.index for c in rank_candidates(candidates, judgments)) for _ in range(20)]
    assert len(set(results)) == 1, "rank_candidates must return an identical list every time"


def test_rank_candidates_is_pure_no_io_no_client_plain_data_only():
    # If this call succeeds with plain dataclasses and no constructed
    # dependency, purity holds -- the strongest thing a unit test can say
    # about "no I/O, no client" is that none was ever given the chance.
    result = rank_candidates([_candidate(0)], [_judgment(0, BAND_CLEAN)])
    assert result == [_candidate(0)]


def test_rank_candidates_drops_a_candidate_with_no_matching_judgment():
    candidates = [_candidate(0), _candidate(1)]
    judgments = [_judgment(0, BAND_CLEAN)]  # candidate 1 was never classified
    result = rank_candidates(candidates, judgments)
    assert [c.index for c in result] == [0]


# --- 06-02-PLAN.md §G2/§G4: proven move-blind and flag-blind, not merely asserted ------


class _MoveBlindCandidate:
    """A candidate proxy exposing only `index`/`text` -- reading `.move`
    raises. If `rank_candidates` ever reads a candidate's move, this fails
    immediately and by name, which is a stronger guarantee than reading the
    function's source: it survives a refactor that keeps the behaviour and
    moves the code (06-02-PLAN.md §G2)."""

    def __init__(self, index: int, text: str = "t") -> None:
        self.index = index
        self.text = text

    @property
    def move(self):
        raise AssertionError("rank_candidates must never read Candidate.move (D-04)")


class _FlagBlindJudgment:
    """A judgment proxy exposing only `index`/`band` -- reading any of the
    named per-rule flags raises. The band is what drops a candidate; the
    flags are diagnostic/auditable only (critic.py:14-20's own contract)."""

    def __init__(self, index: int, band: str) -> None:
        self.index = index
        self.band = band

    @property
    def is_leading(self):
        raise AssertionError("rank_candidates must never read a judgment flag")

    @property
    def is_hypothetical_or_future_tense(self):
        raise AssertionError("rank_candidates must never read a judgment flag")

    @property
    def is_pitch(self):
        raise AssertionError("rank_candidates must never read a judgment flag")

    @property
    def is_opinion_seeking(self):
        raise AssertionError("rank_candidates must never read a judgment flag")

    @property
    def is_confirmation_only(self):
        # 06-02-PLAN.md §G4/T-06-14: the fifth flag must be exactly as
        # invisible to rank_candidates as the original four -- a
        # special-cased direct drop on this flag would be a second ordering
        # path and would violate critic.py:14-20's own contract.
        raise AssertionError("rank_candidates must never read a judgment flag")

    @property
    def is_generic(self):
        # The sixth flag (2026-08-15) is the one most likely to tempt a
        # direct drop, because it is the flag that most directly means "not
        # worth showing". It must still act only through the band.
        raise AssertionError("rank_candidates must never read a judgment flag")


def test_rank_candidates_never_reads_a_candidates_move_or_a_judgments_flags():
    candidates = [_MoveBlindCandidate(0), _MoveBlindCandidate(1), _MoveBlindCandidate(2)]
    judgments = [
        _FlagBlindJudgment(0, BAND_CLEAR_VIOLATION),
        _FlagBlindJudgment(1, BAND_CLEAN),
        _FlagBlindJudgment(2, BAND_UNCERTAIN),
    ]
    result = rank_candidates(candidates, judgments)
    assert [c.index for c in result] == [1, 2], (
        "0 is dropped (clear_violation), 1 (clean) precedes 2 (uncertain) -- reached without "
        "the function ever successfully reading a move or a flag off either proxy"
    )


# --- The critique call itself: request assembly and failure discipline ---------------


def test_critique_instruction_stays_above_the_cache_floor():
    """A character floor standing in for the 4,096-TOKEN one (the real check
    needs `count_tokens` and API credit). If the rubric is trimmed back below
    this, the breakpoint below silently stops caching -- no error, just a
    third of the coach bill quietly returning.

    Measured with NO hypotheses on purpose: the falsifier block is dropped
    when a project has none, which makes it the smaller and therefore the
    binding case -- and it is the common one on this project's real
    interviews, every one of which ran without hypotheses."""
    from app.coach import prompts

    instruction = prompts.critique_instruction(tool_name="x", hypotheses=(), survey=())
    assert len(instruction) > prompts.MIN_CRITIQUE_BLOCK_CHARS


def test_the_critique_system_block_is_a_cache_breakpoint():
    """It is byte-identical for the life of a session -- rubric, falsifiers,
    worked examples -- while the candidates it judges live in the user
    message. Without this it was 184 uncached calls per interview."""
    request = assemble_critique_request(
        candidates=[_candidate(0, "first", MOVE_ADVANCE)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100
    )
    system = request["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_the_cached_critique_prefix_does_not_change_when_the_candidates_do():
    """The whole point of the breakpoint. If the candidates leaked into the
    system block, every cycle would write a fresh entry and none would read."""
    first = assemble_critique_request(
        candidates=[_candidate(0, "a question about mornings", MOVE_ADVANCE)],
        hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100,
    )
    second = assemble_critique_request(
        candidates=[_candidate(0, "a completely different question", MOVE_DEEPEN),
                    _candidate(1, "and another one entirely", MOVE_ADVANCE)],
        hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100,
    )
    assert first["system"][0]["text"] == second["system"][0]["text"]
    assert first["messages"][0]["content"] != second["messages"][0]["content"]


def test_assemble_critique_request_embeds_every_candidate_the_generate_stage_produced():
    candidates = [_candidate(0, "first", MOVE_ADVANCE), _candidate(1, "second", MOVE_DEEPEN)]
    request = assemble_critique_request(candidates=candidates, hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100)
    content = request["messages"][0]["content"]
    assert "first" in content
    assert "second" in content
    assert request["tool_choice"] == {"type": "tool", "name": CRITIC_TOOL_NAME}


def _judgment_payload(
    index: int, band: str, is_confirmation_only: bool = False, is_generic: bool = False
) -> dict:
    return {
        "index": index,
        "is_leading": False,
        "is_hypothetical_or_future_tense": False,
        "is_pitch": False,
        "is_opinion_seeking": False,
        "is_confirmation_only": is_confirmation_only,
        "is_generic": is_generic, "is_off_topic": False,
        "band": band,
        "rationale": "fine",
    }


def test_run_critique_parses_a_normal_response_into_judgments():
    stub = StubAnthropicClient(
        responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [_judgment_payload(0, BAND_CLEAN)]})]
    )
    judgments = run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)
    assert judgments == [_judgment(0, BAND_CLEAN, rationale="fine")]


def test_run_critique_passes_its_own_explicit_timeout():
    stub = StubAnthropicClient(
        responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [_judgment_payload(0, BAND_CLEAN)]})]
    )
    run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.5)
    assert stub.calls[0].timeout_s == 1.5


def test_run_critique_reading_stop_details_on_a_normal_completion_does_not_raise():
    # ToolUseResponse (the stub's duck type) carries no stop_details field at
    # all -- exactly the "absent on the common path" shape the real SDK's
    # Message.stop_details takes on a normal completion. getattr-with-default
    # must never raise reading it.
    stub = StubAnthropicClient(
        responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [_judgment_payload(0, BAND_CLEAN)]})]
    )
    judgments = run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)
    assert len(judgments) == 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_run_critique_treats_refusal_and_truncation_as_untrusted(stop_reason):
    stub = StubAnthropicClient(responses=[ToolUseResponse(content=(), stop_reason=stop_reason)])
    with pytest.raises(CritiqueFailure) as excinfo:
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)
    assert excinfo.value.stop_reason == stop_reason


def test_run_critique_raises_on_an_unparseable_body():
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {})])  # missing "judgments"
    with pytest.raises(CritiqueFailure):
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)


def test_run_critique_wraps_any_exception_from_the_call_itself():
    stub = StubAnthropicClient(responses=[TimeoutError("simulated network timeout")])
    with pytest.raises(CritiqueFailure) as excinfo:
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)
    assert excinfo.value.stop_reason is None


# --- Model-output type validation (review finding, 2026-08-06) ----------------------


def test_a_string_index_is_refused_rather_than_silently_dropping_every_candidate():
    """The stub always returns well-typed blocks, so this is a failure only a
    real model can produce -- which is exactly why it went unnoticed.

    An `index` of "0" (a string) parses fine, then never matches a Candidate's
    int index. Every candidate looks unvetted, `rank_candidates` drops them
    all, and the cycle records SUCCESS with an empty result: the phone freezes
    on stale content while health reports fine, indistinguishable from "every
    candidate was genuinely bad"."""
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {"index": "0", "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
         "band": BAND_CLEAN, "rationale": "fine"},
    ]})])
    with pytest.raises(CritiqueFailure):
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)


def test_a_boolean_index_is_refused_even_though_bool_subclasses_int():
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {"index": True, "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
         "band": BAND_CLEAN, "rationale": "fine"},
    ]})])
    with pytest.raises(CritiqueFailure):
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)


def test_duplicate_indices_make_the_whole_response_untrusted_rather_than_last_winning():
    """Two judgments claiming index 0 -- one clean, one a clear violation.
    Silently last-winning would let the model's EMISSION ORDER decide whether
    a violating question reaches the user. Contradictory output means the
    response cannot be trusted at all (D-05)."""
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {"index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
         "band": BAND_CLEAR_VIOLATION, "rationale": "leading"},
        {"index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False, "is_confirmation_only": False, "is_generic": False, "is_off_topic": False,
         "band": BAND_CLEAN, "rationale": "fine"},
    ]})])
    with pytest.raises(CritiqueFailure):
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)


# --- 06-02-PLAN.md §G4: the fifth flag, is_confirmation_only --------------------------


def _hypothesis(*, id: str = "h1", statement: str = "onboarding delay is the top churn reason",
                 falsifier: str = "churned customers cite a different top reason more often"):
    return SimpleNamespace(id=id, statement=statement, falsifier=falsifier)


def test_critique_schemas_required_list_equals_its_property_names_by_set_equality():
    from app.coach.critic import _critic_tool_schema  # noqa: SLF001

    item_schema = _critic_tool_schema()["input_schema"]["properties"]["judgments"]["items"]
    assert set(item_schema["required"]) == set(item_schema["properties"]), (
        "a flag declared in properties and forgotten in required (or vice versa) must fail here"
    )


def test_a_critique_response_missing_the_fifth_flags_key_makes_the_whole_response_untrusted():
    stub = StubAnthropicClient(responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
        {"index": 0, "is_leading": False, "is_hypothetical_or_future_tense": False,
         "is_pitch": False, "is_opinion_seeking": False,
         # is_confirmation_only deliberately omitted
         "band": BAND_CLEAN, "rationale": "fine"},
    ]})])
    with pytest.raises(CritiqueFailure):
        run_critique(stub, candidates=[_candidate(0)], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100, timeout_s=1.0)


def test_judgment_cannot_be_constructed_without_is_confirmation_only():
    with pytest.raises(TypeError):
        Judgment(  # noqa: no is_confirmation_only -- must raise, no default
            index=0, is_leading=False, is_hypothetical_or_future_tense=False,
            is_pitch=False, is_opinion_seeking=False, band=BAND_CLEAN, rationale="r",
        )


def test_critique_prompt_for_a_targeting_candidate_carries_the_hypothesis_statement_and_falsifier_verbatim():
    hypothesis = _hypothesis()
    candidate = Candidate(index=0, text="does X still happen?", move="test_hypothesis", target_hypothesis_id="h1")
    request = assemble_critique_request(
        candidates=[candidate], hypotheses=(hypothesis,), survey=(),
        trigger_text="the interviewee just said something specific",
        model="m", max_tokens=100,
    )
    # `system` is now a list of content blocks carrying a cache breakpoint.
    system_text = "\n".join(block["text"] for block in request["system"])
    assert hypothesis.statement in system_text
    assert hypothesis.falsifier in system_text
    assert "is_confirmation_only" in system_text


def test_critique_prompt_for_a_candidate_with_no_target_still_assembles():
    candidate = _candidate(0)
    request = assemble_critique_request(candidates=[candidate], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100)
    assert "Candidates to classify" in request["messages"][0]["content"]


def test_critique_prompt_for_a_candidate_whose_target_resolves_to_nothing_loaded_still_assembles():
    candidate = Candidate(index=0, text="does X still happen?", move="test_hypothesis", target_hypothesis_id="h-gone")
    request = assemble_critique_request(candidates=[candidate], hypotheses=(), survey=(), trigger_text="the interviewee just said something specific", model="m", max_tokens=100)
    assert "h-gone" in request["messages"][0]["content"]


def test_assemble_critique_request_requires_its_hypotheses_keyword():
    with pytest.raises(TypeError):
        assemble_critique_request(candidates=[_candidate(0)], model="m", max_tokens=100)  # noqa: missing hypotheses


def test_a_confirmation_only_clear_violation_is_dropped_at_any_rank():
    candidates = [_candidate(0, "a"), _candidate(1, "b"), _candidate(2, "c")]
    for target_position in (0, 1, 2):
        judgments = [
            _judgment(i, BAND_CLEAN if i != target_position else BAND_CLEAR_VIOLATION, is_confirmation_only=(i == target_position))
            for i in range(3)
        ]
        result = rank_candidates(candidates, judgments)
        assert target_position not in [c.index for c in result], (
            f"a confirmation-only clear_violation at position {target_position} must be dropped at any rank"
        )


def test_a_confirmation_only_candidate_banded_clean_or_uncertain_is_not_dropped():
    for band in (BAND_CLEAN, BAND_UNCERTAIN):
        candidates = [_candidate(0)]
        judgments = [_judgment(0, band, is_confirmation_only=True)]
        result = rank_candidates(candidates, judgments)
        assert [c.index for c in result] == [0], (
            f"is_confirmation_only=True banded {band!r} must NOT be dropped -- the drop is band-mediated, "
            "never a direct read of the flag inside rank_candidates"
        )


# =========================================================================================
# 06-03-PLAN.md §H5: A7, the phase's own highest-risk unverified claim -- "a dedicated
# critic flag, given the hypothesis's own falsifier, reliably distinguishes a
# confirmation-only question from a genuinely falsifiable one." Every OTHER test in this
# file drives app/coach/stub.py, which proves the mechanism wires correctly and proves
# NOTHING about judgment quality (06-RESEARCH.md §3). This is the one test that does.
# =========================================================================================


@dataclass(frozen=True)
class _ConfirmationPair:
    """One hand-authored, hand-labeled pair sharing a single hypothesis and
    its real falsifier: a question that could genuinely come back negative
    and move toward the falsifier, and one worded so every plausible answer
    would only ever support the hypothesis. Written at the GRAY-ZONE
    difficulty that matters -- the confirmation-only member is worded to
    SOUND like a genuine test, never an obvious pitch or a textbook-clean
    violation, because that is the shape A7 actually risks getting wrong.
    Module-level, inspectable data: a future reader must be able to read one
    of these and disagree with the label, which is the only way this test
    stays honest."""

    label: str
    hypothesis_statement: str
    falsifier: str
    falsifiable_question: str
    confirmation_only_question: str


# Six pairs, one hypothesis/falsifier each, spanning different discovery-interview
# domains so no single wording pattern is doing all the work. Disagree with any label
# by editing it here -- that IS the intended review mechanism (§H5).
A7_CONFIRMATION_PAIRS: "tuple[_ConfirmationPair, ...]" = (
    _ConfirmationPair(
        label="onboarding-churn",
        hypothesis_statement="Onboarding delay is the top reason customers churn.",
        falsifier="Churned customers cite a different top reason more often than onboarding delay.",
        falsifiable_question=(
            "Walk me through the last time you seriously considered cancelling -- what was the "
            "actual final straw?"
        ),
        confirmation_only_question=(
            "Would you say a slow onboarding was a big part of why you considered leaving?"
        ),
    ),
    _ConfirmationPair(
        label="signup-pricing-confusion",
        hypothesis_statement="Users abandon the signup flow because the pricing page confuses them.",
        falsifier="Users who abandon signup report a different point of confusion, not pricing.",
        falsifiable_question=(
            "Tell me about the last time you started signing up for something online and stopped "
            "partway through -- what actually happened right before you closed the tab?"
        ),
        confirmation_only_question=(
            "Was the pricing page part of what made you hesitate while signing up?"
        ),
    ),
    _ConfirmationPair(
        label="crm-discovery-skip",
        hypothesis_statement=(
            "Sales reps skip the discovery call because the CRM makes it too easy to jump straight "
            "to a demo."
        ),
        falsifier="Reps who skip discovery cite a reason unrelated to the CRM's one-click demo shortcut.",
        falsifiable_question=(
            "Think of the last deal where you skipped straight to a demo -- walk me through why, "
            "step by step."
        ),
        confirmation_only_question=(
            "Does having a one-click demo button in the CRM make it tempting to skip discovery?"
        ),
    ),
    _ConfirmationPair(
        label="mobile-push-tickets",
        hypothesis_statement=(
            "Support tickets about the mobile app spike because push notifications are unreliable."
        ),
        falsifier="The tickets that spike cite something other than notifications.",
        falsifiable_question=(
            "Tell me about the last mobile support ticket you personally filed or saw come in -- "
            "what was actually wrong?"
        ),
        confirmation_only_question=(
            "Do you think unreliable push notifications are a common complaint from mobile users?"
        ),
    ),
    _ConfirmationPair(
        label="renewal-invoice-visibility",
        hypothesis_statement=(
            "Customers renew late because the renewal invoice is hard to find in their inbox."
        ),
        falsifier="Late renewers cite a reason other than not finding the invoice.",
        falsifiable_question=(
            "Walk me through the last time you renewed something -- when did you actually first see "
            "the invoice, and what did you do right after?"
        ),
        confirmation_only_question="Do you feel like renewal invoices are easy to miss in your inbox?",
    ),
    _ConfirmationPair(
        label="test-suite-runtime",
        hypothesis_statement="Engineers skip writing tests because the local test suite takes too long to run.",
        falsifier="Engineers who skip tests cite a reason other than local run time.",
        falsifiable_question=(
            "Tell me about the last time you pushed code without running the full test suite -- "
            "what actually happened right before that?"
        ),
        confirmation_only_question="Would a faster test suite make you more likely to run it before pushing?",
    ),
)


@pytest.mark.live_reasoning
def test_a7_a_dedicated_critic_flag_given_the_falsifier_distinguishes_confirmation_only_from_falsifiable():
    """06-03-PLAN.md §H5 (A7). Drives the REAL critique call, once per pair
    above, and reports -- in this test's own failure message, always, so it
    is never silently swallowed by pytest's default suppression of a passing
    test's output -- how many of each side the model classified correctly.

    Asserts NO pass threshold: there is no data yet to set one from, and a
    threshold invented now would be a number this test is calibrated to
    rather than a fact about the model. This hands a human evidence, never a
    verdict -- read the counts, do not treat this test's own pass/fail state
    as the answer to A7.

    Deselected offline by the `live_reasoning` marker (see
    `test_exactly_three_live_reasoning_marked_tests_exist_...` in
    test_coach_extract.py). `reasoning_client.build_client()` raises
    `CoachConfigError`, by name, when no key is present -- this test fails
    loudly for that reason rather than passing vacuously."""
    from app.coach import reasoning_client

    client = reasoning_client.build_client()

    falsifiable_correct = 0
    confirmation_only_correct = 0
    lines: "list[str]" = []

    for pair in A7_CONFIRMATION_PAIRS:
        hypothesis = SimpleNamespace(
            id="a7-hypothesis", statement=pair.hypothesis_statement, falsifier=pair.falsifier,
        )
        falsifiable_candidate = Candidate(
            index=0, text=pair.falsifiable_question, move=MOVE_TEST_HYPOTHESIS,
            target_hypothesis_id="a7-hypothesis",
        )
        confirmation_candidate = Candidate(
            index=1, text=pair.confirmation_only_question, move=MOVE_TEST_HYPOTHESIS,
            target_hypothesis_id="a7-hypothesis",
        )
        judgments = run_critique(
            client, candidates=[falsifiable_candidate, confirmation_candidate], hypotheses=(hypothesis,),
            model=reasoning_client.DEFAULT_MODEL, max_tokens=1000, timeout_s=60.0,
        )
        judgment_by_index = {j.index: j for j in judgments}
        assert 0 in judgment_by_index and 1 in judgment_by_index, (
            f"[{pair.label}] the critique response did not classify both candidates -- mechanism "
            f"failure, not a judgment-quality question: {judgments}"
        )
        falsifiable_flagged = judgment_by_index[0].is_confirmation_only
        confirmation_flagged = judgment_by_index[1].is_confirmation_only

        if falsifiable_flagged is False:
            falsifiable_correct += 1
        if confirmation_flagged is True:
            confirmation_only_correct += 1
        lines.append(
            f"[{pair.label}] falsifiable question is_confirmation_only={falsifiable_flagged} "
            f"(correct=False); confirmation-only question is_confirmation_only={confirmation_flagged} "
            "(correct=True)"
        )

    total = len(A7_CONFIRMATION_PAIRS)
    report = (
        f"A7 smoke test results ({total} hand-labeled pairs) -- "
        f"falsifiable questions correctly left unflagged: {falsifiable_correct}/{total}; "
        f"confirmation-only questions correctly flagged: {confirmation_only_correct}/{total}. "
        "NO pass threshold is asserted here -- this is evidence for a human to read, not a verdict.\n"
        + "\n".join(lines)
    )
    pytest.fail(report, pytrace=False)


# --- is_generic is a category error for a survey-coverage move --------------------------
#
# The first live coached interview flagged 13 of 29 suggestions `is_generic`
# while every OTHER flag fired zero times, so this one flag was doing 100% of
# the suppression. 8 of those 13 were `advance`/`fill_gap`, and five were the
# operator's own survey questions word for word. A survey question is written
# before the interview and cannot engage with an utterance that had not
# happened yet -- which is exactly what those moves are for.

from dataclasses import replace  # noqa: E402

from app.coach.critic import exempt_survey_coverage_from_generic  # noqa: E402
from app.coach.generate import (  # noqa: E402
    MOVE_ADVANCE,
    MOVE_DEEPEN,
    MOVE_FILL_GAP,
    MOVE_TEST_HYPOTHESIS,
    SURVEY_COVERAGE_MOVES,
)


def _cand(index: int, move: str, text: str = "q") -> Candidate:
    return Candidate(index=index, text=text, move=move)


@pytest.mark.parametrize("move", sorted(SURVEY_COVERAGE_MOVES))
def test_a_survey_coverage_move_is_not_judged_generic(move):
    judged = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAN, is_generic=True)], [_cand(0, move)]
    )
    assert judged[0].is_generic is False


@pytest.mark.parametrize("move", [MOVE_DEEPEN, MOVE_TEST_HYPOTHESIS])
def test_a_move_that_should_engage_keeps_its_generic_flag(move):
    """`deepen` exists to follow what was just said. Being generic IS a real
    failure there, and exempting everything would retire the flag."""
    judged = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAN, is_generic=True)], [_cand(0, move)]
    )
    assert judged[0].is_generic is True


def test_a_violation_that_rested_only_on_generic_is_demoted_not_dropped():
    """Clearing the flag alone would change NOTHING -- `rank_candidates` reads
    the band, so a `clear_violation` still disappears. The band has to move
    too, or the fix is cosmetic."""
    judged = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAR_VIOLATION, is_generic=True)], [_cand(0, MOVE_ADVANCE)]
    )
    assert judged[0].band == BAND_UNCERTAIN
    assert rank_candidates([_cand(0, MOVE_ADVANCE)], judged) == [_cand(0, MOVE_ADVANCE)]


def test_it_never_promotes_anything_to_clean():
    """`uncertain` is still shown, so the operator sees the question -- but the
    critic's doubt survives. Promoting to clean would overrule a judgement
    this only meant to declare inapplicable."""
    judged = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAR_VIOLATION, is_generic=True)], [_cand(0, MOVE_FILL_GAP)]
    )
    assert judged[0].band != BAND_CLEAN


def test_a_real_violation_alongside_generic_is_left_exactly_as_banded():
    """This clears a category error; it does not overrule the critic. A
    candidate that is ALSO leading stays a clear violation and stays dropped."""
    judgment = _judgment(0, BAND_CLEAR_VIOLATION, is_generic=True)
    judgment = replace(judgment, is_leading=True)
    judged = exempt_survey_coverage_from_generic([judgment], [_cand(0, MOVE_ADVANCE)])
    assert judged[0].band == BAND_CLEAR_VIOLATION
    assert rank_candidates([_cand(0, MOVE_ADVANCE)], judged) == []


def test_a_clean_survey_move_is_untouched():
    judged = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAN, is_generic=False)], [_cand(0, MOVE_ADVANCE)]
    )
    assert judged[0] == _judgment(0, BAND_CLEAN, is_generic=False)


def test_a_judgment_with_no_matching_candidate_is_passed_through_unchanged():
    """Defensive: the index space is the generate stage's, and a judgment the
    candidates do not account for must not be silently rewritten."""
    judgment = _judgment(7, BAND_CLEAR_VIOLATION, is_generic=True)
    assert exempt_survey_coverage_from_generic([judgment], [_cand(0, MOVE_ADVANCE)]) == [judgment]


# --- is_generic is a category error for a DECLINE too (2026-08-16) ----------------------
#
# `no_question` means "nothing should be asked right now" and carries an empty
# `text` by construction. The critic still returned `is_generic=true` for some
# declines, which is a flag about a question set on a candidate that has no
# question. This is the same shape as the survey-coverage exemption above --
# the flag has no subject -- but it must NOT borrow that exemption's band
# demotion; see the test below that pins why.

from app.coach.critic import (  # noqa: E402
    _ANNOTATION_MARKER,
    exempt_declines_from_generic,
)
from app.coach.generate import MOVE_NO_QUESTION  # noqa: E402


def test_a_decline_is_not_judged_generic():
    judged = exempt_declines_from_generic(
        [_judgment(0, BAND_CLEAN, is_generic=True)], [_cand(0, MOVE_NO_QUESTION, text="")]
    )
    assert judged[0].is_generic is False


@pytest.mark.parametrize("move", [MOVE_DEEPEN, MOVE_ADVANCE, MOVE_TEST_HYPOTHESIS, MOVE_FILL_GAP])
def test_a_candidate_that_carries_a_question_keeps_its_generic_flag_here(move):
    """This exemption is about the ABSENCE of question text, nothing else. It
    must not become a second, quieter version of the survey-coverage exemption
    -- that one is applied separately and argues its own case."""
    judged = exempt_declines_from_generic(
        [_judgment(0, BAND_CLEAN, is_generic=True)], [_cand(0, move)]
    )
    assert judged[0].is_generic is True


def test_exempting_a_decline_never_moves_its_band():
    """The restraint that matters, and the reason this is not just a call to
    the survey-coverage exemption with a wider move set.

    `pipeline.py` counts declines by walking `rank_candidates`' OUTPUT, and
    that output has already dropped every `clear_violation`. Demoting a
    condemned decline to `uncertain` -- exactly what the survey-coverage
    exemption does -- would push it back into the counted list and inflate the
    `suggestion-declined` event. That is a measurement moving because a flag
    was fixed, which is the one thing this change was told not to do."""
    for band in (BAND_CLEAR_VIOLATION, BAND_UNCERTAIN, BAND_CLEAN):
        judged = exempt_declines_from_generic(
            [_judgment(0, band, is_generic=True)], [_cand(0, MOVE_NO_QUESTION, text="")]
        )
        assert judged[0].band == band, band


def test_a_decline_the_critic_did_not_call_generic_is_untouched():
    judgment = _judgment(0, BAND_CLEAN, is_generic=False)
    assert exempt_declines_from_generic([judgment], [_cand(0, MOVE_NO_QUESTION, text="")]) == [judgment]


def test_a_decline_judgment_with_no_matching_candidate_is_passed_through_unchanged():
    judgment = _judgment(7, BAND_CLEAN, is_generic=True)
    assert exempt_declines_from_generic([judgment], [_cand(0, MOVE_NO_QUESTION, text="")]) == [judgment]


def test_run_critique_applies_both_exemptions_in_one_pass():
    """The wiring, not the rules. Both exemptions have to be reachable through
    the real entry point or each is a function nothing calls -- and a cycle
    routinely produces a decline and a survey move in the same response."""
    candidates = [_cand(0, MOVE_ADVANCE), _cand(1, MOVE_NO_QUESTION, text=""), _cand(2, MOVE_DEEPEN)]
    client = StubAnthropicClient(
        responses=[tool_use_response(CRITIC_TOOL_NAME, {"judgments": [
            _judgment_payload(0, BAND_CLEAN, is_generic=True),
            _judgment_payload(1, BAND_CLEAN, is_generic=True),
            _judgment_payload(2, BAND_CLEAN, is_generic=True),
        ]})]
    )
    judged = run_critique(
        client, candidates=candidates, hypotheses=(), survey=(), trigger_text="they said a specific thing",
        model="m", max_tokens=100, timeout_s=1.0,
    )
    assert [j.is_generic for j in judged] == [False, False, True], (
        "advance exempted as survey coverage, no_question exempted as a decline, "
        "deepen left alone -- being generic IS a real failure there"
    )


# --- The rationale must not contradict the flags it sits beside (2026-08-16) -----------
#
# Both exemptions clear `is_generic` and neither may touch the model's prose.
# The result before this was a rendered row reading `is_generic: false` beside
# a rationale saying "...making it generic in this context", with nothing on
# the page to say a deterministic post-step had overridden the flag on purpose.
# The fix annotates rather than edits: the model's sentence survives verbatim
# and a marked, machine-authored clause is appended after it.


@pytest.mark.parametrize(
    "exempt, move",
    [
        (exempt_survey_coverage_from_generic, MOVE_ADVANCE),
        (exempt_declines_from_generic, MOVE_NO_QUESTION),
    ],
)
def test_an_exempted_rationale_keeps_the_models_own_words_verbatim(exempt, move):
    """The non-negotiable half. Rewriting model prose to say something the
    model did not say destroys the only evidence that the exemption is still
    earning its keep -- and makes "is this override still needed?"
    unanswerable for whoever asks next."""
    said = "This restates the survey item, making it generic in this context."
    judged = exempt([_judgment(0, BAND_CLEAN, rationale=said, is_generic=True)], [_cand(0, move, text="")])
    assert judged[0].rationale.startswith(said), (
        "the critic's sentence must survive byte for byte, and must come first"
    )


@pytest.mark.parametrize(
    "exempt, move",
    [
        (exempt_survey_coverage_from_generic, MOVE_ADVANCE),
        (exempt_declines_from_generic, MOVE_NO_QUESTION),
    ],
)
def test_an_exempted_rationale_says_on_the_page_that_the_flag_was_overridden(exempt, move):
    """The other half. Leaving the prose alone and saying nothing is what
    produced the contradiction; a reviewer needs both artifacts AND the note
    that reconciles them, marked clearly enough that it cannot be mistaken for
    the critic's own voice."""
    judged = exempt(
        [_judgment(0, BAND_CLEAN, rationale="generic in this context", is_generic=True)],
        [_cand(0, move, text="")],
    )
    assert _ANNOTATION_MARKER in judged[0].rationale
    assert "is_generic was cleared" in judged[0].rationale


@pytest.mark.parametrize(
    "exempt, move",
    [
        (exempt_survey_coverage_from_generic, MOVE_ADVANCE),
        (exempt_declines_from_generic, MOVE_NO_QUESTION),
    ],
)
def test_a_judgment_that_was_not_exempted_gets_no_annotation(exempt, move):
    """The note is a record of an override, so a row nothing was done to must
    carry none -- otherwise the marker stops meaning anything and a reviewer
    scanning for it learns nothing."""
    judged = exempt([_judgment(0, BAND_CLEAN, rationale="fine", is_generic=False)], [_cand(0, move, text="")])
    assert judged[0].rationale == "fine"


def test_annotating_twice_cannot_stack_two_notes_on_one_line():
    """The two exemptions run in sequence over the same list. Their move sets
    are disjoint today so no judgment can meet both -- but idempotence is what
    makes that a property rather than a coincidence nobody would notice
    breaking."""
    once = exempt_survey_coverage_from_generic(
        [_judgment(0, BAND_CLEAN, rationale="generic", is_generic=True)], [_cand(0, MOVE_ADVANCE)]
    )
    twice = exempt_survey_coverage_from_generic(
        [replace(once[0], is_generic=True)], [_cand(0, MOVE_ADVANCE)]
    )
    assert twice[0].rationale.count(_ANNOTATION_MARKER) == 1


from app.coach.critic import enforce_off_topic  # noqa: E402


# --- the seventh flag, is_off_topic (2026-08-19) --------------------------------------
#
# The complaint it comes from: the coach kept proposing deeper and deeper
# questions about material the interview was not for, and every one of them
# passed every existing flag, because a sharp question about the wrong subject
# is still a well-formed question. These tests are about the DETERMINISTIC half
# -- `enforce_off_topic` -- because the prompt half can only be measured by
# `judge_critic.py` against the gold set.


def test_an_off_topic_deepen_is_banded_a_violation_whatever_the_critic_said():
    """The point of the whole change. `rank_candidates` reads the band alone,
    so clearing or recording the flag would suppress nothing -- and a
    well-formed off-topic question is exactly the case where a model reaching
    for a band says `clean`."""
    judged = enforce_off_topic(
        [_judgment(0, BAND_CLEAN, is_off_topic=True)], [_cand(0, MOVE_DEEPEN)]
    )
    assert judged[0].band == BAND_CLEAR_VIOLATION
    assert judged[0].is_off_topic is True, "the flag itself is a record, never cleared here"
    assert rank_candidates([_cand(0, MOVE_DEEPEN)], judged) == []


@pytest.mark.parametrize("move", [MOVE_ADVANCE, MOVE_FILL_GAP, MOVE_TEST_HYPOTHESIS])
def test_a_move_that_cannot_wander_has_the_flag_cleared_rather_than_enforced(move):
    """advance and fill_gap ARE the survey and test_hypothesis names a
    hypothesis, so each is on-topic by construction. A flag recorded true
    where it cannot apply is a number that looks like a measurement and is
    not -- and enforcing it would drop the survey coverage the interview
    depends on."""
    judged = enforce_off_topic(
        [_judgment(0, BAND_CLEAN, is_off_topic=True)], [_cand(0, move)]
    )
    assert judged[0].is_off_topic is False
    assert judged[0].band == BAND_CLEAN, "clearing the flag must not move the band"


def test_a_decline_is_never_banded_a_violation_by_this_step():
    """`no_question` carries no question text, so there is nothing for the
    flag to be about. Banding a decline `clear_violation` would also inflate
    the declined count `pipeline.py` derives from `rank_candidates`' output --
    the same trap `exempt_declines_from_generic` documents at length."""
    judged = enforce_off_topic(
        [_judgment(0, BAND_CLEAN, is_off_topic=True)], [_cand(0, MOVE_NO_QUESTION, text="")]
    )
    assert judged[0].is_off_topic is False
    assert judged[0].band == BAND_CLEAN


def test_the_step_never_relaxes_a_band():
    """The cost asymmetry the whole module rests on: this step may only ever
    REMOVE a suggestion from the panel, never add one. A judgment the critic
    already condemned stays condemned whatever this flag says."""
    judged = enforce_off_topic(
        [_judgment(0, BAND_CLEAR_VIOLATION, is_off_topic=False)], [_cand(0, MOVE_DEEPEN)]
    )
    assert judged[0].band == BAND_CLEAR_VIOLATION


def test_an_on_topic_deepen_is_left_exactly_as_the_critic_judged_it():
    """The common case, and the one the flag must not damage. Nothing is
    annotated either, so a reviewer reading a rationale sees the critic's own
    words with nothing appended."""
    judgment = _judgment(0, BAND_CLEAN, is_off_topic=False)
    judged = enforce_off_topic([judgment], [_cand(0, MOVE_DEEPEN)])
    assert judged[0] == judgment


def test_an_enforced_judgment_says_on_the_page_that_the_band_was_overridden():
    """Same contract the two exemptions already keep: the model's prose is
    never edited, and a marked note is appended so a reviewer meeting a
    `clear_violation` whose rationale reads `clean` can tell which artifact
    to believe."""
    judged = enforce_off_topic(
        [_judgment(0, BAND_CLEAN, is_off_topic=True, rationale="a fine question")],
        [_cand(0, MOVE_DEEPEN)],
    )
    assert judged[0].rationale.startswith("a fine question")
    assert "is_off_topic is true" in judged[0].rationale


def test_the_flag_is_judged_against_the_survey_the_request_actually_carries():
    """`is_off_topic` is the one flag that cannot be judged from a question
    and an utterance alone, and until it landed this request carried no
    statement of the plan at all. A survey that never reaches the prompt would
    leave the model answering from nothing."""
    request = assemble_critique_request(
        candidates=[_candidate(0)],
        hypotheses=(),
        survey=("How do you decide what to work on next?",),
        trigger_text="the interviewee just said something specific",
        model="m",
        max_tokens=100,
    )
    system_text = "\n".join(block["text"] for block in request["system"])
    assert "How do you decide what to work on next?" in system_text
    assert "is_off_topic" in system_text


def test_the_survey_stays_in_the_cached_prefix_and_out_of_the_volatile_message():
    """The survey is fixed for the life of a session, exactly like the
    hypotheses it sits beside. Putting it in the user message would cost the
    caching win for information that never changes -- the mirror image of the
    mistake `trigger_text` exists above to avoid in the other direction."""
    kwargs = dict(
        candidates=[_candidate(0)],
        hypotheses=(),
        survey=("a distinctive plan item about sprint planning",),
        model="m",
        max_tokens=100,
    )
    first = assemble_critique_request(trigger_text="one utterance", **kwargs)
    second = assemble_critique_request(trigger_text="an entirely different one", **kwargs)
    assert first["system"] == second["system"]
    assert "sprint planning" not in second["messages"][0]["content"]
