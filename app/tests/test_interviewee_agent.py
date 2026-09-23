"""Tests for the interactive interviewee (`benchmark/eval/interviewee_agent.py`).

**What these prove, and what they cannot.** They prove the HARNESS: that the
persona is read from the one script rather than invented here, that the
pressure classifier fires on the shapes it claims to, that the request forces
the tool and carries the stance instruction for the turn, that an untrusted
response is refused rather than half-parsed, and that the agent's history is
genuinely immutable so a branching run is possible.

They prove NOTHING about what a model does with any of it. Every call here
goes through `StubAnthropicClient`, which returns whatever the test put in it,
so a green run says the instrument is not lying -- not that the interviewee
would answer in character, stay consistent with the record, or push back when
told to. Those are properties of a model against this prompt and can only be
observed by running one. Per the standing note in `app/coach/critic.py`, a
test that exercises one side of a seam proves INTENT, not BEHAVIOUR.

In particular, NO test here can show that the persona actually refuses a
pricing question. The most that is asserted is that the request asked it to,
and that a reply which cooperates anyway is recorded as `caved` rather than
quietly counted as a normal turn.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from app.coach import reasoning_client
from app.coach.openai_seam import OpenAICompatibleClient
from app.coach.stub import StubAnthropicClient, ToolUseResponse, tool_use_response
from benchmark.eval import claudictive_script, interviewee_agent
from benchmark.eval.interviewee_agent import (
    ALL_PRESSURES,
    ALL_STANCES,
    INTERVIEWEE_TOOL_NAME,
    PRESSURE_ABOUT_OTHERS,
    PRESSURE_GENERALISE,
    PRESSURE_LEADING,
    PRESSURE_PRICING,
    PRESSURE_SPECULATIVE,
    REQUIRED_STANCE,
    STANCE_CONTRADICT,
    STANCE_COOPERATE,
    STANCE_CORRECT_PREMISE,
    STANCE_HEDGE,
    STANCE_REFUSE,
    Exchange,
    IntervieweeAgent,
    IntervieweeFailure,
)

MODEL = "test-model"


def _reply(answer: str = "Yeah, that was Thursday.", stance: str = STANCE_COOPERATE) -> ToolUseResponse:
    return tool_use_response(INTERVIEWEE_TOOL_NAME, {"answer": answer, "stance": stance})


def _agent(**kwargs) -> IntervieweeAgent:
    return IntervieweeAgent(model=MODEL, **kwargs)


# ---------------------------------------------------------------------------
# The persona is the script's persona, not a second one.
# ---------------------------------------------------------------------------
def test_the_record_is_read_from_the_script_and_is_not_a_second_copy():
    """The one thing that stops the persona drifting into a different
    character than the one the scripted eval already measures."""
    record = interviewee_agent.established_record()
    assert record
    scripted = [text for speaker, text in claudictive_script.TURNS if speaker == "interviewee"]
    assert list(record) == scripted


def test_the_persona_block_actually_contains_the_record():
    """Reading the record and then forgetting to render it would leave a
    persona with rules and no facts -- which produces confident invention,
    the exact failure the record exists to prevent."""
    block = interviewee_agent.persona_block()
    for line in interviewee_agent.established_record():
        assert line in block


def test_the_persona_block_is_stable_across_calls():
    """It is the one cached breakpoint. A block that varies per call pays the
    cache-write premium every time and is never read back -- the defect
    `generate.assemble_generate_request` was rewritten to remove."""
    assert interviewee_agent.persona_block() == interviewee_agent.persona_block()


# ---------------------------------------------------------------------------
# The pressure classifier.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question, expected",
    [
        ("So the Overwolf install was the frustrating part, right?", PRESSURE_LEADING),
        ("Don't you think this would help?", PRESSURE_LEADING),
        ("Is there anything about how you'd expect to pay for something like this?", PRESSURE_PRICING),
        ("What would the pricing have to be?", PRESSURE_PRICING),
        ("Would you use it every day if it could read your whole codebase?", PRESSURE_SPECULATIVE),
        ("If I gave you a magic wand, what does it do?", PRESSURE_SPECULATIVE),
        ("Is that generally true, or was that one unusual?", PRESSURE_GENERALISE),
        ("Does that always happen?", PRESSURE_GENERALISE),
        ("Don't most developers want something like this?", PRESSURE_ABOUT_OTHERS),
        ("What would your friends have done?", PRESSURE_ABOUT_OTHERS),
    ],
)
def test_each_pressure_is_detected_on_a_question_the_script_actually_contains(question, expected):
    assert expected in interviewee_agent.classify_question(question)


def test_an_ordinary_behavioural_question_carries_no_pressure():
    """The important negative. If the classifier tagged ordinary questions the
    persona would stonewall the whole interview, which is a worse and far less
    visible failure than missing a leading question."""
    assert interviewee_agent.classify_question(
        "Walk me through the last time you sat down to play, from turning the machine on."
    ) == frozenset()
    assert interviewee_agent.classify_question(
        "Take me through what happened when you came back."
    ) == frozenset()


def test_every_pressure_has_a_required_stance_and_every_stance_is_known():
    """Set equality in both directions. A sixth pressure added without a row
    here would silently classify as pressed-but-unconstrained, and the turn
    would ask for nothing."""
    assert set(REQUIRED_STANCE) == set(ALL_PRESSURES)
    assert set(REQUIRED_STANCE.values()) <= ALL_STANCES


def test_the_stance_set_is_pinned():
    """Pinned by equality, like `generate.ALL_MOVES`. A stance may only be
    added by deliberately failing this."""
    assert ALL_STANCES == frozenset(
        {STANCE_COOPERATE, STANCE_HEDGE, STANCE_CONTRADICT, STANCE_REFUSE, STANCE_CORRECT_PREMISE}
    )


def test_non_cooperation_is_reachable_for_every_pressure():
    """`cooperate` must never be an acceptable answer to a pressed question --
    otherwise the whole mechanism reduces to "answer normally"."""
    for pressure in ALL_PRESSURES:
        wanted = interviewee_agent.required_stances(frozenset({pressure}))
        assert wanted
        assert STANCE_COOPERATE not in wanted


# ---------------------------------------------------------------------------
# The regression sweep over the REAL script -- both directions.
#
# This is the test that would have caught the original defect, and no amount of
# mutating the classifier's internals could have. Mutating a wrong rule only
# ever produces a differently wrong rule; a bad pattern is only visible against
# real questions. Two of the entries below (`good-current-spend` and the
# typical-duration follow-up) were tagged by the first version of this
# classifier, which told the persona to refuse and to contradict two of the
# script's BEST questions.
# ---------------------------------------------------------------------------

# Questions the script itself marks as bait, with the pressure each must carry.
SCRIPT_BAIT = {
    "So the Overwolf install was the frustrating part, right?": PRESSURE_LEADING,
    "Would you use it every day if it could also read your whole codebase?": PRESSURE_SPECULATIVE,
    "The whole idea is that you never have to leave the game again, which is what makes it different from anything else out there.": PRESSURE_LEADING,
    "Don't you think most developers would want something like this?": PRESSURE_ABOUT_OTHERS,
    "So you'd say it's useful, then?": PRESSURE_LEADING,
    "Is there anything about how you'd expect to pay for something like this?": PRESSURE_PRICING,
    "Is that generally true, or was that one unusual?": PRESSURE_GENERALISE,
}

# Questions the script treats as COMPETENT interviewing and which must
# therefore carry NO pressure. Tagging any of these makes a cooperative answer
# to a good question count as `caved`, which inverts the metric.
SCRIPT_COMPETENT = (
    # A typical DURATION for a wait already described. Asking for a number is
    # not inviting a generalisation, and the persona must not be told to fight
    # it. (Was tagged `generalise` -> contradict.)
    "How long is that wait, usually?",
    # The script's own REPAIR of its bad pricing question, and the
    # `good-current-spend` gold case in `critic_gold.py`. Present-tense actual
    # spend is the honest way to ask about money. (Was tagged `pricing` ->
    # refuse, i.e. the persona punished the interviewer for asking well.)
    "That's fair, and you're right. Let me ask it differently. What do you currently pay for that sits on this machine?",
    "Walk me through the last time you sat down to play, from turning the machine on to the game actually starting.",
    "What do you do with it when it surfaces?",
    "Tell me about a specific one. The most recent one you can remember.",
    "Take me through the last time in the shooter.",
    "What is different about coming back, between those two?",
    "Walk me through installing a new tool on your gaming machine the last time you did it.",
    "What went through your head at that point?",
    "What happened with the failing build?",
    "Which two?",
)

# Turns this classifier tags that the script does not call bait, KNOWINGLY
# accepted rather than quietly tolerated. Each is a question about a thing that
# does not exist yet or about other people, where hedging or declining to speak
# for others is a defensible answer -- so the false-positive cost is a slightly
# more evasive persona, not an inverted metric on a well-asked question. Pinned
# by equality so this residue can only grow deliberately.
ACCEPTED_OVER_TAGS = {
    "If it were in front of you, in the game, what would you actually do with it?",
    "Would you type code during a game?",
    "If I gave you a magic wand for that specific problem, what does it do?",
    "Would you talk to it, or type it?",
    "Let's go back to the install for a moment. You said your friends would have closed the tab.",
}


def _script_interviewer_turns() -> "list[str]":
    return [text for speaker, text in claudictive_script.TURNS if speaker == "interviewer"]


@pytest.mark.parametrize("question, pressure", sorted(SCRIPT_BAIT.items()))
def test_the_scripts_own_bait_questions_are_still_tagged(question, pressure):
    """The recall direction. The fix must not have been a pattern deletion."""
    assert question in _script_interviewer_turns()
    assert pressure in interviewee_agent.classify_question(question)


@pytest.mark.parametrize("question", SCRIPT_COMPETENT)
def test_a_competent_question_from_the_script_carries_no_pressure(question):
    """The precision direction, and the one that actually matters. A false
    positive here makes a cooperative answer to a GOOD question count as
    `caved`, so a run against a competent interviewer scores worse than one
    against an incompetent one."""
    assert question in _script_interviewer_turns()
    assert interviewee_agent.classify_question(question) == frozenset()


def test_the_over_tagged_set_across_the_whole_script_is_pinned():
    """Everything tagged that the script does not call bait, by equality.
    A new over-tag -- the failure mode that silently corrupts `caved` -- cannot
    appear without failing here, and shrinking the residue must also be
    deliberate."""
    tagged = {
        question
        for question in _script_interviewer_turns()
        if interviewee_agent.classify_question(question)
    }
    assert tagged - set(SCRIPT_BAIT) == ACCEPTED_OVER_TAGS


def test_the_tense_asymmetry_holds_for_money_and_for_typicality():
    """The two exemptions, stated as the asymmetry they encode rather than as
    the two script lines that happened to expose them."""
    for actual in (
        "What do you currently pay for on this machine?",
        "What did you pay for the last tool like this?",
        "Are you paying for anything similar today?",
        "Have you ever paid for an overlay?",
    ):
        assert PRESSURE_PRICING not in interviewee_agent.classify_question(actual)
    for hypothetical in (
        "Would you pay for something like this?",
        "How much would you expect to pay?",
        "What would the pricing have to be?",
        "Would you have paid for it?",
    ):
        assert PRESSURE_PRICING in interviewee_agent.classify_question(hypothetical)

    for typical_value in (
        "How long is that wait, usually?",
        "How often does that usually happen?",
        "How many of those do you typically lose?",
    ):
        assert PRESSURE_GENERALISE not in interviewee_agent.classify_question(typical_value)
    for invitation in (
        "Is that generally true, or was that one unusual?",
        "Does that always happen?",
        "Is that typical of a work night?",
        # A "how" question that asks for no quantity. The exemption is anchored
        # to the VALUE words on purpose -- broadening it to any "how" would
        # exempt this, which is a plain invitation to generalise.
        "How does that generally work for you?",
    ):
        assert PRESSURE_GENERALISE in interviewee_agent.classify_question(invitation)


def test_a_question_carrying_two_pressures_accepts_either_stance():
    pressures = interviewee_agent.classify_question("Would you pay for something like this?")
    assert {PRESSURE_PRICING, PRESSURE_SPECULATIVE} <= pressures
    assert interviewee_agent.required_stances(pressures) == {STANCE_REFUSE, STANCE_HEDGE}


# ---------------------------------------------------------------------------
# The request.
# ---------------------------------------------------------------------------
def test_the_request_forces_the_one_tool_and_offers_every_stance():
    request = interviewee_agent.assemble_interviewee_request(
        question="What happened next?", history=(), model=MODEL
    )
    assert request["tool_choice"] == {"type": "tool", "name": INTERVIEWEE_TOOL_NAME}
    schema = request["tools"][0]
    assert schema["name"] == INTERVIEWEE_TOOL_NAME
    properties = schema["input_schema"]["properties"]
    assert set(properties["stance"]["enum"]) == ALL_STANCES
    # Both required: an answer with no declared stance makes `caved`
    # uncomputable, and any default for it would be a fabricated measurement.
    assert set(schema["input_schema"]["required"]) == {"answer", "stance"}


def test_a_pressed_question_carries_its_stance_instruction_into_the_prompt():
    """The only thing this harness can do about cooperation drift at request
    time. It is a request to the model, not a guarantee -- see `caved`."""
    request = interviewee_agent.assemble_interviewee_request(
        question="So you'd say it's useful, then?", history=(), model=MODEL
    )
    content = request["messages"][0]["content"]
    assert STANCE_CORRECT_PREMISE in content
    assert PRESSURE_LEADING in content


def test_an_unpressed_question_does_not_demand_a_stance():
    request = interviewee_agent.assemble_interviewee_request(
        question="Walk me through the last time.", history=(), model=MODEL
    )
    content = request["messages"][0]["content"]
    assert STANCE_REFUSE not in content
    assert STANCE_CORRECT_PREMISE not in content


def test_the_conversation_so_far_reaches_the_prompt_in_order():
    """Without this the persona answers every question as though it were the
    first, and cannot contradict itself -- which is one of the five stances."""
    history = (
        Exchange(question="First question?", answer="First answer.", stance=STANCE_COOPERATE),
        Exchange(question="Second question?", answer="Second answer.", stance=STANCE_HEDGE),
    )
    content = interviewee_agent.assemble_interviewee_request(
        question="Third question?", history=history, model=MODEL
    )["messages"][0]["content"]
    assert content.index("First answer.") < content.index("Second answer.")
    assert content.index("Second answer.") < content.index("Third question?")


def test_the_first_turn_says_so_rather_than_rendering_an_empty_conversation():
    content = interviewee_agent.assemble_interviewee_request(
        question="Anything?", history=(), model=MODEL
    )["messages"][0]["content"]
    assert "first question" in content.lower()


def test_the_persona_is_the_single_cached_system_block():
    request = interviewee_agent.assemble_interviewee_request(
        question="Anything?", history=(), model=MODEL
    )
    system = request["system"]
    assert len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Driving it against the stub -- no model, no network, no cost.
# ---------------------------------------------------------------------------
def test_a_reply_comes_back_parsed_and_the_call_named_its_own_timeout():
    client = StubAnthropicClient(responses=[_reply("It was Thursday night.")])
    agent = _agent(timeout_s=12.5)
    reply, moved = agent.answer(client, "Walk me through the last time.")
    assert reply.answer == "It was Thursday night."
    assert reply.stance == STANCE_COOPERATE
    assert reply.caved is False
    call = client.calls[0]
    assert call.model == MODEL
    assert call.tool_name == INTERVIEWEE_TOOL_NAME
    # Every call site names its own bound (reasoning_client.with_timeout's
    # contract). Inheriting the SDK's 10-minute default would let one wedged
    # turn hang a whole dry run.
    assert call.timeout_s == 12.5
    assert moved.history[-1].answer == "It was Thursday night."


def test_answering_returns_a_new_agent_and_leaves_the_old_one_untouched():
    """The branching property. Asking A and B from the same point is the only
    reason to have a responsive counterpart at all, and it requires the
    conversation state to survive being used once."""
    client = StubAnthropicClient(responses=[_reply("A."), _reply("B.")])
    agent = _agent()
    _, after_a = agent.answer(client, "Question A?")
    _, after_b = agent.answer(client, "Question B?")
    assert agent.history == ()
    assert len(after_a.history) == 1
    assert len(after_b.history) == 1
    assert after_a.history[0].answer == "A."
    assert after_b.history[0].answer == "B."


def test_a_cooperative_answer_to_a_pressed_question_is_recorded_as_caved():
    """The measurement that makes a degenerate counterpart visible. It is NOT
    an error: a run where the persona folds is still a run, just a weaker
    test, and refusing to produce it would hide the weakness."""
    client = StubAnthropicClient(responses=[_reply("Sure, about ten pounds a month.", STANCE_COOPERATE)])
    reply, _ = _agent().answer(client, "What would you expect to pay for it?")
    assert PRESSURE_PRICING in reply.pressures
    assert reply.caved is True


def test_meeting_the_pressure_is_not_recorded_as_caved():
    client = StubAnthropicClient(responses=[_reply("I'd rather you didn't ask me that.", STANCE_REFUSE)])
    reply, _ = _agent().answer(client, "What would you expect to pay for it?")
    assert reply.caved is False


def test_an_unpressed_question_can_never_cave():
    client = StubAnthropicClient(responses=[_reply("It was Thursday.", STANCE_COOPERATE)])
    reply, _ = _agent().answer(client, "Walk me through the last time.")
    assert reply.pressures == frozenset()
    assert reply.caved is False


# ---------------------------------------------------------------------------
# Everything untrusted is refused, never half-parsed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal", "pause_turn", "model_context_window_exceeded"])
def test_an_untrusted_stop_reason_is_refused_rather_than_parsed(stop_reason):
    """`max_tokens` is the load-bearing row: a truncated answer parsed as a
    whole one puts half a sentence into the transcript, and every later
    measurement then reads a conversation that never happened."""
    response = dataclasses.replace(_reply(), stop_reason=stop_reason)
    client = StubAnthropicClient(responses=[response])
    with pytest.raises(IntervieweeFailure) as excinfo:
        _agent().answer(client, "Anything?")
    assert excinfo.value.reason == "untrusted-stop-reason"
    assert excinfo.value.stop_reason == stop_reason


def test_prose_instead_of_a_tool_call_is_a_failure():
    client = StubAnthropicClient(responses=[ToolUseResponse(content=())])
    with pytest.raises(IntervieweeFailure) as excinfo:
        _agent().answer(client, "Anything?")
    assert excinfo.value.reason == "no-tool-use-block"


def test_an_unrecognized_stance_is_a_failure_and_is_never_coerced():
    """Defaulting it to `cooperate` would make `caved` a fabricated number
    that looks measured."""
    client = StubAnthropicClient(
        responses=[tool_use_response(INTERVIEWEE_TOOL_NAME, {"answer": "Sure.", "stance": "agreeable"})]
    )
    with pytest.raises(IntervieweeFailure) as excinfo:
        _agent().answer(client, "Anything?")
    assert excinfo.value.reason == "parse-error"


@pytest.mark.parametrize(
    "payload",
    [
        {"stance": STANCE_COOPERATE},                       # no answer at all
        {"answer": "", "stance": STANCE_COOPERATE},         # empty answer
        {"answer": "   ", "stance": STANCE_COOPERATE},      # whitespace only
        {"answer": 17, "stance": STANCE_COOPERATE},         # not a string
        {"answer": "Sure."},                                # no stance
    ],
)
def test_a_malformed_tool_payload_is_a_parse_failure(payload):
    client = StubAnthropicClient(responses=[tool_use_response(INTERVIEWEE_TOOL_NAME, payload)])
    with pytest.raises(IntervieweeFailure) as excinfo:
        _agent().answer(client, "Anything?")
    assert excinfo.value.reason == "parse-error"


def test_a_transport_exception_becomes_a_typed_failure_carrying_no_text():
    """The failure must never carry the question or the answer -- an exception
    string is the one place text escapes into logs without anyone deciding
    it should."""
    client = StubAnthropicClient(raises=RuntimeError("connection reset"))
    secret_question = "Walk me through the migration that failed."
    with pytest.raises(IntervieweeFailure) as excinfo:
        _agent().answer(client, secret_question)
    assert excinfo.value.reason == "exception"
    assert excinfo.value.stop_reason is None
    assert secret_question not in str(excinfo.value)


def test_a_failed_turn_does_not_advance_the_history():
    """A hole in the transcript is worse than a short one: every later stage
    reads the remaining turns as though they were contiguous."""
    client = StubAnthropicClient(raises=RuntimeError("down"))
    agent = _agent()
    with pytest.raises(IntervieweeFailure):
        agent.answer(client, "Anything?")
    assert agent.history == ()


# ---------------------------------------------------------------------------
# Construction discipline.
# ---------------------------------------------------------------------------
def test_the_interviewee_can_be_pointed_at_its_own_endpoint(monkeypatch):
    """Separate from the coach's on purpose: one runtime serving both makes
    the coach's measured latency include time spent generating its own test
    subject's answers."""
    monkeypatch.setenv("INTERVIEWEE_BASE_URL", "http://localhost:11435/v1")
    monkeypatch.delenv("INTERVIEWEE_API_KEY", raising=False)
    client = interviewee_agent.build_interviewee_client()
    assert isinstance(client, OpenAICompatibleClient)
    assert client._api_key is None


def test_without_its_own_endpoint_it_defers_to_the_one_construction_site(monkeypatch):
    """Delegated, never re-implemented -- so `reasoning_client`'s two
    invariants (explicit key from the environment, `max_retries=0`) hold here
    for free."""
    monkeypatch.delenv("INTERVIEWEE_BASE_URL", raising=False)
    monkeypatch.setenv("COACH_BASE_URL", "http://localhost:11434/v1")
    called: "list[bool]" = []
    real = reasoning_client.build_client

    def spy():
        called.append(True)
        return real()

    monkeypatch.setattr(interviewee_agent.reasoning_client, "build_client", spy)
    client = interviewee_agent.build_interviewee_client()
    assert called == [True]
    assert isinstance(client, OpenAICompatibleClient)


def test_the_model_id_follows_the_interviewee_endpoint(monkeypatch):
    monkeypatch.setenv("INTERVIEWEE_MODEL", "qwen2.5:7b-instruct")
    assert interviewee_agent.resolve_interviewee_model() == "qwen2.5:7b-instruct"


def test_the_model_id_falls_back_to_the_coach_resolution(monkeypatch):
    monkeypatch.delenv("INTERVIEWEE_MODEL", raising=False)
    monkeypatch.delenv("COACH_MODEL", raising=False)
    assert interviewee_agent.resolve_interviewee_model() == reasoning_client.DEFAULT_MODEL


def test_the_module_never_imports_the_vendor_sdk():
    """Same AST discipline as gate eleven, applied to this module. It is not
    under `app/coach/` so that gate does not walk it, and the whole point of
    routing through `reasoning_client` is lost the moment this file imports
    `anthropic` itself."""
    source = Path(interviewee_agent.__file__).read_text(encoding="utf-8")
    named: "set[str]" = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            named.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            named.add(node.module.split(".")[0])
    assert "anthropic" not in named
