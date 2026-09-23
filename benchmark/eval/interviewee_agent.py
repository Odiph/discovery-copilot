"""An interviewee that ANSWERS THE QUESTION IT WAS ACTUALLY ASKED.

**Why this exists.** `claudictive_script.py` is a fixed list of turns. It is a
good measuring stick for the parts of the system that do not talk back -- the
transcript path, the coverage detector, the critic's banding of a pre-written
run of bad questions -- because a fixed script is perfectly reproducible. But
it is structurally incapable of testing the one thing a coach exists to
change: the interviewer's NEXT question. In a scripted run the interviewee
says sentence N+1 whether the interviewer took the coach's suggestion, ignored
it, or said nothing at all. Every downstream measurement is therefore blind to
whether the suggestion was any good, and a coach that emitted pure noise would
score identically to one that did not.

This module supplies the missing half: a persona-driven counterpart that takes
whatever question was actually asked and returns an in-character answer, so a
dry run can close the loop.

**The persona is not invented here.** It is the SAME Claudictive
gamer-developer already established in `claudictive_script.py`, and
`established_record()` reads that file's own interviewee turns rather than
restating them. There is exactly one persona in this repo and this module does
not get a vote on what it contains. If the script changes, this changes with
it, which is the point -- two hand-maintained copies of a character would
drift and nobody would notice until a run made no sense.

**An agreeable counterpart is a useless counterpart.** The script deliberately
contradicts h1, is ambivalent about h4, and pushes back on the pricing
question, because an interview where everything lines up makes a poor test of
a tool whose whole thesis is disconfirmation. A generative interviewee will
drift towards cooperation unless something holds it away from there: models
are trained to be helpful, and "helpful" reads as "answer the question you
were asked, in the terms it was asked in". So non-cooperation is encoded three
ways here, in increasing order of how much it can be trusted:

1. `classify_question()` names the PRESSURE in a question deterministically,
   offline, with no model involved -- pricing, leading, speculative,
   generalising, or asking about other people.
2. `REQUIRED_STANCE` maps each pressure to the stance the persona must take,
   and that instruction is written into the prompt for that specific turn.
3. The reply's own declared `stance` is a REQUIRED field of the forced tool,
   so `Reply.caved` can record, per turn, that the persona was pressed and
   cooperated anyway.

Only (1) and (3) are facts. (2) is a request.

**`caved` is only as good as the classifier's PRECISION, and a false positive
inverts it.** `caved` means "was pressed AND cooperated anyway". If
`classify_question` tags a question that was in fact well asked, then a
perfectly good cooperative answer to a perfectly good question is counted as
the persona folding -- so a run against a COMPETENT interviewer scores worse
than one against an incompetent one, which is exactly backwards. Recall
matters much less: a missed tag costs one turn of extra cooperation and biases
the metric towards zero, which is the safe direction. This is why the
classifier is deliberately narrow, why `_PRESSURE_EXEMPTIONS` exists, and why
the regression test sweeps the real script in BOTH directions rather than
only checking that the bait is caught.

**Which client, and why it is its own environment variable.** Construction
goes through `reasoning_client.build_client()` -- this module never
constructs a vendor client and never imports `anthropic`. Naming
`INTERVIEWEE_BASE_URL` builds an `openai_seam.OpenAICompatibleClient` instead,
so the counterpart can be pointed at a different runtime from the coach under
test. That separation is not cosmetic: pointing both at one local runtime
makes them contend for the same weights and the same GPU, and the coach's
measured latency then includes time spent generating its own test subject's
answers. Falling back to the coach's own client is the convenient default, not
the correct one for a latency run.

**WHAT THIS HARNESS CANNOT ESTABLISH.**

- It cannot show the interviewee is REALISTIC. It produces a counterpart that
  is responsive and in-character-ish. Whether a real developer would have said
  any of it is not measured here and cannot be, and nothing downstream should
  treat a dry-run transcript as evidence about users.
- It cannot show the coach is GOOD. Closing the loop makes it possible for a
  suggestion to change what comes next; it supplies no judge of whether the
  change was an improvement. `judge_critic.py` scores the critic against gold
  labels; there is no equivalent for a whole conversation, and this module
  does not pretend to be one.
- It cannot show the persona STAYED consistent. `established_record()` puts
  the character's own prior statements in the prompt; it does not verify the
  answer agreed with them. A contradiction with the record and a deliberate
  self-contradiction (which is a wanted behaviour, see `STANCE_CONTRADICT`)
  are indistinguishable from the outside, and no assertion here separates them.
- `classify_question()` is a regex over English. It will miss a leading
  question phrased without any of its markers, and it will tag a well-meant
  one that happens to contain them. It is a floor under non-cooperation, not a
  measurement of interviewer quality -- `critic_gold.py` owns that, with
  hand-labelled cases, precisely because a regex cannot do it.
- Every unit test in `app/tests/test_interviewee_agent.py` runs against
  `StubAnthropicClient`. They prove the request this module builds and the
  parsing it does. They prove NOTHING about what any model returns for that
  request, and a green suite is not evidence that a dry run would produce a
  usable interview.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from benchmark.console import use_utf8_console
from app.coach import openai_seam, reasoning_client
from benchmark.eval import claudictive_script

# --------------------------------------------------------------------------
# Stances: the closed set of ways this persona is allowed to meet a question.
# Closed, and pinned by a test, for the same reason `generate.ALL_MOVES` is: a
# sixth stance appearing without the schema, the prompt and the required-stance
# map growing together is a silent split between what the persona is told to do
# and what it is allowed to report doing.
# --------------------------------------------------------------------------
STANCE_COOPERATE = "cooperate"          # answer straight, with a real instance
STANCE_HEDGE = "hedge"                  # genuinely two-sided; the h4 ambivalence
STANCE_CONTRADICT = "contradict"        # walk back something said earlier
STANCE_REFUSE = "refuse"                # decline this question, and say why
STANCE_CORRECT_PREMISE = "correct_premise"  # reject the framing, not the topic

ALL_STANCES = frozenset(
    {
        STANCE_COOPERATE,
        STANCE_HEDGE,
        STANCE_CONTRADICT,
        STANCE_REFUSE,
        STANCE_CORRECT_PREMISE,
    }
)

# --------------------------------------------------------------------------
# Pressures: what is wrong (or merely loaded) about the question, decided
# offline by regex. These are the four shapes the script already baits the
# interviewer with, plus the generalisation trap the script uses to produce its
# one hard self-contradiction ("Is that generally true, or was that one
# unusual?" -> "No, I think if I'm honest, seeing it is usually not enough.").
# --------------------------------------------------------------------------
PRESSURE_PRICING = "pricing"
PRESSURE_LEADING = "leading"
PRESSURE_SPECULATIVE = "speculative"
PRESSURE_GENERALISE = "generalise"
PRESSURE_ABOUT_OTHERS = "about_others"

ALL_PRESSURES = frozenset(
    {
        PRESSURE_PRICING,
        PRESSURE_LEADING,
        PRESSURE_SPECULATIVE,
        PRESSURE_GENERALISE,
        PRESSURE_ABOUT_OTHERS,
    }
)

# Every pressure maps to exactly one stance, and every pressure has a row. The
# persona's own line for each is in `claudictive_script.py`:
#   pricing       -> "I'd rather you didn't ask me that, because I'll give you
#                     a useless answer."
#   leading       -> "I wouldn't have called it frustrating until you said it."
#   speculative   -> "That's a hard thing to answer about a thing that doesn't
#                     exist."
#   generalise    -> "No, I think if I'm honest, seeing it is usually not
#                     enough."
#   about_others  -> "I genuinely couldn't tell you what most developers want."
REQUIRED_STANCE = {
    PRESSURE_PRICING: STANCE_REFUSE,
    PRESSURE_LEADING: STANCE_CORRECT_PREMISE,
    PRESSURE_SPECULATIVE: STANCE_HEDGE,
    PRESSURE_GENERALISE: STANCE_CONTRADICT,
    PRESSURE_ABOUT_OTHERS: STANCE_CORRECT_PREMISE,
}

# Curated, not clever. Each row is a phrase the script's own bad-question block
# or its bait turns actually use. Deliberately narrow: a false negative costs
# one turn of extra cooperation, while a false positive makes the persona
# stonewall an ordinary behavioural question and the whole dry run turns into
# a wall of refusals, which is a worse and much less obvious failure.
_PRESSURE_PATTERNS = {
    PRESSURE_PRICING: (
        r"\bpay\b|\bpaid\b|\bpaying\b|\bprice\b|\bpricing\b|\bsubscription\b"
        r"|\bsubscribe\b|\bper month\b|\bworth it\b|how much would it cost"
        r"|willing to pay|\bcharge you\b|\bbudget for\b"
    ),
    PRESSURE_LEADING: (
        r",\s*right\?|don'?t you think|wouldn'?t you (say|agree)"
        r"|so you'?d say|isn'?t it (true|fair)\b|\bsurely\b"
        r"|the whole (idea|point) is|that must have been|you must have"
        r"|\bobviously\b"
    ),
    PRESSURE_SPECULATIVE: (
        r"\bwould you\b|\bwould it\b|\bif it could\b|\bimagine\b|hypothetical"
        r"|\bwhat if\b|magic wand|do you think you'?d|\bin future\b"
    ),
    PRESSURE_GENERALISE: (
        r"\bgenerally\b|\bin general\b|\busually\b|\btypically\b|\balways\b"
        r"|\bevery time\b|is that typical|most of the time|\bevery night\b"
    ),
    PRESSURE_ABOUT_OTHERS: (
        r"most (developers|people|users|players)|other people|\byour friends\b"
        r"|\beveryone\b|people like you|\bother developers\b|\banyone else\b"
    ),
}

# The same word can mark the bait or mark a GOOD question, and only the tense
# separates them. An exemption is checked after the pattern matches and drops
# the pressure -- deliberately subtractive rather than a narrower pattern,
# because narrowing the patterns to exclude these would also lose the real
# detections they were written for.
#
# Both rows below exist because running the classifier over
# `claudictive_script.TURNS` found them, which is the only way a bad pattern
# ever surfaces: mutating the classifier's internals cannot reveal one, since
# every mutation of a wrong rule produces a differently wrong rule. The
# regression test in `test_interviewee_agent.py` runs exactly that sweep.
_PRESSURE_EXEMPTIONS = {
    # PAST/PRESENT ACTUAL SPEND IS NOT THE BAIT -- it is the repair.
    # "What do you currently pay for that sits on this machine?" is the script's
    # OWN correction of its bad pricing question, and is a gold case in
    # `critic_gold.py` (`good-current-spend`) precisely because asking what
    # someone already pays is the honest way to ask about money. Only
    # hypothetical future willingness ("would you pay", "what would you expect
    # to pay") is the thing the persona should refuse. The distinguishing
    # feature is an INDICATIVE auxiliary before "you" -- a modal ("would you
    # pay") never matches this, which is what keeps the bait tagged.
    PRESSURE_PRICING: (
        r"\b(do|did|are|were|have|had)\s+you\b[^?]*"
        r"\b(pay|paying|paid|spend|spending|spent|subscrib)"
    ),
    # ASKING FOR A TYPICAL VALUE IS NOT ASKING FOR A GENERALISATION.
    # "How long is that wait, usually?" wants a duration for a wait the
    # interviewee has already described; "Is that generally true?" invites them
    # to speak beyond their own experience. Only the second deserves a
    # contradiction, and telling the persona to fight the first turns a plain
    # factual follow-up into an argument.
    PRESSURE_GENERALISE: r"\bhow\s+(long|often|many|much|frequently)\b",
}


def classify_question(question: str) -> "frozenset[str]":
    """Which pressures this question carries. Offline, deterministic, no model.

    Returns an EMPTY set for an ordinary behavioural question, which is the
    common case and the one that should stay common -- see `_PRESSURE_PATTERNS`
    on why this errs towards missing rather than over-tagging.

    **The asymmetry this encodes.** For money and for typicality, the same
    vocabulary marks both the bait and its repair, and only the tense tells
    them apart. A question about what someone ACTUALLY pays, or how long
    something ACTUALLY takes, is good interviewing and is exempted; a question
    about what they WOULD pay, or whether something is true IN GENERAL, is the
    bait and stays tagged. Getting this backwards is not a cosmetic error --
    see `Reply.caved`, whose meaning inverts on a false positive."""
    lowered = question.lower()
    found = set()
    for pressure, pattern in _PRESSURE_PATTERNS.items():
        if not re.search(pattern, lowered):
            continue
        exemption = _PRESSURE_EXEMPTIONS.get(pressure)
        if exemption and re.search(exemption, lowered):
            continue
        found.add(pressure)
    return frozenset(found)


def required_stances(pressures: "frozenset[str]") -> "frozenset[str]":
    """The stances that would be an acceptable reply to those pressures.

    A set rather than one value because a question can carry two pressures at
    once ("Would you pay for it?" is both speculative and pricing) and there is
    no defensible way to rank them -- meeting EITHER is non-cooperation, which
    is all this harness is entitled to ask for."""
    return frozenset(REQUIRED_STANCE[p] for p in pressures if p in REQUIRED_STANCE)


# --------------------------------------------------------------------------
# The persona itself.
# --------------------------------------------------------------------------
INTERVIEWEE_SPEAKER = "interviewee"


def established_record() -> "tuple[str, ...]":
    """Everything this character has already said, read from
    `claudictive_script.py` rather than restated here.

    Read, never copied, so there is exactly one persona in the repo. A second
    hand-maintained copy would drift from the first and the drift would only
    ever be noticed by a human reading a dry-run transcript and finding it
    slightly off, which is not a detection mechanism."""
    return tuple(
        text for speaker, text in claudictive_script.TURNS
        if speaker == INTERVIEWEE_SPEAKER
    )


_PERSONA_RULES = """You are being interviewed. You are a professional software developer who \
games in the evenings on the same machine you work on. You are not a helpful \
assistant and you are not selling anything -- you are a person answering \
questions about your own evenings, and you have no stake in the interviewer \
being pleased with your answers.

Hard rules, in priority order:

1. Speak only from THE RECORD below. It is everything you have already said \
about yourself. Details it does not contain -- your name, your employer, the \
names of the games, your salary -- you do not volunteer and you do not invent. \
If a question needs a fact the record does not have, say you would rather not \
go into it, or that you cannot remember, the way a real person does.
2. Answer the question you were ACTUALLY asked. If the interviewer changes \
direction, follow them. If they ask something you already answered, notice \
that out loud.
3. One to four sentences. Speech, not prose: contractions, hesitation, \
self-correction mid-sentence. No lists, no headings, no summarising yourself.
4. You are allowed to be unhelpful. Real interviewees are. See the stances."""

_STANCE_RULES = """The stance you take is a required part of your answer, and you must \
report the one you actually took:

- cooperate: answer straight, and ground it in a specific occasion rather than \
a habit. This is the default and most turns should be this.
- hedge: you genuinely do not know, or you can see it both ways. Say both \
sides and do not resolve them. Do not manufacture an opinion about a thing \
that does not exist yet.
- contradict: what you said earlier was too strong, or was one bad week rather \
than a normal one. Walk it back explicitly and say you are walking it back.
- refuse: decline this particular question and say why you think the answer \
would be worthless. You are not refusing the interview, you are refusing the \
question.
- correct_premise: the question has an assumption or a loaded word in it that \
you do not accept. Reject the framing, then answer the honest version of the \
question underneath it if there is one.

Never be more certain than the record supports, and never agree with a \
characterisation of your own experience just because the interviewer offered \
it."""


def persona_block() -> str:
    """The stable system text: rules, stances, and the record.

    Stable BY CONSTRUCTION -- nothing here varies per turn, which is what makes
    it the one cached breakpoint. The turn-varying half (the conversation so
    far, the question, the stance instruction) goes in the user message for
    exactly the reason `generate.assemble_generate_request` keeps the trigger
    out of its cached blocks: a breakpoint over something that changes every
    call pays the cache-write premium and is never once read back."""
    record = "\n".join(f"- {line}" for line in established_record())
    return f"{_PERSONA_RULES}\n\n{_STANCE_RULES}\n\nTHE RECORD (your own words, from before this conversation):\n{record}"


def stance_instruction(pressures: "frozenset[str]") -> str:
    """What this specific turn asks for, given what `classify_question` found.

    A REQUEST, not a guarantee. The model can ignore it, which is why the
    stance comes back as a declared field and `Reply.caved` records the
    disagreement rather than this function pretending to have prevented it."""
    wanted = required_stances(pressures)
    if not wanted:
        return (
            "This question looks like an ordinary behavioural one. Answer it "
            "straight unless something in the record genuinely stops you."
        )
    named = ", ".join(sorted(wanted))
    reasons = ", ".join(sorted(pressures))
    return (
        f"This question carries: {reasons}. A real person would not simply "
        f"answer it. Take one of these stances: {named}. Do not soften it into "
        "cooperation, and do not apologise for the stance you take."
    )


def conversation_block(history: "Sequence[Exchange]") -> str:
    """The live conversation, in order. Separate from THE RECORD on purpose:
    the record is who you are, this is what has been said in the room, and the
    persona is meant to be able to contradict the SECOND one knowingly."""
    if not history:
        return "This is the first question of the interview."
    lines = []
    for exchange in history:
        lines.append(f"INTERVIEWER: {exchange.question}")
        lines.append(f"YOU: {exchange.answer}")
    return "The conversation so far:\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# The call.
# --------------------------------------------------------------------------
INTERVIEWEE_TOOL_NAME = "answer_as_interviewee"

# Same rule as `generate._UNTRUSTED_STOP_REASONS`, and the same reason: a
# truncated answer parsed as a complete one puts half a sentence into the
# transcript, and every downstream stage then measures a conversation that
# never happened.
_UNTRUSTED_STOP_REASONS = frozenset(
    {"refusal", "max_tokens", "pause_turn", "model_context_window_exceeded"}
)

DEFAULT_MAX_TOKENS = 400
# Deliberately far above any coach budget. The counterpart is not on the
# COACH-05 latency path -- it is the thing the coach is being timed AGAINST --
# so a tight bound here would only turn a slow local runtime into a dry run
# that fails for reasons having nothing to do with the coach.
DEFAULT_TIMEOUT_S = 60.0


class IntervieweeFailure(Exception):
    """Raised for anything that makes a reply untrusted: a timeout, a parse
    failure, a refusal by the provider (distinct from the persona's own
    `refuse` stance, which is a successful reply), a truncation, or any other
    exception from the call.

    Carries `reason` and the response's own `stop_reason` only -- never the
    question and never the answer. This mirrors `GenerateFailure` because the
    same argument applies: an exception string is the one place text escapes
    into logs without anybody deciding it should."""

    def __init__(self, *, reason: str, stop_reason: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stop_reason = stop_reason


@dataclass(frozen=True)
class Exchange:
    """One completed question/answer pair. Frozen: the history is rebuilt as a
    new tuple on every turn rather than appended to, so a caller holding an
    earlier agent still sees the interview as it stood then -- which is what
    makes a branching dry run (same history, two different next questions)
    possible at all."""

    question: str
    answer: str
    stance: str


@dataclass(frozen=True)
class Reply:
    """One in-character answer.

    `pressures` is what `classify_question` found in the question, and `stance`
    is what the model SAID it did. `caved` is the disagreement between them,
    and it is the only number in this module worth reporting from a dry run: a
    counterpart that never caves is doing its job, and one that caves on every
    pressed turn has silently degenerated into the agreeable interviewee this
    module exists to avoid. It is not an error and never raises -- a run where
    the persona folds is still a run, it is just a weaker test, and a harness
    that refused to produce it would hide the weakness instead of reporting
    it.

    **It is meaningful only to the extent `classify_question` is PRECISE.**
    Every false positive inverts this field: a cooperative answer to a
    well-asked question is recorded as caving, so a better interviewer
    produces a worse-looking number. Read a `caved` count as a lower bound on
    the persona's cooperation AND an upper bound on the classifier's
    precision, never as a clean measurement of either."""

    answer: str
    stance: str
    pressures: "frozenset[str]"
    caved: bool


def _interviewee_tool_schema() -> dict:
    """Fixed, module-level shape -- never built per-call from anything that
    varies, matching `generate._generate_tool_schema`'s reasoning about the
    API's day-long compiled-schema cache.

    `stance` is REQUIRED. Making it optional would let a model return an answer
    with no stance, which `_parse_interviewee_response` would have to default
    to something -- and any default is a lie about what the model did."""
    return {
        "name": INTERVIEWEE_TOOL_NAME,
        "description": (
            "Answer the interviewer's question in character, and report which "
            "stance you took towards the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "stance": {"type": "string", "enum": sorted(ALL_STANCES)},
            },
            "required": ["answer", "stance"],
        },
    }


def assemble_interviewee_request(
    *,
    question: str,
    history: "Sequence[Exchange]",
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Builds the exact kwargs for `client.messages.create(...)`.

    Pure: no I/O, no client, no clock. Every assertion worth making about what
    this harness asks a model for can therefore be made by inspecting the
    returned dict, with nothing running -- the same reason
    `assemble_generate_request` is factored out of `run_generate`."""
    pressures = classify_question(question)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": persona_block(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": (
                    conversation_block(history)
                    + "\n\nINTERVIEWER: "
                    + question
                    + "\n\n"
                    + stance_instruction(pressures)
                    + f"\n\nReply by calling the `{INTERVIEWEE_TOOL_NAME}` tool "
                    "and nothing else."
                ),
            }
        ],
        "tools": [_interviewee_tool_schema()],
        "tool_choice": {"type": "tool", "name": INTERVIEWEE_TOOL_NAME},
    }


def _parse_interviewee_response(response: Any, *, pressures: "frozenset[str]") -> Reply:
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason in _UNTRUSTED_STOP_REASONS:
        raise IntervieweeFailure(reason="untrusted-stop-reason", stop_reason=stop_reason)

    for block in getattr(response, "content", None) or ():
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == INTERVIEWEE_TOOL_NAME
        ):
            try:
                answer = block.input["answer"]
                stance = block.input["stance"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("answer must be a non-empty string")
                if stance not in ALL_STANCES:
                    # NOT coerced to `cooperate`. An unrecognized stance means
                    # the reply's own self-report is unreadable, and `caved` is
                    # computed from it -- silently defaulting would produce a
                    # dry-run number that looks measured and is not.
                    raise ValueError("unrecognized stance")
                wanted = required_stances(pressures)
                return Reply(
                    answer=answer,
                    stance=stance,
                    pressures=pressures,
                    caved=bool(wanted) and stance not in wanted,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise IntervieweeFailure(reason="parse-error", stop_reason=stop_reason) from exc

    raise IntervieweeFailure(reason="no-tool-use-block", stop_reason=stop_reason)


@dataclass(frozen=True)
class IntervieweeAgent:
    """The counterpart, plus everything it has said in this room.

    Frozen, and `answer()` returns a NEW agent rather than mutating this one.
    That is not ceremony: the reason to have a responsive interviewee at all is
    to ask what happens when the interviewer asks A instead of B from the same
    point in the conversation, and that comparison is only possible if the
    conversation state at that point survives being used once."""

    model: str
    history: "tuple[Exchange, ...]" = ()
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: float = DEFAULT_TIMEOUT_S

    def answer(self, client: Any, question: str) -> "tuple[Reply, IntervieweeAgent]":
        """Drives ONE call through the shared client seam. `client` is injected
        (real, `OpenAICompatibleClient`, or `StubAnthropicClient`) and is never
        constructed here.

        Raises `IntervieweeFailure` for anything untrusted, and in that case
        returns nothing -- the caller decides whether to retry, ask a different
        question, or end the run. Deliberately NOT degrade-and-continue like
        the coach pipeline: a coach that skips a cycle costs one suggestion,
        while an interviewee that skips a turn leaves a hole in the transcript
        every later measurement then reads as real."""
        pressures = classify_question(question)
        request = assemble_interviewee_request(
            question=question,
            history=self.history,
            model=self.model,
            max_tokens=self.max_tokens,
        )
        bound = reasoning_client.with_timeout(client, self.timeout_s)
        try:
            response = bound.messages.create(**request)
        except IntervieweeFailure:
            raise
        except Exception as exc:  # noqa: BLE001 -- timeout, connection, anything
            raise IntervieweeFailure(reason="exception", stop_reason=None) from exc
        reply = _parse_interviewee_response(response, pressures=pressures)
        moved = replace(
            self,
            history=self.history
            + (Exchange(question=question, answer=reply.answer, stance=reply.stance),),
        )
        return reply, moved


# --------------------------------------------------------------------------
# Construction. Mirrors `reasoning_client.build_client()` exactly, and defers
# to it for the vendor path -- this module must never import `anthropic`.
# --------------------------------------------------------------------------
def build_interviewee_client() -> Any:
    """The counterpart's client.

    `INTERVIEWEE_BASE_URL` wins when named, so the interviewee can run on a
    different endpoint from the coach under test. When it is not named this
    falls back to `reasoning_client.build_client()`, which means the two share
    a runtime -- convenient, and wrong for any run whose output is a latency
    number, since the coach would then be timed while the same machine
    generates its test subject's answers.

    The Anthropic path is NEVER constructed here. It is delegated, so
    `reasoning_client` remains the one construction site (04-CONTEXT.md §C4)
    and its two invariants -- key read explicitly from the environment,
    `max_retries=0` -- hold for the interviewee too, for free."""
    base_url = os.environ.get("INTERVIEWEE_BASE_URL")
    if base_url:
        return openai_seam.OpenAICompatibleClient(
            base_url=base_url,
            api_key=os.environ.get("INTERVIEWEE_API_KEY") or None,
        )
    return reasoning_client.build_client()


def resolve_interviewee_model() -> str:
    """`INTERVIEWEE_MODEL`, else whatever the coach resolved to.

    The fallback carries the same trap `resolve_model()` documents: naming
    `INTERVIEWEE_BASE_URL` without `INTERVIEWEE_MODEL` sends a candidate
    endpoint a dated Anthropic model id it has never heard of."""
    return os.environ.get("INTERVIEWEE_MODEL") or reasoning_client.resolve_model()


if __name__ == "__main__":
    use_utf8_console()
    # Offline only. Prints what the harness would ask for; calls no model, by
    # design -- there is no code path in this file's `__main__` that opens a
    # socket.
    print(f"{len(established_record())} established interviewee turns")
    for probe in (
        "Walk me through the last time you sat down to play.",
        "So the Overwolf install was the frustrating part, right?",
        "Is there anything about how you'd expect to pay for something like this?",
        "Is that generally true, or was that one unusual?",
        "Don't you think most developers would want something like this?",
    ):
        found = classify_question(probe)
        print(f"  {sorted(found) or ['-']} -> {sorted(required_stances(found)) or ['-']}")
