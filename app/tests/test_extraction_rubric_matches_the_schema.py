"""The extraction prompt must instruct the model about fields that EXIST.

Commit `bc113f3` replaced `Fact.related_hypothesis_ids` with `bearings`,
carrying a direction, because a field that records only "bears on" can produce
counts that go one way -- and a monotonic evidence count is a confirmation
machine. The tool schema was changed. **The rubric that tells the model what to
fill in was not.**

So `extraction_rubric_block` went on saying "set related_hypothesis_ids to the
ids of any this claim bears on", naming a field the schema does not have, while
never once mentioning `bearings`, `stance`, `supports`, `undermines` or
`unclear`. The model was being instructed about a dead field and left to infer
the live one from the JSON schema alone.

Measured consequence on the only real corpus in the repo: 23 facts extracted
from a real interview with 3 hypotheses loaded, and **every single one carried
zero bearings**. The hub then rendered "Nothing in the corpus supports this
yet" and "No disconfirming evidence found -- was any sought?" for all three
hypotheses, which reads as a finding rather than as a broken pipeline.

The narrow test here is that the rubric names the live field and every stance.
The GENERAL guard is `test_the_rubric_never_instructs_a_field_the_schema_does
_not_have`: it scans the rubric for identifiers and requires each to be a real
property of the extraction tool schema. That is the one that would have caught
this the day the schema changed, and it keeps catching the next rename.
"""

from __future__ import annotations

import re

from app.coach import corpus, extract, prompts


class _Hypothesis:
    def __init__(self, id: str, statement: str, falsifier: str) -> None:
        self.id = id
        self.statement = statement
        self.falsifier = falsifier


_HYPOTHESES = (
    _Hypothesis("h1", "Researchers keep a manual fallback", "nobody describes a manual fallback"),
)

# Identifiers, not prose: two or more lowercase words joined by underscores.
# Deliberately not a hand-listed set of dead names -- the point is to catch the
# NEXT rename, not this one.
_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _fact_schema_properties() -> "set[str]":
    schema = extract._extract_tool_schema()
    facts = schema["input_schema"]["properties"]["facts"]
    return set(facts["items"]["properties"])


def test_the_rubric_names_the_live_field_and_every_stance():
    text = prompts.extraction_rubric_block(hypotheses=_HYPOTHESES)
    assert "bearings" in text, "the rubric must name the field the schema actually requires"
    for stance in sorted(corpus.ALL_STANCES):
        assert stance in text, (
            f"the rubric never mentions the stance {stance!r}, so the model is left to infer "
            "the direction vocabulary from the JSON schema alone"
        )


def test_the_rubric_warns_against_manufacturing_a_direction():
    # `corpus.ALL_STANCES`' own comment says the manufactured direction will
    # almost always be `supports`, because the model is completing a sentence
    # about a hypothesis someone wrote down BECAUSE THEY BELIEVED IT. That
    # reasoning was written for humans reading the source and never told to the
    # model doing the judging. A rubric that offers `unclear` without saying
    # when to reach for it will not get it used.
    text = prompts.extraction_rubric_block(hypotheses=_HYPOTHESES)
    lowered = text.lower()
    assert "unclear" in lowered
    assert any(word in lowered for word in ("do not guess", "never guess", "do not force")), (
        "the rubric must tell the model not to guess a direction, not merely offer it the option"
    )


def test_the_rubric_never_instructs_a_field_the_schema_does_not_have():
    # THE GUARD. Not a check that one dead name is gone -- a check that every
    # field-shaped identifier the rubric mentions is a real property of the
    # tool schema the model is forced to call. Rename a schema property without
    # touching the prompt and this fails immediately, which is exactly what did
    # not happen when `related_hypothesis_ids` became `bearings`.
    text = prompts.extraction_rubric_block(hypotheses=_HYPOTHESES)
    properties = _fact_schema_properties()
    mentioned = {name for name in _IDENTIFIER.findall(text)}
    unknown = {name for name in mentioned if name not in properties}
    assert not unknown, (
        f"the extraction rubric instructs the model about {sorted(unknown)}, which "
        f"{'is' if len(unknown) == 1 else 'are'} not "
        f"{'a property' if len(unknown) == 1 else 'properties'} of the extract_facts schema "
        f"({sorted(properties)}). A prompt naming a field the schema does not have is worse "
        "than a prompt naming none: it is a contradictory instruction."
    )


def test_the_guard_also_covers_the_no_hypotheses_branch():
    # The empty branch is its own string and drifted the same way -- it told
    # the model to "leave related_hypothesis_ids empty". A guard that only ever
    # renders the populated branch would have passed while half the prompt
    # stayed wrong.
    text = prompts.extraction_rubric_block(hypotheses=())
    properties = _fact_schema_properties()
    unknown = {name for name in _IDENTIFIER.findall(text) if name not in properties}
    assert not unknown, f"the no-hypotheses branch instructs unknown fields: {sorted(unknown)}"
