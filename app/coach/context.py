"""The grounding seam (04-CONTEXT.md D-01, §C6; CTX-01/02/03).

`load_brief()` and `load_survey()` are the ONLY readers of
`context/brief.json` and `context/survey.json` anywhere in this codebase.
Both take the context directory as an injectable argument so tests never
touch the real one.

`load_brief()` is the CTX-03 seam: v2 retrieval replaces this function's
BODY, never its callers and never its return type (`tuple[ContextItem, ...]`).
A caller that only ever calls `load_brief(context_dir)` -- or, for
substitutability, accepts a loader callable with this same signature -- is
what proves the seam holds; see `app/tests/test_coach_context.py`.

`provenance` has no default anywhere in this module. CTX-02 exists so a
guess is never silently presented to the model as fact -- a validator that
coerced a missing or malformed value to `"evidence"` would defeat the
requirement more thoroughly than having no validator at all. An invalid
item raises `ContextValidationError` naming the offending item's id.

Validated with `pydantic`, which arrives as `anthropic`'s own transitive
dependency (04-01-PLAN.md §C7) -- not a separately pinned package, and not
the situation `03-01-PLAN.md` §P9 rejected (that refused to build tests on a
library arriving through an UNRELATED vendor SDK's dependency tree; here the
library is a declared dependency of the package this plan pins deliberately
for its own vendor call, which is why counting it separately would be double
counting, not the missing-justification problem §P9 named).

This module must never write `context/brief.json` or `context/survey.json`
-- writing is 04-03's authoring page. It must never guess a default
provenance, and it must never be the second place either file is parsed.

**06-01-PLAN.md: v2 retrieval, in this function's BODY only.** `load_brief`'s
signature and its return type (`tuple[ContextItem, ...]`) are UNCHANGED --
this is the CTX-03 seam Phase 4 was built around, and `app/main.py:358`, its
one production caller, is untouched. The body now also reads
`context/facts.json` (via `app.coach.corpus.load_facts`, never a second
parser) and merges the facts that score relevant against the survey and the
hypotheses into the SAME returned tuple, after the hand-written brief items.

**Why the corpus import below is function-local, not at module scope.**
`app/coach/corpus.py:35` already does `from app.coach.context import
PROVENANCE_VALUES` -- a module-scope `from app.coach import corpus` here
would therefore be circular. The fix is a function-local import inside
`load_brief`'s own body, following `corpus.py:36`'s own "the lower-level
module imports the higher-level one, never the reverse" precedent, applied
here as "the import happens only inside the one function that needs it,
never at import time." Proven in `app/tests/test_coach_context.py` in TWO
fresh subprocess interpreters, in both import orders -- an in-process
assertion would prove nothing, since another test module has typically
already put `app.coach.corpus` into `sys.modules` by the time this one runs.

**`sessions_root=None`, deliberately, on this live path.** Resolving every
cited fact's citation against its session's own `transcript.jsonl` is real
I/O sixty seconds before a customer call, over data the loud loader at
extraction time (`05-02`'s `preflight_check_corpus_parses`) already
resolved once. See `load_facts`'s own docstring for the full reasoning.

**What the mapping never carries forward, in the same voice `corpus.py:82-89`
uses for the fields it deliberately omits: a fact's verbatim `quoted_text`
has NO HOME here.** The retrieval helpers below read `Fact.id`, `.claim`,
`.session_id`, `.seqs` and `.provenance` only -- never `.quoted_text`, which
is a previous, possibly different, customer's exact words and must never
reach a live prompt (GC-08). `.related_hypothesis_ids` is scoring input,
consumed before a `ContextItem` is ever built, exactly as `extract.py`
consumes its own copy of the same field. Gate 20
(`app/tests/test_phase_gates.py`) pins, by equality, the exact set of
modules allowed to read `Fact.quoted_text` at all -- this module is
deliberately not in it, and that is what keeps this true by construction,
not merely by care.

**STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: `Fact.seqs` is a SPAN, not
a single `seq`.** `_fact_to_context_item`'s citation string reads
`min(fact.seqs)`/`max(fact.seqs)` to render `session:{id}#seq:{first}` for a
single-segment span or `session:{id}#seq:{first}-{last}` for a multi-segment
one; `_select_and_map_facts`'s total-order sort key uses `min(fact.seqs)` in
`.seq`'s old ordinal position, so the sort remains a stable total order over
plain data exactly as before."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from app import config as config_mod

BRIEF_FILENAME = "brief.json"
SURVEY_FILENAME = "survey.json"

PROVENANCE_EVIDENCE = "evidence"
PROVENANCE_INFERENCE = "inference"
PROVENANCE_VALUES = (PROVENANCE_EVIDENCE, PROVENANCE_INFERENCE)

# A fact-derived `ContextItem.id` is namespaced with this prefix so it can
# never collide with a hand-authored `brief.json` id in the merged tuple,
# and so `app/coach/prompts.py` can tell a retrieved item from a hand-written
# one using only the five fields `ContextItem` already carries -- no sixth
# field, no separate container, nothing that would threaten the D-03 seam's
# single `tuple[ContextItem, ...]` return type.
FACT_ID_PREFIX = "fact:"


class ContextValidationError(Exception):
    """Raised for a malformed `context/brief.json` or `context/survey.json`
    -- a missing file, invalid JSON, a missing/misspelled/empty `provenance`,
    or a duplicate survey item id. The message always names the offending
    item's id (or the file, when the error is not item-scoped)."""


class ContextItem(BaseModel):
    """One brief entry (CTX-02): text, source, citation, and a required,
    non-defaultable evidence-vs-inference marker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    text: str
    source: Optional[str] = None
    citation: Optional[str] = None
    provenance: str  # "evidence" | "inference" -- no default, ever (CTX-02)

    @field_validator("provenance")
    @classmethod
    def _provenance_must_be_recognized(cls, value: str) -> str:
        if value not in PROVENANCE_VALUES:
            raise ValueError(
                f"provenance must be one of {PROVENANCE_VALUES!r}, got {value!r} -- "
                "never silently defaulted to 'evidence' (CTX-02)"
            )
        return value


class SurveyItem(BaseModel):
    """One survey question: a stable id (never reused -- the coverage event
    schema references it by `question_id`, and that reference must survive
    the user rewording the question later), an order, and the text itself.
    Deliberately NOT a `ContextItem` -- D-01/CTX-01/02 reinterpretation:
    the survey is a separate artifact with no provenance concept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    order: int
    text: str


def _item_label(entry: object, index: int) -> str:
    if isinstance(entry, dict) and "id" in entry:
        return repr(entry["id"])
    return f"<item at index {index}>"


# --- 06-01-PLAN.md §F3: retrieval over the fact corpus, private to this module ------
#
# Nothing outside `load_brief` needs to call any of the functions below -- a public
# seam here would invite a second caller that reintroduces per-cycle cost (06-RESEARCH
# §2's Design B, deliberately not adopted for v1). Every helper is pure, deterministic,
# no I/O, no client, plain data in and out -- the `rank_candidates` shape
# (`app/coach/critic.py:222-258`) applied to fact selection instead of candidate order.
#
# `fact`/`hypothesis` parameters below are typed loosely (`Any`) rather than as
# `corpus.Fact`/`corpus.Hypothesis` -- importing those TYPES at module scope would be
# the exact circular import `load_brief`'s own function-local import exists to avoid
# (mirrors `prompts.py`'s own "typed loosely here to avoid a hard import-time coupling"
# precedent for transcript segments).

# A small, fixed bump for a fact whose OWN `related_hypothesis_ids` intersect the
# CURRENTLY loaded hypothesis set. "Bounded" (06-RESEARCH.md §F3) means exactly this:
# never large enough to pull in a fact with zero textual overlap on its own -- see
# `_score_fact` below, where the bonus is added only AFTER a genuine overlap already
# scored above zero, never in place of one.
_RELATED_HYPOTHESIS_BONUS = 2


def _query_tokens(survey: "tuple[SurveyItem, ...]", hypotheses: "tuple[Any, ...]") -> "Counter[str]":
    """The retrieval query: every survey question's text, plus every loaded
    hypothesis's `statement` and `falsifier`, casefolded and split on
    whitespace. Stdlib only (06-RESEARCH.md §1 -- no embeddings, no BM25
    library, no network call; the corpus is small by construction, D-02).
    A `Counter`, never a `set` -- see `_select_and_map_facts` for why the
    distinction is load-bearing, not stylistic."""
    tokens: "Counter[str]" = Counter()
    for item in survey:
        tokens.update(item.text.casefold().split())
    for hypothesis in hypotheses:
        tokens.update(hypothesis.statement.casefold().split())
        tokens.update(hypothesis.falsifier.casefold().split())
    return tokens


def _score_fact(fact: Any, query_tokens: "Counter[str]", hypothesis_ids: "frozenset[str]") -> int:
    """Casefolded token-multiset overlap between `fact.claim` and the query,
    plus `_RELATED_HYPOTHESIS_BONUS` when the fact's own
    `related_hypothesis_ids` intersect `hypothesis_ids`. A fact scoring ZERO
    overlap is not selected at all -- the early return below fires before the
    bonus is ever considered, so a fact sharing no vocabulary with the survey
    or the hypotheses is excluded even when it is hypothesis-linked."""
    fact_tokens = Counter(fact.claim.casefold().split())
    overlap = sum((fact_tokens & query_tokens).values())
    if overlap == 0:
        return 0
    if hypothesis_ids and set(fact.related_hypothesis_ids) & hypothesis_ids:
        overlap += _RELATED_HYPOTHESIS_BONUS
    return overlap


def _fact_to_context_item(fact: Any, *, source_node: "Optional[str]" = None) -> ContextItem:
    """§F3's mapping, four of `Fact`'s seven fields: `provenance` is a direct
    passthrough of the SAME imported enum (no translation table -- gate 18
    exists to prevent exactly that drift); `claim` becomes `text`, verbatim,
    never a superstring; `id` is namespaced with `FACT_ID_PREFIX`;
    `session_id`+`seqs` become `citation`, free text by contract --
    `session:{id}#seq:{first}` when `fact.seqs` is a single segment, or
    `session:{id}#seq:{first}-{last}` for a multi-segment span (STATE.md §
    FIRST REAL-SPEECH EVIDENCE FINDING 1), reading `min`/`max` rather than
    assuming any particular stored order. Never reads `fact.quoted_text` (no
    home in `ContextItem` -- see the module docstring and gate 20) or
    `fact.related_hypothesis_ids` (scoring input, already consumed by
    `_score_fact` before this point).

    Phase A: `source_node` qualifies a fact drawn from a DIFFERENT project
    node than the one this interview is filed under. `None` means the filing
    node itself and produces byte-identical output to before this parameter
    existed -- which is what keeps `prompts.static_block`'s byte-stability
    contract intact for every single-project run.

    Qualification is not cosmetic. Session ids are `%Y%m%dT%H%M%S` and
    `create_session_dir` only checks collisions WITHIN one root, so two
    projects can mint the identical session id. An unqualified merged id or
    citation would then be genuinely ambiguous at render time -- and, worse,
    would read as a single provenance chain across two customers."""
    first_seq, last_seq = min(fact.seqs), max(fact.seqs)
    seq_part = f"{first_seq}" if first_seq == last_seq else f"{first_seq}-{last_seq}"
    if source_node is None:
        return ContextItem(
            id=f"{FACT_ID_PREFIX}{fact.id}",
            text=fact.claim,
            source=f"past interview {fact.session_id}",
            citation=f"session:{fact.session_id}#seq:{seq_part}",
            provenance=fact.provenance,
        )
    return ContextItem(
        id=f"{FACT_ID_PREFIX}{source_node}:{fact.id}",
        text=fact.claim,
        source=f"past interview {fact.session_id} (project {source_node})",
        citation=f"project:{source_node}#session:{fact.session_id}#seq:{seq_part}",
        provenance=fact.provenance,
    )


def _select_and_map_facts(
    sourced_facts: "tuple[tuple[Optional[str], Any], ...]",
    *,
    survey: "tuple[SurveyItem, ...]",
    hypotheses: "tuple[Any, ...]",
    max_selected: int = config_mod.DEFAULT_COACH_MAX_RETRIEVED_FACTS,
) -> "tuple[ContextItem, ...]":
    """Scores every fact, drops the zero-overlap ones, and sorts the survivors
    by a TOTAL ORDER over plain data -- `(-score, session_id, min(seqs), id)`
    (STATE.md § FIRST REAL-SPEECH EVIDENCE FINDING 1: `min(fact.seqs)` fills
    `.seq`'s old ordinal position in this key, unchanged otherwise) --
    capping at `max_selected`. NEVER sorts or selects by iterating a `set` or
    a `dict` in hash-derived order: Python randomizes `str` hashing per
    process, so a set-ordered selection would make `prompts.static_block`'s
    byte-stability contract (its own module docstring) differ between two
    interpreters with different `PYTHONHASHSEED` values -- silently doubling
    the prompt's cache-miss rate with no error and no warning (T-06-05).
    Two contradicting facts that both score relevant are BOTH kept, each with
    its own distinct citation -- never deduplicated, never resolved to a
    single 'winner' (06-RESEARCH.md §5; D-01 forbids a second silent write
    path into hypothesis state, and picking a winner here would be exactly
    that in miniature).

    Phase A: takes `(source_node, fact)` pairs from ALL selected project
    nodes POOLED, and selects once over the pool. Per-directory selection was
    the obvious alternative and is wrong twice over: `max_selected` would be
    applied once per node, so choosing three source nodes would put four
    times the facts in the brief -- spending cache tokens and diluting it,
    the exact effect this cap exists to prevent -- and a highly relevant fact
    in one node could not outrank an irrelevant one in another.

    `source_node` enters the sort key immediately after the score, ahead of
    `session_id`, because two nodes can carry the SAME session id (see
    `_fact_to_context_item`). Without it the key would stop being a total
    order over distinct facts and the tie would fall through to a uuid4,
    which is stable within a run but meaningless across one. Placing it after
    `-score` leaves the single-node ordering (where it is constant)
    byte-identical."""
    query_tokens = _query_tokens(survey, hypotheses)
    hypothesis_ids = frozenset(h.id for h in hypotheses)

    scored: "list[tuple[int, Optional[str], Any]]" = []
    for source_node, fact in sourced_facts:
        score = _score_fact(fact, query_tokens, hypothesis_ids)
        if score > 0:
            scored.append((score, source_node, fact))

    scored.sort(
        key=lambda triple: (
            -triple[0],
            triple[1] or "",
            triple[2].session_id,
            min(triple[2].seqs),
            triple[2].id,
        )
    )
    selected = scored[:max_selected]
    return tuple(
        _fact_to_context_item(fact, source_node=source_node)
        for _, source_node, fact in selected
    )


def load_brief(context_dir: "Path | str") -> "tuple[ContextItem, ...]":
    """The CTX-03 seam: v1 read `context/brief.json` verbatim; v2 retrieval
    (06-01-PLAN.md) extends this function's BODY ONLY -- the signature and
    the return type are unchanged. Raises `ContextValidationError` naming the
    offending item's id if `brief.json` is missing, unreadable, not a JSON
    array, or any item's `provenance` is missing/misspelled/empty.

    The second half -- retrieval over `context/facts.json` -- never raises
    for an ABSENT facts file (D-02/D-05: interview #1's empty corpus is the
    normal case, not an edge case) but DOES raise, via `corpus.load_facts`,
    for a MALFORMED one, naming the offending fact. See the module docstring
    for the function-local corpus import and the `sessions_root=None`
    decision.

    **Phase A: this signature is still exactly D-03's single positional
    argument, and that is deliberate.** Cross-project reads are
    `load_grounded_brief` below; this function is defined as that one with no
    extra sources. Widening this signature was the obvious move and would
    have broken the CTX-03 seam the whole of Phase 6 is built to preserve --
    the seam is proven by a caller that only ever calls
    `load_brief(context_dir)`, or accepts a loader callable of this shape."""
    return load_grounded_brief(context_dir, extra_fact_dirs=())


def load_grounded_brief(
    context_dir: "Path | str",
    *,
    extra_fact_dirs: "Sequence[tuple[str, Path | str]]",
) -> "tuple[ContextItem, ...]":
    """`load_brief` plus facts drawn from other project nodes in the same
    hierarchy.

    `extra_fact_dirs` is a sequence of `(source_node, context_dir)` pairs and
    has **no default**, on purpose. The forcing function gate 22 applied to
    `session_start_grounding_s` and gate 26 later had to add for
    `hypotheses`: a compatibility default here is exactly the shape that let
    the live coach run with zero hypotheses for an entire phase while
    reporting itself healthy. Every caller names its source set, even when
    the set is empty.

    **Reads merge; writes never do.** The brief and the survey always come
    from `context_dir` alone -- they are what THIS interview is about, and
    merging another node's survey would silently change the questions.
    Only facts and hypotheses merge, and only as read-only grounding.

    Callers are responsible for having established that every extra node
    shares a root with the filing node (`app.projects.share_a_root`); this
    function does not re-derive that, because it takes directories rather
    than project paths and cannot see the hierarchy."""
    context_dir = Path(context_dir)
    path = context_dir / BRIEF_FILENAME
    if not path.is_file():
        raise ContextValidationError(f"{path} does not exist -- author it before `start`.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ContextValidationError(f"{path} must be a JSON array of brief items.")

    items: "list[ContextItem]" = []
    for index, entry in enumerate(raw):
        try:
            items.append(ContextItem.model_validate(entry))
        except ValidationError as exc:
            raise ContextValidationError(
                f"{path} item {_item_label(entry, index)} is invalid: {exc}"
            ) from exc

    # v2 retrieval. Function-local, deliberately -- see the module docstring:
    # `corpus.py:35` imports PROVENANCE_VALUES from THIS module at module
    # scope, so a module-scope import in the other direction would be
    # circular. Importing only inside this function, after the hand-written
    # brief above is already fully parsed, breaks the cycle at zero cost to
    # anything else in this module.
    from app.coach import corpus

    # `None` marks the filing node, and is what keeps its mapped facts
    # byte-identical to the pre-Phase-A output.
    sources: "list[tuple[Optional[str], Path]]" = [(None, context_dir)]
    sources.extend((node, Path(directory)) for node, directory in extra_fact_dirs)

    sourced_facts: "list[tuple[Optional[str], Any]]" = []
    merged_hypotheses: "list[Any]" = []
    for source_node, directory in sources:
        # D-02/D-05: an absent facts file is the normal empty-corpus case and
        # never an error. Checked PER SOURCE -- an extra node with no corpus
        # yet must not stop the filing node's own facts being read, and vice
        # versa. The pre-Phase-A early return on an absent file was safe only
        # because there was exactly one source.
        if (directory / corpus.FACTS_FILENAME).is_file():
            sourced_facts.extend(
                (source_node, fact)
                for fact in corpus.load_facts(directory, sessions_root=None)
            )
        # Hypotheses from every source are scoring input only. They widen
        # what counts as relevant; they are never presented as this
        # interview's own open questions, and nothing here writes them.
        merged_hypotheses.extend(corpus.load_hypotheses(directory, missing_ok=True))

    if not sourced_facts:
        # An explicit empty array is the identical degrade case as an absent
        # file -- both must reduce to exactly the hand-written brief.
        return tuple(items)

    survey = load_survey(context_dir)
    retrieved = _select_and_map_facts(
        tuple(sourced_facts), survey=survey, hypotheses=tuple(merged_hypotheses)
    )
    return tuple(items) + retrieved


def load_survey(context_dir: "Path | str") -> "tuple[SurveyItem, ...]":
    """Reads `context/survey.json`, returns items ordered by `order`. Raises
    `ContextValidationError` naming the offending item's id if the file is
    missing, unreadable, not a JSON array, any item is malformed, or an id
    is duplicated."""
    path = Path(context_dir) / SURVEY_FILENAME
    if not path.is_file():
        raise ContextValidationError(f"{path} does not exist -- author it before `start`.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContextValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise ContextValidationError(f"{path} must be a JSON array of survey items.")

    items: "list[SurveyItem]" = []
    seen_ids: "set[str]" = set()
    for index, entry in enumerate(raw):
        try:
            item = SurveyItem.model_validate(entry)
        except ValidationError as exc:
            raise ContextValidationError(
                f"{path} item {_item_label(entry, index)} is invalid: {exc}"
            ) from exc
        if item.id in seen_ids:
            raise ContextValidationError(f"{path} item {item.id!r} is a duplicate id.")
        seen_ids.add(item.id)
        items.append(item)
    return tuple(sorted(items, key=lambda i: i.order))
