"""Script-equivalence aliases for WER scoring (added 2026-08-09 from real-call evidence).

## Why this exists

Two live bilingual calls showed the engine TRANSLITERATES English into Hebrew
about as often as it preserves Latin script: "Claude" arrived as `קלוד`, while
`PM` survived as Latin inside `ה-PMים`. Only 8-14% of segments carried any
Latin characters at all. The user, reading both transcripts, judged the two
forms equivalent: "both are great because they both mean the same here."

Without this module a reference saying `claude` and a hypothesis saying `קלוד`
scores as a substitution, so the engine is penalised for a choice that carries
no meaning -- and it is penalised *precisely at code-switch boundaries*, which
is where `switch_adjacent_wer` does its measuring. The headline metric would be
distorted by the one phenomenon the benchmark exists to study.

## The two rules that keep this honest

**1. Applied at COMPARISON time only -- never during normalization or tagging.**
`normalize.tag_lang()` derives a token's language from its Unicode script, and
`normalize.find_switch_points()` defines switch points from the REFERENCE's
language sequence. Canonicalising `קלוד` to `claude` inside `normalize_text`
would flip that token from Hebrew to English and MOVE the switch points --
the benchmark would then measure a window around boundaries that were never
spoken, and would look perfectly healthy while doing it. So `score.build_alignment`
substitutes canonical forms ONLY in the strings handed to the aligner, and keeps
the original `Token` objects, with their original script and language tags, in
`Alignment.reference` / `Alignment.hypothesis`.

**2. An explicit hand-maintained table -- NEVER algorithmic transliteration.**
Hebrew<->Latin transliteration is many-to-many and unstable (`ק`/`כ`, `ו`/`וו`,
absent vowels). An algorithm would invent equivalences nobody authored and would
silently LOWER WER in ways no reader could audit -- the opposite failure to the
one above, and harder to notice because the number improves. Every entry below is
a claim a human can read and disagree with. When in doubt, leave it out: a
missing alias costs one visible substitution, an invented one silently forgives
a real error.

## How to extend it

Add a pair only when both forms are actually observed for the same spoken word,
and say where. The natural moment is while hand-typing `reference.txt`: type
what you heard in whichever script you heard it, and add an entry here whenever
the engine chose the other script for the same word. `score.Alignment.alias_hits`
counts how many tokens were canonicalised, and the benchmark report prints it, so
a reader can always see how much work this table did for a given number.
"""

from __future__ import annotations

# Canonical form is the lowercase Latin spelling. Every alias maps TO it.
# Each entry names the evidence for its existence.
_ALIAS_TO_CANONICAL: "dict[str, str]" = {
    # Observed 2026-08-09, sessions/20260809T140347 + 20260809T145730: the
    # engine rendered the spoken product name "Claude" in Hebrew script while
    # the same speaker used Latin for other product names in the same call.
    "קלוד": "claude",
    "קלאוד": "claude",
}

# Terms observed in Latin script in the same calls and therefore NOT aliased
# here -- they need no entry unless a Hebrew-script rendering is actually seen:
#   product, ai, plg, harness, llm, pm, x
# Adding speculative Hebrew spellings for these would be exactly the invented
# equivalence the module docstring rejects.


# Single-letter Hebrew prefixes that attach directly to a following word with no
# space: ב(in/with) ל(to) ה(the) ו(and) מ(from) ש(that) כ(as). Latin gets the
# same job done with a hyphen (`ה-PMים`, `ל-llm`), which `normalize._classify_raw_token`
# already splits -- so `ב-Claude` becomes two tokens while `בקלוד` stays one.
# Without the split below the alias can never fire on the form real speech
# actually produces, and the token counts differ too, so the pair scores as an
# error twice over. Found 2026-08-09 by mutation-testing this module's own guard.
_HEBREW_PREFIXES = ("ב", "ל", "ה", "ו", "מ", "ש", "כ")


def split_prefixed_alias(token_text: str) -> "tuple[str, ...]":
    """Split a single attached Hebrew prefix off a KNOWN alias stem, mirroring
    the hyphen split Latin already gets. Returns a 1-tuple unchanged when the
    remainder is not a known alias -- this must never split ordinary Hebrew
    words, only the handful of stems the table already names.
    """
    if token_text in _ALIAS_TO_CANONICAL:
        return (token_text,)
    if len(token_text) > 1 and token_text[0] in _HEBREW_PREFIXES:
        stem = token_text[1:]
        if stem in _ALIAS_TO_CANONICAL:
            return (token_text[0], stem)
    return (token_text,)


def canonical(token_text: str) -> str:
    """Return the canonical comparison form of one already-normalized token.

    Identity for anything not in the table -- this must never guess. Callers
    pass text that has already been through `normalize.normalize_text`, so
    Latin is lowercased and punctuation is stripped before lookup.
    """
    return _ALIAS_TO_CANONICAL.get(token_text, token_text)


def alias_pairs() -> "tuple[tuple[str, str], ...]":
    """The table as an ordered, inspectable tuple -- for the report, and for the
    test that pins it so a future edit is a deliberate, reviewed change."""
    return tuple(sorted(_ALIAS_TO_CANONICAL.items()))
