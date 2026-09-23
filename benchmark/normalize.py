"""Tokenization, normalization and switch-point detection (phase_specification S1-S3).

Pure and free of THIRD-PARTY dependencies by design (see 01-01-PLAN.md Task 2): every
number produced anywhere in the benchmark rests on this module treating the reference
and every engine hypothesis identically. `normalize_text()` then `tokenize()` is the only
sanctioned path from raw text to scored tokens — score.py accepts only the token
lists this module produces, and run.py is the only caller allowed to build them.

The one in-package import is `benchmark.aliases` (added 2026-08-09), used ONLY by
`_classify_raw_token` to split a single attached Hebrew prefix off a known
transliterated stem — the mirror of the hyphen split Latin already receives. It is
deliberately not a general transliteration facility; see that module's docstring for
why an algorithmic one would be worse than none. Language tagging still derives from
Unicode script alone and no alias ever changes a token's `lang`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from benchmark import aliases

# --- S2: language tags --------------------------------------------------------------

LANG_HEBREW = "he"
LANG_ENGLISH = "en"
LANG_NEUTRAL = "neutral"

# Hebrew Unicode block, including punctuation/points (U+0590-U+05FF). Used for
# script detection (S2) and covers the niqqud/cantillation range stripped in S1.
_HEBREW_BLOCK_START = 0x0590
_HEBREW_BLOCK_END = 0x05FF

# Niqqud and cantillation marks (U+0591-U+05C7). Pointing is never spoken and no
# engine emits it consistently, so it is stripped entirely (not replaced with a
# space) since it combines onto the consonant it follows.
_NIQQUD_START = 0x0591
_NIQQUD_END = 0x05C7

# Hebrew geresh/gershayim, mapped onto ASCII apostrophe/double-quote (S1 step 3),
# alongside the common curly-quote codepoints that both engines and humans use
# interchangeably with a plain ASCII quote.
_QUOTE_MAP = {
    "׳": "'",  # HEBREW PUNCTUATION GERESH
    "״": '"',  # HEBREW PUNCTUATION GERSHAYIM
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "‚": "'",  # SINGLE LOW-9 QUOTATION MARK
    "‛": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "′": "'",  # PRIME
    "ʼ": "'",  # MODIFIER LETTER APOSTROPHE
    "`": "'",
    "“": '"',  # LEFT DOUBLE QUOTATION MARK
    "”": '"',  # RIGHT DOUBLE QUOTATION MARK
    "„": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "″": '"',  # DOUBLE PRIME
}

# Characters kept when intra-word (S1 step 5); every other punctuation codepoint is
# removed.
_INTRAWORD_KEEP = {"'", "-"}

_WHITESPACE_RE = re.compile(r"\s+")


def _is_hebrew(ch: str) -> bool:
    return _HEBREW_BLOCK_START <= ord(ch) <= _HEBREW_BLOCK_END


def _is_niqqud(ch: str) -> bool:
    return _NIQQUD_START <= ord(ch) <= _NIQQUD_END


def _is_latin_letter(ch: str) -> bool:
    if not ch.isalpha() or _is_hebrew(ch):
        return False
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def _is_word_char(ch: str) -> bool:
    return ch.isalnum()


# --- S1: normalization ---------------------------------------------------------------


def _strip_niqqud(text: str) -> str:
    return "".join(ch for ch in text if not _is_niqqud(ch))


def _map_quotes(text: str) -> str:
    return "".join(_QUOTE_MAP.get(ch, ch) for ch in text)


def _lowercase_latin(text: str) -> str:
    # Hebrew is caseless and final-form letters are standard orthography, not a
    # variant spelling, so only Latin letters are folded.
    return "".join(ch if _is_hebrew(ch) else ch.lower() for ch in text)


def _strip_punctuation(text: str) -> str:
    chars = list(text)
    n = len(chars)
    out: list[str] = []
    for i, ch in enumerate(chars):
        if ch in _INTRAWORD_KEEP:
            prev_word = i > 0 and _is_word_char(chars[i - 1])
            next_word = i < n - 1 and _is_word_char(chars[i + 1])
            out.append(ch if (prev_word and next_word) else " ")
            continue
        category = unicodedata.category(ch)
        out.append(" " if category.startswith("P") else ch)
    return "".join(out)


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Apply S1 steps 1-6. Step 7 (digits stay as digits) is a no-op by
    construction — nothing above touches digit codepoints."""
    text = unicodedata.normalize("NFC", text)
    text = _strip_niqqud(text)
    text = _map_quotes(text)
    text = _lowercase_latin(text)
    text = _strip_punctuation(text)
    text = _collapse_whitespace(text)
    return text


# --- S2: language tagging + tokenization ----------------------------------------------


@dataclass(frozen=True)
class Token:
    text: str
    lang: str  # LANG_HEBREW | LANG_ENGLISH | LANG_NEUTRAL
    ambiguous: bool = False


def tag_lang(token_text: str) -> str:
    """S2: derived by Unicode script, never hand-authored."""
    if any(_is_hebrew(ch) for ch in token_text):
        return LANG_HEBREW
    if any(_is_latin_letter(ch) for ch in token_text):
        return LANG_ENGLISH
    return LANG_NEUTRAL


def _classify_raw_token(raw: str) -> list[Token]:
    has_he = any(_is_hebrew(ch) for ch in raw)
    has_en = any(_is_latin_letter(ch) for ch in raw)

    if has_he and has_en and "-" in raw:
        parts = [p for p in raw.split("-") if p]
        result: list[Token] = []
        for part in parts:
            result.extend(_classify_raw_token(part))
        return result

    if has_he and has_en:
        # Mixed-script token that survives the hyphen split (or never had a
        # hyphen): tag by majority letter count and flag for human review.
        he_count = sum(1 for ch in raw if _is_hebrew(ch))
        en_count = sum(1 for ch in raw if _is_latin_letter(ch))
        lang = LANG_HEBREW if he_count >= en_count else LANG_ENGLISH
        return [Token(text=raw, lang=lang, ambiguous=True)]

    # A transliterated English word carrying an attached Hebrew prefix
    # (`בקלוד`) is the mirror image of the hyphenated Latin case handled above
    # (`ב-Claude` -> `ב` + `claude`), and must split the same way or the two
    # forms can never align. Only stems the alias table already names are
    # split -- ordinary Hebrew words are never touched. Language tagging is
    # unchanged: both halves stay Hebrew-script here, so switch points are
    # exactly where the speaker put them (benchmark/aliases.py explains why
    # that separation is load-bearing).
    parts = aliases.split_prefixed_alias(raw)
    if len(parts) > 1:
        return [Token(text=part, lang=tag_lang(part)) for part in parts]

    return [Token(text=raw, lang=tag_lang(raw))]


def tokenize(normalized_text: str) -> tuple[list[Token], list[str]]:
    """Whitespace split after S1, with the mixed-script hyphen-split rule.

    Returns (tokens, ambiguous_tokens) where ambiguous_tokens lists the raw text
    of every mixed-script token that survived the hyphen split, for human review.
    """
    tokens: list[Token] = []
    ambiguous: list[str] = []
    for raw in normalized_text.split():
        classified = _classify_raw_token(raw)
        tokens.extend(classified)
        ambiguous.extend(t.text for t in classified if t.ambiguous)
    return tokens, ambiguous


def normalize_and_tokenize(text: str) -> tuple[list[Token], list[str]]:
    """Convenience wrapper: the only path run.py should use for reference and
    hypothesis text alike."""
    return tokenize(normalize_text(text))


# --- S3: switch points -----------------------------------------------------------------


def find_switch_points(tokens: list[Token]) -> list[tuple[int, int]]:
    """A switch point is the boundary between reference tokens i and j where
    lang(i) != lang(j), both are in {he, en}, and every token strictly between
    them is neutral. Neutral tokens are transparent."""
    non_neutral = [idx for idx, tok in enumerate(tokens) if tok.lang != LANG_NEUTRAL]
    switches: list[tuple[int, int]] = []
    for a, b in zip(non_neutral, non_neutral[1:]):
        if tokens[a].lang != tokens[b].lang:
            switches.append((a, b))
    return switches
