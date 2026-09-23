"""Script-equivalence aliases (benchmark/aliases.py), added 2026-08-09.

The load-bearing test in this file is `test_aliasing_does_not_move_switch_points`.
Everything else checks that the equivalence works; that one checks it did not
break the metric it was added to protect.
"""

from __future__ import annotations

from benchmark import aliases, score
from benchmark.normalize import (
    LANG_ENGLISH,
    LANG_HEBREW,
    find_switch_points,
    normalize_and_tokenize,
)


def test_canonical_is_identity_for_anything_not_in_the_table():
    # The table must never guess. A missing alias costs one visible
    # substitution; an invented one silently forgives a real error.
    for text in ("shalom", "מילה", "product", "1200", "llm"):
        assert aliases.canonical(text) == text


def test_the_alias_table_is_pinned_so_a_future_edit_is_deliberate():
    # Not a containment check: an equality pin means adding an entry is a
    # reviewed change, which is the whole discipline this table depends on.
    assert aliases.alias_pairs() == (
        ("קלאוד", "claude"),
        ("קלוד", "claude"),
    )


def test_a_transliterated_product_name_is_not_scored_as_an_error():
    """The real-speech case: `ב-Claude` in the reference against `בקלוד` from
    the engine. Asserted as ZERO total errors, not "no substitution mentioning
    claude" -- the weaker form passed vacuously when the mismatch showed up as
    a delete plus an insert instead of a substitution, which is exactly how the
    first version of this module shipped broken.
    """
    ref_tokens, _ = normalize_and_tokenize("אני משתמש ב-Claude כל יום")
    hyp_tokens, _ = normalize_and_tokenize("אני משתמש בקלוד כל יום")

    # Both sides must tokenize to the same length, or the prefix split is not
    # doing its job and jiwer will score the length difference as errors.
    assert len(ref_tokens) == len(hyp_tokens) == 6, (
        f"prefix split failed: ref={[t.text for t in ref_tokens]} hyp={[t.text for t in hyp_tokens]}"
    )

    alignment = score.build_alignment(ref_tokens, hyp_tokens)
    assert score.overall_wer(alignment) == 0.0, (
        "a meaning-preserving script choice was scored as an error: "
        f"{[(c.type, c.ref_start_idx, c.ref_end_idx) for c in alignment.chunks]}"
    )


def test_an_ordinary_hebrew_word_that_merely_starts_with_a_prefix_letter_is_never_split():
    # `בקלוב` shares its first two letters with `בקלוד` and is an ordinary word.
    # Splitting it would corrupt every reference containing it.
    tokens, _ = normalize_and_tokenize("בקלוב משהו")
    assert [t.text for t in tokens] == ["בקלוב", "משהו"]


def test_alias_hits_are_counted_so_the_report_can_show_them():
    ref_tokens, _ = normalize_and_tokenize("Claude")
    hyp_tokens, _ = normalize_and_tokenize("קלוד")
    alignment = score.build_alignment(ref_tokens, hyp_tokens)
    # One token canonicalised on the hypothesis side; the reference was already
    # canonical. A WER number is not auditable without this count.
    assert alignment.alias_hits == 1


def test_aliasing_does_not_move_switch_points():
    """THE regression this module could plausibly cause.

    `find_switch_points` reads language tags off the tokens stored on the
    alignment. If `build_alignment` canonicalised `קלוד` to `claude` in the
    stored tokens rather than only in the aligner's input, that token would
    flip Hebrew -> English and a switch point would appear (or vanish) that was
    never spoken. `switch_adjacent_wer` would then measure a window around a
    boundary that does not exist, and would look entirely healthy doing it.
    """
    tokens, _ = normalize_and_tokenize("אני משתמש בקלוד כל יום")
    before = find_switch_points(tokens)

    alignment = score.build_alignment(tokens, list(tokens))
    after = find_switch_points(list(alignment.reference))

    assert after == before, "aliasing moved the switch points it must never touch"
    # And the stored token really does keep its original script and tag.
    stored = [t for t in alignment.reference if "קלוד" in t.text]
    assert stored, "the Hebrew-script token was rewritten in the stored alignment"
    assert all(t.lang == LANG_HEBREW for t in stored)


def test_a_real_switch_point_is_still_found_when_the_reference_uses_latin():
    # The complementary case: when the SPEAKER used Latin, the switch is real
    # and must survive. Aliasing must not erase genuine code-switches either.
    tokens, _ = normalize_and_tokenize("אני משתמש ב Claude כל יום")
    langs = {t.lang for t in tokens}
    assert LANG_HEBREW in langs and LANG_ENGLISH in langs
    assert find_switch_points(tokens), "a genuine he->en switch disappeared"
