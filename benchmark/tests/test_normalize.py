"""S1 normalization + S2 tagging + S3 switch-point behaviors (Task 3)."""

from __future__ import annotations

from benchmark import normalize as nz


def test_strips_niqqud():
    # Interleave every consonant of שלום with a niqqud mark (QAMATS, U+05B8) —
    # stripping must remove every mark and recover the bare consonants exactly.
    bare = "שלום"
    qamats = "ָ"
    with_niqqud = qamats.join(bare) + qamats
    assert nz.normalize_text(with_niqqud) == bare


def test_folds_latin_case_but_not_hebrew():
    text = "HELLO שלום World"
    normalized = nz.normalize_text(text)
    assert normalized == "hello שלום world"


def test_preserves_hebrew_final_forms():
    # ך (final kaf) must not be normalized to כ (regular kaf) — final forms are
    # standard orthography, not a variant spelling.
    text = "ך"
    assert nz.normalize_text(text) == "ך"
    tokens, _ = nz.tokenize(nz.normalize_text("ראיתי אותך"))
    assert tokens[-1].text == "אותך"


def test_digits_stay_as_digits_no_expansion():
    # Standalone digits (not hyphen-glued to a prefix) survive untouched — no
    # number-word expansion ("5" never becomes "five" or "חמש").
    text = "יש לי 5 פגישות ויש גם 3 באגים"
    normalized = nz.normalize_text(text)
    assert "5" in normalized
    assert "3" in normalized
    tokens, _ = nz.tokenize(normalized)
    digit_tokens = [t.text for t in tokens if t.text.isdigit()]
    assert digit_tokens == ["5", "3"]


def test_hyphenated_hebrew_prefix_english_noun_splits_into_two_tokens():
    tokens, ambiguous = nz.tokenize(nz.normalize_text("עבדתי ב-Google השבוע"))
    texts_and_langs = [(t.text, t.lang) for t in tokens]
    assert ("ב", nz.LANG_HEBREW) in texts_and_langs
    assert ("google", nz.LANG_ENGLISH) in texts_and_langs
    assert ambiguous == []  # the hyphen split resolved the mix; nothing ambiguous survives


def test_neutral_digit_does_not_suppress_switch():
    tokens, _ = nz.tokenize(nz.normalize_text("א ב 5 google"))
    switches = nz.find_switch_points(tokens)
    assert switches == [(1, 3)]


def test_mixed_script_token_without_hyphen_is_ambiguous_and_tagged_by_majority():
    # a token with both scripts and no hyphen to split on: 3 Hebrew chars vs 1
    # Latin char -> majority is Hebrew. Latin case-folding happens in step 4, so
    # the surviving ambiguous token is lowercased.
    tokens, ambiguous = nz.tokenize(nz.normalize_text("אבגX"))
    assert len(tokens) == 1
    assert tokens[0].lang == nz.LANG_HEBREW
    assert tokens[0].ambiguous is True
    assert ambiguous == ["אבגx"]
