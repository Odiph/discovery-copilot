"""S5 metric family tests (Task 3, TDD). All fixtures are hand-authored
(reference, hypothesis) token pairs under benchmark/fixtures/text/cases.json —
no audio, no network, no model. Each case's `note` states the arithmetic it
proves; see that file and its README for how the numbers were derived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import normalize as nz
from benchmark import score

_CASES_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "text" / "cases.json"
_CASES = json.loads(_CASES_PATH.read_text(encoding="utf-8"))


def _tokens(text: str):
    tokens, _ = nz.tokenize(nz.normalize_text(text))
    return tokens


def _build(case_name: str, hyp_override: str | None = None):
    case = _CASES[case_name]
    ref_tokens = _tokens(case["reference"])
    hyp_text = hyp_override if hyp_override is not None else case["hypothesis"]
    hyp_tokens = _tokens(hyp_text)
    alignment = score.build_alignment(ref_tokens, hyp_tokens)
    switch_points = nz.find_switch_points(ref_tokens)
    return alignment, switch_points


# --- identical hypothesis: every rate 0, ratio undefined ------------------------


def test_identical_hypothesis_scores_zero_everywhere_and_ratio_is_undefined():
    alignment, switches = _build("identical_hypothesis")
    assert score.overall_wer(alignment) == 0.0
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches)
    assert sa_wer == 0.0
    ns_wer, _ = score.non_switch_wer(alignment, switches)
    # No tokens fall outside the window in this tiny fixture, so non_switch_wer's
    # n is 0 -> None. Either way, the ratio must be reported as undefined, not a
    # ZeroDivisionError.
    ratio = score.switch_penalty_ratio(sa_wer, ns_wer)
    assert ratio == score.UNDEFINED


# --- S3: known switch count, neutral digit transparency -------------------------


def test_neutral_digit_does_not_suppress_switch_point():
    tokens = _tokens(_CASES["neutral_digit_does_not_suppress_switch"]["reference"])
    switches = nz.find_switch_points(tokens)
    assert switches == [(1, 3)]


# --- S4: overlapping windows merge into a union ----------------------------------


def test_overlapping_windows_merge_into_one_union():
    alignment, switches = _build("overlapping_windows_merge")
    _, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    assert n_window == 6  # not 5 + 5


# --- errors inside vs. outside the window ----------------------------------------


def test_errors_inside_window_only_leave_non_switch_wer_at_zero():
    alignment, switches = _build("errors_inside_window_only")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, n_outside = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 8
    assert n_outside == 2
    assert sa_wer == pytest.approx(1 / 8)
    assert ns_wer == 0.0
    ratio = score.switch_penalty_ratio(sa_wer, ns_wer)
    assert ratio == score.INFINITE  # non-zero / zero, not a crash


def test_errors_outside_window_only_leave_switch_adjacent_wer_at_zero():
    alignment, switches = _build("errors_outside_window_only")
    sa_wer, _ = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, _ = score.non_switch_wer(alignment, switches, window=3)
    assert sa_wer == 0.0
    assert ns_wer == pytest.approx(1 / 2)


# --- S5 insertion attribution ------------------------------------------------------


def test_insertion_both_bounding_tokens_in_window_is_attributed():
    alignment, switches = _build("insertion_both_bounding_in_window")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, _ = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 8
    assert sa_wer == pytest.approx(1 / 8)
    assert ns_wer == 0.0


def test_insertion_no_window_token_either_side_is_not_attributed():
    alignment, switches = _build("insertion_no_window_token_either_side")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, n_outside = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 12
    assert sa_wer == 0.0
    assert n_outside == 2
    assert ns_wer == pytest.approx(1 / 2)


def test_insertion_mixed_boundary_is_attributed_to_window():
    """S5's 'at least one' rule: a gap bounded by one in-window and one
    out-of-window reference token IS attributed to the window."""
    alignment, switches = _build("insertion_mixed_boundary")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, _ = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 7
    assert sa_wer == pytest.approx(1 / 7)
    assert ns_wer == 0.0


def test_insertion_before_first_token_decided_by_single_bounding_token_in_window():
    alignment, switches = _build("insertion_before_first_token_attributed")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, _ = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 4
    assert sa_wer == pytest.approx(1 / 4)
    assert ns_wer == 0.0


def test_insertion_after_last_token_decided_by_single_bounding_token_in_window():
    alignment, switches = _build("insertion_after_last_token_attributed")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, _ = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 4
    assert sa_wer == pytest.approx(1 / 4)
    assert ns_wer == 0.0


def test_insertion_after_last_token_decided_by_single_bounding_token_outside_window():
    alignment, switches = _build("insertion_after_last_token_not_attributed")
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switches, window=3)
    ns_wer, n_outside = score.non_switch_wer(alignment, switches, window=3)
    assert n_window == 7
    assert sa_wer == 0.0
    assert n_outside == 3
    assert ns_wer == pytest.approx(1 / 3)


# --- S5 en/he side split (BENCH-03b discriminator) ---------------------------------


def test_english_side_errors_raise_en_side_and_leave_he_side_at_zero():
    alignment, switches = _build("side_split_english_only_errors")
    en_wer, en_n = score.switch_adjacent_wer_en_side(alignment, switches, window=3)
    he_wer, he_n = score.switch_adjacent_wer_he_side(alignment, switches, window=3)
    assert en_n == 2
    assert he_n == 6
    assert en_wer == pytest.approx(1 / 2)
    assert he_wer == 0.0


# --- wrong_script_rate is not exclusive with substitution WER ----------------------


def test_wrong_script_transliteration_counts_in_both_metrics():
    alignment, switches = _build("wrong_script_transliteration")
    rate, n = score.wrong_script_rate(alignment, switches, window=3)
    sa_wer, _ = score.switch_adjacent_wer(alignment, switches, window=3)
    assert n == 8
    assert rate == pytest.approx(1 / 8)
    assert sa_wer == pytest.approx(1 / 8)  # same error also counts as a WER substitution


# --- W = 2/3/5 sensitivity ----------------------------------------------------------


def test_sensitivity_window_token_counts_are_monotonic_across_w():
    alignment, switches = _build("sensitivity_w2_w3_w5")
    _, n2 = score.switch_adjacent_wer(alignment, switches, window=2)
    _, n3 = score.switch_adjacent_wer(alignment, switches, window=3)
    _, n5 = score.switch_adjacent_wer(alignment, switches, window=5)
    assert n2 == 6
    assert n3 == 8
    assert n5 == 10
    assert n2 <= n3 <= n5


# --- overall_cer -----------------------------------------------------------------


def test_overall_cer_zero_for_identical_and_positive_for_a_typo():
    ref = _tokens("שלום עולם")
    identical_cer = score.overall_cer(ref, _tokens("שלום עולם"))
    assert identical_cer == 0.0

    typo_cer = score.overall_cer(_tokens("hello world"), _tokens("hallo world"))
    assert typo_cer > 0.0
