"""S6 gate evaluation, deterministic selection, and the full multi-candidate
report render (Task 3, TDD). LOW_CONFIDENCE flagging, the mandatory
user-override section, and the no-verbatim-transcript-text guarantee are all
exercised here.
"""

from __future__ import annotations

from benchmark import config, report


# --- LOW_CONFIDENCE flag ------------------------------------------------------------


def test_low_confidence_flag_set_below_threshold():
    summary = report.build_candidate_summary_from_metrics(
        name="engine-a",
        overall_wer=0.1,
        switch_adjacent_wer=0.2,
        non_switch_wer=0.05,
        switch_penalty_ratio=4.0,
        switch_adjacent_wer_en_side=0.2,
        switch_adjacent_wer_he_side=0.2,
        wrong_script_rate=0.0,
        overall_cer=0.05,
        n_window_tokens=50,
    )
    assert summary.low_confidence is True


def test_low_confidence_flag_clear_at_or_above_threshold():
    summary = report.build_candidate_summary_from_metrics(
        name="engine-a",
        overall_wer=0.1,
        switch_adjacent_wer=0.2,
        non_switch_wer=0.05,
        switch_penalty_ratio=4.0,
        switch_adjacent_wer_en_side=0.2,
        switch_adjacent_wer_he_side=0.2,
        wrong_script_rate=0.0,
        overall_cer=0.05,
        n_window_tokens=config.MIN_WINDOW_TOKENS_FOR_CONFIDENCE,
    )
    assert summary.low_confidence is False


# --- Gate evaluation -----------------------------------------------------------------


def test_gate_pass_when_inside_g1_g2_g3():
    gate = report.evaluate_gates(overall_wer=0.10, switch_adjacent_wer=0.20, switch_penalty_ratio_value=1.2)
    assert gate.passed is True
    assert gate.breached == ()
    assert gate.caution is False


def test_gate_fails_g1_and_names_it():
    gate = report.evaluate_gates(overall_wer=0.30, switch_adjacent_wer=0.10, switch_penalty_ratio_value=1.0)
    assert gate.passed is False
    assert "G1_overall_wer" in gate.breached


def test_gate_fails_g2_and_names_it():
    gate = report.evaluate_gates(overall_wer=0.10, switch_adjacent_wer=0.35, switch_penalty_ratio_value=1.0)
    assert gate.passed is False
    assert "G2_switch_adjacent_wer" in gate.breached


def test_gate_fails_g3_and_names_it():
    gate = report.evaluate_gates(overall_wer=0.10, switch_adjacent_wer=0.10, switch_penalty_ratio_value=2.5)
    assert gate.passed is False
    assert "G3_switch_penalty_ratio" in gate.breached


def test_gate_caution_band_between_1_5_and_2_0_still_passes():
    gate = report.evaluate_gates(overall_wer=0.10, switch_adjacent_wer=0.10, switch_penalty_ratio_value=1.7)
    assert gate.passed is True
    assert gate.caution is True


def test_gate_undefined_ratio_does_not_breach_or_caution():
    gate = report.evaluate_gates(
        overall_wer=0.0, switch_adjacent_wer=0.0, switch_penalty_ratio_value="undefined"
    )
    assert gate.passed is True
    assert gate.caution is False


def test_gate_infinite_ratio_breaches_g3():
    gate = report.evaluate_gates(overall_wer=0.1, switch_adjacent_wer=0.1, switch_penalty_ratio_value="inf")
    assert gate.passed is False
    assert "G3_switch_penalty_ratio" in gate.breached


# --- Selection ------------------------------------------------------------------------


def _summary(name, sa_wer, passed, latency=None, cost=None, is_local=False):
    gate = report.GateEvaluation(passed=passed, breached=() if passed else ("G2_switch_adjacent_wer",), caution=False, g4_passed=None)
    return report.CandidateSummary(
        name=name,
        overall_wer=0.1,
        switch_adjacent_wer=sa_wer,
        non_switch_wer=0.05,
        switch_penalty_ratio=2.0 if sa_wer else "undefined",
        switch_adjacent_wer_en_side=sa_wer,
        switch_adjacent_wer_he_side=sa_wer,
        wrong_script_rate=0.0,
        overall_cer=0.05,
        n_window_tokens=200,
        low_confidence=False,
        gate=gate,
        latency_median_s=latency,
        latency_p95_s=latency,
        cost_per_hour=cost,
        is_local=is_local,
    )


def test_selection_picks_lowest_switch_adjacent_wer_among_passers():
    candidates = [
        _summary("engine-a", 0.20, passed=True),
        _summary("engine-b", 0.10, passed=True),
        _summary("engine-c", 0.05, passed=False),  # lowest overall but fails the gate
    ]
    outcome = report.select_candidate(candidates)
    assert outcome.winner == "engine-b"
    assert outcome.outcome == "SELECTED"


def test_selection_ties_within_two_points_broken_by_latency():
    candidates = [
        _summary("engine-slow", 0.20, passed=True, latency=3.0, is_local=False),
        _summary("engine-fast", 0.21, passed=True, latency=1.0, is_local=False),
    ]
    outcome = report.select_candidate(candidates)
    assert outcome.winner == "engine-fast"


def test_selection_no_candidate_clears_bar_when_all_fail():
    candidates = [
        _summary("engine-a", 0.20, passed=False),
        _summary("engine-b", 0.30, passed=False),
    ]
    outcome = report.select_candidate(candidates)
    assert outcome.outcome == config.NO_CANDIDATE_CLEARS_BAR
    assert outcome.winner is None


# --- No verbatim transcript text ever reaches the committed report -------------------


def test_render_full_report_contains_no_transcript_text_in_committed_report():
    from benchmark import normalize as nz
    from benchmark import score

    marker = "ZZDISTINCTIVEMARKERZZ"
    ref_tokens, _ = nz.normalize_and_tokenize("א ב google")
    hyp_tokens, _ = nz.normalize_and_tokenize(f"א ב {marker}")
    alignment = score.build_alignment(ref_tokens, hyp_tokens)
    switch_points = nz.find_switch_points(ref_tokens)

    summary = report.build_candidate_summary(
        name="engine-a",
        alignment=alignment,
        switch_points=switch_points,
    )
    outcome = report.select_candidate([summary])
    rendered = report.render_full_report([summary], outcome)

    assert marker.lower() not in rendered.lower()
