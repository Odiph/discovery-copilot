"""Renders scored results to a JSON artifact plus a markdown table, evaluates
the S6 gates, applies the deterministic selection rule, and renders the full
multi-candidate report.

The committed report (`render_full_report`) carries metrics, gate outcomes and
verdicts only. `CandidateSummary` — the only object `render_full_report` reads —
has no field capable of holding verbatim transcript text; `build_candidate_summary`
consumes a `score.Alignment` (which does carry token text) but immediately
reduces it to numbers before returning. See phase_specification S5-S6 and the
T-01-05 threat entry.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

from benchmark import aliases, config, score
from benchmark.engines.base import Segment


@dataclass(frozen=True)
class ResultRow:
    engine_name: str
    overall_wer: Optional[float]
    switch_adjacent_wer: Optional[float]
    n_window_tokens: int
    latencies_s: tuple[float, ...]
    detected_languages: tuple[Optional[str], ...]
    metadata: dict = field(default_factory=dict)


def build_result_row(
    engine_name: str,
    overall_wer: Optional[float],
    switch_adjacent_wer: Optional[float],
    n_window_tokens: int,
    segments: tuple[Segment, ...],
    metadata: dict,
) -> ResultRow:
    latencies = tuple(seg.ready_wall_s - seg.audio_end_s for seg in segments)
    languages = tuple(seg.detected_language for seg in segments)
    return ResultRow(
        engine_name=engine_name,
        overall_wer=overall_wer,
        switch_adjacent_wer=switch_adjacent_wer,
        n_window_tokens=n_window_tokens,
        latencies_s=latencies,
        detected_languages=languages,
        metadata=metadata,
    )


def write_json_report(rows: list[ResultRow], out_path: str) -> None:
    payload = {"results": [asdict(r) for r in rows]}
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown_table(rows: list[ResultRow]) -> str:
    lines = [
        "| Engine | Overall WER | Switch-Adjacent WER (W=3) | N window tokens |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.engine_name} | {_fmt_pct(row.overall_wer)} | "
            f"{_fmt_pct(row.switch_adjacent_wer)} | {row.n_window_tokens} |"
        )

    # Script-equivalence disclosure. Every number above was computed with the
    # alias table in force, so a reader who cannot see it cannot audit the WER
    # -- an alias silently forgives a real substitution, and does so most often
    # at exactly the code-switch boundaries the second column measures.
    pairs = aliases.alias_pairs()
    if pairs:
        hits = sum(int(row.metadata.get("alias_hits", 0) or 0) for row in rows)
        rendered = ", ".join(f"`{alias}`=`{canon}`" for alias, canon in pairs)
        lines.extend([
            "",
            f"**Script-equivalence aliases applied ({len(pairs)} pairs, "
            f"{hits} token(s) canonicalised):** {rendered}",
            "",
            "These forms are scored as identical because they are the same spoken word "
            "written in different scripts. Language tags and switch points are unaffected "
            "-- see `benchmark/aliases.py`.",
        ])
    return "\n".join(lines)


# --- S6: gate evaluation --------------------------------------------------------------


@dataclass(frozen=True)
class GateEvaluation:
    passed: bool  # True iff G1 AND G2 AND G3 all pass
    breached: tuple[str, ...]
    caution: bool  # switch_penalty_ratio in the [1.5, 2.0] CAUTION band
    g4_passed: Optional[bool]  # None when no latency data was supplied


def _ratio_breaches_g3(ratio: Union[float, str, None]) -> tuple[bool, bool]:
    """Returns (breaches, caution). `score.UNDEFINED` (0/0) is neither a breach
    nor a caution — there is no switch-adjacent signal to penalize. `score.INFINITE`
    breaches (a positive switch-adjacent rate over zero non-switch rate is the
    worst possible penalty)."""
    if ratio == score.UNDEFINED or ratio is None:
        return False, False
    if ratio == score.INFINITE:
        return True, False
    if ratio > config.GATE_G3_SWITCH_PENALTY_RATIO_MAX:
        return True, False
    if ratio >= config.GATE_G3_SWITCH_PENALTY_CAUTION_MIN:
        return False, True
    return False, False


def evaluate_gates(
    overall_wer: Optional[float],
    switch_adjacent_wer: Optional[float],
    switch_penalty_ratio_value: Union[float, str, None],
    latency_median_s: Optional[float] = None,
    latency_p95_s: Optional[float] = None,
) -> GateEvaluation:
    """A result inside all of G1/G2/G3 is PASS; breaching any one is FAIL with
    the breached gate named. G4 (latency) is reported and disqualifying for live
    coaching only — it never affects `passed` (S6)."""
    breached: list[str] = []

    if overall_wer is None or overall_wer > config.GATE_G1_OVERALL_WER_MAX:
        breached.append("G1_overall_wer")

    if switch_adjacent_wer is None or switch_adjacent_wer > config.GATE_G2_SWITCH_ADJACENT_WER_MAX:
        breached.append("G2_switch_adjacent_wer")

    g3_breach, g3_caution = _ratio_breaches_g3(switch_penalty_ratio_value)
    if g3_breach:
        breached.append("G3_switch_penalty_ratio")

    g4_passed: Optional[bool] = None
    if latency_median_s is not None and latency_p95_s is not None:
        g4_passed = (
            latency_median_s <= config.GATE_G4_LATENCY_MEDIAN_MAX_S
            and latency_p95_s <= config.GATE_G4_LATENCY_P95_MAX_S
        )

    return GateEvaluation(
        passed=len(breached) == 0, breached=tuple(breached), caution=g3_caution, g4_passed=g4_passed
    )


# --- CandidateSummary: metrics + gate outcome, NEVER raw transcript text -------------


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    overall_wer: Optional[float]
    switch_adjacent_wer: Optional[float]
    non_switch_wer: Optional[float]
    switch_penalty_ratio: Union[float, str, None]
    switch_adjacent_wer_en_side: Optional[float]
    switch_adjacent_wer_he_side: Optional[float]
    wrong_script_rate: Optional[float]
    overall_cer: Optional[float]
    n_window_tokens: int
    low_confidence: bool
    gate: GateEvaluation
    latency_median_s: Optional[float] = None
    latency_p95_s: Optional[float] = None
    cost_per_hour: Optional[float] = None
    is_local: bool = False
    device_note: Optional[str] = None
    sensitivity: dict = field(default_factory=dict)  # {window: (wer, n)}


def build_candidate_summary_from_metrics(
    *,
    name: str,
    overall_wer: Optional[float],
    switch_adjacent_wer: Optional[float],
    non_switch_wer: Optional[float],
    switch_penalty_ratio: Union[float, str, None],
    switch_adjacent_wer_en_side: Optional[float],
    switch_adjacent_wer_he_side: Optional[float],
    wrong_script_rate: Optional[float],
    overall_cer: Optional[float],
    n_window_tokens: int,
    latency_median_s: Optional[float] = None,
    latency_p95_s: Optional[float] = None,
    cost_per_hour: Optional[float] = None,
    is_local: bool = False,
    device_note: Optional[str] = None,
    sensitivity: Optional[dict] = None,
) -> CandidateSummary:
    """Build a CandidateSummary from already-computed numbers (used by tests and
    by any caller that scored metrics itself)."""
    gate = evaluate_gates(overall_wer, switch_adjacent_wer, switch_penalty_ratio, latency_median_s, latency_p95_s)
    return CandidateSummary(
        name=name,
        overall_wer=overall_wer,
        switch_adjacent_wer=switch_adjacent_wer,
        non_switch_wer=non_switch_wer,
        switch_penalty_ratio=switch_penalty_ratio,
        switch_adjacent_wer_en_side=switch_adjacent_wer_en_side,
        switch_adjacent_wer_he_side=switch_adjacent_wer_he_side,
        wrong_script_rate=wrong_script_rate,
        overall_cer=overall_cer,
        n_window_tokens=n_window_tokens,
        low_confidence=n_window_tokens < config.MIN_WINDOW_TOKENS_FOR_CONFIDENCE,
        gate=gate,
        latency_median_s=latency_median_s,
        latency_p95_s=latency_p95_s,
        cost_per_hour=cost_per_hour,
        is_local=is_local,
        device_note=device_note,
        sensitivity=sensitivity or {},
    )


def build_candidate_summary(
    *,
    name: str,
    alignment: "score.Alignment",
    switch_points: list[tuple[int, int]],
    latencies_s: Optional[list[float]] = None,
    cost_per_hour: Optional[float] = None,
    is_local: bool = False,
    device_note: Optional[str] = None,
) -> CandidateSummary:
    """Scores `alignment` with every S5 metric and reduces it to a
    CandidateSummary. `alignment` carries real reference/hypothesis token text;
    nothing from it is copied onto the returned object — only numbers survive.
    """
    overall = score.overall_wer(alignment)
    sa_wer, n_window = score.switch_adjacent_wer(alignment, switch_points, window=config.WINDOW_HEADLINE)
    ns_wer, _ = score.non_switch_wer(alignment, switch_points, window=config.WINDOW_HEADLINE)
    ratio = score.switch_penalty_ratio(sa_wer, ns_wer)
    en_wer, _ = score.switch_adjacent_wer_en_side(alignment, switch_points, window=config.WINDOW_HEADLINE)
    he_wer, _ = score.switch_adjacent_wer_he_side(alignment, switch_points, window=config.WINDOW_HEADLINE)
    wrong_script, _ = score.wrong_script_rate(alignment, switch_points, window=config.WINDOW_HEADLINE)
    cer = score.overall_cer(list(alignment.reference), list(alignment.hypothesis))

    sensitivity = {}
    for w in config.ALL_WINDOWS:
        wer_w, n_w = score.switch_adjacent_wer(alignment, switch_points, window=w)
        sensitivity[w] = (wer_w, n_w)

    latency_median = statistics.median(latencies_s) if latencies_s else None
    latency_p95 = (
        statistics.quantiles(latencies_s, n=100)[94] if latencies_s and len(latencies_s) >= 2
        else (latencies_s[0] if latencies_s else None)
    )

    return build_candidate_summary_from_metrics(
        name=name,
        overall_wer=overall,
        switch_adjacent_wer=sa_wer,
        non_switch_wer=ns_wer,
        switch_penalty_ratio=ratio,
        switch_adjacent_wer_en_side=en_wer,
        switch_adjacent_wer_he_side=he_wer,
        wrong_script_rate=wrong_script,
        overall_cer=cer,
        n_window_tokens=n_window,
        latency_median_s=latency_median,
        latency_p95_s=latency_p95,
        cost_per_hour=cost_per_hour,
        is_local=is_local,
        device_note=device_note,
        sensitivity=sensitivity,
    )


# --- S6: deterministic selection --------------------------------------------------------


@dataclass(frozen=True)
class SelectionOutcome:
    winner: Optional[str]
    outcome: str  # "SELECTED" | config.NO_CANDIDATE_CLEARS_BAR
    tie_break_used: Optional[str] = None


def select_candidate(candidates: list[CandidateSummary]) -> SelectionOutcome:
    """S6 selection rule:
    1. Keep candidates passing G1, G2 and G3.
    2. Rank survivors by switch_adjacent_wer ascending.
    3. Break ties within 2 absolute percentage points by G4 latency, then by
       cost, then prefer local over cloud.
    4. If no candidate passes G1-G3, emit NO_CANDIDATE_CLEARS_BAR.
    """
    passers = [c for c in candidates if c.gate.passed and c.switch_adjacent_wer is not None]
    if not passers:
        return SelectionOutcome(winner=None, outcome=config.NO_CANDIDATE_CLEARS_BAR)

    best_wer = min(c.switch_adjacent_wer for c in passers)
    tie_band = config.SELECTION_TIE_BAND_ABS_PERCENTAGE_POINTS / 100.0
    tied = [c for c in passers if (c.switch_adjacent_wer - best_wer) <= tie_band]

    if len(tied) == 1:
        return SelectionOutcome(winner=tied[0].name, outcome="SELECTED")

    tie_break_used = "latency"
    tied_sorted = sorted(
        tied,
        key=lambda c: (
            c.latency_median_s if c.latency_median_s is not None else float("inf"),
            c.cost_per_hour if c.cost_per_hour is not None else float("inf"),
            0 if c.is_local else 1,
        ),
    )
    return SelectionOutcome(winner=tied_sorted[0].name, outcome="SELECTED", tie_break_used=tie_break_used)


# --- Full multi-candidate report render (committed — metrics/gates/verdicts only) ------


def _fmt_ratio(value: Union[float, str, None]) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:.2f}x"


def _gate_cell(gate: GateEvaluation) -> str:
    if gate.passed:
        return "PASS (CAUTION)" if gate.caution else "PASS"
    return f"FAIL ({', '.join(gate.breached)})"


def _ranking_by_window(candidates: list[CandidateSummary], window: int) -> list[str]:
    ranked = [
        (c.name, c.sensitivity.get(window, (None, 0))[0])
        for c in candidates
        if c.sensitivity.get(window, (None, 0))[0] is not None
    ]
    ranked.sort(key=lambda pair: pair[1])
    return [name for name, _ in ranked]


def render_full_report(candidates: list[CandidateSummary], selection: SelectionOutcome) -> str:
    lines: list[str] = ["# STT Benchmark Report", ""]

    lines.append("## Raw numbers (printed above the verdict)")
    lines.append("")
    lines.append(
        "| Engine | Overall WER | Switch-Adj WER (W=3) | N window | LOW-CONF | "
        "Non-switch WER | Switch penalty | EN-side | HE-side | Wrong-script | CER | Gate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in candidates:
        lines.append(
            f"| {c.name} | {_fmt_pct(c.overall_wer)} | {_fmt_pct(c.switch_adjacent_wer)} | "
            f"{c.n_window_tokens} | {'YES' if c.low_confidence else 'no'} | "
            f"{_fmt_pct(c.non_switch_wer)} | {_fmt_ratio(c.switch_penalty_ratio)} | "
            f"{_fmt_pct(c.switch_adjacent_wer_en_side)} | {_fmt_pct(c.switch_adjacent_wer_he_side)} | "
            f"{_fmt_pct(c.wrong_script_rate)} | {_fmt_pct(c.overall_cer)} | {_gate_cell(c.gate)} |"
        )
    lines.append("")

    lines.append("## Sensitivity (W = 2 / 3 / 5)")
    lines.append("")
    lines.append("| Engine | W=2 WER (n) | W=3 WER (n) | W=5 WER (n) |")
    lines.append("|---|---|---|---|")
    for c in candidates:
        cells = []
        for w in config.ALL_WINDOWS:
            wer_w, n_w = c.sensitivity.get(w, (None, 0))
            cells.append(f"{_fmt_pct(wer_w)} (n={n_w})")
        lines.append(f"| {c.name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lo, hi = config.WINDOW_SENSITIVITY
    ranking_lo = _ranking_by_window(candidates, lo)
    ranking_hi = _ranking_by_window(candidates, hi)
    if ranking_lo and ranking_hi and ranking_lo != ranking_hi:
        lines.append("")
        lines.append(
            f"**Ranking is NOT stable across W={lo} and W={hi}** — a conclusion that only "
            "survives at one window is not a conclusion. Treat the selection below as sensitive "
            "to the W=3 choice."
        )
    lines.append("")

    lines.append("## BENCH-03b — finetune vs. base (English-side switch-adjacent WER)")
    lines.append("")
    lines.append(
        "The finetune-versus-base verdict is read on the English-side switch-adjacent number, "
        "not on overall WER — a Hebrew finetune can post a better average while destroying "
        "exactly the English tokens the coach needs most."
    )
    lines.append("")
    lines.append("| Engine | EN-side switch-adjacent WER | Overall WER (context only, not decisive) |")
    lines.append("|---|---|---|")
    for c in candidates:
        lines.append(f"| {c.name} | {_fmt_pct(c.switch_adjacent_wer_en_side)} | {_fmt_pct(c.overall_wer)} |")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if selection.outcome == config.NO_CANDIDATE_CLEARS_BAR:
        lines.append(f"**{config.NO_CANDIDATE_CLEARS_BAR}** — no candidate passed G1, G2 and G3 together.")
    else:
        tie_note = f" (tie broken by {selection.tie_break_used})" if selection.tie_break_used else ""
        lines.append(f"**Selected: {selection.winner}**{tie_note}")
    lines.append("")

    lines.append("## User override")
    lines.append("")
    lines.append(
        "The G1-G4 thresholds above are a stated prior, not a law of nature. The user may move "
        "any gate or pick any candidate other than the one selected above; the only requirement "
        "is that the override and its reason are written down next to the numbers it overrides."
    )

    return "\n".join(lines)
