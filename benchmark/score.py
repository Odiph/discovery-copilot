"""Metric family, all sliced from ONE global word-level alignment per pair
(phase_specification S5).

`build_alignment()` computes that single alignment via jiwer's word-level
processing; every metric function below slices `Alignment.chunks` — never
recomputes an alignment — so all figures stay mutually consistent by
construction.

Accepts only pre-normalized token lists from `normalize.py` — this module never
calls `normalize_text()` or `tokenize()` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jiwer

from benchmark import aliases, config
from benchmark.normalize import LANG_ENGLISH, LANG_HEBREW, Token

# switch_penalty_ratio sentinels (JSON-safe strings, never a ZeroDivisionError):
# UNDEFINED is 0/0 (both rates are zero); INFINITE is a non-zero switch-adjacent
# rate over a zero non-switch rate.
UNDEFINED = "undefined"
INFINITE = "inf"


@dataclass(frozen=True)
class Alignment:
    """One global word-level alignment. `chunks` are jiwer AlignmentChunk objects
    (type in {'equal', 'substitute', 'delete', 'insert'}) with ref_start_idx /
    ref_end_idx / hyp_start_idx / hyp_end_idx into `reference` / `hypothesis`."""

    reference: tuple[Token, ...]
    hypothesis: tuple[Token, ...]
    chunks: tuple
    # How many tokens the script-equivalence table canonicalised before
    # alignment (benchmark/aliases.py). Defaults to 0 so an alignment built
    # without aliases is still constructible. The report PRINTS this: a WER
    # figure is not auditable unless a reader can see how much the alias table
    # contributed to it.
    alias_hits: int = 0


def build_alignment(reference: list[Token], hypothesis: list[Token]) -> Alignment:
    """Script-equivalence aliases (benchmark/aliases.py) are applied ONLY to the
    strings handed to the aligner. The `Token` objects stored on the returned
    `Alignment` keep their original text, script and language tag -- because
    `normalize.find_switch_points` reads those tags off `alignment.reference`,
    and canonicalising a Hebrew-script token to its Latin form would flip its
    language and MOVE the switch points `switch_adjacent_wer` measures around.
    The equivalence must change what counts as a MATCH, never where the
    boundaries are."""
    ref_canon = [aliases.canonical(t.text) for t in reference]
    hyp_canon = [aliases.canonical(t.text) for t in hypothesis]
    alias_hits = sum(1 for original, canon in zip(reference, ref_canon) if original.text != canon)
    alias_hits += sum(1 for original, canon in zip(hypothesis, hyp_canon) if original.text != canon)

    out = jiwer.process_words(" ".join(ref_canon), " ".join(hyp_canon))
    chunks = tuple(out.alignments[0]) if out.alignments else tuple()
    return Alignment(
        reference=tuple(reference),
        hypothesis=tuple(hypothesis),
        chunks=chunks,
        alias_hits=alias_hits,
    )


def switch_window_indices(
    reference: tuple[Token, ...],
    switch_points: list[tuple[int, int]],
    window: int,
) -> set[int]:
    """S4: the union of the W-token windows around every switch point, over
    reference tokens only. Overlapping windows merge into one union."""
    n = len(reference)
    indices: set[int] = set()
    for i, j in switch_points:
        indices.update(range(max(0, i - window + 1), i + 1))
        indices.update(range(j, min(n, j + window)))
    return indices


def _insertion_bounding_indices(ref_start_idx: int, n_ref: int) -> list[int]:
    """An insert chunk's ref_start_idx == ref_end_idx is the gap position `p`
    between reference token p-1 and reference token p. A gap at the very start or
    end of the reference has only one bounding token."""
    p = ref_start_idx
    bounding = []
    if p > 0:
        bounding.append(p - 1)
    if p < n_ref:
        bounding.append(p)
    return bounding


def _slice_error_counts(
    chunks: tuple,
    n_ref: int,
    window: Optional[set[int]],
    want_in_window: bool = True,
) -> tuple[int, int, int, int]:
    """Sum (substitutions, deletions, insertions, n_ref_tokens) restricted to the
    reference-token membership predicate. `window=None` means "no restriction" —
    used for the overall metric, which is computed over the whole alignment.

    Insertion attribution (S5): an insertion is attributed to the window when AT
    LEast ONE of the reference tokens bounding its gap is inside the window —
    "either side" means one or both. A gap with only one bounding token (start or
    end of the reference) is decided by that token alone.
    """
    s = d = ins = n = 0
    for chunk in chunks:
        if chunk.type == "insert":
            bounding = _insertion_bounding_indices(chunk.ref_start_idx, n_ref)
            if window is None:
                count_it = True
            elif not bounding:
                count_it = False
            else:
                in_window = any(b in window for b in bounding)
                count_it = in_window == want_in_window
            if count_it:
                ins += chunk.hyp_end_idx - chunk.hyp_start_idx
            continue

        for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
            if window is not None and (idx in window) != want_in_window:
                continue
            n += 1
            if chunk.type == "substitute":
                s += 1
            elif chunk.type == "delete":
                d += 1
            # 'equal' contributes to n only (a hit)
    return s, d, ins, n


def _wer_from_counts(s: int, d: int, ins: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    return (s + d + ins) / n


def overall_wer(alignment: Alignment) -> Optional[float]:
    s, d, ins, n = _slice_error_counts(alignment.chunks, len(alignment.reference), window=None)
    return _wer_from_counts(s, d, ins, n)


def switch_adjacent_wer(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int = config.WINDOW_HEADLINE,
) -> tuple[Optional[float], int]:
    """Returns (switch_adjacent_wer, n_window_tokens)."""
    window_indices = switch_window_indices(alignment.reference, switch_points, window)
    s, d, ins, n = _slice_error_counts(
        alignment.chunks, len(alignment.reference), window=window_indices, want_in_window=True
    )
    return _wer_from_counts(s, d, ins, n), n


def non_switch_wer(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int = config.WINDOW_HEADLINE,
) -> tuple[Optional[float], int]:
    """The same computation as switch_adjacent_wer, over reference tokens
    OUTSIDE every window. Returns (non_switch_wer, n_outside_tokens)."""
    window_indices = switch_window_indices(alignment.reference, switch_points, window)
    s, d, ins, n = _slice_error_counts(
        alignment.chunks, len(alignment.reference), window=window_indices, want_in_window=False
    )
    return _wer_from_counts(s, d, ins, n), n


def switch_penalty_ratio(
    switch_wer: Optional[float], non_switch_wer_value: Optional[float]
) -> Union[float, str]:
    """switch_adjacent_wer / non_switch_wer — compared against non-switch rather
    than overall because overall already contains the switch tokens and
    therefore understates the gap (S5). Division-by-zero-safe: 0/0 is
    `UNDEFINED`; a positive numerator over a zero denominator is `INFINITE`.
    Never raises ZeroDivisionError."""
    numerator_is_zero = switch_wer is None or switch_wer == 0
    denominator_is_zero = non_switch_wer_value is None or non_switch_wer_value == 0
    if denominator_is_zero:
        return UNDEFINED if numerator_is_zero else INFINITE
    if switch_wer is None:
        return UNDEFINED
    return switch_wer / non_switch_wer_value


def _side_window_indices(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int,
    side_lang: str,
) -> set[int]:
    window_indices = switch_window_indices(alignment.reference, switch_points, window)
    return {idx for idx in window_indices if alignment.reference[idx].lang == side_lang}


def switch_adjacent_wer_en_side(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int = config.WINDOW_HEADLINE,
) -> tuple[Optional[float], int]:
    """switch_adjacent_wer restricted to window tokens whose own tag is `en`.
    This is the BENCH-03b discriminator — read the finetune-vs-base verdict from
    this, never from overall WER (phase_specification S5)."""
    side_indices = _side_window_indices(alignment, switch_points, window, LANG_ENGLISH)
    s, d, ins, n = _slice_error_counts(
        alignment.chunks, len(alignment.reference), window=side_indices, want_in_window=True
    )
    return _wer_from_counts(s, d, ins, n), n


def switch_adjacent_wer_he_side(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int = config.WINDOW_HEADLINE,
) -> tuple[Optional[float], int]:
    """switch_adjacent_wer restricted to window tokens whose own tag is `he`."""
    side_indices = _side_window_indices(alignment, switch_points, window, LANG_HEBREW)
    s, d, ins, n = _slice_error_counts(
        alignment.chunks, len(alignment.reference), window=side_indices, want_in_window=True
    )
    return _wer_from_counts(s, d, ins, n), n


def wrong_script_rate(
    alignment: Alignment,
    switch_points: list[tuple[int, int]],
    window: int = config.WINDOW_HEADLINE,
) -> tuple[Optional[float], int]:
    """Share of window reference tokens whose aligned hypothesis token is in the
    opposite script (he/en) — the transliteration failure mode named in
    research §1. Not exclusive with the WER substitution count: the same
    mismatch counts once here AND once as a substitution.

    Only reference tokens with an aligned hypothesis token (equal or substitute
    chunks) and a he/en tag of their own are eligible — a deleted reference
    token has no hypothesis counterpart to compare, and a neutral reference
    token (digit/symbol) has no meaningful "wrong script"."""
    window_indices = switch_window_indices(alignment.reference, switch_points, window)
    eligible = 0
    wrong = 0
    for chunk in alignment.chunks:
        if chunk.type not in ("equal", "substitute"):
            continue
        span = chunk.ref_end_idx - chunk.ref_start_idx
        for offset in range(span):
            ref_idx = chunk.ref_start_idx + offset
            if ref_idx not in window_indices:
                continue
            ref_tag = alignment.reference[ref_idx].lang
            if ref_tag not in (LANG_HEBREW, LANG_ENGLISH):
                continue
            eligible += 1
            if chunk.type == "equal":
                continue  # exact text match, trivially same script
            hyp_idx = chunk.hyp_start_idx + offset
            hyp_tag = alignment.hypothesis[hyp_idx].lang
            if hyp_tag in (LANG_HEBREW, LANG_ENGLISH) and hyp_tag != ref_tag:
                wrong += 1
    if eligible == 0:
        return None, 0
    return wrong / eligible, eligible


def overall_cer(reference: list[Token], hypothesis: list[Token]) -> float:
    """Character error rate, reported as a secondary column because word-level
    WER punishes Hebrew clitic-attachment differences that carry no meaning
    loss (S5)."""
    ref_str = " ".join(t.text for t in reference)
    hyp_str = " ".join(t.text for t in hypothesis)
    return jiwer.cer(ref_str, hyp_str)
