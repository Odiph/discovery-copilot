"""CLI: engine name, audio path, reference path, optional model/device overrides,
output path.

`normalize.tokenize()` feeds BOTH the reference and every engine hypothesis — this
module is the only caller of `normalize.normalize_and_tokenize()`. Adding or
swapping an engine must never require touching score.py or report.py; only the
`_ENGINE_FACTORIES` registry below changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from benchmark.console import use_utf8_console
from benchmark import config, normalize as nz
from benchmark import report, score
from benchmark.engines import assemblyai as assemblyai_mod
from benchmark.engines import soniox as soniox_mod
from benchmark.engines.base import Engine
from benchmark.engines.local_whisper import make_base_engine, make_finetune_engine

_DEFAULT_SWITCH_AUDIT_DIR = Path("benchmark/data/reports")


def _make_soniox(model_id: Optional[str] = None, device: Optional[str] = None) -> Engine:
    del model_id, device  # not applicable to the cloud candidates
    return soniox_mod.SonioxEngine()


def _make_assemblyai(model_id: Optional[str] = None, device: Optional[str] = None) -> Engine:
    del model_id, device
    return assemblyai_mod.AssemblyAIEngine()


# The four candidates fixed in the phase brief (B-01) — exactly these, behind
# one identical Engine protocol. Adding or swapping an engine must never
# require touching score.py or report.py; only this registry changes.
_ENGINE_FACTORIES: dict[str, Callable[..., Engine]] = {
    "soniox": _make_soniox,
    "assemblyai": _make_assemblyai,
    "local-finetune": make_finetune_engine,
    "local-base": make_base_engine,
}


def available_engines() -> list[str]:
    return sorted(_ENGINE_FACTORIES.keys())


def get_engine(
    name: str, *, model_id: Optional[str] = None, device: Optional[str] = None
) -> Engine:
    factory = _ENGINE_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown engine '{name}'. Available: {available_engines()}")
    return factory(model_id=model_id, device=device)


def run_single(
    engine_name: str,
    audio_path: str,
    reference_path: str,
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    output_path: Optional[str] = None,
) -> report.ResultRow:
    reference_text = Path(reference_path).read_text(encoding="utf-8")
    reference_tokens, ref_ambiguous = nz.normalize_and_tokenize(reference_text)
    switch_points = nz.find_switch_points(reference_tokens)

    engine = get_engine(engine_name, model_id=model_id, device=device)
    result = engine.transcribe(audio_path)

    hypothesis_tokens, hyp_ambiguous = nz.normalize_and_tokenize(result.hypothesis_text)

    alignment = score.build_alignment(reference_tokens, hypothesis_tokens)
    overall = score.overall_wer(alignment)
    sa_wer, n_window_tokens = score.switch_adjacent_wer(alignment, switch_points)

    metadata = dict(result.metadata)
    metadata["ambiguous_tokens"] = sorted(set(ref_ambiguous) | set(hyp_ambiguous))
    metadata["n_switch_points"] = len(switch_points)
    # How many tokens the script-equivalence table canonicalised before scoring.
    # Recorded because a WER figure is not auditable unless a reader can see how
    # much of it the alias table paid for (benchmark/aliases.py).
    metadata["alias_hits"] = alignment.alias_hits

    row = report.build_result_row(
        engine_name=engine.name,
        overall_wer=overall,
        switch_adjacent_wer=sa_wer,
        n_window_tokens=n_window_tokens,
        segments=result.segments,
        metadata=metadata,
    )

    if output_path:
        report.write_json_report([row], output_path)

    return row


def run_all(
    audio_path: str, reference_path: str, *, output_path: Optional[str] = None
) -> tuple[list[report.CandidateSummary], report.SelectionOutcome, str]:
    """`--all`: run every configured candidate over one audio plus reference
    pair and hand the collected results to report.py — gate evaluation,
    selection and the full committed report (metrics/gates/verdicts only, no
    verbatim transcript text)."""
    reference_text = Path(reference_path).read_text(encoding="utf-8")
    reference_tokens, ref_ambiguous = nz.normalize_and_tokenize(reference_text)
    switch_points = nz.find_switch_points(reference_tokens)

    summaries: list[report.CandidateSummary] = []
    for name in available_engines():
        engine = get_engine(name)
        result = engine.transcribe(audio_path)
        hypothesis_tokens, _ = nz.normalize_and_tokenize(result.hypothesis_text)
        alignment = score.build_alignment(reference_tokens, hypothesis_tokens)
        latencies = [seg.ready_wall_s - seg.audio_end_s for seg in result.segments]
        summaries.append(
            report.build_candidate_summary(
                name=engine.name,
                alignment=alignment,
                switch_points=switch_points,
                latencies_s=latencies,
                is_local=name.startswith("local"),
            )
        )

    outcome = report.select_candidate(summaries)
    rendered = report.render_full_report(summaries, outcome)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")

    return summaries, outcome, rendered


def run_switch_audit(reference_path: str, output_path: Optional[str] = None) -> str:
    """`--switch-audit --reference <path>`: for each detected switch point, print
    the three reference tokens either side and the derived language tags, then
    the total switch count and the resulting n_window_tokens (W = headline).

    This is the artifact a human eyeballs in plan 01-02 to confirm the metric is
    measuring what they think it is — the cheapest possible defence against a
    silently wrong ground truth. It contains verbatim transcript text, so it is
    ALWAYS written under benchmark/data/, never inlined into the committed
    report.
    """
    reference_text = Path(reference_path).read_text(encoding="utf-8")
    reference_tokens, ambiguous = nz.normalize_and_tokenize(reference_text)
    switch_points = nz.find_switch_points(reference_tokens)
    window_indices = score.switch_window_indices(
        tuple(reference_tokens), switch_points, config.WINDOW_HEADLINE
    )

    lines = ["SWITCH AUDIT", "=" * 60, f"Reference: {reference_path}", ""]
    for idx, (i, j) in enumerate(switch_points, start=1):
        left = reference_tokens[max(0, i - 2) : i + 1]
        right = reference_tokens[j : min(len(reference_tokens), j + 3)]
        left_str = " ".join(f"{t.text}[{t.lang}]" for t in left)
        right_str = " ".join(f"{t.text}[{t.lang}]" for t in right)
        lines.append(f"#{idx:03d} switch at ref[{i}]->ref[{j}]: ...{left_str} || {right_str}...")

    lines.append("")
    lines.append(f"Total switch points: {len(switch_points)}")
    lines.append(f"n_window_tokens (W={config.WINDOW_HEADLINE}): {len(window_indices)}")
    if ambiguous:
        lines.append(f"Ambiguous (mixed-script) tokens for human review: {ambiguous}")

    output_text = "\n".join(lines)

    target_path = Path(output_path) if output_path else (
        _DEFAULT_SWITCH_AUDIT_DIR / f"switch_audit_{Path(reference_path).stem}.txt"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(output_text, encoding="utf-8")

    return output_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.run",
        description="STT benchmark harness — score one engine against a reference transcript.",
    )
    parser.add_argument("engine", nargs="?", help="Engine name (see --list-engines)")
    parser.add_argument("--audio", help="Path to the audio file")
    parser.add_argument("--reference", help="Path to the plain-text reference transcript")
    parser.add_argument("--output", help="Path to write the JSON report", default=None)
    parser.add_argument("--model-id", default=None, help="Override the local engine's model id")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument(
        "--list-engines", action="store_true", help="Print available engine names and exit"
    )
    parser.add_argument(
        "--switch-audit",
        action="store_true",
        help="Print/write a switch-point audit for --reference (verbatim text; always written under benchmark/data/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every configured candidate over --audio/--reference and render the full gated report",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_engines:
        for name in available_engines():
            print(name)
        return 0

    if args.switch_audit:
        if not args.reference:
            parser.error("--switch-audit requires --reference")
        print(run_switch_audit(args.reference, output_path=args.output))
        return 0

    if args.all:
        if not args.audio or not args.reference:
            parser.error("--all requires --audio and --reference")
        _, _, rendered = run_all(args.audio, args.reference, output_path=args.output)
        print(rendered)
        return 0

    if not args.engine or not args.audio or not args.reference:
        parser.error("engine, --audio and --reference are required unless --list-engines is given")

    row = run_single(
        args.engine,
        args.audio,
        args.reference,
        model_id=args.model_id,
        device=args.device,
        output_path=args.output,
    )
    print(report.render_markdown_table([row]))
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
