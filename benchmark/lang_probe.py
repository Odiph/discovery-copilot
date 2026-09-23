"""Language-lock probe — per-chunk detected language, WITHOUT ground truth.

Why this exists
---------------
`wrong_script_rate` and every WER figure need a hand-typed reference. The
per-chunk DETECTED LANGUAGE does not — it comes from the decoder itself. On
audio known to be bilingual, an engine that reports one language for every
chunk is language-locking, and that is visible with zero transcription work.

This is diagnostic only. It produces NO WER, NO gate result, and nothing it
emits may enter 01-BENCHMARK.md or any G1-G4 decision. It exists to tell the
user early whether a candidate is obviously broken, before they spend an hour
hand-transcribing for the real benchmark.

Usage:
    python -m benchmark.lang_probe --engine local-finetune --audio X.mp3 \
        --start 0 --end 60 --model-id tiny
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time

from benchmark.console import use_utf8_console
from benchmark import config as config_mod
from benchmark.engines.local_whisper import LocalWhisperConfig, LocalWhisperEngine

FINETUNE_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
BASE_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"

ENGINES = {"local-finetune": FINETUNE_MODEL, "local-base": BASE_MODEL}


def build_engine(name: str, model_id: str | None, device: str) -> LocalWhisperEngine:
    resolved = model_id or ENGINES[name]
    cfg = LocalWhisperConfig(
        model_id=resolved,
        device=device,
        compute_type="int8",
        beam_size=1,
        temperature=0.0,
        chunk_min_s=1.0,
        chunk_max_s=3.0,
        condition_on_previous_text=False,  # required by BENCH-03a
        base_model_architecture="large-v3-turbo",
    )
    return LocalWhisperEngine(name=name, config=cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=sorted(ENGINES))
    ap.add_argument("--audio", required=True)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--model-id", default=None, help="Override (e.g. 'tiny' for a cheap path test)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    engine = build_engine(args.engine, args.model_id, args.device)
    slice_ = (args.start, args.end) if args.start is not None or args.end is not None else None

    t0 = time.perf_counter()
    result = engine.transcribe(args.audio, slice_=slice_)
    elapsed = time.perf_counter() - t0

    langs = [s.detected_language for s in result.segments]
    counts = collections.Counter(l for l in langs if l)
    switches = sum(1 for a, b in zip(langs, langs[1:]) if a and b and a != b)

    report = {
        "PROBE_ONLY": "diagnostic — no WER, no gate result, never cite as a benchmark number",
        "engine": args.engine,
        "model_id": args.model_id or ENGINES[args.engine],
        "device": args.device,
        "audio": args.audio,
        "slice_s": [args.start, args.end],
        "wall_seconds": round(elapsed, 1),
        "n_chunks": len(result.segments),
        "language_counts": dict(counts),
        "distinct_languages": len(counts),
        "language_changes_between_chunks": switches,
        "language_sequence": langs,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    # Never print transcript text — only the language trace.
    print(text)
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
