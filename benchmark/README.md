# STT benchmark harness (Phase 1 — Prove the Hebrew Assumption)

This package scores up to four speech-to-text candidates against a shared
normalizer and a shared alignment, and reports overall WER, switch-adjacent WER
(and its sensitivity/side-split variants), a wrong-script rate, and latency. It
exists to answer one question before any product architecture is committed:
does any candidate survive Hebrew/English code-switching well enough to drive a
live coaching transcript?

See `.planning/phases/01-prove-the-hebrew-assumption/01-01-PLAN.md` for the full
specification (S1-S8) this package implements.

## Synthetic audio is a harness fixture — NEVER a benchmark result

**`benchmark/fixtures/make_synthetic.py` renders a scripted, TTS-spoken bilingual
dialogue (`benchmark/fixtures/synthetic_script.json`) to `benchmark/data/fixtures/`.
This synthesized speech exists solely to exercise the pipeline end-to-end before
real audio exists. It is not representative of spontaneous interview speech —
it has none of the disfluency, prosody, accent variation, overlapping speech or
genuine mid-thought code-switching a real bilingual interview has. Any number this
fixture produces measures whether the harness works, never whether a candidate
is good enough for a real interview. Do not cite it, quote it, or paste it into a
decision record as evidence about which STT engine to pick — see
`01-02-PLAN.md` for the real recorded fixture that decision is made on.**

If `edge-tts` synthesis fails for any reason (no network, endpoint change), the
generator falls back to printing a labelled read-aloud sheet and instructs the
operator to record ~60 seconds of it themselves to the same path — the harness
never silently produces empty output.

## What this is not

No live audio capture, no UI, no FastAPI shell, no streaming service (S7). This
is a throwaway benchmark harness — kept at most as a Phase 2 test fixture — not
application architecture.

## Layout

- `normalize.py` — tokenization, normalization (S1), script-derived language
  tagging (S2), switch-point detection (S3). Pure, dependency-free; every number
  in this package rests on it.
- `audio.py` — decode/resample via `av`, VAD speech segmentation (Silero, bundled
  in faster-whisper), and the chunk driver that groups speech spans into 1-3
  second independent decode chunks.
- `engines/` — the `Engine` protocol (`base.py`) and the four candidates:
  `soniox.py`, `assemblyai.py`, and `local_whisper.py` (both local candidates).
  Nothing outside `engines/` and `run.py`'s dispatch table imports a concrete
  engine module.
- `score.py` — ONE global word-level alignment per (reference, hypothesis) pair;
  every metric is a slice of that one alignment (S5).
- `report.py` — S6 gate evaluation, deterministic selection, and the full
  multi-candidate report render (metrics/gates/verdicts only, never verbatim
  transcript text).
- `run.py` — the CLI: single-engine scoring, `--switch-audit`, and `--all`
  (every configured candidate over one audio+reference pair).
- `fixtures/` — the synthetic script and its generator.
- `data/` (gitignored, never committed) — audio, reference transcripts, switch
  audits and any other artifact containing verbatim transcript text.

## The four candidates

| Name | Vendor / model | Notes |
|---|---|---|
| `soniox` | Soniox `stt-rt-v5` streaming | Raw `websockets` against `wss://stt-rt.soniox.com/transcribe-websocket` — no vendor SDK is installed (S8); no language pre-declared |
| `assemblyai` | AssemblyAI Universal-3.5 Pro streaming | Via the official `assemblyai` SDK's `streaming.v3` client; multilingual detection on, no language pinned |
| `local-finetune` | ivrit.ai Hebrew finetune, `large-v3-turbo` architecture | Pinned revision; chunked independent VAD decode (BENCH-03a) |
| `local-base` | Matching base weights, SAME `large-v3-turbo` architecture | The BENCH-03b control — everything but model id is frozen identical to `local-finetune` (`config.FROZEN_LOCAL_COMPARISON`, guarded by `engines.local_whisper.assert_frozen_comparison`) |

**AssemblyAI bills wall-clock connection time.** The adapter closes the socket
promptly (`disconnect(terminate=True)`) as soon as streaming ends — leaving a
connection open after the audio finishes would bill idle time for nothing.

## The driver is not on the critical path

The maintainer's GPU driver (511.81, CUDA 11.6 era) predates the CUDA >=12.3 /
cuDNN 9 that current `ctranslate2` wants. `config.CUDA_FALLBACK_LADDER` /
`config.resolve_cuda_device()` walk four rungs — pinned-current GPU,
`ctranslate2==4.4.0` GPU, `ctranslate2==3.24.0` GPU, and CPU int8 — and always
resolve to a working device without raising. **WER is device-independent: a
driver problem can only degrade the local pair's LATENCY number (BENCH-05) and
can never block BENCH-03, BENCH-03a, BENCH-03b or BENCH-04.** When the local
pair runs on the CPU floor rung, the report labels its latency figures as a
floor rather than the shipping number — declining to update the driver (or
having no usable GPU at all) never gates this phase.

## Data handling

`benchmark/data/` is gitignored (`.gitignore`, asserted via `git check-ignore`).
Real customer audio, human reference transcripts and per-error detail output
must never reach git. The committed report (`report.py`) carries metrics, gate
outcomes and verdicts only — never verbatim transcript text.

## Quickstart

```
# Score one candidate against a reference:
python -m benchmark.run local-finetune --audio benchmark/data/fixtures/synthetic_cs.wav \
    --reference benchmark/data/fixtures/synthetic_cs.txt \
    --output benchmark/data/reports/local_finetune_tracer.json

# Run every configured candidate and render the full gated report:
python -m benchmark.run --all --audio <audio.wav> --reference <reference.txt> \
    --output benchmark/data/reports/full_report.md

# Eyeball the switch-point detection before trusting any number it produces:
python -m benchmark.run --switch-audit --reference <reference.txt>
```

Run the test suite (no API keys, no GPU required):

```
python -m pytest benchmark/tests -q -m "not live"
```

`soniox` and `assemblyai` also have a `@pytest.mark.live` smoke test each,
skipped automatically when `SONIOX_API_KEY` / `ASSEMBLYAI_API_KEY` are absent
— this plan never requires either to run its test suite.
