"""Task 2 done criteria: generate the synthetic fixture if absent, run
benchmark/run.py with a local engine on CPU int8 using a small model (tiny)
for speed, and assert the emitted JSON contains a non-null numeric overall_wer,
a non-null numeric switch_adjacent_wer, an n_window_tokens greater than zero, at
least one recorded per-chunk detected language, and at least one latency record.

Requires no GPU, no API key and no user-supplied audio — only network access to
download the tiny Whisper model and synthesize the fixture via edge-tts.

Task 4 note: Task 2 originally targeted a standalone "local" registry entry
(engines/local_whisper.make_tracer_engine). Task 4 replaced run.py's registry
with the four final candidates (soniox, assemblyai, local-finetune,
local-base) per the phase brief's fixed candidate list (B-01), which does not
include a bare "local" entry. This test now targets "local-finetune" with an
explicit `model_id="tiny"` override — the factory's model_id override exists
for exactly this purpose, so the tracer still exercises the full pipeline on a
small, fast model without downloading the real ivrit.ai weights.
"""

from __future__ import annotations

import json

import pytest

from benchmark import run as run_module
from benchmark.fixtures import make_synthetic


@pytest.fixture(scope="module")
def synthetic_fixture():
    wav_path, txt_path = make_synthetic.generate()
    return wav_path, txt_path


def test_tracer_end_to_end(tmp_path, synthetic_fixture):
    wav_path, txt_path = synthetic_fixture
    assert wav_path.exists(), "synthetic fixture wav was not generated"
    assert txt_path.exists(), "synthetic fixture reference text was not generated"

    output_path = tmp_path / "tracer_result.json"
    run_module.run_single(
        "local-finetune",
        str(wav_path),
        str(txt_path),
        model_id="tiny",
        device="cpu",
        output_path=str(output_path),
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    results = payload["results"]
    assert len(results) == 1
    result = results[0]

    assert result["overall_wer"] is not None
    assert isinstance(result["overall_wer"], (int, float))

    assert result["switch_adjacent_wer"] is not None
    assert isinstance(result["switch_adjacent_wer"], (int, float))

    assert result["n_window_tokens"] > 0

    assert len(result["detected_languages"]) >= 1
    assert any(lang is not None for lang in result["detected_languages"])

    assert len(result["latencies_s"]) >= 1
