"""Renders `synthetic_script.json` to a synthetic bilingual WAV fixture plus its
matching reference transcript.

HARNESS FIXTURE ONLY. Because the script text is authored by the executor, this
fixture's ground truth is known by construction — that is the entire reason for
using synthesis here. It is never representative of spontaneous interview speech
and MUST NEVER be reported as a benchmark result (see benchmark/README.md).

Voices are enumerated at runtime via `edge_tts.list_voices()` rather than
hardcoded, selecting one he-IL voice and one en-US voice. If edge-tts fails for
any reason (no network, endpoint change, etc.), a labelled fallback is printed:
the script is emitted as a read-aloud sheet and the operator is instructed to
record ~60 seconds of it themselves to the target path.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.console import use_utf8_console
from benchmark import audio as audio_mod

SAMPLE_RATE = audio_mod.DEFAULT_SAMPLE_RATE
SILENCE_MS = 400

_FIXTURES_DIR = Path(__file__).resolve().parent
_SCRIPT_PATH = _FIXTURES_DIR / "synthetic_script.json"
_DATA_DIR = _FIXTURES_DIR.parent / "data" / "fixtures"
WAV_PATH = _DATA_DIR / "synthetic_cs.wav"
TXT_PATH = _DATA_DIR / "synthetic_cs.txt"


def load_script() -> dict:
    return json.loads(_SCRIPT_PATH.read_text(encoding="utf-8"))


async def _pick_voices() -> dict[str, str]:
    import edge_tts

    voices = await edge_tts.list_voices()
    he_voice = next((v["ShortName"] for v in voices if v["Locale"] == "he-IL"), None)
    en_voice = next((v["ShortName"] for v in voices if v["Locale"] == "en-US"), None)
    if not he_voice or not en_voice:
        raise RuntimeError(
            f"Could not find both a he-IL and an en-US edge-tts voice "
            f"(found he-IL={he_voice!r}, en-US={en_voice!r})"
        )
    return {"he-IL": he_voice, "en-US": en_voice}


async def _synthesize_utterance(text: str, voice: str, out_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


async def _synthesize_all(script: dict, voices: dict[str, str], tmp_dir: Path) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for idx, utt in enumerate(script["utterances"]):
        voice = voices[utt["locale"]]
        mp3_path = tmp_dir / f"utt_{idx:03d}.mp3"
        await _synthesize_utterance(utt["text"], voice, mp3_path)
        arrays.append(audio_mod.load_audio(str(mp3_path), sampling_rate=SAMPLE_RATE))
    return arrays


def _concatenate_with_silence(
    arrays: list[np.ndarray], sample_rate: int, silence_ms: int
) -> np.ndarray:
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for idx, arr in enumerate(arrays):
        if idx > 0:
            pieces.append(silence)
        pieces.append(arr)
    return np.concatenate(pieces)


def _write_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _write_reference_text(script: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [utt["text"] for utt in script["utterances"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_readaloud_fallback(script: dict, wav_path: Path, txt_path: Path) -> Path:
    _write_reference_text(script, txt_path)
    sheet_path = wav_path.with_suffix(".readaloud.txt")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [utt["text"] for utt in script["utterances"]]
    sheet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[make_synthetic] FALLBACK: edge-tts synthesis failed.")
    print(f"[make_synthetic] Read-aloud sheet written to: {sheet_path}")
    print(
        f"[make_synthetic] Record ~60 seconds reading those lines aloud and save "
        f"the recording as: {wav_path}"
    )
    return sheet_path


def generate(force: bool = False) -> tuple[Path, Path]:
    """Ensure the synthetic fixture exists. Idempotent unless force=True."""
    if not force and WAV_PATH.exists() and TXT_PATH.exists():
        return WAV_PATH, TXT_PATH

    script = load_script()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            voices = asyncio.run(_pick_voices())
            arrays = asyncio.run(_synthesize_all(script, voices, tmp_dir))
        combined = _concatenate_with_silence(arrays, SAMPLE_RATE, SILENCE_MS)
        _write_wav(combined, SAMPLE_RATE, WAV_PATH)
        _write_reference_text(script, TXT_PATH)
        duration_s = len(combined) / SAMPLE_RATE
        print(f"[make_synthetic] Wrote {WAV_PATH} ({duration_s:.1f}s) and {TXT_PATH}")
    except Exception as exc:  # noqa: BLE001 - any synthesis failure triggers the fallback
        _write_readaloud_fallback(script, WAV_PATH, TXT_PATH)
        raise RuntimeError(f"edge-tts synthesis failed: {exc}") from exc

    return WAV_PATH, TXT_PATH


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic bilingual harness fixture")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the fixture exists")
    args = parser.parse_args(argv)
    generate(force=args.force)
    return 0


if __name__ == "__main__":
    use_utf8_console()
    raise SystemExit(main())
