"""Turns the script into one continuous WAV, and writes the ground truth beside it.

The gap between turns is not cosmetic. Segmentation is the engine's own
decision, and a file with no pauses would be one unbroken utterance -- which
would test nothing about how this pipeline actually cuts a conversation up.
A slightly longer gap goes in where the speaker changes, because that is what
a real turn boundary sounds like.
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claudictive_script import TURNS  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

TURN_DIR = HERE / "turns"
OUT_WAV = HERE / "claudictive_interview.wav"
TRUTH = HERE / "claudictive_truth.json"

# A turn change gets a real beat. Two reasons, and the second is the one that
# matters for a coach eval: it is what a handover actually sounds like, and it
# is the window the coach has to produce a suggestion before the next thing is
# said. Compressed gaps would measure the coach against a conversation nobody
# has ever had.
GAP_SAME_SPEAKER_S = 0.5
GAP_TURN_CHANGE_S = 1.9


def synthesise() -> None:
    payload = [{"speaker": speaker, "text": text} for speaker, text in TURNS]
    turns_json = HERE / "turns.json"
    turns_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(HERE / "synth_interview.ps1"),
         "-TurnsJson", str(turns_json), "-OutDir", str(TURN_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise SystemExit(f"synthesis failed: {result.stderr}")


def concatenate() -> None:
    parts = sorted(TURN_DIR.glob("turn_*.wav"))
    if len(parts) != len(TURNS):
        raise SystemExit(f"expected {len(TURNS)} turn files, found {len(parts)}")

    with wave.open(str(parts[0]), "rb") as probe:
        params = probe.getparams()
    rate, width, channels = params.framerate, params.sampwidth, params.nchannels
    print(f"format: {rate} Hz, {width * 8}-bit, {channels}ch")

    truth = []
    elapsed = 0.0
    with wave.open(str(OUT_WAV), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        previous_speaker = None
        for path, (speaker, text) in zip(parts, TURNS):
            if previous_speaker is not None:
                gap = GAP_TURN_CHANGE_S if speaker != previous_speaker else GAP_SAME_SPEAKER_S
                out.writeframes(b"\x00" * int(rate * width * channels * gap))
                elapsed += gap
            with wave.open(str(path), "rb") as part:
                frames = part.readframes(part.getnframes())
                duration = part.getnframes() / part.getframerate()
            out.writeframes(frames)
            truth.append({
                "index": len(truth), "speaker": speaker, "text": text,
                "start_s": round(elapsed, 3), "end_s": round(elapsed + duration, 3),
            })
            elapsed += duration
            previous_speaker = speaker

    TRUTH.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    minutes, seconds = divmod(elapsed, 60)
    print(f"wrote {OUT_WAV.name}: {int(minutes)}m {seconds:04.1f}s over {len(truth)} turns")
    print(f"ground truth: {TRUTH.name}")


if __name__ == "__main__":
    use_utf8_console()
    if "--skip-synth" not in sys.argv:
        synthesise()
    concatenate()

