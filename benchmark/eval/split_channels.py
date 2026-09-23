"""Splits the interview into one WAV per SPEAKER, aligned on the same clock.

The pipeline separates speakers by device, so an eval of it has to hand each
side its own stream. Both files are the full length of the interview and carry
silence wherever the other person is talking, which is exactly what two real
microphones in a room would record if there were no bleed at all.

No bleed is deliberately unrealistic. It isolates what is being measured here
-- the coach, coverage detection and the writer -- from a separate problem the
project already knows it has.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.console import use_utf8_console  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "claudictive_interview.wav"
TRUTH = HERE / "claudictive_truth.json"


def main() -> None:
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    with wave.open(str(SOURCE), "rb") as source:
        rate, width, channels = source.getframerate(), source.getsampwidth(), source.getnchannels()
        frames = source.readframes(source.getnframes())

    frame_bytes = width * channels
    total = len(frames) // frame_bytes
    tracks = {"interviewer": bytearray(b"\x00" * len(frames)),
              "interviewee": bytearray(b"\x00" * len(frames))}

    for turn in truth:
        start = min(total, int(turn["start_s"] * rate))
        end = min(total, int(turn["end_s"] * rate))
        lo, hi = start * frame_bytes, end * frame_bytes
        tracks[turn["speaker"]][lo:hi] = frames[lo:hi]

    for speaker, data in tracks.items():
        path = HERE / f"channel_{speaker}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(width)
            out.setframerate(rate)
            out.writeframes(bytes(data))
        speaking = sum(t["end_s"] - t["start_s"] for t in truth if t["speaker"] == speaker)
        print(f"{path.name}: {total / rate / 60:.1f} min total, {speaking / 60:.1f} min of speech")


if __name__ == "__main__":
    use_utf8_console()
    main()

