"""Test-only playback helper: plays a caller-supplied mono waveform out of the
default render device, on its own thread, so loopback capture can be verified
against known audio with no human present and no call running. Not part of
the application -- imported only by tests under `app/tests/`."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import pyaudiowpatch as pyaudio


def make_tone(
    freq_hz: float, duration_s: float, *, sample_rate: int, amplitude: float = 0.5
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s))
    return (amplitude * np.sin(2 * np.pi * freq_hz * t / sample_rate)).astype(np.float64)


class TonePlayer:
    """Plays `samples` (float64, range [-1, 1], mono) out of the default
    render device on a background thread."""

    def __init__(self, samples: np.ndarray, *, sample_rate: int) -> None:
        self._samples = samples
        self._sample_rate = sample_rate
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        pcm16 = (np.clip(self._samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._sample_rate,
                output=True,
            )
            try:
                stream.write(pcm16)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            pa.terminate()


def play_tone(freq_hz: float, duration_s: float, *, sample_rate: int = 48000) -> TonePlayer:
    samples = make_tone(freq_hz, duration_s, sample_rate=sample_rate)
    player = TonePlayer(samples, sample_rate=sample_rate)
    player.start()
    return player
