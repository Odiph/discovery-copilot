"""Test-only PortAudio doubles: a `pyaudio`-module-shaped object and a
`PyAudio`-instance-shaped object whose `.open()` returns a stream that never
blocks and never touches real hardware. Shared by test_dual_stream.py,
test_windows_traps.py and test_precheck.py so every deterministic trap test
drives the SAME production `CaptureStream`/`StreamWorker` code the real
interview uses, per this plan's own instruction to test the real modules
rather than a parallel code path. Not part of the application -- imported
only by tests under `app/tests/`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class StubPyAudioModule:
    """Stands in for the `pyaudiowpatch` module itself -- only the constant
    `CaptureStream`/`open_capture_stream` actually reference (`paInt16`)."""

    paInt16 = 8  # arbitrary sentinel; the stub stream never inspects it


class StubStream:
    """One fake PortAudio stream. `available_frames` is how many frames
    `get_read_available()` reports; `amplitude` controls whether `read()`
    returns a 440 Hz tone (loud, tagged `real`) or all-zero silence (still
    tagged `real` by CaptureStream, since it IS "available" audio -- the
    zero/synthesized distinction CaptureStream itself makes is about whether
    the device had ANY samples available, not about their content).

    `on_read`/`on_get_available` let a test raise or reprogram behaviour
    mid-stream (e.g. simulate a device that goes idle, or one whose read
    starts failing)."""

    def __init__(
        self,
        *,
        frames_per_buffer: int,
        channels: int,
        amplitude: int = 0,
        available_frames: Optional[int] = None,
        on_read: Optional[Callable[[], None]] = None,
        on_get_available: Optional[Callable[[], None]] = None,
    ) -> None:
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.amplitude = amplitude
        self.available_frames = (
            frames_per_buffer if available_frames is None else available_frames
        )
        self._on_read = on_read
        self._on_get_available = on_get_available
        self.closed = False
        self.read_calls = 0
        self.written: list[bytes] = []

    def get_read_available(self) -> int:
        if self._on_get_available is not None:
            self._on_get_available()
        return self.available_frames

    def read(self, n: int, exception_on_overflow: bool = False) -> bytes:
        if self._on_read is not None:
            self._on_read()
        self.read_calls += 1
        count = n * self.channels
        if self.amplitude:
            t = np.arange(count)
            wave = (self.amplitude * np.sin(2 * np.pi * 440.0 * t / 16000.0)).astype(np.int16)
            return wave.tobytes()
        return b"\x00" * (count * 2)

    def write(self, data: bytes) -> None:
        self.written.append(data)

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class StubPyAudio:
    """Stands in for a `pyaudiowpatch.PyAudio()` instance. `stream_factory`,
    when given, is called with the same kwargs `.open()` receives so a test
    can hand back a purpose-built `StubStream` (e.g. one that raises) instead
    of the default always-available one."""

    def __init__(
        self,
        *,
        amplitude: int = 0,
        stream_factory: Optional[Callable[..., "StubStream"]] = None,
        open_error: Optional[Exception] = None,
    ) -> None:
        self._amplitude = amplitude
        self._stream_factory = stream_factory
        self._open_error = open_error
        self.opened_streams: list[StubStream] = []
        self.open_calls: list[dict] = []

    def open(self, **kwargs) -> StubStream:
        self.open_calls.append(kwargs)
        if self._open_error is not None:
            raise self._open_error
        if self._stream_factory is not None:
            stream = self._stream_factory(**kwargs)
        else:
            stream = StubStream(
                frames_per_buffer=kwargs.get("frames_per_buffer", 160),
                channels=kwargs.get("channels", 1),
                amplitude=self._amplitude,
            )
        self.opened_streams.append(stream)
        return stream

    def terminate(self) -> None:
        pass
