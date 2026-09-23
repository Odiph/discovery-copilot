"""Runs the REAL interview pipeline with two WAV files standing in for the two
capture devices.

**What is substituted, and nothing more.** Exactly two seams are replaced:
`construct_pyaudio`, which is the only call site in `app/` outside tests, and
`_resolve_devices`, so no driver is asked for a device that is not there.
Everything downstream is the shipping code -- `CaptureStream`, the resampler,
the Soniox session, the coach, the coverage detector, the critic, the session
writer. What is NOT exercised is PortAudio itself and loopback capture, and
that is the honest cost of getting two clean channels.

**Paced in real time, deliberately.** `get_read_available()` reports only what
the wall clock says should have arrived. Feeding the file as fast as it can be
read would hand the engine a 25-minute conversation in seconds, and every
latency number, every coach cycle and every queue-depth guard in the system
would be measuring a situation that cannot occur.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Derived, never a drive letter. `record-test-session.bat` was fixed for this
# exact bug: a hardcoded path is a silent failure on anyone else's clone.
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from app.capture.devices import DeviceInfo  # noqa: E402
import app.main as main_mod  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

RATE = 16000
CHANNELS = 1
WIDTH = 2

CHANNEL_FILES = {
    11: HERE / "channel_interviewee.wav",   # loopback side
    22: HERE / "channel_interviewer.wav",   # microphone side
}


class FileStream:
    """One capture device, backed by a WAV and a clock."""

    def __init__(self, path: Path, frames_per_buffer: int) -> None:
        with wave.open(str(path), "rb") as handle:
            self._frames = handle.readframes(handle.getnframes())
        self._position = 0            # in FRAMES, not bytes
        self._frames_per_buffer = frames_per_buffer
        self._started = time.monotonic()
        self.closed = False

    @property
    def _total_frames(self) -> int:
        return len(self._frames) // (WIDTH * CHANNELS)

    def get_read_available(self) -> int:
        elapsed = time.monotonic() - self._started
        due = min(int(elapsed * RATE), self._total_frames)
        return max(0, due - self._position)

    def read(self, n: int, exception_on_overflow: bool = False) -> bytes:
        lo = self._position * WIDTH * CHANNELS
        hi = lo + n * WIDTH * CHANNELS
        chunk = self._frames[lo:hi]
        self._position = min(self._total_frames, self._position + n)
        if len(chunk) < n * WIDTH * CHANNELS:
            # Past the end of the file: silence, never a short read. A short
            # read is a device fault to everything downstream, and the file
            # simply ending is not one.
            chunk = chunk + b"\x00" * (n * WIDTH * CHANNELS - len(chunk))
        return chunk

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FilePyAudio:
    """A `PyAudio()` instance whose devices are files."""

    def __init__(self) -> None:
        self.streams: "list[FileStream]" = []

    def open(self, **kwargs) -> FileStream:
        index = kwargs.get("input_device_index")
        path = CHANNEL_FILES.get(index)
        if path is None:
            raise RuntimeError(f"no file bound to device index {index!r}")
        stream = FileStream(path, kwargs.get("frames_per_buffer", 160))
        self.streams.append(stream)
        print(f"[harness] opened {path.name} as device {index}", flush=True)
        return stream

    def terminate(self) -> None:
        pass

    def get_sample_size(self, fmt) -> int:
        return WIDTH


LOOPBACK = DeviceInfo(
    index=11, name="Mock loopback (synthetic interviewee)",
    native_sample_rate=RATE, channels=CHANNELS, is_loopback=True, identity="mock-loopback",
)
MICROPHONE = DeviceInfo(
    index=22, name="Mock microphone (synthetic interviewer)",
    native_sample_rate=RATE, channels=CHANNELS, is_loopback=False, identity="mock-mic",
)
RENDER = DeviceInfo(
    index=33, name="Mock render", native_sample_rate=RATE, channels=CHANNELS,
    is_loopback=False, identity="mock-render",
)


def main() -> int:
    # The DEVICE MODULE is stubbed, not `_resolve_devices`. That function stays
    # real, so the mapping from "which sides were asked for" to "which devices
    # get opened" -- including its refusal to substitute one side for the
    # other -- is still the shipping code. The supervisor also re-resolves on
    # every poll to detect a device change, and it must get the same answer or
    # it will report a device that vanished mid-interview.
    main_mod.construct_pyaudio = FilePyAudio
    main_mod.resolve_default_loopback_device = lambda pa: LOOPBACK
    main_mod.resolve_default_input_device = lambda pa: MICROPHONE
    main_mod.resolve_default_render_device = lambda pa: RENDER
    seconds = sys.argv[1] if len(sys.argv) > 1 else "1560"
    return main_mod.main([
        "start", "--project", "demo-claudictive", "--no-serve", "--seconds", seconds,
    ])


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
