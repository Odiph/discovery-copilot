"""The phase_specification S3 pacer and S5 capture thread. One bounded queue
per stream is the seam that makes a slow engine structurally unable to stall
capture (S5, T-02-07) -- this module never touches the network.

Every `frame_ms` period, exactly one frame is emitted: device samples when
`get_read_available()` reports at least a frame's worth, zero-fill otherwise,
tagged `real` or `synthesized`. It never asks the device for more samples
than it reports available, which is what makes a blocking read structurally
impossible rather than merely unlikely (the pyaudio_portaudio #20 gotcha this
mitigates -- see .planning/research/runtime-shell.md). This is the AUDIO-04
mitigation, the mechanism a streaming vendor needs to detect end-of-utterance
during natural pauses, and what keeps the audio clock and session wall clock
aligned for a meaningful latency number (phase_specification S3).

No STT vendor is ever named in this module's executable code (STT-03).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

from benchmark.audio import LivePCMResampler
from app.capture.devices import DeviceInfo, DeviceOpenError, open_capture_stream
from app.capture.levels import RecentLevel

logger = logging.getLogger(__name__)

FrameKind = Literal["real", "synthesized"]
_SAMPLE_WIDTH_BYTES = 2  # paInt16


@dataclass(frozen=True)
class CaptureFrame:
    """One paced frame (S3): resampled 16 kHz mono PCM16, whether it came
    from the device or was zero-filled. `seq` is a monotonic frame index that
    never skips -- it counts frame *periods* on the pacer clock, not bytes."""

    seq: int
    pcm16: bytes
    kind: FrameKind
    captured_monotonic: float


@dataclass
class CaptureHealth:
    """Truthful, live state a supervisor can poll without touching the queue.
    Nothing consumes this yet -- plan 02-02 turns it into visible events --
    but it must be accurate now (phase_specification S5)."""

    last_real_audio_monotonic: Optional[float] = None
    consecutive_synthesized_frames: int = 0
    dropped_frame_count: int = 0
    queue_depth: int = 0


class CaptureStream:
    """Owns one PortAudio input stream (mic or loopback). Runs its own thread;
    paces frames on a real wall clock (S3), resamples to 16 kHz mono PCM16
    (S4), and pushes onto a bounded queue (S5). Never blocks on anything but
    its own frame clock."""

    def __init__(
        self,
        pyaudio_module,
        pa,
        device: DeviceInfo,
        *,
        target_sample_rate: int,
        frame_ms: int,
        queue_max_seconds: float,
        label: str = "",
    ) -> None:
        self._pyaudio_module = pyaudio_module
        self._pa = pa
        self._device = device
        self._target_sample_rate = target_sample_rate
        self._frame_ms = frame_ms
        self._label = label
        queue_max_frames = max(1, int(queue_max_seconds * 1000 / frame_ms))
        self.queue: "queue.Queue[CaptureFrame]" = queue.Queue(maxsize=queue_max_frames)
        self.health = CaptureHealth()
        # The live level probe. Read by the status poller to tell "this side
        # has gone silent because it is broken" apart from "this side has gone
        # silent because nobody is talking" -- see `app/session/feed_watch.py`.
        self._recent_level = RecentLevel()
        self._resampler = LivePCMResampler(target_rate=target_sample_rate)
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._dead_event = threading.Event()
        self._stream_lock = threading.Lock()
        self._seq = 0
        self._native_frames_per_period = max(
            1, int(device.native_sample_rate * frame_ms / 1000)
        )

    @property
    def device(self) -> DeviceInfo:
        return self._device

    @property
    def is_dead(self) -> bool:
        return self._dead_event.is_set()

    def start(self) -> None:
        self._stream = open_capture_stream(
            self._pyaudio_module,
            self._pa,
            self._device,
            frames_per_buffer=self._native_frames_per_period,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self._label or self._device.index}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        tail = self._resampler.flush()
        if tail:
            self._enqueue(CaptureFrame(seq=self._seq, pcm16=tail, kind="real", captured_monotonic=time.monotonic()))
            self._seq += 1

    def reopen(self, new_device: DeviceInfo) -> None:
        """Close the current native stream and open a new one against
        `new_device`, reusing the SAME PyAudio instance passed at
        construction (phase_specification S13) -- never reconstructs the
        API, only the stream. Exactly one attempt; the caller
        (`StreamSupervisor`) decides what a failure means and records the
        outcome. The new stream is opened BEFORE the old one is torn down,
        so a failed reopen leaves the prior stream reference untouched.
        Thread-safe against the capture thread's own frame-emission loop via
        `_stream_lock`."""
        native_frames = max(1, int(new_device.native_sample_rate * self._frame_ms / 1000))
        new_stream = open_capture_stream(
            self._pyaudio_module, self._pa, new_device, frames_per_buffer=native_frames,
        )  # DeviceOpenError propagates to the caller on failure
        with self._stream_lock:
            old_stream = self._stream
            self._stream = new_stream
            self._device = new_device
            self._native_frames_per_period = native_frames
            self._resampler = LivePCMResampler(target_rate=self._target_sample_rate)
            self.health.consecutive_synthesized_frames = 0
        if old_stream is not None:
            try:
                old_stream.stop_stream()
                old_stream.close()
            except Exception:  # noqa: BLE001 -- best-effort close of the superseded stream
                pass

    def mark_dead(self) -> None:
        """Called by the supervisor after a failed reopen attempt: stops the
        capture thread's loop (it exits cleanly on its next iteration) so a
        stream against a device that is genuinely gone stops trying to read
        from it, without crashing the process or the other stream
        (T-02-13)."""
        self._dead_event.set()

    def _run(self) -> None:
        period_s = self._frame_ms / 1000.0
        next_deadline = time.monotonic() + period_s
        while not self._stop_event.is_set() and not self._dead_event.is_set():
            sleep_s = next_deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Behind schedule (system stall, slow resample). Re-anchor
                # rather than spin trying to catch up -- a caught-up clock
                # that lies about being on time is worse than a clock that
                # admits it slipped.
                next_deadline = time.monotonic()
            self._emit_one_frame()
            next_deadline += period_s
            # Waiting BEFORE every emission -- including frame 0 -- matters
            # for more than pacing: each frame's audio-duration accounting
            # (S4/S6, StreamWorker/StreamingSession) assumes cumulative bytes
            # sent tracks cumulative real wall time 1:1. Emitting frame 0
            # immediately (no wait) would credit it a full `period_s` of
            # "elapsed audio" while zero real time had passed, permanently
            # skewing every downstream `audio_end_s` ahead of `ready_wall_s`
            # by one frame period -- discovered live: it made every measured
            # latency clamp to zero (STT-04 requires a MEASURED latency).
            # Waiting first also gives the device a full period to accumulate
            # real samples before frame 0's `get_read_available()` check,
            # making frame 0 more likely to be `real` on a freshly-opened
            # stream instead of `synthesized`.

    def _emit_one_frame(self) -> None:
        # Held only for the duration of one read+resample -- `reopen()` (S13,
        # T1/T-02-13) swaps `_stream`/`_device`/`_resampler` under the same
        # lock, so a device-change reopen can never race a frame in flight.
        with self._stream_lock:
            stream = self._stream
            device = self._device
            resampler = self._resampler
            native_frames = self._native_frames_per_period

        available = stream.get_read_available()
        now = time.monotonic()
        if available >= native_frames:
            raw = stream.read(native_frames, exception_on_overflow=False)
            kind: FrameKind = "real"
            self.health.last_real_audio_monotonic = now
            self.health.consecutive_synthesized_frames = 0
            # Only REAL frames are measured. A synthesized frame is the
            # ABSENCE of audio, not a measurement of it, and pushing its zeros
            # would let a stream that has stopped delivering read as a quiet
            # room -- which is the one reading `feed_watch` treats as innocent.
            self._recent_level.push(raw, at_s=now)
        else:
            raw = b"\x00" * (native_frames * device.channels * _SAMPLE_WIDTH_BYTES)
            kind = "synthesized"
            self.health.consecutive_synthesized_frames += 1

        pcm16 = resampler.push(
            raw, source_rate=device.native_sample_rate, source_channels=device.channels
        )
        frame = CaptureFrame(seq=self._seq, pcm16=pcm16, kind=kind, captured_monotonic=now)
        self._seq += 1
        self._enqueue(frame)
        self.health.queue_depth = self.queue.qsize()

    def recent_peak_dbfs(self, *, at_s: "Optional[float]" = None) -> "Optional[float]":
        """The loudest real audio this stream has carried recently, or `None`
        if it has carried none to measure. Safe to call from the poller
        thread: the probe is append-and-trim over a deque, and a read that
        races a push sees either the frame or not, never a torn value."""
        return self._recent_level.peak_dbfs(
            at_s=time.monotonic() if at_s is None else at_s
        )

    def _enqueue(self, frame: CaptureFrame) -> None:
        try:
            self.queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        # Overflow: drop the oldest, count it, and record that it happened --
        # never a silent drop (OPS-05, S5, T-02-07).
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        self.health.dropped_frame_count += 1
        logger.warning(
            "capture queue overflow, dropped oldest frame speaker=%s total_dropped=%d",
            self._label, self.health.dropped_frame_count,
        )
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass  # a concurrent consumer drained it first; the next frame will succeed
