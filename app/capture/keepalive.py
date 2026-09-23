"""The keep-alive render stream: the second, independent layer under the
silent-loopback trap (phase_specification T1), on top of the §S3 pacer's own
non-blocking read. Writes near-silent PCM16 to the default WASAPI render
device on a slow cadence for as long as capture is running -- the same
workaround NAudio's own docs describe for the identical OS behaviour on a
loopback session with nothing rendering.

Uses the SAME `pa` instance passed in -- never constructs its own
(phase_specification S13). Its own failure is a recorded WARNING event,
never a raised exception: the pacer alone is already sufficient, so a
keep-alive that cannot open must not take capture down with it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.capture.devices import DeviceInfo

logger = logging.getLogger(__name__)

KEEPALIVE_WRITE_INTERVAL_S = 0.5


class KeepAliveStream:
    """Owns one PortAudio OUTPUT stream on the render device. Writes
    zero-filled PCM16 on a slow cadence until stopped."""

    def __init__(self, pyaudio_module, pa, render_device: DeviceInfo) -> None:
        self._pyaudio_module = pyaudio_module
        self._pa = pa
        self._device = render_device
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.opened = False
        self.warning: Optional[str] = None

    def start(self) -> None:
        # WASAPI shared-mode streams must match the device's own mix-format
        # sample rate (a fixed low rate like 8000 Hz raised "Invalid sample
        # rate" against the maintainer's real render device, measured
        # 2026-08-05) -- so the keep-alive opens at the device's OWN native
        # rate, exactly like every capture stream already does (S4).
        channels = max(1, min(self._device.channels, 2))
        rate = self._device.native_sample_rate
        try:
            self._stream = self._pa.open(
                format=self._pyaudio_module.paInt16,
                channels=channels,
                rate=rate,
                output=True,
                output_device_index=self._device.index,
            )
        except Exception as exc:  # noqa: BLE001 -- keep-alive is best-effort; must never take down capture
            self.warning = f"keep-alive render stream failed to open on '{self._device.name}': {exc}"
            logger.warning(self.warning)
            return
        self.opened = True
        self._channels = channels
        self._chunk_frames = max(1, int(rate * KEEPALIVE_WRITE_INTERVAL_S))
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="keepalive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # noqa: BLE001 -- best-effort close during shutdown
                pass
            self._stream = None

    def _run(self) -> None:
        silence = b"\x00" * (self._chunk_frames * self._channels * 2)
        while not self._stop_event.is_set():
            try:
                self._stream.write(silence)
            except Exception as exc:  # noqa: BLE001 -- a write failure is a warning, never fatal
                self.warning = f"keep-alive write failed: {exc}"
                logger.warning(self.warning)
                break
            self._stop_event.wait(KEEPALIVE_WRITE_INTERVAL_S)
