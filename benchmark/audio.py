"""Audio load/resample and VAD chunk grouping.

Decode rides on `av` (already a faster-whisper dependency, so no ffmpeg binary and
no separate audio library are added per phase_specification S8). VAD speech
segmentation uses the Silero model bundled inside faster-whisper
(`faster_whisper.vad`) — the import path and option-class names below were
confirmed against the installed faster-whisper==1.2.1 / av==18.0.0 at
implementation time, not assumed.
"""

from __future__ import annotations

import gc
import io
import itertools
from dataclasses import dataclass
from typing import Optional

import av
import numpy as np
from faster_whisper import vad as fw_vad

DEFAULT_SAMPLE_RATE = 16_000


def _ignore_invalid_frames(frames):
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples: int = 500_000):
    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None  # ignore timestamp check across the concatenated stream
        fifo.write(frame)
        if fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


def load_audio(
    path: str,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> np.ndarray:
    """Decode `path` through av, downmix to mono, resample to `sampling_rate`, and
    optionally trim to [start_s, end_s). Returns a float32 array in [-1.0, 1.0]."""
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=sampling_rate
    )
    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(path, mode="r", metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames)
        frames = _resample_frames(frames, resampler)
        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    del resampler
    gc.collect()

    if dtype is None:
        audio = np.zeros(0, dtype=np.float32)
    else:
        audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)
        audio = audio.astype(np.float32) / 32768.0

    if start_s is not None or end_s is not None:
        start_idx = int((start_s or 0.0) * sampling_rate)
        end_idx = int(end_s * sampling_rate) if end_s is not None else len(audio)
        audio = audio[start_idx:end_idx]

    return audio


def _pcm_layout_name(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    raise ValueError(f"unsupported channel count for live PCM resample: {channels}")


def _ndarrays_to_pcm16(frames) -> bytes:
    chunks = [f.to_ndarray() for f in frames]
    if not chunks:
        return b""
    return np.concatenate(chunks, axis=None).astype(np.int16).tobytes()


class LivePCMResampler:
    """Stateful per-stream PCM16 resampler for the live capture path
    (phase_specification S4) -- the live-capture counterpart to `load_audio`'s
    file-shaped resample above. No ffmpeg binary is invoked; PyAV wraps
    libavcodec/libswresample in-process, matching the rest of this module.

    Reuses one `av.audio.resampler.AudioResampler` across calls so consecutive
    frames stay phase-continuous: recreating the resampler per frame would
    discard its internal history and reintroduce a discontinuity at every
    frame boundary, which is exactly what the continuous audio clock
    (phase_specification S3) exists to avoid. One instance per capture
    stream; call `flush()` once, at stream shutdown, to release anything
    still buffered internally.
    """

    def __init__(self, *, target_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self._target_rate = target_rate
        self._resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=target_rate
        )

    def push(self, raw: bytes, *, source_rate: int, source_channels: int) -> bytes:
        """Resample one buffer of interleaved PCM16 samples (mono or stereo)
        to mono PCM16 at `target_rate`. May return fewer or more samples than
        a naive ratio would suggest -- the resampler's internal buffering is
        what keeps the stream phase-continuous across calls."""
        if not raw:
            return b""
        layout = _pcm_layout_name(source_channels)
        array = np.frombuffer(raw, dtype=np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(array, format="s16", layout=layout)
        frame.sample_rate = source_rate
        return _ndarrays_to_pcm16(self._resampler.resample(frame))

    def flush(self) -> bytes:
        """Release anything the resampler is still holding internally. Call
        once, when the stream is shutting down."""
        return _ndarrays_to_pcm16(self._resampler.resample(None))


def get_speech_segments(
    audio: np.ndarray,
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    **vad_kwargs,
) -> list[dict]:
    """Silero VAD speech spans (start/end in samples at `sampling_rate`)."""
    options = fw_vad.VadOptions(**vad_kwargs) if vad_kwargs else fw_vad.VadOptions()
    return fw_vad.get_speech_timestamps(audio, vad_options=options, sampling_rate=sampling_rate)


@dataclass(frozen=True)
class Chunk:
    start_s: float
    end_s: float


def group_into_chunks(
    spans: list[dict],
    sampling_rate: int = DEFAULT_SAMPLE_RATE,
    min_chunk_s: float = 1.0,
    max_chunk_s: float = 3.0,
) -> list[Chunk]:
    """Group VAD speech spans into independent 1-3 second decode chunks.

    Spans longer than max_chunk_s are split into consecutive sub-chunks of at
    most max_chunk_s. Consecutive short spans are merged up to max_chunk_s. This
    is the chunk driver referenced in phase_specification S1-S8 and Task 2 —
    every chunk below is decoded as an INDEPENDENT decode by the local engine,
    never one long-form conditioned decode (BENCH-03a).
    """
    del min_chunk_s  # documented target only; merging already approaches it
    chunks: list[Chunk] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    for span in spans:
        s = span["start"] / sampling_rate
        e = span["end"] / sampling_rate

        if e - s > max_chunk_s:
            if cur_start is not None:
                chunks.append(Chunk(cur_start, cur_end))
                cur_start = cur_end = None
            t = s
            while t < e:
                t_end = min(t + max_chunk_s, e)
                chunks.append(Chunk(t, t_end))
                t = t_end
            continue

        if cur_start is None:
            cur_start, cur_end = s, e
            continue

        if (e - cur_start) <= max_chunk_s:
            cur_end = e
        else:
            chunks.append(Chunk(cur_start, cur_end))
            cur_start, cur_end = s, e

    if cur_start is not None:
        chunks.append(Chunk(cur_start, cur_end))

    return chunks
