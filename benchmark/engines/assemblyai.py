"""AssemblyAI Universal-3.5 Pro streaming adapter (candidate 2) — via the
official `assemblyai` SDK (this is the one cloud vendor where the approved
dependency list, Task 1, includes the vendor's own package rather than raw
`websockets`). Multilingual detection is enabled and no language is pinned,
matching Soniox's "no pre-declared language" property so the two cloud
candidates are compared on equal footing.

AssemblyAI bills wall-clock connection time (see benchmark/README.md) — the
socket is closed promptly via `disconnect(terminate=True)` as soon as
streaming ends.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from benchmark import audio as audio_mod
from benchmark import config as config_mod
from benchmark.engines.base import EngineResult, Segment

DEFAULT_SPEECH_MODEL = "universal-3-5-pro"
FRAME_MS = 150  # within the plan's ~100-200ms real-time pacing window
SAMPLE_RATE = 16000


class AssemblyAIConfigError(Exception):
    """Raised for missing/invalid configuration. Message never includes key material."""


@dataclass(frozen=True)
class AssemblyAIConfig:
    speech_model: str = DEFAULT_SPEECH_MODEL
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = FRAME_MS


def turn_event_to_segment(
    event: Any, session_start_wall: float, now_wall: float
) -> Optional[Segment]:
    """Pure message handler: reduces one finalized AssemblyAI v3 `TurnEvent` to
    a Segment. Returns None for non-final turns or empty text. No network, no
    side effects — this is exactly what the contract test replays recorded
    payloads through (via the SDK's own offline `StreamingClient._parse_message`,
    then this function), with no live call.
    """
    if not event.end_of_turn:
        return None
    text = event.transcript.strip()
    if not text:
        return None
    audio_end_s = (event.words[-1].end / 1000.0) if event.words else 0.0
    return Segment(
        text=text,
        audio_end_s=audio_end_s,
        ready_wall_s=now_wall - session_start_wall,
        detected_language=event.language_code,
    )


class AssemblyAIEngine:
    """Satisfies the Engine protocol. Construction never touches the network
    or reads the API key — only `transcribe()` does."""

    name = "assemblyai"

    def __init__(self, config: Optional[AssemblyAIConfig] = None):
        self._config = config or AssemblyAIConfig()

    def _get_api_key(self) -> str:
        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise AssemblyAIConfigError(
                "ASSEMBLYAI_API_KEY is not set. Read from the environment only — see benchmark/README.md."
            )
        return api_key

    def transcribe(
        self, audio_path: str, slice_: Optional[tuple[float, float]] = None
    ) -> EngineResult:
        from assemblyai.streaming.v3 import (
            Encoding,
            SpeechModel,
            StreamingClient,
            StreamingClientOptions,
            StreamingEvents,
            StreamingParameters,
        )

        api_key = self._get_api_key()
        cfg = self._config
        start_s, end_s = slice_ if slice_ else (None, None)
        samples = audio_mod.load_audio(
            audio_path, sampling_rate=cfg.sample_rate, start_s=start_s, end_s=end_s
        )
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        frame_bytes = int(cfg.sample_rate * cfg.frame_ms / 1000) * 2  # 16-bit mono

        segments: list[Segment] = []
        hyp_parts: list[str] = []
        session_start_wall = time.perf_counter()

        def on_turn(client, event):
            seg = turn_event_to_segment(event, session_start_wall, time.perf_counter())
            if seg is not None:
                segments.append(seg)
                hyp_parts.append(seg.text)

        client = StreamingClient(
            StreamingClientOptions(
                api_key=api_key,
                # Bound connection establishment and teardown. The SDK defaults are
                # unbounded; behind the live interface that would hang the surface.
                connect_timeout=config_mod.NET_OPEN_TIMEOUT_S,
                terminate_timeout=config_mod.NET_CLOSE_TIMEOUT_S,
            )
        )
        client.on(StreamingEvents.Turn, on_turn)
        client.connect(
            StreamingParameters(
                sample_rate=cfg.sample_rate,
                encoding=Encoding.pcm_s16le,
                speech_model=SpeechModel(cfg.speech_model),
                language_detection=True,  # no pinned language -- multilingual/code-switching
                format_turns=True,
            )
        )

        for offset in range(0, len(pcm16), frame_bytes):
            chunk = pcm16[offset : offset + frame_bytes]
            client.stream(chunk)
            time.sleep(cfg.frame_ms / 1000.0)  # pace to wall clock

        client.disconnect(terminate=True)  # closes promptly -- billed on wall-clock connection time

        return EngineResult(
            hypothesis_text=" ".join(hyp_parts),
            segments=tuple(segments),
            metadata={"speech_model": cfg.speech_model, "decode_mode": "streaming"},
        )
