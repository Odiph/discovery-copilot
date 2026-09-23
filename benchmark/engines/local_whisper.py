"""Chunked local faster-whisper runner (candidates 3 and 4: local-finetune / local-base).

Each VAD chunk is transcribed as an INDEPENDENT decode — language auto-detection
left on, previous-text conditioning switched off — which is the whole point of
BENCH-03a: faster-whisper's own long-form VAD filter still performs one
conditioned decode across the file, which is exactly the batch behaviour the
requirement forbids. See `transcribe_batch_diagnostic` for the excluded-from-gates
diagnostic comparison added in Task 4.

Latency is measured on a virtual real-time clock: chunk k's audio ends at its VAD
end time; decoding starts at the later of that time and the previous chunk's
completion; the recorded latency is completion minus audio-end. An engine that
cannot keep up therefore shows queueing rather than a flattering per-call number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from benchmark import config as config_mod
from benchmark import audio as audio_mod
from benchmark.engines.base import EngineResult, Segment

# Resolved AT IMPLEMENTATION TIME (01-01-PLAN.md Task 4) against the Hugging
# Face Hub, not assumed. Both share the SAME base architecture
# (large-v3-turbo) -- ivrit.ai publishes a large-v3-turbo finetune, so per the
# plan's CRITICAL CONTROL VALIDITY rule neither candidate drops to plain
# large-v3. Revisions are pinned to an exact commit SHA so a later upload to
# either repo can never silently change what was measured.
IVRIT_AI_FINETUNE_REPO = "ivrit-ai/whisper-large-v3-turbo-ct2"
IVRIT_AI_FINETUNE_REVISION = "72ad623a37947395efcc3933132353790e5a12f5"
BASE_CONTROL_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
BASE_CONTROL_REVISION = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8"
SHARED_BASE_ARCHITECTURE = "large-v3-turbo"


class FrozenComparisonViolation(Exception):
    """Raised when the local finetune/base pair diverges on anything other
    than model id (+ revision) — the BENCH-03b control is invalid otherwise."""


_FROZEN_FIELDS = (
    "device",
    "compute_type",
    "beam_size",
    "temperature",
    "chunk_min_s",
    "chunk_max_s",
    "vad_options",
    "condition_on_previous_text",
    "base_model_architecture",
)


def assert_frozen_comparison(a: "LocalWhisperConfig", b: "LocalWhisperConfig") -> None:
    """Raises FrozenComparisonViolation if `a` and `b` differ in device, compute
    type, beam size, temperature, chunk length, VAD settings, previous-text
    conditioning, or base model architecture. Passes when they differ only in
    model id/revision (config.FROZEN_LOCAL_COMPARISON is the canonical shared
    settings object both configs should be built from)."""
    mismatches = [field_name for field_name in _FROZEN_FIELDS if getattr(a, field_name) != getattr(b, field_name)]
    if mismatches:
        raise FrozenComparisonViolation(
            f"Frozen local comparison violated: {', '.join(mismatches)} differ between "
            "the finetune and base configs. BENCH-03b is only valid while everything but "
            "model id is identical between the two local candidates."
        )


@dataclass(frozen=True)
class LocalWhisperConfig:
    model_id: str
    revision: Optional[str] = None
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    temperature: float = 0.0
    chunk_min_s: float = 1.0
    chunk_max_s: float = 3.0
    vad_options: dict = field(default_factory=dict)
    condition_on_previous_text: bool = False
    base_model_architecture: str = "unknown"


class LocalWhisperEngine:
    """Satisfies the Engine protocol. Construction never touches the network or a
    GPU; the model is loaded lazily on first `transcribe()` call (which does)."""

    def __init__(self, name: str, config: LocalWhisperConfig):
        self.name = name
        self._config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._config.model_id,
                device=self._config.device,
                compute_type=self._config.compute_type,
                revision=self._config.revision,
            )
        return self._model

    def transcribe(
        self, audio_path: str, slice_: Optional[tuple[float, float]] = None
    ) -> EngineResult:
        cfg = self._config
        start_s, end_s = slice_ if slice_ else (None, None)
        samples = audio_mod.load_audio(
            audio_path, sampling_rate=audio_mod.DEFAULT_SAMPLE_RATE, start_s=start_s, end_s=end_s
        )
        spans = audio_mod.get_speech_segments(
            samples, sampling_rate=audio_mod.DEFAULT_SAMPLE_RATE, **cfg.vad_options
        )
        chunks = audio_mod.group_into_chunks(
            spans,
            sampling_rate=audio_mod.DEFAULT_SAMPLE_RATE,
            min_chunk_s=cfg.chunk_min_s,
            max_chunk_s=cfg.chunk_max_s,
        )
        model = self._ensure_model()

        sr = audio_mod.DEFAULT_SAMPLE_RATE
        queue_ready = 0.0
        emitted: list[Segment] = []
        hyp_parts: list[str] = []

        for chunk in chunks:
            chunk_audio = samples[int(chunk.start_s * sr) : int(chunk.end_s * sr)]
            decode_start = max(chunk.end_s, queue_ready)

            t0 = time.perf_counter()
            segments_iter, info = model.transcribe(
                chunk_audio,
                beam_size=cfg.beam_size,
                temperature=cfg.temperature,
                condition_on_previous_text=cfg.condition_on_previous_text,
                vad_filter=False,
                language=None,  # per-chunk language auto-detection, left on
            )
            decoded_segments = list(segments_iter)
            decode_duration = time.perf_counter() - t0
            decode_complete = decode_start + decode_duration

            text = "".join(s.text for s in decoded_segments).strip()
            if text:
                hyp_parts.append(text)

            emitted.append(
                Segment(
                    text=text,
                    audio_end_s=chunk.end_s,
                    ready_wall_s=decode_complete,
                    detected_language=getattr(info, "language", None),
                )
            )
            queue_ready = decode_complete

        return EngineResult(
            hypothesis_text=" ".join(hyp_parts),
            segments=tuple(emitted),
            metadata={
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "device": cfg.device,
                "compute_type": cfg.compute_type,
                "beam_size": cfg.beam_size,
                "temperature": cfg.temperature,
                "chunk_min_s": cfg.chunk_min_s,
                "chunk_max_s": cfg.chunk_max_s,
                "condition_on_previous_text": cfg.condition_on_previous_text,
                "base_model_architecture": cfg.base_model_architecture,
                "n_chunks": len(chunks),
                "decode_mode": "independent_vad_chunks",
            },
        )

    def transcribe_batch_diagnostic(
        self, audio_path: str, slice_: Optional[tuple[float, float]] = None
    ) -> EngineResult:
        """One long-form conditioned decode over the whole file, instead of VAD
        chunks. DIAGNOSTIC ONLY — turns research §7's flagged assumption (whether
        short-chunking actually mitigates language lock) into a measurement.
        Explicitly excluded from gate evaluation and candidate selection; the
        chunked configuration above remains the scored one (BENCH-03a)."""
        cfg = self._config
        start_s, end_s = slice_ if slice_ else (None, None)
        samples = audio_mod.load_audio(
            audio_path, sampling_rate=audio_mod.DEFAULT_SAMPLE_RATE, start_s=start_s, end_s=end_s
        )
        model = self._ensure_model()

        t0 = time.perf_counter()
        segments_iter, info = model.transcribe(
            samples,
            beam_size=cfg.beam_size,
            temperature=cfg.temperature,
            condition_on_previous_text=True,
            vad_filter=True,
            language=None,
        )
        decoded_segments = list(segments_iter)
        decode_duration = time.perf_counter() - t0

        text = " ".join(s.text.strip() for s in decoded_segments if s.text.strip())
        audio_duration_s = len(samples) / audio_mod.DEFAULT_SAMPLE_RATE
        emitted = tuple(
            Segment(
                text=s.text.strip(),
                audio_end_s=s.end,
                ready_wall_s=audio_duration_s + decode_duration,
                detected_language=getattr(info, "language", None),
            )
            for s in decoded_segments
        )

        return EngineResult(
            hypothesis_text=text,
            segments=emitted,
            metadata={
                "model_id": cfg.model_id,
                "revision": cfg.revision,
                "device": cfg.device,
                "compute_type": cfg.compute_type,
                "decode_mode": "batch_long_form_diagnostic",
                "excluded_from_gates": True,
            },
        )


def make_tracer_engine(
    model_id: Optional[str] = None, device: Optional[str] = None
) -> LocalWhisperEngine:
    """CPU int8 local engine for the Task 2 tracer — small model for speed, no GPU
    and no credentials required."""
    resolved_model_id = model_id or "tiny"
    config = LocalWhisperConfig(
        model_id=resolved_model_id,
        device=device or "cpu",
        compute_type="int8",
        beam_size=1,
        temperature=0.0,
        base_model_architecture=resolved_model_id,
    )
    return LocalWhisperEngine(name="local", config=config)


def _frozen_config(
    *,
    model_id: str,
    revision: Optional[str],
    device: Optional[str] = None,
    compute_type_override: Optional[str] = None,
) -> LocalWhisperConfig:
    """Build a LocalWhisperConfig from config.FROZEN_LOCAL_COMPARISON, resolving
    the device/compute_type via the CUDA fallback ladder unless overridden."""
    frozen = config_mod.FROZEN_LOCAL_COMPARISON
    if device is not None:
        resolved_device, resolved_compute_type = device, (compute_type_override or frozen.compute_type)
    else:
        rung = config_mod.resolve_cuda_device()
        resolved_device, resolved_compute_type = rung["device"], rung["compute_type"]

    return LocalWhisperConfig(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        compute_type=resolved_compute_type,
        beam_size=frozen.beam_size,
        temperature=frozen.temperature,
        chunk_min_s=frozen.chunk_min_s,
        chunk_max_s=frozen.chunk_max_s,
        vad_options=dict(frozen.vad_options),
        condition_on_previous_text=frozen.condition_on_previous_text,
        base_model_architecture=frozen.base_model_architecture,
    )


def make_finetune_engine(
    model_id: Optional[str] = None, device: Optional[str] = None
) -> LocalWhisperEngine:
    """Candidate 4 (local-finetune): the ivrit.ai Hebrew finetune, pinned to an
    exact revision, built from the shared frozen decode/VAD settings."""
    resolved_model_id = model_id or IVRIT_AI_FINETUNE_REPO
    revision = IVRIT_AI_FINETUNE_REVISION if resolved_model_id == IVRIT_AI_FINETUNE_REPO else None
    cfg = _frozen_config(model_id=resolved_model_id, revision=revision, device=device)
    return LocalWhisperEngine(name="local-finetune", config=cfg)


def make_base_engine(
    model_id: Optional[str] = None, device: Optional[str] = None
) -> LocalWhisperEngine:
    """Candidate 3 (local-base): the matching base weights (same
    large-v3-turbo architecture as the finetune), pinned to an exact revision,
    built from the SAME shared frozen decode/VAD settings — only model id
    differs from make_finetune_engine, which is what makes this a valid
    control (BENCH-03b)."""
    resolved_model_id = model_id or BASE_CONTROL_REPO
    revision = BASE_CONTROL_REVISION if resolved_model_id == BASE_CONTROL_REPO else None
    cfg = _frozen_config(model_id=resolved_model_id, revision=revision, device=device)
    return LocalWhisperEngine(name="local-base", config=cfg)
