"""Named constants for the STT benchmark harness.

Every threshold and window size used anywhere in this package is defined here, once,
so no number is ever spelled inline in score.py, report.py or run.py (phase spec S6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- S4: switch-adjacent window -------------------------------------------------
#
# W = 3 is the headline window (see phase_specification S4 for the four-point
# justification). W = 2 and W = 5 are computed over the same alignment and printed
# as sensitivity columns.
WINDOW_HEADLINE = 3
WINDOW_SENSITIVITY = (2, 5)
ALL_WINDOWS = (WINDOW_SENSITIVITY[0], WINDOW_HEADLINE, WINDOW_SENSITIVITY[1])

# --- S5: confidence floor ---------------------------------------------------------
#
# Below this many reference tokens inside the union of switch windows, every
# switch-adjacent figure is labelled LOW-CONFIDENCE in the report.
MIN_WINDOW_TOKENS_FOR_CONFIDENCE = 100

# --- Network timeouts (cloud engines) ---------------------------------------
# No network call may block indefinitely. In this benchmark a hung vendor merely
# stalls a run, but STT-03 puts these same engines behind the live interface, where
# a blocking read would freeze the coaching surface mid-interview — the exact
# failure SURF-06 ("degrades to transcript-only, never blank") and OPS-05
# ("failures are visible, never silent") forbid. Fix it here, once, at the source.
#
# Generous relative to the ~2s coaching budget on purpose: these bound
# PATHOLOGY (a dead socket), not latency. Latency is measured by G4, not enforced here.
NET_OPEN_TIMEOUT_S = 15.0   # connection establishment
NET_RECV_TIMEOUT_S = 30.0   # single blocking read — the one that could hang forever
NET_CLOSE_TIMEOUT_S = 10.0  # graceful close before abandoning the socket

# --- S6: the bar -------------------------------------------------------------------
#
# G1: overall WER must not exceed this to remain a usable LLM substrate.
GATE_G1_OVERALL_WER_MAX = 0.25

# G2: switch-adjacent WER must not exceed this — at least two of every three words
# around a switch must survive.
GATE_G2_SWITCH_ADJACENT_WER_MAX = 0.30

# G3: switch-penalty ratio. Between the caution floor and the fail ceiling is a
# CAUTION band that the report must call out explicitly, not silently pass.
GATE_G3_SWITCH_PENALTY_CAUTION_MIN = 1.5
GATE_G3_SWITCH_PENALTY_RATIO_MAX = 2.0

# G4: latency. Reported and disqualifying for live coaching only — a candidate that
# fails only G4 still passes for durable transcription (routes to the Phase 4
# fallback rather than disqualifying the transcript).
GATE_G4_LATENCY_MEDIAN_MAX_S = 2.0
GATE_G4_LATENCY_P95_MAX_S = 3.0

# --- Tie-breaking (S6 selection rule step 3) ---------------------------------------
#
# Candidates whose switch_adjacent_wer are within this many absolute percentage
# points of each other are considered tied and broken by latency, then cost, then
# a preference for local over cloud.
SELECTION_TIE_BAND_ABS_PERCENTAGE_POINTS = 2.0

NO_CANDIDATE_CLEARS_BAR = "NO_CANDIDATE_CLEARS_BAR"

# --- S1: normalization ranges -------------------------------------------------------
HEBREW_NIQQUD_RANGE = ("֑", "ׇ")
HEBREW_BLOCK_RANGE = ("֐", "׿")
HEBREW_GERESH = "׳"
HEBREW_GERSHAYIM = "״"


# --- Task 4: frozen local-pair comparison -------------------------------------------
#
# The finetune-vs-base control (BENCH-03b) is only valid while every decode and VAD
# setting is identical between the two local candidates and only the model id (and
# therefore the weights) differs. This object holds the shared settings; the guard
# function in engines/local_whisper.py refuses to run the pair if anything else
# diverges.
@dataclass(frozen=True)
class FrozenLocalComparisonSettings:
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    temperature: float = 0.0
    chunk_min_s: float = 1.0
    chunk_max_s: float = 3.0
    vad_options: dict = field(default_factory=dict)
    condition_on_previous_text: bool = False
    base_model_architecture: str = "large-v3-turbo"


FROZEN_LOCAL_COMPARISON = FrozenLocalComparisonSettings()

# --- Task 4: CUDA fallback ladder ---------------------------------------------------
#
# Tried in order; each rung logs which one it landed on. WER is device-independent —
# a driver problem can only degrade the local pair's LATENCY number (BENCH-05) and
# never blocks BENCH-03 / BENCH-03a / BENCH-03b / BENCH-04.
CUDA_FALLBACK_LADDER = (
    {"device": "cuda", "compute_type": "int8_float16", "ctranslate2_pin": None,
     "reason": "GPU with the pinned ctranslate2 (current CUDA/cuDNN)"},
    {"device": "cuda", "compute_type": "int8_float16", "ctranslate2_pin": "4.4.0",
     "reason": "GPU with ctranslate2==4.4.0 (CUDA 12 + cuDNN 8)"},
    {"device": "cuda", "compute_type": "int8", "ctranslate2_pin": "3.24.0",
     "reason": "GPU with ctranslate2==3.24.0 (CUDA 11 + cuDNN 8, matches an 511.81-era driver)"},
    {"device": "cpu", "compute_type": "int8", "ctranslate2_pin": None,
     "reason": "CPU int8 — the floor rung, always available"},
)


def resolve_cuda_device(gpu_available: Optional[bool] = None) -> dict:
    """Walk CUDA_FALLBACK_LADDER and resolve to a working (device, compute_type)
    pair plus a human-readable reason, WITHOUT raising. Never blocks the phase —
    a driver problem degrades only the local pair's latency number (BENCH-05).

    `gpu_available` overrides detection (used by tests); when None, detection
    is attempted via `ctranslate2.get_cuda_device_count()` and any import or
    runtime failure is treated as "no usable GPU" rather than propagated.
    """
    if gpu_available is None:
        try:
            import ctranslate2

            gpu_available = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            gpu_available = False

    if not gpu_available:
        return dict(CUDA_FALLBACK_LADDER[-1])

    # A GPU is present. A full per-rung CUDA/cuDNN compatibility probe is out
    # of scope for this harness (01-01-PLAN.md Task 4) -- the ladder's job is
    # to document the fallback policy and guarantee the CPU floor when no GPU
    # exists; the top rung is offered as the starting point for a live run.
    return dict(CUDA_FALLBACK_LADDER[0])
