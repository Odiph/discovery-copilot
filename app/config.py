"""The single place any runtime knob is named (phase_specification S1). Nothing
else in `app/` reads `os.environ` for one of these values, and nothing here
reads an API key -- keys stay inside the engine adapters (S7).

Engine resolution order (finished in plan 02-01 Task 3, started here):
command-line override beats the environment, which beats the default. An
unknown engine name must fail before any device is opened or session
directory created -- see `app/stt/live.get_engine` and `app/main.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Safe at module level, unlike `SupervisorConfig` below: `app.coach.repetition`
# imports nothing but the standard library and `app/coach/__init__.py` is inert,
# so there is no cycle back into this module.
from app.coach import repetition

DEFAULT_ENGINE = "soniox"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 150  # matches the cloud adapters' own real-time pacing window
DEFAULT_QUEUE_MAX_SECONDS = 30.0
DEFAULT_IDLE_THRESHOLD_S = 5.0
DEFAULT_SESSIONS_ROOT = "sessions"

# --- plan 02-03: per-stream audio retention (phase_specification U3) -----------------
# Defaults to OFF -- a privacy decision, not a preference. The common case for
# `start` is a real customer call; retained audio is benchmark input ONLY,
# and the benchmark uploads to a vendor (AssemblyAI) never approved for real
# customer audio. Enabled per-run only via the explicit `--retain-audio` CLI
# flag in app/main.py -- there is deliberately no environment-variable
# override for this one, so it can never be left on silently in a shared
# .env file.
DEFAULT_RETAIN_AUDIO = False

# --- plan 02-02: supervisor / keep-alive / device-watch knobs ------------------------
DEFAULT_KEEPALIVE_ENABLED = True
DEFAULT_SUPERVISOR_POLL_INTERVAL_S = 0.5
DEFAULT_LAG_QUEUE_FRACTION = 0.5  # queue_depth / maxsize beyond this -> "lagging"
DEFAULT_DEVICE_POLL_INTERVAL_S = 2.0

# --- plan 02-02: pre-call check knobs -------------------------------------------------
DEFAULT_PRECHECK_SECONDS = 3.0
DEFAULT_PRECHECK_REAL_FRACTION_MIN = 0.5  # share of frames tagged `real` required to PASS liveness
# Floor for "carries real audio", not just "opened", applied to the LOUDEST 500ms of the
# window (app/capture/levels.py) -- the old whole-window int16 RMS scored a longer window
# lower for the same voice and failed a working microphone. -36 dBFS is the top of the
# "too quiet" band: below it, an engine is being fed something a human would ask to repeat.
DEFAULT_PRECHECK_LEVEL_DBFS_MIN = -36.0
DEFAULT_PRECHECK_ENGINE_TIMEOUT_S = 20.0  # bounded deadline for the engine-reachability row
# The self-test `start` runs immediately before an interview. Shorter than
# `precheck`: this one is paid on EVERY recording, with the operator already
# waiting, and it is judging a room that was set up moments ago rather than
# talking someone through setting one up.
DEFAULT_SELF_TEST_SECONDS = 2.0

# --- plan 03-01: the phone surface (D-05, OPS-01) --------------------------------------
# Pinned, never auto-chosen -- a silently-different bound port would desynchronize the
# printed/QR'd join URL from what is actually listening (03-RESEARCH §4, OPS-05).
DEFAULT_APP_PORT = 8765
# Auto-open the local join URL in a browser window at `start` time so the second-monitor
# path (SURF-03) costs the user nothing (03-RESEARCH §5). Off in tests/CI via env override.
DEFAULT_OPEN_BROWSER = True

# The phone join QR, OFF by default (user decision, 2026-08-10: "stop
# generating QR code for now"). A toggle rather than a deletion: the phone
# surface itself is untouched and still reachable -- `create_join` prints the
# join URL and the resolved LAN IP exactly as before -- so re-enabling is
# `APP_EMIT_QR=1` and nothing else. Deleting the renderers would also have
# stranded `qrcode[pil]`, a pinned dependency the gate-3 count is calibrated
# against.
DEFAULT_EMIT_QR = False

# --- plan 03-03: precheck's phone-reachability row (OPS-05, T-03-27) -----------------
# Bounded wait for a real client to connect during `precheck --phone`, BEFORE reporting
# FAIL. A pre-call check that hangs one minute before an interview is worse than one that
# fails -- this is what keeps it bounded rather than an indefinite wait.
DEFAULT_PRECHECK_PHONE_WAIT_S = 30.0

# --- plan 04-01: the coach's coverage detector (04-CONTEXT.md §C3/§C4/§C5) -----------
# The coverage-match call sits off the ~1.5-2s coaching critical path (COACH-05) --
# it has the whole span of the interviewee's subsequent answer to complete in, hence
# a looser bound than anything else in this file. `max_retries=0` at the client
# construction site (app/coach/reasoning_client.py) is what makes this timeout the
# REAL wall-clock bound rather than up to 3x it (the SDK retries timeouts by default).
DEFAULT_COACH_COVERAGE_TIMEOUT_S = 3.0
# Bounded so a sustained vendor outage cannot grow an unbounded backlog of pending
# interviewer segments (T-04-07). Drop-oldest, not reject-newest -- a stale coverage
# match is worth less than a fresh one (§C3).
DEFAULT_COACH_QUEUE_MAXSIZE = 16
# Where `context/brief.json` and `context/survey.json` (§C6, CTX-01/02) live --
# git-ignored, read once at `start`, never written by anything in `app/coach/`.
DEFAULT_CONTEXT_DIR = "context"

# --- plan 04-03: the authoring page (04-CONTEXT.md D-07, §E4) ------------------------
# A SEPARATE port from DEFAULT_APP_PORT -- the authoring server is a genuinely
# separate app instance, bound to the loopback address only (never the phone
# surface's wildcard bind). Not printed on a QR and never shared -- pinned
# only so re-running `python -m app author` prints a stable, memorable URL.
DEFAULT_AUTHOR_PORT = 8766

# --- plan 04-02: the generate/critic pipeline (D1-D9) --------------------------------
# Each stage's OWN explicit per-call timeout (D1/D2). With `max_retries=0` at the
# client construction site, these are the REAL wall-clock bound, not a third of it.
#
# RAISED 2026-08-10 from 2.5/1.5 after the first real measurement. Both were set
# from an estimate, and both were too low to ever succeed: measured medians are
# generate ~3.5s and critique ~4.4s, with observed maxima of 6.6s and 6.1s. The
# old critique bound failed EVERY cycle, and because the pipeline returns without
# broadcasting on a critique failure, the surface showed nothing at all -- the
# coach had never once put a suggestion on a screen.
#
# The previous comment here claimed the two values "differ deliberately -- generate
# has more work to do (candidate text) than critique (a small structured verdict)".
# Measurement inverts that: critique emits ~452 output tokens to generate's ~230,
# because it writes a rationale string per candidate. It is the SLOWER of the two.
# Output tokens are what cost time, so they are set equal now rather than shaped by
# an assumption about which stage "does more".
#
# 10s each is ~50% headroom over the observed maxima, which the growing prompt eats
# into as an interview runs long. Bounded rather than generous-forever so a hung
# vendor call cannot stall the cycle indefinitely; the pipeline's drop-oldest queue
# handles the backlog if one does.
DEFAULT_COACH_GENERATE_TIMEOUT_S = 10.0
DEFAULT_COACH_CRITIQUE_TIMEOUT_S = 10.0
# Output token caps (D1's "keep the prompts and the token caps doing that work" --
# no `effort` parameter exists on this model tier). Critique is capped generously
# per §D5 -- "headroom is cheap and a truncated verdict is expensive".
DEFAULT_COACH_GENERATE_MAX_TOKENS = 700
DEFAULT_COACH_CRITIQUE_MAX_TOKENS = 1000
# One primary plus up to two alternatives (D-05).
DEFAULT_COACH_MAX_SHOWN = 3
# Consecutive-failure threshold before CoachHealth reports unhealthy (§D7). A
# single slow round trip is normal; this is what tells a real outage apart from
# one bad turn.
DEFAULT_COACH_FAILURE_THRESHOLD = 3
# Language hysteresis (D-06, `04-RESEARCH.md` Open Question 2 -- no measurement
# stands behind these starting values, so they must be tunable without a code
# change). Window bounded by elapsed INTERVIEWEE speech time, not wall-clock time
# or segment count. The flip margin is a clear-majority share, not a bare 50%.
DEFAULT_COACH_LANGUAGE_WINDOW_S = 90.0
DEFAULT_COACH_LANGUAGE_FLIP_MARGIN = 0.65
DEFAULT_COACH_LANGUAGE_MIN_CHARS = 40
# The pipeline's own trigger queue (§D9) -- maxsize=1 is what implements
# "replace the queued-but-unstarted trigger, never queue behind it": a second
# arrival while one is already queued drops the queued one and records a named
# event, exactly like the coverage detector's own drop-oldest queue.
DEFAULT_COACH_PIPELINE_QUEUE_MAXSIZE = 1

# --- plan 03-02: heartbeat / reconnect / status-poll knobs (D-01/D-02, §Q2-Q4, §Q6/§Q7) -----
# The client cannot originate or observe protocol-level WebSocket ping/pong frames from page
# JS (03-RESEARCH §2) -- an application-level JSON heartbeat is the only lever that detects a
# half-open mobile socket. Served to the client inside the `hello` message so both sides read
# one source of truth; never hardcoded in app.js.
DEFAULT_HEARTBEAT_INTERVAL_MS = 15000
DEFAULT_HEARTBEAT_TIMEOUT_MS = 5000
# min(cap, base * 2^n) with +-20% jitter (03-RESEARCH §2's "Reconnect backoff shape").
DEFAULT_RECONNECT_BACKOFF_BASE_MS = 500
DEFAULT_RECONNECT_BACKOFF_CAP_MS = 10000
# How often the async status poller re-checks worker.failed / capture.is_dead and pushes a
# server_status message on change (§Q6/§Q7). Reads state each StreamPair already exposes
# publicly -- no new callback into app/capture/supervisor.py or app/stt/live.py.
DEFAULT_STATUS_POLL_INTERVAL_S = 1.0

# --- plan 05-01: extraction (05-CONTEXT.md D-06/D-07, §K6/§K10) ---------------------
# This runs AFTER an interview, off the ~1.5-2s live coaching budget entirely -- so its
# timeout is a large, explicitly named BATCH value in the tens of seconds, not a
# live-call bound. Still named rather than inherited (`reasoning_client.py:77-82`):
# every call site binds its OWN timeout, even a generous one.
DEFAULT_COACH_EXTRACT_TIMEOUT_S = 60.0
# One call per finished interview, output sized for a full session's worth of facts
# (§K10: ~2-4k output tokens projected for a full transcript's extraction).
DEFAULT_COACH_EXTRACT_MAX_TOKENS = 4000

# --- plan 06-01: retrieval over the fact corpus (06-CONTEXT.md D-02/D-05, §F3) -------
# Caps how many past-interview facts `load_brief()`'s own v2 retrieval selects into the
# merged brief -- padding to a fixed N with facts that scored zero relevance would spend
# cache tokens and dilute the brief for no benefit (06-RESEARCH.md §1's "the corpus is
# small by construction" finding). `load_brief(context_dir)`'s signature is fixed to a
# single positional argument by D-03 -- there is no config object it could accept without
# breaking the CTX-03 seam this whole phase is built to preserve. So this constant is
# imported directly by `app/coach/context.py`'s own selection helper as a keyword-only
# default, exactly like `app/web/hub.py:68`'s own
# `max_shown: int = config_mod.DEFAULT_COACH_MAX_SHOWN` pattern, rather than threaded
# through `AppConfig`, which would add a field no caller could ever route to `load_brief`.
DEFAULT_COACH_MAX_RETRIEVED_FACTS = 5

# --- independent review finding B: uvicorn's websocket keepalive, pinned explicitly ------
# `app/web/hub.py::_broadcast_payload` fans out SEQUENTIALLY with only a per-connection
# try/except -- that protects against a send that RAISES, not one that never returns (a
# half-open phone whose write buffer is full parks `send_json()` on backpressure and
# starves every other connected client, e.g. a second monitor). The only thing that
# bounds that today is `websockets`' own ping/pong timeout, applied by uvicorn's
# `ws_ping_interval`/`ws_ping_timeout` -- which `uvicorn.Config(...)` never set, so it
# rode entirely on the library's undocumented default. Naming the values explicitly here
# means a future dependency bump can never silently remove the bound.
DEFAULT_WS_PING_INTERVAL_S = 20.0
DEFAULT_WS_PING_TIMEOUT_S = 20.0

_ENV_LOADED = False


def _load_dotenv_once(path: "str | Path" = ".env") -> None:
    """Minimal, dependency-free `.env` loader (no `python-dotenv` is
    installed). Only sets variables not already present in the environment --
    the environment always wins. Silently does nothing if the file is absent
    and never raises on a malformed line; this is a convenience, not a
    validator."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppConfig:
    engine_name: str
    sample_rate: int
    frame_ms: int
    queue_max_seconds: float
    idle_threshold_s: float
    sessions_root: Path
    keepalive_enabled: bool
    supervisor_poll_interval_s: float
    lag_queue_fraction: float
    device_poll_interval_s: float
    precheck_seconds: float
    precheck_real_fraction_min: float
    precheck_level_dbfs_min: float
    self_test_seconds: float
    precheck_engine_timeout_s: float
    precheck_phone_wait_s: float
    coach_coverage_timeout_s: float
    coach_queue_maxsize: int
    context_dir: Path
    coach_generate_timeout_s: float
    coach_critique_timeout_s: float
    coach_generate_max_tokens: int
    coach_critique_max_tokens: int
    coach_max_shown: int
    coach_failure_threshold: int
    coach_language_window_s: float
    coach_language_flip_margin: float
    coach_language_min_chars: int
    coach_repeat_window: int
    coach_repeat_threshold: float
    coach_repeat_prompt_recent: int
    coach_pipeline_queue_maxsize: int
    coach_extract_timeout_s: float
    coach_extract_max_tokens: int
    author_port: int
    app_port: int
    open_browser: bool
    emit_qr: bool
    heartbeat_interval_ms: int
    heartbeat_timeout_ms: int
    reconnect_backoff_base_ms: int
    reconnect_backoff_cap_ms: int
    status_poll_interval_s: float
    ws_ping_interval_s: float
    ws_ping_timeout_s: float


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def load_config(*, engine_override: Optional[str] = None) -> AppConfig:
    """Resolve every runtime knob once. `engine_override` is a CLI flag and
    beats `STT_ENGINE`, which beats `DEFAULT_ENGINE` -- the order plan 02-01
    Task 3 asserts explicitly."""
    _load_dotenv_once()
    engine_name = engine_override or os.environ.get("STT_ENGINE") or DEFAULT_ENGINE
    return AppConfig(
        engine_name=engine_name,
        sample_rate=int(os.environ.get("APP_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)),
        frame_ms=int(os.environ.get("APP_FRAME_MS", DEFAULT_FRAME_MS)),
        queue_max_seconds=float(
            os.environ.get("APP_QUEUE_MAX_SECONDS", DEFAULT_QUEUE_MAX_SECONDS)
        ),
        idle_threshold_s=float(os.environ.get("APP_IDLE_THRESHOLD_S", DEFAULT_IDLE_THRESHOLD_S)),
        sessions_root=Path(os.environ.get("APP_SESSIONS_ROOT", DEFAULT_SESSIONS_ROOT)),
        keepalive_enabled=_bool_env("APP_KEEPALIVE_ENABLED", DEFAULT_KEEPALIVE_ENABLED),
        supervisor_poll_interval_s=float(
            os.environ.get("APP_SUPERVISOR_POLL_INTERVAL_S", DEFAULT_SUPERVISOR_POLL_INTERVAL_S)
        ),
        lag_queue_fraction=float(os.environ.get("APP_LAG_QUEUE_FRACTION", DEFAULT_LAG_QUEUE_FRACTION)),
        device_poll_interval_s=float(
            os.environ.get("APP_DEVICE_POLL_INTERVAL_S", DEFAULT_DEVICE_POLL_INTERVAL_S)
        ),
        precheck_seconds=float(os.environ.get("APP_PRECHECK_SECONDS", DEFAULT_PRECHECK_SECONDS)),
        precheck_real_fraction_min=float(
            os.environ.get("APP_PRECHECK_REAL_FRACTION_MIN", DEFAULT_PRECHECK_REAL_FRACTION_MIN)
        ),
        precheck_level_dbfs_min=float(
            os.environ.get("APP_PRECHECK_LEVEL_DBFS_MIN", DEFAULT_PRECHECK_LEVEL_DBFS_MIN)
        ),
        self_test_seconds=float(
            os.environ.get("APP_SELF_TEST_SECONDS", DEFAULT_SELF_TEST_SECONDS)
        ),
        precheck_engine_timeout_s=float(
            os.environ.get("APP_PRECHECK_ENGINE_TIMEOUT_S", DEFAULT_PRECHECK_ENGINE_TIMEOUT_S)
        ),
        precheck_phone_wait_s=float(
            os.environ.get("APP_PRECHECK_PHONE_WAIT_S", DEFAULT_PRECHECK_PHONE_WAIT_S)
        ),
        coach_coverage_timeout_s=float(
            os.environ.get("APP_COACH_COVERAGE_TIMEOUT_S", DEFAULT_COACH_COVERAGE_TIMEOUT_S)
        ),
        coach_queue_maxsize=int(
            os.environ.get("APP_COACH_QUEUE_MAXSIZE", DEFAULT_COACH_QUEUE_MAXSIZE)
        ),
        context_dir=Path(os.environ.get("APP_CONTEXT_DIR", DEFAULT_CONTEXT_DIR)),
        coach_generate_timeout_s=float(
            os.environ.get("APP_COACH_GENERATE_TIMEOUT_S", DEFAULT_COACH_GENERATE_TIMEOUT_S)
        ),
        coach_critique_timeout_s=float(
            os.environ.get("APP_COACH_CRITIQUE_TIMEOUT_S", DEFAULT_COACH_CRITIQUE_TIMEOUT_S)
        ),
        coach_generate_max_tokens=int(
            os.environ.get("APP_COACH_GENERATE_MAX_TOKENS", DEFAULT_COACH_GENERATE_MAX_TOKENS)
        ),
        coach_critique_max_tokens=int(
            os.environ.get("APP_COACH_CRITIQUE_MAX_TOKENS", DEFAULT_COACH_CRITIQUE_MAX_TOKENS)
        ),
        coach_max_shown=int(os.environ.get("APP_COACH_MAX_SHOWN", DEFAULT_COACH_MAX_SHOWN)),
        coach_failure_threshold=int(
            os.environ.get("APP_COACH_FAILURE_THRESHOLD", DEFAULT_COACH_FAILURE_THRESHOLD)
        ),
        coach_language_window_s=float(
            os.environ.get("APP_COACH_LANGUAGE_WINDOW_S", DEFAULT_COACH_LANGUAGE_WINDOW_S)
        ),
        coach_language_flip_margin=float(
            os.environ.get("APP_COACH_LANGUAGE_FLIP_MARGIN", DEFAULT_COACH_LANGUAGE_FLIP_MARGIN)
        ),
        coach_language_min_chars=int(
            os.environ.get("APP_COACH_LANGUAGE_MIN_CHARS", DEFAULT_COACH_LANGUAGE_MIN_CHARS)
        ),
        # Defaults live in `app/coach/repetition.py` beside the measurement
        # that produced them, never re-spelled here as bare numbers.
        coach_repeat_window=int(
            os.environ.get("APP_COACH_REPEAT_WINDOW", repetition.DEFAULT_WINDOW)
        ),
        coach_repeat_threshold=float(
            os.environ.get("APP_COACH_REPEAT_THRESHOLD", repetition.DEFAULT_THRESHOLD)
        ),
        coach_repeat_prompt_recent=int(
            os.environ.get("APP_COACH_REPEAT_PROMPT_RECENT", repetition.DEFAULT_PROMPT_RECENT)
        ),
        coach_pipeline_queue_maxsize=int(
            os.environ.get("APP_COACH_PIPELINE_QUEUE_MAXSIZE", DEFAULT_COACH_PIPELINE_QUEUE_MAXSIZE)
        ),
        coach_extract_timeout_s=float(
            os.environ.get("APP_COACH_EXTRACT_TIMEOUT_S", DEFAULT_COACH_EXTRACT_TIMEOUT_S)
        ),
        coach_extract_max_tokens=int(
            os.environ.get("APP_COACH_EXTRACT_MAX_TOKENS", DEFAULT_COACH_EXTRACT_MAX_TOKENS)
        ),
        author_port=int(os.environ.get("APP_AUTHOR_PORT", DEFAULT_AUTHOR_PORT)),
        app_port=int(os.environ.get("APP_PORT", DEFAULT_APP_PORT)),
        open_browser=_bool_env("APP_OPEN_BROWSER", DEFAULT_OPEN_BROWSER),
        emit_qr=_bool_env("APP_EMIT_QR", DEFAULT_EMIT_QR),
        heartbeat_interval_ms=int(
            os.environ.get("APP_HEARTBEAT_INTERVAL_MS", DEFAULT_HEARTBEAT_INTERVAL_MS)
        ),
        heartbeat_timeout_ms=int(
            os.environ.get("APP_HEARTBEAT_TIMEOUT_MS", DEFAULT_HEARTBEAT_TIMEOUT_MS)
        ),
        reconnect_backoff_base_ms=int(
            os.environ.get("APP_RECONNECT_BACKOFF_BASE_MS", DEFAULT_RECONNECT_BACKOFF_BASE_MS)
        ),
        reconnect_backoff_cap_ms=int(
            os.environ.get("APP_RECONNECT_BACKOFF_CAP_MS", DEFAULT_RECONNECT_BACKOFF_CAP_MS)
        ),
        status_poll_interval_s=float(
            os.environ.get("APP_STATUS_POLL_INTERVAL_S", DEFAULT_STATUS_POLL_INTERVAL_S)
        ),
        ws_ping_interval_s=float(
            os.environ.get("APP_WS_PING_INTERVAL_S", DEFAULT_WS_PING_INTERVAL_S)
        ),
        ws_ping_timeout_s=float(
            os.environ.get("APP_WS_PING_TIMEOUT_S", DEFAULT_WS_PING_TIMEOUT_S)
        ),
    )


def supervisor_config(cfg: AppConfig) -> "SupervisorConfig":
    """Builds a `StreamSupervisor` config from the resolved `AppConfig` --
    the one place CLI/env knobs become supervisor thresholds."""
    from app.capture.supervisor import SupervisorConfig

    return SupervisorConfig(
        poll_interval_s=cfg.supervisor_poll_interval_s,
        idle_threshold_s=cfg.idle_threshold_s,
        lag_queue_fraction=cfg.lag_queue_fraction,
        device_poll_interval_s=cfg.device_poll_interval_s,
    )
