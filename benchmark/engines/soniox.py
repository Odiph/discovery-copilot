"""Soniox real-time WebSocket adapter (candidate 1) — no vendor SDK.

Per phase_specification S8, no PyPI package named `soniox`/`sonix` is installed
(that collision is the T-01-07 threat entry); Soniox is accessed directly over
the approved `websockets` package against Soniox's own endpoint.

Endpoint, auth and message schema confirmed AT IMPLEMENTATION TIME against
Soniox's own example code (github.com/soniox/soniox_examples,
speech_to_text/python/soniox_realtime.py), not assumed:
- wss://stt-rt.soniox.com/transcribe-websocket
- first message: a JSON config frame — api_key, model, audio_format,
  sample_rate, num_channels, enable_language_identification,
  enable_endpoint_detection. `language_hints` is deliberately OMITTED: Soniox
  is candidate 1 precisely BECAUSE it needs no pre-declared language, and
  setting a hint would bias exactly the property under test.
- audio sent as raw binary frames; an empty string "" signals end-of-audio
- responses: JSON with "tokens" (each: text, is_final, language, ...), an
  optional "error_code"/"error_message" pair, and "finished": true at the end

RE-CONFIRMED against a live response 2026-08-10 (25s of real call audio). The
exact observed shapes, which the two facts below depend on:
- top-level keys: `tokens`, `finished`, `total_audio_proc_ms`, `final_audio_proc_ms`
- token keys: `text`, `start_ms`, `end_ms`, `confidence`, `is_final`, `language`
- endpoint detection emits a token whose text is the literal `<end>`, carrying
  `start_ms: 0, end_ms: 0` and NO `language`. It is a MARKER, not speech.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from websockets.exceptions import ConnectionClosed

from benchmark import audio as audio_mod
from benchmark import config as config_mod
from benchmark.engines.base import EngineResult, Segment, StreamingSession

SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
DEFAULT_MODEL = "stt-rt-v5"
FRAME_MS = 150  # within the plan's ~100-200ms real-time pacing window
SAMPLE_RATE = 16000

# Endpoint detection (`enable_endpoint_detection: True` in the start frame)
# marks an utterance boundary with a token whose text is this literal. It is
# not speech: it must never reach segment text, and its zeroed `end_ms` must
# never reach the audio clock. Both happened -- see `tokens_to_segment`.
ENDPOINT_MARKER_TEXT = "<end>"

logger = logging.getLogger(__name__)


class SonioxConfigError(Exception):
    """Raised for missing/invalid configuration. Message never includes key material."""


class SonioxServerError(Exception):
    """Raised when Soniox returns an error_code/error_message frame."""


@dataclass(frozen=True)
class SonioxConfig:
    model: str = DEFAULT_MODEL
    sample_rate: int = SAMPLE_RATE
    frame_ms: int = FRAME_MS


def build_start_message(api_key: str, config: SonioxConfig) -> str:
    """The first frame sent over the socket."""
    payload = {
        "api_key": api_key,
        "model": config.model,
        "audio_format": "pcm_s16le",
        "sample_rate": config.sample_rate,
        "num_channels": 1,
        "enable_language_identification": True,
        "enable_endpoint_detection": True,
    }
    return json.dumps(payload)


def process_response(response: dict) -> tuple[list[dict], bool, Optional[tuple[int, str]]]:
    """Pure message handler: given ONE parsed JSON response from the socket,
    returns (finalized_tokens, is_finished, error_or_none). No network, no
    side effects — this is exactly what the contract test replays recorded
    payloads through, with no live call.
    """
    if response.get("error_code") is not None:
        return [], False, (response["error_code"], response.get("error_message", ""))
    finalized = [t for t in response.get("tokens", []) if t.get("text") and t.get("is_final")]
    return finalized, bool(response.get("finished")), None


def speech_tokens(tokens: list[dict]) -> list[dict]:
    """`tokens` minus the endpoint markers. Everything that reads token text or
    token timings must go through here."""
    return [t for t in tokens if t.get("text") != ENDPOINT_MARKER_TEXT]


def dominant_language(tokens: list[dict]) -> Optional[str]:
    """The language MOST OF THE SEGMENT is in, weighted by characters.

    This used to be `languages[0]` -- the FIRST token's tag labelling the whole
    segment -- and on 2026-08-12 a real recording showed what that costs. The
    segment `"fa, "` + nineteen characters of Hebrew was reported as ITALIAN
    on the strength of a leading two-character token, with every Hebrew
    character after it outvoted by position.

    **Weighted by CHARACTER COUNT, not token count, and that is the same rule
    `app/coach/language.py` already reached** for the same question about the
    same audio. Its reasoning applies unchanged here: a one-character token is
    not evidence equal to a long word, and short-segment language
    identification is unreliable across engines generally. This module has no
    business answering the question a different way from the module that
    already thought about it.

    Ties break toward the language seen FIRST, so the answer is deterministic
    for a genuinely balanced code-switch rather than depending on dict order.

    **What this deliberately does NOT do is decide anything.** The coach reads
    none of this -- `app/coach/language.py` counts script from the text itself
    and its docstring forbids reading `detected_language` at all, because the
    vendor tag was measured wrong three times out of three. This tag is for the
    RECORD: the transcript, the render, the phone's language badge. Making it
    right makes the record right, and nothing else moves.
    """
    weights: "dict[str, int]" = {}
    order: "list[str]" = []
    for token in tokens:
        language = token.get("language")
        if not language:
            continue
        if language not in weights:
            weights[language] = 0
            order.append(language)
        # `or ""` because a token may carry a language and no text; it then
        # contributes nothing, which is right -- it is not evidence about what
        # was said, only about what the vendor guessed.
        weights[language] += len(token.get("text") or "")
    if not weights:
        return None
    return max(order, key=lambda language: weights[language])


def token_audio_end_s(tokens: list[dict]) -> Optional[float]:
    """When the last real SPEECH in `tokens` ended, on the audio clock, in
    seconds. `None` when no token carries a usable `end_ms`.

    This is the number the latency metric needs and the only one Soniox
    actually knows. Taking `max` rather than the last token's value is
    deliberate: it survives an out-of-order response without silently
    reporting an earlier time, and it ignores the endpoint marker's `end_ms: 0`
    even if `speech_tokens` were ever bypassed.
    """
    ends = [
        t["end_ms"]
        for t in speech_tokens(tokens)
        if isinstance(t.get("end_ms"), (int, float)) and not isinstance(t.get("end_ms"), bool)
    ]
    if not ends:
        return None
    return max(ends) / 1000.0


def tokens_to_segment(
    tokens: list[dict], *, fallback_audio_end_s: float, ready_wall_s: float
) -> Optional[Segment]:
    """Reduce one response's worth of finalized Soniox tokens to a Segment.
    Returns None when there is no speech (an empty/interim response, or a
    response carrying only an endpoint marker).

    `audio_end_s` comes from the tokens' OWN `end_ms`, never from the caller's
    push position. This is the fix for the invalidated latency metric
    (`.planning/STATE.md` § THE LATENCY METRIC MEASURES THE WRONG THING): the
    push position is "how much audio we have SENT", so `ready_wall_s -
    audio_end_s` measured the skew between the capture path's push cadence and
    the wall clock -- a property of buffering, not of the engine. It
    understated true latency by however much audio was pushed after the
    utterance actually ended, which is precisely the trailing audio VAD needs
    before it finalizes. Measured on the same 25s of real call audio: 1.83s
    median from token timings vs 1.09s from the push counter.

    `fallback_audio_end_s` is used ONLY if no token carries a usable `end_ms`,
    which has never been observed. It keeps a live interview running rather
    than raising over a metric -- but it silently restores the broken quantity,
    so it warns. Treat that warning in a session log as invalidating that
    session's latency figures.
    """
    speech = speech_tokens(tokens)
    text = "".join(t["text"] for t in speech).strip()
    if not text:
        return None
    audio_end_s = token_audio_end_s(speech)
    if audio_end_s is None:
        logger.warning(
            "soniox response carried speech with no usable end_ms; falling back to the "
            "push position, which measures buffering skew and NOT latency"
        )
        audio_end_s = fallback_audio_end_s
    detected_language = dominant_language(speech)
    return Segment(
        text=text, audio_end_s=audio_end_s, ready_wall_s=ready_wall_s, detected_language=detected_language
    )


def _describe_early_close(ws, exc: Exception) -> str:
    """Turn an early server-side close into something the user can act on.

    Soniox sends a JSON error frame and then closes. If that happens while we are
    still streaming audio, the send raises before the recv loop ever runs. Drain
    whatever the server already queued so the real reason survives.
    """
    try:
        while True:
            raw = ws.recv(timeout=2.0)
            payload = json.loads(raw)
            code = payload.get("error_code")
            if code is not None:
                message = payload.get("error_message", "")
                more = payload.get("more_info", "")
                detail = f"Soniox error {code}: {message}"
                if more:
                    detail += f" — {more}"
                return detail
    except Exception:  # noqa: BLE001 — best-effort drain; original close still reported below
        pass
    return (
        "Soniox closed the connection during audio upload without a readable error frame "
        f"({exc}). Common causes: exhausted account balance (HTTP 402), an invalid API key, "
        "or an unrecognised model id. Check https://console.soniox.com."
    )


class SonioxEngine:
    """Satisfies the Engine protocol. Construction never touches the network
    or reads the API key — only `transcribe()` does."""

    name = "soniox"

    def __init__(self, config: Optional[SonioxConfig] = None):
        self._config = config or SonioxConfig()

    def _get_api_key(self) -> str:
        # `.strip()`, and it is load-bearing rather than tidiness: `"   "` is
        # truthy, so an exported-but-blank variable (`set SONIOX_API_KEY= `)
        # passed this check, passed the pre-flight built on it, and delivered
        # the whole original failure -- devices opened, session directory
        # written, traceback from the worker thread. Reproduced 2026-08-13.
        # `app/config.py`'s own .env loader already strips whitespace and
        # matched quotes, which is why this only ever bit a shell export.
        api_key = (os.environ.get("SONIOX_API_KEY") or "").strip()
        if not api_key:
            raise SonioxConfigError(
                "SONIOX_API_KEY is not set. Read from the environment only — see benchmark/README.md."
            )
        return api_key

    def preflight(self) -> None:
        """Raise if this engine could not open a session, WITHOUT opening one.

        Optional on the `StreamingEngine` protocol: an engine that needs no
        credentials simply does not define it, and `app.stt.live.preflight_engines`
        skips it. It exists because the check `open_session` already performs
        happens far too late -- inside a worker thread, after `start` has opened
        the audio devices and created the session directory. A missing key then
        produced a traceback and an orphaned session directory with no
        `closed_at`, which is indistinguishable on disk from a crashed
        interview.

        Deliberately NOT a network call. It answers "is this configured", not
        "is the vendor up" -- checking the latter would spend a connection (and
        on some accounts, money) on every single start, and would turn a
        transient vendor blip into a refusal to record. Reachability is still
        discovered where it always was, at `open_session`.
        """
        self._get_api_key()

    def transcribe(
        self, audio_path: str, slice_: Optional[tuple[float, float]] = None
    ) -> EngineResult:
        from websockets.sync.client import connect as ws_connect

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
        # Set when the FIRST audio frame is sent, not here -- see the comment on
        # `_SonioxStreamingSession._audio_clock_origin`. Soniox's `end_ms` counts
        # from the first audio it received, so the wall clock must share that
        # origin or the connect handshake is billed as model latency.
        audio_clock_origin: Optional[float] = None

        def ready_wall_s() -> float:
            origin = audio_clock_origin if audio_clock_origin is not None else time.perf_counter()
            return time.perf_counter() - origin

        with ws_connect(
            SONIOX_WEBSOCKET_URL,
            open_timeout=config_mod.NET_OPEN_TIMEOUT_S,
            close_timeout=config_mod.NET_CLOSE_TIMEOUT_S,
        ) as ws:
            ws.send(build_start_message(api_key, cfg))

            audio_position_s = 0.0

            def _drain(block_s: float) -> bool:
                """Consume whatever the server has already queued. Returns True when the
                server signalled `finished`.

                This exists because Soniox streams interim results for the WHOLE upload.
                `websockets` buffers incoming frames in a queue of `max_queue` (default 16);
                once it fills, the connection's reader stalls, PONG frames stop being
                processed, and the client's own keepalive fires — killing a healthy session
                with `sent 1011 (internal error) keepalive ping timeout`.

                Observed 2026-08-05: 8s and 30s uploads passed, a ~3.5 minute upload failed.
                The bug is duration-dependent, so short tests never see it and a real
                interview always would. Draining as we send is the fix; raising max_queue
                only moves the cliff.
                """
                nonlocal segments, hyp_parts
                deadline = time.perf_counter() + block_s
                while True:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return False
                    try:
                        message = ws.recv(timeout=remaining)
                    except TimeoutError:
                        return False
                    response = json.loads(message)
                    finalized, finished, error = process_response(response)
                    if error is not None:
                        code, msg = error
                        raise SonioxServerError(f"Soniox error {code}: {msg}")
                    seg = tokens_to_segment(
                        finalized,
                        fallback_audio_end_s=audio_position_s,
                        ready_wall_s=ready_wall_s(),
                    )
                    if seg is not None:
                        segments.append(seg)
                        hyp_parts.append(seg.text)
                    if finished:
                        return True

            try:
                for offset in range(0, len(pcm16), frame_bytes):
                    chunk = pcm16[offset : offset + frame_bytes]
                    ws.send(chunk)
                    if audio_clock_origin is None:
                        audio_clock_origin = time.perf_counter()
                    audio_position_s += cfg.frame_ms / 1000.0
                    # Pace to wall clock so vendor buffering is measured honestly — but
                    # spend that pause DRAINING rather than sleeping, so the receive queue
                    # never fills. Same wall-clock cost, no starvation.
                    if _drain(cfg.frame_ms / 1000.0):
                        break

                ws.send("")  # end-of-audio signal
            except ConnectionClosed as exc:
                # The server rejected the session and closed while we were still
                # streaming, so we never reached the recv loop that decodes error
                # frames. Without this branch the caller sees a bare
                # `ConnectionClosedOK: received 1000 (OK)` — technically accurate and
                # completely useless. Observed 2026-08-05 with error_code 402
                # (organization_balance_exhausted), which cost real debugging time.
                # OPS-05: a failure the user cannot act on is a silent failure.
                raise SonioxServerError(_describe_early_close(ws, exc)) from exc

            while True:
                # Bounded read. Without a timeout this blocks forever on a stalled
                # vendor — harmless-looking here, but fatal behind the live interface.
                message = ws.recv(timeout=config_mod.NET_RECV_TIMEOUT_S)
                response = json.loads(message)
                finalized, finished, error = process_response(response)
                if error is not None:
                    code, msg = error
                    raise SonioxServerError(f"Soniox error {code}: {msg}")
                seg = tokens_to_segment(
                    finalized,
                    fallback_audio_end_s=audio_position_s,
                    ready_wall_s=ready_wall_s(),
                )
                if seg is not None:
                    segments.append(seg)
                    hyp_parts.append(seg.text)
                if finished:
                    break

        return EngineResult(
            hypothesis_text=" ".join(hyp_parts),
            segments=tuple(segments),
            metadata={"model": cfg.model, "decode_mode": "streaming"},
        )

    # --- Phase 2 (app/): the live-source extension (base.StreamingEngine) --------
    #
    # Reuses build_start_message / process_response / tokens_to_segment above
    # unchanged -- the vendor's message schema is described exactly once in this
    # repository (phase_specification S2). Only open_session() touches the
    # network; construction above already never does.

    def open_session(self, *, sample_rate: int, label: str = "") -> StreamingSession:
        del label  # Soniox needs no per-stream label; kept for protocol symmetry
        from websockets.sync.client import connect as ws_connect

        api_key = self._get_api_key()
        cfg = SonioxConfig(
            model=self._config.model, sample_rate=sample_rate, frame_ms=self._config.frame_ms
        )
        ws = ws_connect(
            SONIOX_WEBSOCKET_URL,
            open_timeout=config_mod.NET_OPEN_TIMEOUT_S,
            close_timeout=config_mod.NET_CLOSE_TIMEOUT_S,
        )
        try:
            ws.send(build_start_message(api_key, cfg))
        except ConnectionClosed as exc:
            raise SonioxServerError(_describe_early_close(ws, exc)) from exc
        return _SonioxStreamingSession(ws=ws, config=cfg, session_start=time.monotonic())


class _SonioxStreamingSession:
    """Satisfies base.StreamingSession. Owns one live session's socket, from
    exactly one thread (S5) -- send_audio/poll/close must all be called from
    the same thread.

    A segment's `audio_end_s` comes from the vendor's own token timings; its
    `ready_wall_s` is measured from `_audio_clock_origin`, so both clocks in a
    segment share one origin and their difference is a real latency."""

    def __init__(self, *, ws, config: SonioxConfig, session_start: float) -> None:
        self._ws = ws
        self._config = config
        self._session_start = session_start
        # Soniox's token `end_ms` counts from the first audio the SERVER
        # received, so the wall clock has to be anchored to the first audio we
        # SENT -- not to `session_start`, which is stamped when open_session()
        # returns, before any audio flows. Anchoring at open time would bill
        # the connect handshake and the gap before the capture thread's first
        # frame as model latency. Stays None until the first non-empty frame.
        self._audio_clock_origin: Optional[float] = None
        # Kept ONLY as `tokens_to_segment`'s fallback. This is "how much audio
        # we have SENT", which is not when an utterance ended -- reading it as
        # the audio clock is the defect fixed 2026-08-10.
        self._audio_position_s = 0.0
        self._finished = False

    def _ready_wall_s(self) -> float:
        origin = self._audio_clock_origin
        if origin is None:  # a segment before any audio was sent; not observed
            origin = self._session_start
        return time.monotonic() - origin

    def send_audio(self, pcm16: bytes) -> None:
        if self._finished or not pcm16:
            return
        if self._audio_clock_origin is None:
            # Stamped BEFORE the send deliberately: if the send blocks, that
            # delay belongs in the measurement. An honest metric errs toward
            # overstating latency, never toward understating it -- understating
            # is exactly how the previous metric passed unnoticed.
            self._audio_clock_origin = time.monotonic()
        self._ws.send(pcm16)
        self._audio_position_s += len(pcm16) / 2 / self._config.sample_rate  # 16-bit mono

    def poll(self, timeout_s: float = 0.0) -> tuple[Segment, ...]:
        """Never raises on 'nothing yet' -- a read timeout is normal. Drains
        every message already buffered (first read honours `timeout_s`, the
        rest are non-blocking) so a poll after a burst of vendor activity
        returns everything at once rather than trickling it out one call at a
        time."""
        if self._finished:
            return ()
        segments: list[Segment] = []
        timeout = timeout_s
        while True:
            try:
                message = self._ws.recv(timeout=timeout)
            except TimeoutError:
                break
            except ConnectionClosed:
                self._finished = True
                break
            segments.extend(self._handle_message(message))
            if self._finished:
                break
            timeout = 0.0  # subsequent reads: drain what's already arrived, don't wait again
        return tuple(segments)

    def close(self) -> tuple[Segment, ...]:
        """Sends the end-of-audio signal, drains the tail, and closes the
        socket. Safe to call more than once."""
        segments: list[Segment] = []
        if not self._finished:
            try:
                self._ws.send("")
            except ConnectionClosed:
                self._finished = True
            else:
                while not self._finished:
                    try:
                        message = self._ws.recv(timeout=config_mod.NET_RECV_TIMEOUT_S)
                    except TimeoutError:
                        break
                    except ConnectionClosed:
                        break
                    segments.extend(self._handle_message(message))
        self._finished = True
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001 -- best-effort close, session already ending
            pass
        return tuple(segments)

    def _handle_message(self, message) -> list[Segment]:
        response = json.loads(message)
        finalized, finished, error = process_response(response)
        if error is not None:
            code, msg = error
            raise SonioxServerError(f"Soniox error {code}: {msg}")
        if finished:
            self._finished = True
        seg = tokens_to_segment(
            finalized,
            fallback_audio_end_s=self._audio_position_s,
            ready_wall_s=self._ready_wall_s(),
        )
        return [seg] if seg is not None else []
