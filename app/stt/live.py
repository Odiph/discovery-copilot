"""The engine registry and the engine-agnostic stream worker. This is where
`app/config.STT_ENGINE` becomes an actual `StreamingEngine` instance -- the
one place that chain runs through (phase_specification key_links, STT-03).

`StreamWorker` may name no vendor and must work unchanged for any registered
factory: it owns a `StreamingSession` and therefore its socket, from exactly
one thread (S5), drains the capture queue, sends audio, polls for segments,
and hands each finalized segment to the store.

Phase 3 (03-01-PLAN.md §P3): after `append_segment`, a finalized record
optionally crosses the thread -> asyncio boundary via `asyncio.run_coroutine_threadsafe`,
to whatever `broadcast` callable was supplied at construction. Both `loop` and
`broadcast` default to `None`, so every pre-Phase-3 test keeps passing
unchanged. `StreamWorker` must NEVER import anything from `app.web` -- it
takes a plain callable, which is also what makes this bridge unit-testable
without any server at all. The one rule that matters most: `_persist` must
NEVER call `.result()` on the future this produces -- that would block this
real-time STT thread until every connected client's `send()` completes,
coupling STT throughput to a possibly-dead phone (OPS-05). A done-callback
records any broadcast failure as a named store event instead.

Phase 4 (04-01-PLAN.md §C3): `StreamWorker` also gains one optional
`segment_hook`, called synchronously after the unchanged `append_segment` and
after the broadcast bridge above. It defaults to `None`, exactly like `loop`/
`broadcast`, so every pre-Phase-4 test keeps passing unchanged. It takes a
plain callable and this module must NEVER import anything from `app.coach` --
in production the hook IS `CoverageDetector.submit`, which only enqueues and
returns (never blocks, never touches the network), so calling it
synchronously here is safe; a raising hook is caught and logged, never
allowed to kill this thread.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
from typing import Awaitable, Callable, Iterable, Optional

from benchmark.engines.base import Segment, StreamingEngine
from app.capture.stream import CaptureFrame
from app.session.audio_writer import AudioWriter
from app.session.store import SegmentRecord, SessionStore

logger = logging.getLogger(__name__)

EngineFactory = Callable[[], StreamingEngine]

_REGISTRY: dict[str, EngineFactory] = {}


class UnknownEngineError(Exception):
    """Raised for an engine name with no registered factory. The message
    names the available engines so the caller can fix it immediately."""


class EngineNotReadyError(Exception):
    """Raised when a resolved engine cannot open a session -- a missing API
    key, most often -- discovered BEFORE anything is opened or created.

    A separate type from `UnknownEngineError` because the two have different
    fixes: one means the engine NAME is wrong, the other means the environment
    is. Both are the operator's to fix, and neither should ever reach them as
    a traceback.

    Vendor-agnostic on purpose. `preflight_engines` below converts whatever the
    engine raised into this type, which is what lets `app/main.py` catch a
    missing Soniox key without importing a vendor module -- the rule this
    module's own docstring sets out and which `StreamWorker` already keeps.
    """


def register_engine(name: str, factory: EngineFactory) -> None:
    _REGISTRY[name] = factory


def available_engines() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def get_engine(name: str) -> StreamingEngine:
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise UnknownEngineError(
            f"Unknown STT engine '{name}'. Available engines: {sorted(_REGISTRY)}."
        ) from exc
    return factory()


def preflight_engines(engines: "Iterable[StreamingEngine]") -> None:
    """Check every resolved engine is usable, before a device is opened or a
    session directory created. Raises `EngineNotReadyError`; returns `None`.

    **The ordering is the whole point.** `start` used to discover a missing
    `SONIOX_API_KEY` inside the STT worker thread, by which time it had already
    opened both audio devices and written `sessions/<id>/` with metadata. The
    operator got a traceback and an orphaned session directory that -- having
    no `closed_at` -- is byte-for-byte the shape of a recording that crashed
    mid-interview. Two bad outcomes from one missing environment variable, and
    the second one poisons the session corpus permanently.

    **`preflight` is optional and duck-typed, not added to the protocol.** The
    `fake` and `replay` engines need no credentials and define nothing; adding
    an abstract method would have forced two empty implementations and a
    benchmark-side protocol change to fix an `app`-side ordering bug.

    An engine is only checked ONCE even when several speakers resolved their
    own instance: the factories are the same class and the check reads process
    environment, so N calls can only ever produce N copies of one answer.
    """
    seen: set[type] = set()
    for engine in engines:
        preflight = getattr(engine, "preflight", None)
        if preflight is None or type(engine) in seen:
            continue
        seen.add(type(engine))
        try:
            preflight()
        except Exception as exc:  # noqa: BLE001 -- any vendor error becomes one named type
            # `str(exc)`, never the exception object: the message is written to
            # be read by the operator (it names the variable and where to read
            # about it) and chaining keeps the original for the log. The vendor
            # type is deliberately not named in the surfaced text -- it would
            # be the one piece of this the operator cannot act on.
            raise EngineNotReadyError(str(exc)) from exc


def _register_default_engines() -> None:
    from benchmark.engines.soniox import SonioxEngine
    from app.stt.fake import FakeStreamingEngine
    from app.stt.replay import ReplayStreamingEngine

    register_engine("soniox", SonioxEngine)
    register_engine("fake", FakeStreamingEngine)
    register_engine("replay", ReplayStreamingEngine)


_register_default_engines()


class StreamWorker:
    """Owns one `StreamingSession` (and therefore its socket) from exactly
    one thread. Engine-agnostic by construction -- it never imports a vendor
    module and works unchanged for any registered `StreamingEngine`."""

    def __init__(
        self,
        *,
        engine: StreamingEngine,
        frame_queue: "queue.Queue[CaptureFrame]",
        store: SessionStore,
        speaker: str,
        sample_rate: int,
        poll_timeout_s: float = 0.05,
        queue_get_timeout_s: float = 0.2,
        audio_writer: Optional[AudioWriter] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        broadcast: Optional[Callable[..., Awaitable[None]]] = None,
        segment_hook: Optional[Callable[[SegmentRecord], None]] = None,
    ) -> None:
        self._engine = engine
        self._frame_queue = frame_queue
        self._store = store
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._poll_timeout_s = poll_timeout_s
        self._queue_get_timeout_s = queue_get_timeout_s
        # Retention (phase_specification U3): fed the SAME frames the engine
        # consumes, including zero-filled idle frames, so the WAV's timeline
        # matches the transcript's. `None` when retention is off (the
        # default) -- `write()` is simply never called.
        self._audio_writer = audio_writer
        # Phase 3 thread->asyncio bridge (§P3). Both default to `None` so
        # this worker stays fully usable, unit-testable and unaware of any
        # server whenever no live phone surface is being served.
        self._loop = loop
        self._broadcast = broadcast
        # Phase 4 (04-01-PLAN.md §C3): coach-agnostic by construction -- a
        # plain callable, never `app.coach` imported here. `None` for the
        # interviewee side and for every pre-Phase-4 caller.
        self._segment_hook = segment_hook
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._last_audio_end_s = 0.0
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        self._session = self._engine.open_session(sample_rate=self._sample_rate, label=self._speaker)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"stt-{self._speaker}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None

    def _run(self) -> None:
        assert self._session is not None
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self._frame_queue.get(timeout=self._queue_get_timeout_s)
                except queue.Empty:
                    continue
                self._send_and_poll(frame)

            # Drain whatever is left in the queue without blocking, then close.
            while True:
                try:
                    frame = self._frame_queue.get_nowait()
                except queue.Empty:
                    break
                self._send_and_poll(frame)

            for segment in self._session.close():
                self._persist(segment)
        except Exception as exc:  # noqa: BLE001 -- confine this stream's failure to itself;
            # never let one engine's exception kill the process (T2, T-02-14,
            # OPS-05). Recorded as a named event, never swallowed silently.
            self._failed = True
            logger.error(
                "stream worker failed speaker=%s error_type=%s", self._speaker, type(exc).__name__,
            )
            self._store.append_event(
                event="stream-failure",
                speaker=self._speaker,
                detail={"error_type": type(exc).__name__, "message": str(exc)},
            )

    def _send_and_poll(self, frame: CaptureFrame) -> None:
        if frame.pcm16:
            self._session.send_audio(frame.pcm16)
            if self._audio_writer is not None:
                self._audio_writer.write(frame.pcm16)
        for segment in self._session.poll(timeout_s=self._poll_timeout_s):
            self._persist(segment)

    def _persist(self, segment: Segment) -> None:
        # ⚠️ THIS is the fabrication every downstream module routes around.
        # `Segment` (benchmark/engines/base.py) carries `audio_end_s` and no
        # start, so there is nothing real to record and this tiles the field from
        # the same speaker's previous end. It is written because
        # `SegmentRecord` requires a float, NOT because it means anything: the
        # gap it spans is that speaker's silence. Gate 35 fails the build on
        # any read. A real start time has to come from the engine contract
        # first; there is no cleverer sum available downstream.
        audio_start_s = self._last_audio_end_s
        self._last_audio_end_s = segment.audio_end_s
        record = self._store.append_segment(
            speaker=self._speaker,
            text=segment.text,
            audio_start_s=audio_start_s,
            audio_end_s=segment.audio_end_s,
            ready_wall_s=segment.ready_wall_s,
            detected_language=segment.detected_language,
            engine=self._engine.name,
        )
        # Never logs segment.text -- only its length (S7, T-02-05).
        logger.debug(
            "worker persisted segment speaker=%s text_len=%d audio_end_s=%.2f "
            "language=%s engine=%s",
            self._speaker, len(segment.text), segment.audio_end_s,
            segment.detected_language, self._engine.name,
        )
        # Phase 3 bridge (§P3): schedule-and-forget onto the loop thread.
        # NEVER call .result() here -- that would block THIS thread (a
        # real-time STT worker) until a possibly-dead phone's send()
        # completes. The done-callback records failure without ever making
        # this thread wait for it.
        if self._loop is not None and self._broadcast is not None:
            future = asyncio.run_coroutine_threadsafe(self._broadcast(record), self._loop)
            future.add_done_callback(self._log_broadcast_failure)

        # Phase 4 (04-01-PLAN.md §C3): called SYNCHRONOUSLY, on purpose --
        # the hook (in production, `CoverageDetector.submit`) must itself be
        # an enqueue-and-return, never a network call, so this stays fast.
        # Wrapped so a raising hook can never kill this real-time STT thread.
        if self._segment_hook is not None:
            try:
                self._segment_hook(record)
            except Exception as exc:  # noqa: BLE001 -- one hook failure must never take down STT
                logger.error(
                    "segment hook failed speaker=%s error_type=%s", self._speaker, type(exc).__name__,
                )

    def _log_broadcast_failure(self, future: "concurrent.futures.Future") -> None:
        try:
            exc = future.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            # Never logs record.text -- error_type only (S7, OPS-05: a
            # broadcast failure must be visible, never silently swallowed).
            logger.error(
                "broadcast failed speaker=%s error_type=%s", self._speaker, type(exc).__name__,
            )
            self._store.append_event(
                event="broadcast-failure",
                speaker=self._speaker,
                detail={"error_type": type(exc).__name__},
            )
