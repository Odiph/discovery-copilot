"""The three commands: `python -m app start`, `python -m app render`, and
`python -m app precheck`.

`start` resolves both devices (loopback + microphone, unless `--only`
restricts to one), opens the store, launches one capture and one STT worker
PER SIDE, supervises both for the duration of the run, serves the live phone
surface (Phase 3, D-05), shuts down cleanly, and prints where the transcript
was written. A keyboard interrupt closes the session cleanly, not abandons it
(SESS-01). `--retain-audio` (default OFF -- phase_specification U3) writes one
WAV per speaker alongside the transcript, for a REHEARSAL session only.

Phase 3 (03-01-PLAN.md §P2): the asyncio event loop hosting uvicorn runs on
the MAIN thread, via `run_start`'s own top-level `asyncio.run()` -- never
nested inside uvicorn's own `uvicorn.run()`, which calls `asyncio.run()`
internally. Device resolution, store creation, worker/capture construction
and the entire ordered shutdown block are UNCHANGED and in the SAME order as
before this phase; the only structural change to the run loop is
`time.sleep(0.2)` becoming `await asyncio.sleep(0.2)` (03-RESEARCH.md §1).

`render` reads a closed OR still-open session (by id under the sessions root,
or by full path) and writes a readable, speaker-labelled, time-ordered
transcript document into the session directory (SESS-03).

Exactly ONE `pyaudiowpatch.PyAudio()` instance is constructed per invocation
(`construct_pyaudio()` below is the only call site in `app/` outside tests --
phase_specification S13, enforced by a standing source-scan gate in
test_windows_traps.py). Both streams, the keep-alive render stream, and
device re-resolution after a change all share it; nothing ever constructs a
second one.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
import webbrowser
from dataclasses import dataclass, replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pyaudiowpatch as pyaudio

from app import config as config_mod
from app import project_meta
from app import projects as projects_mod
from app.interrupt import ignore_stop_signals, install_stop_handler
from app.control import ControlChannel, MicrophoneState, microphone_state_line
from app.capture.devices import (
    DeviceInfo,
    DeviceOpenError,
    resolve_capture_device_by_name,
    resolve_default_input_device,
    resolve_default_loopback_device,
    resolve_default_render_device,
)
from app.capture.keepalive import KeepAliveStream
from app.capture.precheck import (
    build_precheck_report,
    print_precheck_report,
    print_self_test_report,
    run_self_test,
)
from app.capture.stream import CaptureStream
from app.capture.supervisor import StreamSupervisor
from app.coach import context as coach_context
from app.coach import corpus as coach_corpus
from app.coach import corpus_store as coach_corpus_store
from app.coach import asked as coach_asked
from app.coach import asked_critique as coach_asked_critique
from app.coach import asked_store as coach_asked_store
from app.coach import demotion_store as coach_demotion_store
from app.coach import extract as coach_extract
from app.coach import interview_summary as summary_mod
from app.coach import reasoning_client
from app.coach import rejection_store as coach_rejection_store
from app.coach.context import ContextItem, ContextValidationError, SurveyItem
from app.coach.coverage import CoverageDetector, CoverageState
from app.coach.pipeline import SuggestionPipeline
from app.coach.trigger import TurnTrigger
from app.session.audio_writer import AudioWriter
from app.session.participant import PARTICIPANT_KEY, normalize_participant
from app.session.reader import read_session
from app.session.render import write_rendered_transcript
from app.session.feed_watch import FeedWatch, FeedWatchConfig
from app.session.store import SegmentRecord, SessionStore, create_session_dir
from app.stt.live import (
    EngineNotReadyError,
    StreamWorker,
    UnknownEngineError,
    get_engine,
    preflight_engines,
)
from app.web import author as web_author
from app.web import console as web_console
from app.web import review as web_review
from app.web import server as web_server
from app.web.hub import ConnectionHub, survey_payload
from app.web.survey_hold import SurveyDisplayHold
from app.web.join import create_join
from app.web.status import StatusBoard, StreamHealth, derive_status

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
# 04-03-PLAN.md §E4: the authoring page's OWN static subdirectory, beside the
# phone surface's four named assets -- adding it changes nothing about what
# the phone app's asset loader reads (it names its four files explicitly).
STATIC_AUTHOR_DIR = STATIC_DIR / "author"
# Phase B: the console's own static subdirectory, a sibling of the
# authoring page's -- same reasoning, and it changes nothing about what
# either of the other two asset loaders reads.
STATIC_CONSOLE_DIR = STATIC_DIR / "console"
STATIC_REVIEW_DIR = STATIC_DIR / "review"

# What `--no-coach` says, and the reason it is a named constant rather than a
# literal at the one call site: it must stay DISTINGUISHABLE from
# "Coach disabled: ANTHROPIC_API_KEY ...", which an existing test greps for.
# The two mean different things -- "I turned it off" versus "your key is
# missing" -- and have different fixes.
NO_COACH_NOTICE = (
    "Coach off (--no-coach): recording only. No brief or survey is loaded, and no "
    "next question or suggestion will be produced."
)

SPEAKER_INTERVIEWEE = "interviewee"
SPEAKER_INTERVIEWER = "interviewer"
ALL_SPEAKERS = (SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER)


def _fan_out(*hooks: "Optional[Callable[[SegmentRecord], None]]"):
    """One `segment_hook` that feeds several consumers, isolating each.

    Returns `None` when nothing is wired, so `StreamWorker`'s pre-Phase-4
    default path is untouched, and the single hook itself when only one is --
    a wrapper around one consumer would only add a frame to every segment on
    the real-time STT thread.

    Each consumer is called inside its own guard so that one raising cannot
    silently cost the other its segment. `StreamWorker` wraps the hook too,
    but that outer guard would stop at the FIRST failure and drop the rest,
    which is exactly the coverage-loses-a-turn bug this shape prevents."""
    live = [hook for hook in hooks if hook is not None]
    if not live:
        return None
    if len(live) == 1:
        return live[0]

    def call(record: "SegmentRecord") -> None:
        for hook in live:
            try:
                hook(record)
            except Exception as exc:  # noqa: BLE001 -- never kill the STT thread
                logger.error(
                    "segment hook consumer failed error_type=%s", type(exc).__name__
                )

    return call


async def _cancel_and_await(task: "asyncio.Task", *, task_name: str) -> None:
    """Cancels `task` and awaits it, absorbing whatever it raises -- never
    just `asyncio.CancelledError` (independent review finding C).

    Called from `_async_main`'s `finally` block, BEFORE `pair.supervisor.
    stop()`, `pa.terminate()`, the embedded server's own stop, and
    `store.close()`. If the awaited task had already failed with anything
    OTHER than `CancelledError` (e.g. a bug in `_status_poll_loop`), a bare
    `await task` re-raises that exception right there -- aborting the
    `finally` block and skipping every ordered-shutdown step listed above,
    on the one code path whose entire purpose is a clean close (SESS-01).
    Degrading and continuing here, the same discipline this codebase already
    applies to `app/coach/coverage.py`'s worker thread, is what keeps a
    future poll-loop bug from also costing an unflushed session file and
    dangling capture threads."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001 -- see docstring: must never abort ordered shutdown
        logger.warning("%s failed during shutdown error_type=%s", task_name, type(exc).__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Establishes the logging discipline once, for the whole application:
    every call site logs counts, byte lengths, durations, sequence numbers,
    language tags, speaker labels and event names -- never segment text
    (phase_specification S7, finished in plan 02-01 Task 3)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def construct_pyaudio() -> "pyaudio.PyAudio":
    """The ONE place `pyaudiowpatch.PyAudio()` is constructed in this
    application outside tests (phase_specification S13). Constructing a
    second instance concurrently from another thread crashes the process
    with an unrecoverable Windows access violation -- discovered in plan
    02-01's live verification. Called once per CLI invocation, on the main
    thread, before any capture worker starts; both streams, the keep-alive
    stream, and device re-resolution after a change all reuse this same
    instance -- reopen the *stream*, never the API. See
    test_windows_traps.py's source-scan gate, which asserts this is the
    only construction site in `app/` outside tests."""
    return pyaudio.PyAudio()


async def _status_poll_loop(
    pairs: "list[StreamPair]",
    board: StatusBoard,
    hub: ConnectionHub,
    interval_s: float,
    *,
    coach_health_provider: "Optional[Callable[[], object]]" = None,
    coach_failure_threshold: int = config_mod.DEFAULT_COACH_FAILURE_THRESHOLD,
    feed_watch: "Optional[FeedWatch]" = None,
    elapsed_provider: "Optional[Callable[[], float]]" = None,
    store: "Optional[SessionStore]" = None,
) -> None:
    """Reads each stream pair's ALREADY-public `worker.failed`/`capture.is_dead`
    state on an interval and pushes a `server_status` change to every
    connected client -- only when the derived status differs from the last
    one broadcast (03-02-PLAN.md Task 2 §Q6/§Q7). Adds no new callback into
    `app/capture/supervisor.py` or `app/stt/live.py`; this is polling, not a
    subscription.

    04-03-PLAN.md §E5: `coach_health_provider`, when given, is polled on the
    SAME interval and folded into the SAME `derive_status()` call -- one more
    input to the mechanism Phase 3 already built, never a second poller or a
    second message type. `None` (the default, and every pre-Phase-4 caller)
    means "no coach running" -- never treated as unhealthy."""
    while True:
        # The feed watch, polled on the SAME interval and folded into the SAME
        # `derive_status()` call -- the shape `coach_health_provider` already
        # established, and for the identical reason: a second poller and a
        # second message type would be a second opinion about the banner.
        if feed_watch is not None and elapsed_provider is not None:
            # The level probe, read from the capture streams that already
            # measure it. It answers the one question the transcript cannot:
            # a side producing no segments while carrying real sound is a
            # fault, while a side producing no segments while carrying silence
            # is just a room nobody is speaking in. `None` for a side means
            # not measured, and `feed_watch` reports rather than excuses it.
            levels = {
                pair.speaker: pair.capture.recent_peak_dbfs() for pair in pairs
            }
            for alert in feed_watch.poll(elapsed_s=elapsed_provider(), levels=levels):
                logger.warning(
                    "feed %s %s -- %.0fs without a transcribed segment",
                    alert.speaker, alert.kind, alert.since_s,
                )
                if store is not None:
                    store.append_event(
                        speaker=alert.speaker,
                        event=f"feed-{alert.kind}",
                        detail={"since_s": round(alert.since_s, 1)},
                    )
        healths = [
            StreamHealth(
                speaker=pair.speaker,
                worker_failed=pair.worker.failed,
                capture_dead=pair.capture.is_dead,
                # Read from the supervisor that already computes them,
                # debounced, on its own 0.5s cadence. This poller ran beside
                # it for three phases without ever asking, which is why a
                # muted interviewee showed a green banner: the system knew
                # and had no way to say so.
                idle=pair.supervisor.is_idle,
                lagging=pair.supervisor.is_lagging,
                not_transcribing=(
                    feed_watch.is_silent(pair.speaker) if feed_watch is not None else False
                ),
            )
            for pair in pairs
        ]
        coach_unhealthy = False
        if coach_health_provider is not None:
            coach_unhealthy = coach_health_provider().is_unhealthy(threshold=coach_failure_threshold)
        status = derive_status(healths, coach_unhealthy=coach_unhealthy)
        if board.observe(status):
            await hub.broadcast_status(status)
        await asyncio.sleep(interval_s)


def _suggestion_broadcast_callback(
    loop: asyncio.AbstractEventLoop,
    hub: ConnectionHub,
    store: SessionStore,
    *,
    max_shown: int = config_mod.DEFAULT_COACH_MAX_SHOWN,
) -> "Callable[[list], None]":
    """Builds the `on_ranked` callback `SuggestionPipeline` invokes from its
    OWN worker thread (04-03-PLAN.md §E1 "wire the callback"). Schedule-and-
    forget onto the loop thread via `run_coroutine_threadsafe`, with a
    done-callback that records a failure -- the SAME thread->asyncio bridge
    shape `StreamWorker._persist` already uses for the segment broadcast
    (app/stt/live.py), never `.result()`, which would couple the coach's own
    worker thread to a possibly-dead phone's `send()` completing.

    `max_shown` (independent review finding E) is the SAME `cfg.coach_max_shown`
    value already passed to `critic.rank_candidates()` -- threaded through to
    `hub.broadcast_suggestion()` so the wire's alternatives cap can never
    drift from the knob that already governs how many candidates survive
    ranking."""

    def _on_ranked(ranked: "list") -> None:
        if not ranked:
            # Nothing survived ranking this cycle (D-05/COACH-06): the
            # region keeps its last content rather than being blanked by an
            # empty broadcast -- a healthy pipeline must never look dead.
            return
        future = asyncio.run_coroutine_threadsafe(
            hub.broadcast_suggestion(ranked, max_shown=max_shown), loop
        )

        def _log_failure(completed: "concurrent.futures.Future") -> None:
            try:
                exc = completed.exception()
            except concurrent.futures.CancelledError:
                return
            if exc is not None:
                logger.error("suggestion broadcast failed error_type=%s", type(exc).__name__)
                store.append_event(
                    event="broadcast-failure",
                    speaker=SPEAKER_INTERVIEWEE,
                    detail={"error_type": type(exc).__name__, "stage": "suggestion"},
                )

        future.add_done_callback(_log_failure)

    return _on_ranked


def _schedule_survey_broadcast(
    loop: asyncio.AbstractEventLoop,
    hub: ConnectionHub,
    store: SessionStore,
    survey: "Sequence[SurveyItem]",
    asked: "frozenset[str]",
) -> None:
    """The schedule-and-forget thread->asyncio bridge both survey callbacks
    below share -- never `.result()`, which would couple a worker thread to a
    sleeping phone's `send()` completing."""
    future = asyncio.run_coroutine_threadsafe(
        hub.broadcast_survey(survey, asked=asked), loop
    )

    def _log_failure(completed: "concurrent.futures.Future") -> None:
        try:
            exc = completed.exception()
        except concurrent.futures.CancelledError:
            return
        if exc is not None:
            logger.error("survey broadcast failed error_type=%s", type(exc).__name__)
            store.append_event(
                event="broadcast-failure",
                speaker=SPEAKER_INTERVIEWER,
                detail={"error_type": type(exc).__name__, "stage": "survey"},
            )

    future.add_done_callback(_log_failure)


def _survey_broadcast_callback(
    loop: asyncio.AbstractEventLoop,
    hub: ConnectionHub,
    store: SessionStore,
    survey: "Sequence[SurveyItem]",
    hold_provider: "Callable[[], Optional[SurveyDisplayHold]]",
) -> "Callable[[CoverageState], None]":
    """Builds the `on_change` callback `CoverageDetector` invokes from its OWN
    worker thread when a survey question moves to `asked`.

    **This no longer broadcasts on every change, and that is the point.** The
    coverage change itself fires the instant the INTERVIEWER's own utterance
    matches -- while they are still speaking -- so broadcasting here moved the
    question out from under them mid-read (reported 2026-08-15, the first real
    coached interview). `SurveyDisplayHold` decides whether the panel may
    advance yet; `None` from it means "the panel is already correct, send
    nothing". See `app/web/survey_hold.py` for the three ways a hold releases.

    Coverage STATE is untouched by any of this -- only the display lags.

    `hold_provider` is a callable rather than the hold itself because the hold
    is constructed from the detector's own replayed state, which does not
    exist until the detector does. It is resolved on first use, long after
    both exist.

    Unlike the suggestion callback there is no empty-set early return. An
    exhausted survey is a state the panel must SHOW ("no questions left"),
    not a reason to leave the last question on screen -- see
    `hub.survey_payload()`'s own docstring."""

    def _on_change(state: "CoverageState") -> None:
        hold = hold_provider()
        if hold is None:
            _schedule_survey_broadcast(loop, hub, store, survey, state.asked)
            return
        to_show = hold.note_coverage_change(state.asked)
        if to_show is None:
            return
        _schedule_survey_broadcast(loop, hub, store, survey, to_show)

    return _on_change


def _survey_hold_release_hook(
    loop: asyncio.AbstractEventLoop,
    hub: ConnectionHub,
    store: SessionStore,
    survey: "Sequence[SurveyItem]",
    hold: "SurveyDisplayHold",
) -> "Callable[[SegmentRecord], None]":
    """The other half of the hold: a finalized segment is what RELEASES a held
    question, so the panel advances when the interviewee answers rather than
    when the interviewer asks.

    Wired onto BOTH streams. The interviewee's segment is the release that
    matters; the interviewer's exists only so a hold nobody ever answers can
    expire (the timeout is evaluated lazily, on the next event -- there is no
    ticker on the server, see `survey_hold`'s own module docstring).

    Runs on the STT stream thread, so it does exactly what
    `coverage_detector.submit` does there: a little pure state, then a
    schedule-and-forget broadcast. No network, no blocking."""

    def _on_segment(record: SegmentRecord) -> None:
        to_show = hold.note_segment(speaker=record.speaker)
        if to_show is None:
            return
        _schedule_survey_broadcast(loop, hub, store, survey, to_show)

    return _on_segment


@dataclass
class StreamPair:
    speaker: str
    device: DeviceInfo
    capture: CaptureStream
    worker: StreamWorker
    supervisor: StreamSupervisor
    audio_writer: Optional[AudioWriter] = None


def _apply_project(cfg: "config_mod.AppConfig", args: argparse.Namespace) -> "config_mod.AppConfig":
    """Points the config's two roots at one project node, or returns it
    untouched when no `--project` was given.

    This is the whole of "selecting a project" on the read side, and it is
    deliberately this small. Nothing under `app/coach/`, `app/session/` or
    `app/web/` reads config internally -- every loader and writer already
    takes its directory as a parameter, and `cfg.context_dir` /
    `cfg.sessions_root` are named only in this module. So substituting the
    two roots in the frozen config reaches every consumer at once, and no
    call site below needs to know a project exists.

    Returns a NEW config rather than mutating one: `AppConfig` is frozen, and
    a project selection that could be applied twice, or partially, is exactly
    the kind of state a long-lived console process would get wrong.

    Not applied to `precheck` -- see `_add_project_argument`."""
    project = getattr(args, "project", None)
    if project is None:
        return cfg
    paths = projects_mod.resolve(project)
    return dataclass_replace(
        cfg, context_dir=paths.context_dir, sessions_root=paths.sessions_root
    )


def _resolve_extra_fact_dirs(args: argparse.Namespace) -> "tuple[tuple[str, Path], ...]":
    """Turns `--read-from` into the `(source_node, context_dir)` pairs
    `load_grounded_brief` takes, enforcing the user's rule: cross-project
    reads only WITHIN the same hierarchy, and only on explicit request.

    Every rejection here is a raise, never a skip. A source node quietly
    dropped would mean the coach ran grounded in less than the operator
    selected while reporting success -- and the operator would attribute the
    coach's thinner suggestions to the model."""
    requested = getattr(args, "read_from", None) or ()
    if not requested:
        return ()

    project = getattr(args, "project", None)
    if project is None:
        raise projects_mod.ProjectPathError(
            "--read-from needs --project: without a filing node there is no hierarchy to "
            "read within, and the root context/ directory is not part of the project tree"
        )

    filing = projects_mod.resolve(project)
    resolved: "list[tuple[str, Path]]" = []
    seen: "set[str]" = set()
    for candidate in requested:
        paths = projects_mod.resolve(candidate)
        if paths.root != filing.root:
            raise projects_mod.ProjectPathError(
                f"--read-from {candidate!r} is under a different root ({paths.root!r}) than "
                f"--project {project!r} ({filing.root!r}) -- knowledge may be shared only "
                f"within one hierarchy"
            )
        if paths.path == filing.path:
            raise projects_mod.ProjectPathError(
                f"--read-from {candidate!r} is the filing node itself, which is always read; "
                f"naming it again would double-count its facts"
            )
        if paths.path in seen:
            raise projects_mod.ProjectPathError(
                f"--read-from {candidate!r} was given twice -- a repeated source would put the "
                f"same facts into the brief twice"
            )
        seen.add(paths.path)
        resolved.append((paths.path, paths.context_dir))
    return tuple(resolved)


def _add_project_argument(subparser: argparse.ArgumentParser) -> None:
    """Adds `--project` to one subcommand.

    Deliberately NOT added to `precheck`: it touches neither root, so a
    project would be an argument it silently ignored -- worse than absent,
    because an operator who passed it would reasonably believe the check ran
    against that project's devices.

    Default `None` means "unchanged", which is what keeps every existing
    invocation, `record-test-session.bat`, and the gates that read
    `REPO_ROOT / "context"` working exactly as before."""
    subparser.add_argument(
        "--project",
        default=None,
        metavar="PATH",
        help=(
            "Project to file this under, as a nested label path "
            "(e.g. acme/onboarding/activation-delay). Selects that node's own "
            "context/ and sessions/ under projects/. Omit to use the root "
            "context/ and sessions/ directories."
        ),
    )


def _participant_arg(value: str) -> "Optional[str]":
    """`--participant`'s own validator, run by argparse at PARSE time.

    Deliberately validated here rather than deep inside `_async_main`, and the
    difference is the whole point: by the time the session metadata is written,
    the engines have been pre-flighted, both audio devices are open and a
    session directory exists on disk. Refusing a 500-character label at that
    point would mean tearing all of it back down over a typo, and the operator
    is one minute from a call. Argparse raising here costs nothing and happens
    before anything is opened.

    Returns the NORMALIZED label, so what argparse hands `_async_main` is
    already exactly what will land in `session.json` -- there is no second
    place where trimming could be forgotten."""
    label, error = normalize_participant(value)
    if error is not None:
        raise argparse.ArgumentTypeError(error)
    # An empty or whitespace-only `--participant ""` normalizes to `None`,
    # which means the same thing as not passing the flag at all: nobody named.
    # Returning it rather than refusing keeps a shell script that interpolates
    # a possibly-empty variable from failing over a value it did not have.
    return label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start a capture + live-transcription session")
    start.add_argument(
        "--engine", default=None, help="Engine name override (beats STT_ENGINE, beats the default)"
    )
    start.add_argument(
        "--seconds", type=float, default=None, help="Stop automatically after N seconds"
    )
    start.add_argument(
        "--only",
        choices=ALL_SPEAKERS,
        default=None,
        help=(
            "Run only one side (interviewee=loopback, interviewer=microphone) -- "
            "for the trap tests, or for a user debugging one half of a broken setup"
        ),
    )
    start.add_argument(
        "--retain-audio",
        action="store_true",
        default=config_mod.DEFAULT_RETAIN_AUDIO,
        help=(
            "Retain per-stream WAVs. Use for REHEARSAL recordings only -- "
            "retained audio is intended as benchmark input, and the "
            "benchmark uploads to third-party vendors."
        ),
    )
    start.add_argument(
        "--participant",
        type=_participant_arg,
        default=None,
        metavar="LABEL",
        help=(
            "Who is being interviewed -- a name, a role or a pseudonym. Optional, "
            "and omitting it is normal: the label is recorded in the session's "
            "session.json and can also be filled in afterwards on the hub. It "
            "exists because per-interview counting counts SESSIONS, which is the "
            "same as counting people only while nobody is re-interviewed."
        ),
    )
    start.add_argument(
        "--interviewee-device",
        default=None,
        metavar="NAME",
        help=(
            "Pin the interviewee side to a named system-sound (loopback) device "
            "instead of following the Windows default output. Matched "
            "case-insensitively against the device name, and REFUSES rather than "
            "guessing if the name matches none or several. Use this when the "
            "default output renegotiates mid-call -- a Bluetooth headset moves "
            "playback from its stereo endpoint to its hands-free endpoint the "
            "moment a call app opens the microphone, and the stereo endpoint it "
            "leaves behind captures pure silence."
        ),
    )
    start.add_argument(
        "--interviewer-device",
        default=None,
        metavar="NAME",
        help=(
            "Pin the interviewer side to a named microphone (input) device "
            "instead of following the Windows default input. Same matching and "
            "same refusal-to-guess as --interviewee-device."
        ),
    )
    start.add_argument(
        "--require-live-audio",
        action="store_true",
        default=False,
        help=(
            "Refuse to start unless the self-test measures real sound on every "
            "side, instead of warning and continuing. For a rig that must not "
            "fail quietly -- a recorded demo, a one-shot interview -- where a "
            "false stop costs less than a silent take."
        ),
    )
    start.add_argument(
        "--skip-self-test",
        action="store_true",
        default=False,
        help=(
            "Record without first checking that both sides carry sound. The "
            "self-test costs about two seconds and is what stops an interview "
            "being recorded against a silent channel; skip it for rehearsals, "
            "replays and automated runs, where there is no live audio to measure."
        ),
    )
    start.add_argument(
        "--no-coach",
        action="store_true",
        default=False,
        help=(
            "Record with no coach at all -- no brief or survey is loaded, no next "
            "question is derived and no suggestion is generated. For an interview "
            "filed under a project that does not exist yet (projects/_unfiled), "
            "where there is nothing to ground a coach in."
        ),
    )
    start.add_argument(
        "--no-serve",
        action="store_true",
        default=False,
        help=(
            "Skip the web layer entirely -- no port bound, no browser opened, no QR "
            "produced. For the single-side trap tests, which need neither."
        ),
    )

    _add_project_argument(start)
    start.add_argument(
        "--read-from",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "Additionally ground the coach in another project node's facts and "
            "hypotheses, read-only. Repeatable. Must share a root with --project: "
            "knowledge is shared only within one hierarchy. The brief and survey "
            "always come from --project alone, and nothing is ever written here."
        ),
    )

    render = subparsers.add_parser(
        "render", help="Render a session's transcript into a readable document"
    )
    render.add_argument(
        "session", help="Session id (a name under the sessions root) or a full path to a session directory"
    )
    _add_project_argument(render)

    precheck = subparsers.add_parser(
        "precheck", help="Go/no-go check: both streams carry real audio, the engine answers"
    )
    precheck.add_argument("--seconds", type=float, default=None, help="Liveness window, in seconds")
    precheck.add_argument(
        "--skip-stt", action="store_true", help="Skip the engine-reachability row (reported as SKIPPED)"
    )
    precheck.add_argument("--engine", default=None, help="Engine name override for the reachability row")
    precheck.add_argument(
        "--interviewee-device", default=None, metavar="NAME",
        help="Check the same pinned loopback device `start --interviewee-device` would open.",
    )
    precheck.add_argument(
        "--interviewer-device", default=None, metavar="NAME",
        help="Check the same pinned input device `start --interviewer-device` would open.",
    )
    precheck.add_argument(
        "--phone",
        action="store_true",
        default=False,
        help=(
            "Add a bounded real-phone reachability row: starts the server, shows the "
            "QR, and waits for a real connection before reporting go/no-go. Off by "
            "default so the everyday precheck stays fast -- run it before a real call."
        ),
    )

    author = subparsers.add_parser(
        "author",
        help=(
            "Author the brief, survey and hypotheses before a call -- opens a "
            "loopback-only browser page that saves to context/brief.json, "
            "context/survey.json and context/hypotheses.json"
        ),
    )
    _add_project_argument(author)

    hub = subparsers.add_parser(
        "hub",
        help=(
            "Review a finished interview -- where each hypothesis stands, the "
            "evidence for and against it, and the transcript"
        ),
    )
    hub.add_argument(
        "session", nargs="?", default=None,
        help=(
            "Session id to open first. Optional -- the page is scoped to the "
            "project and its picker selects the newest interview when this is "
            "omitted"
        ),
    )
    _add_project_argument(hub)

    subparsers.add_parser(
        "overlay",
        help=(
            "Open the floating record overlay -- translucent, always on top, "
            "picks a project and runs the interview"
        ),
    )

    subparsers.add_parser(
        "console",
        help=(
            "Pick a project and start the interview -- opens a loopback-only "
            "browser page showing the project tree, each node's survey, "
            "hypotheses, fact count and past interviews"
        ),
    )

    extract = subparsers.add_parser(
        "extract",
        help=(
            "Extract cited, provenance-tagged facts from a finished session into "
            "the fact corpus (context/facts.json)"
        ),
    )
    extract.add_argument(
        "session", help="Session id (a name under the sessions root) or a full path to a session directory"
    )
    _add_project_argument(extract)

    critique = subparsers.add_parser(
        "critique-questions",
        help=(
            "Band the INTERVIEWER's own questions from a finished session against "
            "the same Mom Test rubric the coach is held to "
            "(writes questions_critique.json into the session directory)"
        ),
    )
    critique.add_argument(
        "session", help="Session id (a name under the sessions root) or a full path to a session directory"
    )
    critique.add_argument(
        "--min-words", type=int, default=coach_asked.DEFAULT_MIN_WORDS,
        help="floor for a turn to be worth judging; a turn ending in '?' is kept "
             "however short (default: %(default)s)",
    )
    _add_project_argument(critique)

    # `summary` and `standing` read what extraction already wrote and print it.
    # Neither makes a model call, and neither may grow one: every number they
    # show is arithmetic over `context/facts.json`, so they cost nothing, cannot
    # fail on the network, and give the same answer twice.
    summary = subparsers.add_parser(
        "summary",
        help=(
            "What one interview produced: how many claims, how many were "
            "reported versus inferred, and which hypotheses they bore on "
            "(reads the corpus; no model call)"
        ),
    )
    summary.add_argument(
        "session", help="Session id (a name under the sessions root) or a full path to a session directory"
    )
    _add_project_argument(summary)

    standing = subparsers.add_parser(
        "standing",
        help=(
            "Where every hypothesis stands across all extracted interviews; "
            "name a session to also see how that one interview moved each "
            "(reads the corpus; no model call)"
        ),
    )
    standing.add_argument(
        "session", nargs="?", default=None,
        help="Optional session id (or full path). Given, each hypothesis also "
             "shows what it stood at before that interview and what the "
             "interview contributed",
    )
    _add_project_argument(standing)

    export = subparsers.add_parser(
        "export",
        help=(
            "Write a verifiable copy of a project's irreplaceable data (transcripts, "
            "facts, hypotheses, pre-registration boundaries) to a zip"
        ),
    )
    export.add_argument("--to", required=True, type=Path,
                        help="destination .zip; refused if git can see it")
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--verify", action="store_true",
                        help="re-check every checksum after writing -- a backup nobody "
                             "has ever read is a hope, not a backup")
    _add_project_argument(export)

    return parser


def _device_pin(args, speaker: str) -> "Optional[str]":
    """The pinned capture-device name for one side -- the CLI flag if one was
    typed, otherwise `APP_INTERVIEWEE_DEVICE` / `APP_INTERVIEWER_DEVICE`.

    The environment fallback exists for the OVERLAY, which spawns `start`
    itself (`app/overlay/session.py`) and has no device panel yet: without it
    an operator recording from the overlay has no way to pin a side at all,
    and reverts to the Windows default -- which is exactly the endpoint that
    goes silent when a call app renegotiates a Bluetooth headset.

    The flag wins over the environment on purpose. A name typed for THIS
    interview outranks one left in a shell from some earlier one, and the
    asymmetry matches `resolve_capture_device_by_name`'s own posture: the
    operator's most recent instruction is the one that must not be
    second-guessed."""
    flag = getattr(args, f"{speaker}_device", None)
    if flag:
        return flag
    return os.environ.get(f"APP_{speaker.upper()}_DEVICE") or None


def _side_resolver(speaker: str, *, pin: "Optional[str]") -> "Callable[[Any], DeviceInfo]":
    """How ONE side's capture device is chosen -- the only place in this module
    where a default resolver may be named.

    Three separate seams used to choose a device independently: session start,
    the supervisor's re-resolution after a device change, and the lazy
    microphone build for a run that began `--only interviewee`. Each named a
    default resolver directly, so a `--interviewee-device` pin honoured at the
    first would have been discarded by the other two at exactly the moment it
    mattered -- Windows renegotiating a Bluetooth endpoint when a call app
    opens the microphone is a device change, and reverting to "the default"
    there means reverting to the endpoint that has just gone silent.

    Returning a callable rather than a device is what lets all three share one
    decision: seam one calls it once, seams two and three call it again later,
    and a pin cannot be honoured by some of them.

    `test_capture_device_pin_seams.py` pins this structurally by asserting no
    other function in this module names a default resolver."""
    is_loopback = speaker == SPEAKER_INTERVIEWEE
    if pin:
        return lambda pa: resolve_capture_device_by_name(
            pyaudio, pa, pin, is_loopback=is_loopback
        )
    default = resolve_default_loopback_device if is_loopback else resolve_default_input_device
    return lambda pa: default(pa)


def _resolve_devices(
    pa,
    speakers: Sequence[str],
    *,
    resolvers: "Mapping[str, Callable[[Any], DeviceInfo]]",
) -> dict[str, DeviceInfo]:
    """Resolves each requested side through the SAME device module the
    interview uses. A missing side raises a `DeviceOpenError` whose message
    already names which side is missing (phase_specification T1) -- neither
    side is ever silently substituted for the other.

    `resolvers` has NO DEFAULT on purpose. A compatibility default here would
    let a caller that has not thought about pinning fall back to the built-in
    behaviour while reading as though it had -- which is the same shape as the
    seams this parameter exists to close."""
    devices: dict[str, DeviceInfo] = {}
    for speaker in (SPEAKER_INTERVIEWEE, SPEAKER_INTERVIEWER):
        if speaker in speakers:
            devices[speaker] = resolvers[speaker](pa)
    return devices


def _session_metadata(
    devices: dict[str, DeviceInfo],
    *,
    engine_name: str,
    sample_rate: int,
    participant: Optional[str] = None,
) -> dict:
    """Session provenance (S6): both device names and both native rates AS
    REPORTED by the driver at open time -- the two rates being different is
    normal and is never reconciled against each other (T1).

    `participant` is WHO was interviewed, and the key is OMITTED entirely when
    there is none rather than written as `null`. The product's central counting
    rule is participants-by-session and nothing recorded the participant, so
    this is the field that makes the rule checkable -- but the overwhelming
    majority of sessions will be recorded without one, and a stored `null`
    would assert a difference between "nobody said" and "somebody said nobody"
    that no byte on disk supports. See `app/session/participant.py`."""
    metadata: dict = {"engine": engine_name, "target_sample_rate": sample_rate}
    for speaker, device in devices.items():
        metadata[f"{speaker}_device"] = device.name
        metadata[f"{speaker}_native_sample_rate"] = device.native_sample_rate
    if participant is not None:
        metadata[PARTICIPANT_KEY] = participant
    return metadata


def _load_coach_context(
    context_dir: Path,
    *,
    extra_fact_dirs: "Sequence[tuple[str, Path]]",
) -> "tuple[tuple[ContextItem, ...], tuple[SurveyItem, ...], tuple[coach_corpus.Hypothesis, ...]]":
    """Loads the brief, survey and hypotheses together (04-01-PLAN.md §C6,
    extended by 06-01-PLAN.md §F4) so a failure on any of them fails loudly,
    before any device is opened and before any session directory is created.
    Raises `ContextValidationError` (brief/survey) or
    `coach_corpus.CorpusValidationError` (a malformed hypotheses.json --
    `missing_ok=True`, so an ABSENT one is tolerated, D-02) naming the
    offending file/item -- the caller decides what to do with it.

    The third return element is NOT the D-03 seam -- `_load_coach_context` is
    a private helper local to THIS module, and `load_brief` itself (the
    actual seam) is untouched, still `(context_dir) ->
    tuple[ContextItem, ...]`. Loading hypotheses here, alongside the brief
    and survey this helper already reads together, is what lets a later move
    (06-02, not this plan) use them on the live path without a second parse
    of `hypotheses.json`."""
    brief = coach_context.load_grounded_brief(
        context_dir, extra_fact_dirs=extra_fact_dirs
    )
    survey = coach_context.load_survey(context_dir)
    hypotheses = coach_corpus.load_hypotheses(context_dir, missing_ok=True)
    return brief, survey, hypotheses


def _previous_session_started_at(sessions_root: Path) -> Optional[datetime]:
    """The most recent `started_at` among existing session directories, or
    `None` if there is no prior session -- used only to decide whether the
    brief/survey are STALE (§C6), never to decide whether they are valid."""
    if not sessions_root.is_dir():
        return None
    latest: Optional[datetime] = None
    for entry in sessions_root.iterdir():
        metadata_path = entry / "session.json"
        if not entry.is_dir() or not metadata_path.is_file():
            continue
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            started_at = datetime.fromisoformat(data["started_at"])
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
        if latest is None or started_at > latest:
            latest = started_at
    return latest


def _warn_if_context_stale(context_dir: Path, previous_started_at: Optional[datetime]) -> None:
    """A loud warning (not a failure -- only the user can tell 'deliberately
    reused' from 'forgotten edit', §C6) when the brief or survey predates
    the previous session's start."""
    if previous_started_at is None:
        return
    for filename in (coach_context.BRIEF_FILENAME, coach_context.SURVEY_FILENAME):
        path = context_dir / filename
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < previous_started_at:
            print(
                f"WARNING: {path} was last modified before the previous session started -- "
                "confirm this isn't a stale brief/survey left over from a previous customer.",
                file=sys.stderr,
            )


async def _async_main(args: argparse.Namespace) -> int:
    cfg = _apply_project(config_mod.load_config(engine_override=args.engine), args)
    extra_fact_dirs = _resolve_extra_fact_dirs(args)
    speakers: tuple[str, ...] = (args.only,) if getattr(args, "only", None) else ALL_SPEAKERS
    # OPS-01's one escape hatch: serving is not a mode, it is what `start` does
    # -- `--no-serve` is the one flag that skips it entirely, for the
    # single-side trap tests, which need no browser, no QR and no socket.
    no_serve: bool = getattr(args, "no_serve", False)

    if not no_serve:
        # Fail loudly, before any device/session setup, if the pinned port is
        # already taken -- never silently bind elsewhere, which would
        # desynchronize the printed/QR'd join URL from what is actually
        # listening (03-RESEARCH.md §4, OPS-05).
        try:
            web_server.check_port_available(cfg.app_port)
        except web_server.PortInUseError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print("Serving disabled (--no-serve): no port bound, no browser opened, no QR produced.")

    # Phase 3 (03-01-PLAN.md §P2/§P3): moved up from where these three were
    # originally constructed (right before the join surface, below) so that
    # the coach's suggestion-broadcast callback (04-03-PLAN.md §E1's "wire
    # the callback" step) can be attached at `SuggestionPipeline` construction
    # time instead of needing a setter this plan's own `<files>` list does
    # not authorize adding to `app/coach/pipeline.py`. `asyncio.get_running_loop()`
    # is valid anywhere inside this coroutine, so moving it earlier changes
    # nothing about what it returns; `ConnectionHub()`/`StatusBoard()` have no
    # construction-order dependencies on anything below either.
    loop: Optional[asyncio.AbstractEventLoop] = None if no_serve else asyncio.get_running_loop()
    hub: Optional[ConnectionHub] = None if no_serve else ConnectionHub()
    status_board: Optional[StatusBoard] = None if no_serve else StatusBoard()

    # Phase 4 (04-01-PLAN.md §C4/§C6): resolved before any device is opened or
    # session directory created, exactly like the port check above. No key ->
    # the coach is disabled and `start` proceeds anyway (SURF-06's documented
    # degrade path -- refusing to run would be a worse failure than the one
    # this prevents). A key WITH a missing/malformed brief or survey fails
    # loudly here, before anything else is touched.
    coach_client = None
    coach_brief: "tuple[ContextItem, ...]" = ()
    coach_survey: "tuple[SurveyItem, ...]" = ()
    coach_hypotheses: "tuple[coach_corpus.Hypothesis, ...]" = ()
    coach_enabled = False
    # 06-01-PLAN.md §F4: measured around the CALL SITE, not inside the loader
    # -- `_load_coach_context`'s (and `load_brief`'s) own signature and return
    # type are untouched. This is retrieval's own session-start cost, closing
    # the gap `pipeline.py:268-277`'s per-cycle measurement structurally
    # cannot see -- see `SuggestionPipeline.__init__`'s own docstring.
    coach_context_load_s = 0.0
    if getattr(args, "no_coach", False):
        # An EXPLICIT flag, branching BEFORE `build_client()`, and deliberately
        # not reached by tolerating a missing `context/`: a standing test
        # (test_coach_context.py) requires `start` to fail loudly and create NO
        # session directory when the brief or survey cannot load, and that rule
        # protects a real interview from running ungrounded by accident. This
        # is the other case -- an interview with nothing to be grounded IN,
        # chosen on purpose.
        #
        # The wording is deliberately unlike "Coach disabled: ANTHROPIC_API_KEY
        # ...", which an existing test greps for and which means a DIFFERENT
        # thing with a different fix: the operator turned this off, rather than
        # their key being absent.
        print(NO_COACH_NOTICE)
    else:
        try:
            coach_client = reasoning_client.build_client()
        except reasoning_client.CoachConfigError as exc:
            print(f"Coach disabled: {exc}")
        else:
            try:
                context_load_started = time.perf_counter()
                coach_brief, coach_survey, coach_hypotheses = _load_coach_context(
                    cfg.context_dir, extra_fact_dirs=extra_fact_dirs
                )
                coach_context_load_s = time.perf_counter() - context_load_started
            except (ContextValidationError, coach_corpus.CorpusValidationError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            _warn_if_context_stale(cfg.context_dir, _previous_session_started_at(cfg.sessions_root))
            coach_enabled = True
            print(
                f"Coach enabled: brief has {len(coach_brief)} item(s), survey has "
                f"{len(coach_survey)} item(s), hypotheses has {len(coach_hypotheses)} item(s)."
            )

    try:
        engines = {speaker: get_engine(cfg.engine_name) for speaker in speakers}
    except UnknownEngineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Resolved engine: {cfg.engine_name}")

    # HERE, and the position is the entire fix: after the engine is resolved,
    # before `construct_pyaudio()` two lines below and before
    # `create_session_dir` further down. A missing `SONIOX_API_KEY` used to
    # surface inside the STT worker thread -- long after both audio devices
    # were open and `sessions/<id>/` had been written -- as a traceback plus an
    # orphaned session directory. A directory with no `closed_at` is exactly
    # what a crashed recording looks like, so the corpus gained a permanent
    # fake casualty every time someone forgot an environment variable.
    #
    # Same shape as the `check_port_available` guard at the top of this
    # function and placed for the same reason: everything that can be known
    # before the interview costs anything should be known before it does.
    try:
        preflight_engines(engines.values())
    except EngineNotReadyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Built ONCE, before any device is opened, and used by every seam that
    # chooses a device for the rest of the run -- see `_side_resolver`.
    device_resolvers = {
        speaker: _side_resolver(speaker, pin=_device_pin(args, speaker))
        for speaker in ALL_SPEAKERS
    }

    pa = construct_pyaudio()
    try:
        devices = _resolve_devices(pa, speakers, resolvers=device_resolvers)
    except DeviceOpenError as exc:
        pa.terminate()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for speaker, device in devices.items():
        print(f"Resolved {speaker} device: {device.name} ({device.native_sample_rate} Hz, {device.channels} ch)")

    # The self-test, BEFORE `create_session_dir` and for the same reason the
    # engine pre-flight above sits where it does: a run that is going to record
    # a silent channel should not first leave a session directory behind that
    # looks like a crashed interview.
    #
    # This is the gate that was missing on 2026-08-17, when the interviewee
    # channel captured two minutes of digital silence. The MEASUREMENT already
    # existed -- `precheck` had reported `-69.5 dBFS (silence)` on that very
    # device -- but nothing stood between it and the recording.
    if not getattr(args, "skip_self_test", False):
        try:
            self_test = run_self_test(
                pyaudio_module=pyaudio, pa=pa, devices=devices, cfg=cfg
            )
        except DeviceOpenError as exc:
            pa.terminate()
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print_self_test_report(self_test)
        if self_test.exit_code != 0:
            # WARNS by default; blocks only when asked. Refusing to start would
            # be the more satisfying gate and the wrong one: the ordinary flow
            # is to press record and THEN join the call, so a quiet room at the
            # instant of pressing is the common case, not the fault. A record
            # button that refuses a third of the time gets worked around, and
            # then the check is gone for the cases that are real.
            #
            # What makes warning sufficient is `feed_watch`: it catches the same
            # fault 45s in, from the transcript rather than from a level meter,
            # and replayed over four real sessions it fired on the broken one
            # and on none of the three healthy ones. The blocking gate is
            # available for a rig that must not fail -- a recorded demo, say --
            # where a false stop costs less than a silent take.
            message = (
                "a side of this interview is not carrying sound yet. If the call has "
                "not started, this is expected -- it will be re-checked from the "
                "transcript once people are talking."
            )
            if getattr(args, "require_live_audio", False):
                pa.terminate()
                print(f"ERROR: {message} Refusing to start (--require-live-audio).", file=sys.stderr)
                return 1
            print(f"WARNING: {message}", file=sys.stderr)

    retain_audio: bool = getattr(args, "retain_audio", config_mod.DEFAULT_RETAIN_AUDIO)

    session_dir = create_session_dir(cfg.sessions_root)
    store = SessionStore(session_dir)
    metadata = _session_metadata(
        devices,
        engine_name=cfg.engine_name,
        sample_rate=cfg.sample_rate,
        # Already normalized by `_participant_arg` at parse time, and read with
        # `getattr` because every other command's namespace lacks the flag
        # entirely -- `start` is the only subparser that defines it.
        participant=getattr(args, "participant", None),
    )
    metadata["audio_retained"] = retain_audio
    store.write_metadata(metadata)
    # Printed at START, not only at close -- a user who cannot see what the
    # system is doing cannot tell a working system from a dead one
    # (phase_specification Task 3).
    print(f"Session path: {store.session_dir}")

    # Phase 4 (04-01-PLAN.md §C6): the brief/survey are read ONCE, at `start`,
    # and copied into the session's own metadata for provenance -- a later
    # edit to context/brief.json must never silently re-ground a call already
    # in progress.
    #
    # 06-01-PLAN.md §F4/06-RESEARCH.md open question 1: a "coach_grounding"
    # detail recorded so a future decision about the scoring function has
    # evidence rather than an impression -- the measured session-start cost,
    # counts of hand-written vs. retrieved items, and the SELECTED FACT IDS
    # ONLY. Never a claim, never a quote -- gate 12 stays green by
    # construction, and this is derived entirely from `coach_brief`, the
    # already-returned tuple, never a second read of facts.json.
    coach_grounding_fact_ids = tuple(
        item.id for item in coach_brief if item.id.startswith(coach_context.FACT_ID_PREFIX)
    )
    store.update_metadata({
        "coach_enabled": coach_enabled,
        "context_brief": [item.model_dump() for item in coach_brief],
        "context_survey": [item.model_dump() for item in coach_survey],
        "coach_grounding": {
            "session_start_grounding_s": coach_context_load_s,
            "hand_written_brief_items": len(coach_brief) - len(coach_grounding_fact_ids),
            "retrieved_fact_items": len(coach_grounding_fact_ids),
            "retrieved_fact_ids": list(coach_grounding_fact_ids),
        },
        # Phase A: which project this interview is filed under, and the EXACT
        # source set the coach was allowed to draw on.
        #
        # `grounding_sources` lists every node the operator selected, not just
        # the ones that happened to contribute a selected fact. Those two
        # differ constantly -- a node whose facts all scored zero overlap
        # contributes nothing to `retrieved_fact_ids` while still having been
        # part of the request -- and only the full list answers "what was this
        # coach allowed to see", which is the question a later reader of a
        # session actually has. Deriving it from the fact ids would answer a
        # different question and look like the same one.
        #
        # Node PATHS only. A project path is a label the operator typed, never
        # customer speech, so this carries no more than the session directory's
        # own location already does -- and gate 12's no-text rule is unaffected.
        "project": args.project,
        "grounding_sources": [node for node, _dir in extra_fact_dirs],
    })

    coverage_detector: Optional[CoverageDetector] = None
    suggestion_pipeline: Optional[SuggestionPipeline] = None
    turn_trigger: Optional[TurnTrigger] = None
    # The next-question panel's display policy (app/web/survey_hold.py). Assigned
    # immediately after the detector below, because it is seeded from the
    # detector's OWN replayed coverage state -- a resumed session starts part way
    # down the survey, and an assumed-empty hold would show question one to an
    # interviewer who is half way through.
    survey_display_hold: Optional[SurveyDisplayHold] = None
    if coach_enabled:
        coverage_detector = CoverageDetector(
            client=coach_client,
            store=store,
            survey=coach_survey,
            # `resolve_model()`, not the constant: a `COACH_BASE_URL` pointing
            # at another provider needs a model id that provider recognises,
            # and the dated Anthropic id would be meaningless there.
            model=reasoning_client.resolve_model(),
            timeout_s=cfg.coach_coverage_timeout_s,
            queue_maxsize=cfg.coach_queue_maxsize,
            # Phase C: re-broadcast the next-question panel whenever a survey
            # question moves to `asked`. `None` under `--no-serve` (no hub to
            # broadcast to), exactly like `on_ranked` below. Gate 27 pins this
            # call's keyword set against the constructor's parameter set by
            # equality, for the reason gate 26's comment gives at length --
            # `on_change` has a compatibility default, which is precisely the
            # shape that let `hypotheses` go unpassed and unnoticed.
            on_change=(
                # `lambda: survey_display_hold` is late-binding on purpose: the
                # hold is built from this detector's replayed state on the very
                # next line, and a coverage change cannot arrive before
                # `.start()` below, let alone before the first segment.
                _survey_broadcast_callback(
                    loop, hub, store, coach_survey, lambda: survey_display_hold
                )
                if hub is not None else None
            ),
        )
        survey_display_hold = SurveyDisplayHold(
            initial_asked=coverage_detector.state.asked
        )
        coverage_detector.start()
        # Phase 4 (04-02-PLAN.md §D9): the generate->critic pipeline's own
        # worker thread, wired below onto ONLY the interviewee stream's
        # segment_hook -- the two triggers (coverage vs. suggestion) stay
        # structurally separate, exactly like `coverage_detector` above stays
        # separate from the interviewee side.
        suggestion_pipeline = SuggestionPipeline(
            client=coach_client,
            store=store,
            brief=coach_brief,
            survey=coach_survey,
            # Passed EXPLICITLY, and this line is load-bearing: `hypotheses`
            # defaults to `()` for backwards compatibility, so omitting it here
            # is silent -- the coach starts, reports healthy, and runs every
            # cycle with an empty hypothesis set. That is the whole Phase 6
            # premise (propose the question that would validate or KILL an open
            # hypothesis) quietly switched off. `session_start_grounding_s`
            # above was deliberately given no default as gate 22's forcing
            # function; `hypotheses` was not, and this caller was missed for
            # exactly that reason. Found 2026-08-10.
            hypotheses=coach_hypotheses,
            session_start_grounding_s=coach_context_load_s,
            coverage_state_provider=lambda: coverage_detector.state,
            # `resolve_model()`, not the constant: a `COACH_BASE_URL` pointing
            # at another provider needs a model id that provider recognises,
            # and the dated Anthropic id would be meaningless there.
            model=reasoning_client.resolve_model(),
            generate_timeout_s=cfg.coach_generate_timeout_s,
            critique_timeout_s=cfg.coach_critique_timeout_s,
            generate_max_tokens=cfg.coach_generate_max_tokens,
            critique_max_tokens=cfg.coach_critique_max_tokens,
            max_shown=cfg.coach_max_shown,
            failure_threshold=cfg.coach_failure_threshold,
            language_window_s=cfg.coach_language_window_s,
            language_flip_margin=cfg.coach_language_flip_margin,
            language_min_chars=cfg.coach_language_min_chars,
            # Passed explicitly, like every other pipeline parameter: gate 26
            # pins this call's keyword set against the constructor's, precisely
            # so a repetition guard cannot be added and then left unwired here,
            # silently running on its own defaults or not at all.
            repeat_window=cfg.coach_repeat_window,
            repeat_threshold=cfg.coach_repeat_threshold,
            repeat_prompt_recent=cfg.coach_repeat_prompt_recent,
            queue_maxsize=cfg.coach_pipeline_queue_maxsize,
            # 04-03-PLAN.md §E1 "wire the callback": schedule-and-forget onto
            # the loop thread through the SAME thread->asyncio bridge shape
            # StreamWorker._persist already uses for the segment broadcast --
            # `None` under `--no-serve` (no hub to broadcast to), never a
            # blocking call from the pipeline's own worker thread.
            on_ranked=(
                _suggestion_broadcast_callback(loop, hub, store, max_shown=cfg.coach_max_shown)
                if hub is not None else None
            ),
        )
        suggestion_pipeline.start()
        # `turn+3s`, decided by `benchmark/eval/trigger_sim.py` over the real
        # 26-minute transcript: 71 cycles against every-segment's 184, and 81%
        # of interviewer turns covered against 50%. Cheaper AND better, which
        # is why it is a gate and not a rate limit -- see `app/coach/trigger.py`
        # for why `turn-end`, the obvious choice, scores zero.
        #
        # It sits BETWEEN the STT threads and `suggestion_pipeline.submit`, and
        # it has to see the interviewer too: a turn boundary is a speaker
        # change, so wiring it on the interviewee alone silently degrades it
        # back to every-segment.
        turn_trigger = TurnTrigger(downstream=suggestion_pipeline.submit)

    keepalive: Optional[KeepAliveStream] = None
    if cfg.keepalive_enabled and SPEAKER_INTERVIEWEE in devices:
        try:
            render_device = resolve_default_render_device(pa)
            keepalive = KeepAliveStream(pyaudio, pa, render_device)
            keepalive.start()
            if not keepalive.opened:
                store.append_event(event="keepalive-warning", detail={"reason": keepalive.warning})
        except DeviceOpenError as exc:
            store.append_event(event="keepalive-warning", detail={"reason": str(exc)})

    # Phase 3 (03-01-PLAN.md §P2/§P3): the running loop and the hub's
    # broadcast are handed to each worker so a finalized segment can cross
    # the thread -> asyncio boundary. `loop`/`hub`/`status_board` themselves
    # were constructed earlier (see the comment above the coach section) so
    # the suggestion-broadcast callback could be wired at `SuggestionPipeline`
    # construction time; nothing about their VALUES changes here. None of
    # this is constructed at all under `--no-serve` -- StreamWorker's
    # `loop`/`broadcast` already default to `None` (03-01-PLAN.md), so
    # capture and transcription run unchanged.
    # One side transcribing nothing while the other flows. Its own clock, taken
    # here and read by both the segment hook and the status poller, rather than
    # `SegmentRecord`'s timestamps: this session's own transcript recorded
    # `at_s: 0` on all 24 segments, so a watch that trusted that field would
    # have measured every gap as zero and never fired.
    feed_watch = FeedWatch(speakers=ALL_SPEAKERS, config=FeedWatchConfig())
    feed_watch_origin = time.monotonic()

    def _feed_watch_elapsed() -> float:
        return time.monotonic() - feed_watch_origin

    pairs: list[StreamPair] = []

    def _build_pair(speaker: str, device: "DeviceInfo") -> StreamPair:
        """One speaker's capture, worker and supervisor.

        A function rather than a loop body because a pair is now also built
        MID-INTERVIEW: a run started with `--only interviewee` has no
        interviewer pair at all, and the microphone control used to loop over
        the existing pairs, find none, and do nothing -- silently. The operator
        clicked it and nothing happened, twice, with no error anywhere.

        Nothing here opens a device or a socket. `CaptureStream.__init__` only
        records what to open; `start()` is what touches the microphone, and
        `worker.start()` is what opens the STT session. That separation is what
        makes building a pair on demand safe."""
        capture = CaptureStream(
            pyaudio, pa, device,
            target_sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms,
            queue_max_seconds=cfg.queue_max_seconds, label=speaker,
        )
        audio_writer: Optional[AudioWriter] = None
        if retain_audio:
            audio_writer = AudioWriter(
                session_dir, speaker=speaker, sample_rate=cfg.sample_rate, store=store
            )
        worker = StreamWorker(
            engine=engines[speaker], frame_queue=capture.queue, store=store,
            speaker=speaker, sample_rate=cfg.sample_rate, audio_writer=audio_writer,
            loop=loop, broadcast=(hub.broadcast if hub is not None else None),
            # Phase 4 (04-01-PLAN.md §C2/D-03, 04-02-PLAN.md §D9): the two
            # triggers stay structurally separate -- ONLY the interviewer's
            # stream gets the coverage hook (coverage is written from what
            # the interviewer actually said and from nothing else), and ONLY
            # the interviewee's stream gets the suggestion-pipeline hook
            # (a suggestion is triggered by what the interviewee just said).
            # The interviewer's stream now feeds TWO consumers, and the second
            # one is easy to misread as a layering violation: coverage is still
            # written from interviewer speech and nothing else, and the
            # suggestion side is still triggered by interviewee speech and
            # nothing else. `turn_trigger` is not a third trigger -- it only
            # needs the interviewer's segments to know where the interviewee's
            # TURN ends, and it forwards nothing it hears from him.
            segment_hook=_fan_out(
                # On BOTH streams, and unconditionally: the whole judgement is
                # comparative, so a watch fed only one side could never tell a
                # broken feed from a quiet room.
                lambda record: feed_watch.note_segment(
                    record.speaker, at_s=_feed_watch_elapsed()
                ),
                coverage_detector.submit
                if coverage_detector is not None and speaker == SPEAKER_INTERVIEWER
                else None,
                turn_trigger.submit if turn_trigger is not None else None,
                # The next-question hold, on BOTH streams (2026-08-15). This is
                # not a third trigger and not a layering violation: it writes no
                # coverage state and reads no text. The interviewee's segment is
                # the release that matters -- the panel advances when they
                # ANSWER, not when the interviewer asks. The interviewer's is
                # only what lets an unanswered hold expire, since the timeout is
                # evaluated lazily and there is no ticker on the server.
                _survey_hold_release_hook(
                    loop, hub, store, coach_survey, survey_display_hold
                )
                if (survey_display_hold is not None and hub is not None and loop is not None)
                else None,
            ),
        )
        supervisor = StreamSupervisor(
            speaker=speaker,
            capture=capture,
            store=store,
            # The SAME resolver session start used, so a pin survives a device
            # change instead of reverting to the default endpoint here.
            device_resolver=(lambda pa=pa, fn=device_resolvers[speaker]: fn(pa)),
            config=config_mod.supervisor_config(cfg),
        )
        return StreamPair(
            speaker=speaker, device=device, capture=capture, worker=worker,
            supervisor=supervisor, audio_writer=audio_writer,
        )

    for speaker, device in devices.items():
        pairs.append(_build_pair(speaker, device))

    # STT sessions open (network round-trip) BEFORE any capture stream starts
    # producing frames -- same ordering rule plan 02-01 established, applied
    # per side now, so no pre-session backlog skews either side's audio clock.
    for pair in pairs:
        pair.worker.start()
    for pair in pairs:
        pair.capture.start()
        pair.supervisor.start()

    # Seeded from what was actually opened: a `--only interviewee` run has no
    # interviewer pair, so the microphone starts closed.
    microphone = MicrophoneState(
        on=any(pair.speaker == SPEAKER_INTERVIEWER for pair in pairs)
    )

    def _set_microphone(enabled: bool) -> None:
        """Releases or re-opens the INTERVIEWER device while the interview
        runs. Called from the control channel's thread.

        **The supervisor is stopped FIRST and started LAST, and that ordering
        is the whole correctness argument.** `StreamSupervisor` exists to
        notice a stream that has gone quiet -- its own docstring says a muted
        interviewee used to produce a green banner "during the one event that
        cannot be re-run". A microphone the operator switched off is quiet on
        purpose, and leaving the supervisor watching it would fire exactly the
        alarm it was built for, at exactly the moment it is wrong.

        The STT worker is deliberately left running. It reads from the capture
        queue and simply receives nothing; stopping and restarting an engine
        session mid-interview is a network round-trip and a new session id for
        no gain the operator can see. If the vendor drops an idle session, the
        existing failure path reports it honestly rather than this code
        pretending it cannot happen.
        """
        pair = next((p for p in pairs if p.speaker == SPEAKER_INTERVIEWER), None)
        if pair is None:
            if not enabled:
                # Already off, and nothing to release. Report it so the panel
                # agrees rather than waiting for an answer that never comes.
                print(microphone_state_line(microphone.is_on), flush=True)
                return
            # A run started with `--only interviewee` has no interviewer pair.
            # This loop used to find none and fall through in SILENCE -- the
            # operator clicked the control twice and nothing happened, with no
            # event, no state line and no error. Build it now.
            try:
                device = device_resolvers[SPEAKER_INTERVIEWER](pa)
                pair = _build_pair(SPEAKER_INTERVIEWER, device)
                pairs.append(pair)
                # The STT session opens here, not at construction. A run that
                # never enables the microphone therefore never opens one --
                # which is why this is lazy rather than built up front and left
                # idle for the whole recording.
                pair.worker.start()
            except Exception as exc:  # noqa: BLE001 -- no microphone is a condition, not a crash
                logger.error(
                    "could not add a microphone to this run error_type=%s", type(exc).__name__
                )
                store.append_event(
                    event="microphone",
                    detail={"enabled": True, "error_type": type(exc).__name__},
                )
                print(microphone_state_line(microphone.is_on), flush=True)
                return
        try:
            if enabled:
                pair.capture.start()
                pair.supervisor.start()
            else:
                pair.supervisor.stop()
                pair.capture.stop()
        except Exception as exc:  # noqa: BLE001 -- a failed toggle must not end the interview
            logger.error(
                "toggling the microphone failed enabled=%s error_type=%s",
                enabled, type(exc).__name__,
            )
            store.append_event(
                event="microphone",
                detail={"enabled": enabled, "error_type": type(exc).__name__},
            )
            # The state we ACTUALLY reached -- read from the tracker, which
            # did not move, rather than computed as the opposite of the
            # request. `not enabled` would be wrong the moment someone asks
            # for a state they are already in and the attempt fails.
            print(microphone_state_line(microphone.is_on), flush=True)
            return

        # Recorded in the transcript, because it changes what the transcript
        # MEANS: a gap on the interviewer track is either a silent operator or
        # a released device, and nothing else on disk can tell those apart
        # afterwards.
        microphone.record(enabled)
        store.append_event(event="microphone", detail={"enabled": enabled})
        logger.info("microphone %s", "on" if enabled else "off (device released)")
        print(microphone_state_line(microphone.is_on), flush=True)

    control = ControlChannel(sys.stdin, on_microphone=_set_microphone)
    control.start()

    # 03-02-PLAN.md Task 2 §Q6/§Q7: a laptop-side failure must survive a
    # reconnect, so it is server-authoritative state polled from what each
    # pair already exposes publicly, not something the client can miss.
    # Skipped entirely under `--no-serve` -- there is no hub to push to.
    status_poll_task: Optional[asyncio.Task] = None
    if not no_serve:
        status_poll_task = asyncio.create_task(
            _status_poll_loop(
                pairs, status_board, hub, cfg.status_poll_interval_s,
                coach_health_provider=(
                    (lambda: suggestion_pipeline.health) if suggestion_pipeline is not None else None
                ),
                coach_failure_threshold=cfg.coach_failure_threshold,
                feed_watch=feed_watch,
                elapsed_provider=_feed_watch_elapsed,
                store=store,
            )
        )

    # The join surface (D-05) and the embedded server. Constructed only now
    # -- after every capture/STT thread is already running -- so a QR/LAN-IP
    # degradation (join.py never raises) can never prevent capture from
    # starting. Entirely skipped under `--no-serve`: no port bound, no
    # browser opened, no QR produced (OPS-01's one escape hatch).
    server: Optional[object] = None
    serve_task: Optional[asyncio.Task] = None
    if not no_serve:
        join_info = create_join(
            port=cfg.app_port, session_dir=session_dir, store=store, emit_qr=cfg.emit_qr
        )
        server_started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
        app = web_server.create_app(
            token=join_info.token,
            hub=hub,
            static_dir=STATIC_DIR,
            store=store,
            session_id=session_dir.name,
            server_started_at=server_started_at,
            status_board=status_board,
            # Phase C: called on every fresh connect so a browser joining
            # mid-interview (or waking from sleep) shows the right next
            # question immediately, rather than at the next coverage change.
            # Reads `coverage_detector.state` live -- never a value captured
            # at construction time, which would freeze the panel at "nothing
            # asked yet" for any client that connected after the first match.
            # Reads the HOLD, never live coverage state. A phone waking up
            # mid-interview would otherwise be handed the newest state and skip
            # the question still being answered -- the same defect the hold
            # exists to fix, arriving through the reconnect door instead.
            survey_provider=(
                (lambda: survey_payload(
                    coach_survey, asked=survey_display_hold.asked_for_display()
                ))
                if survey_display_hold is not None
                else (lambda: survey_payload(coach_survey, asked=coverage_detector.state.asked))
                if coverage_detector is not None else None
            ),
            # Phase A: which knowledge is in play, fixed for the whole run.
            project=args.project,
            grounding_sources=[node for node, _dir in extra_fact_dirs],
            # The value computed by the coach-startup block above, not a
            # restatement of the flag that asked for one. `--no-coach` and a
            # missing ANTHROPIC_API_KEY both land here as False, which is the
            # whole point: a client must be able to tell that no suggestion is
            # coming, and the overlay used to infer that from the project's
            # NAME.
            coach_enabled=coach_enabled,
            heartbeat_interval_ms=cfg.heartbeat_interval_ms,
            heartbeat_timeout_ms=cfg.heartbeat_timeout_ms,
            reconnect_backoff_base_ms=cfg.reconnect_backoff_base_ms,
            reconnect_backoff_cap_ms=cfg.reconnect_backoff_cap_ms,
        )
        server = web_server.build_uvicorn_server(
            app,
            host="0.0.0.0",
            port=cfg.app_port,
            ws_ping_interval_s=cfg.ws_ping_interval_s,
            ws_ping_timeout_s=cfg.ws_ping_timeout_s,
        )
        # BEFORE uvicorn starts, because uvicorn RESTORES whatever handler it
        # found here and then re-raises the signal it captured
        # (`uvicorn/server.py:340`). Leaving `default_int_handler` in place
        # means that re-raise becomes a `KeyboardInterrupt` delivered in the
        # middle of the ordered shutdown, which is what killed every overlay
        # recording before `store.close()` could run.
        #
        # With `SIG_IGN` here, the sequence is: uvicorn's OWN handler takes the
        # break and sets `should_exit`; the wait loop above breaks on it; the
        # shutdown below runs to completion; and uvicorn's parting re-raise
        # lands on `SIG_IGN` and does nothing.
        #
        # Nothing is lost by ignoring it directly: the break still stops the
        # interview, via `should_exit` rather than via an exception. See
        # `app/interrupt.py::ignore_stop_signals`.
        ignore_stop_signals()
        serve_task = asyncio.create_task(server.serve())

        if cfg.open_browser:
            try:
                webbrowser.open(join_info.join_url)
            except Exception as exc:  # noqa: BLE001 -- a browser-open failure must never abort a session
                logger.warning("browser auto-open failed error_type=%s", type(exc).__name__)

    try:
        start_time = time.monotonic()
        while True:
            if args.seconds is not None and (time.monotonic() - start_time) >= args.seconds:
                break
            if server is not None and server.should_exit:
                # e.g. a fatal ASGI error, or uvicorn's own SIGINT handler
                # having already set this on Ctrl-C -- either way, stop
                # waiting immediately rather than sleeping out the rest of
                # `--seconds`. Genuine failures are recorded below, once
                # `serve_task` is awaited and can report what happened.
                break
            await asyncio.sleep(0.2)  # was time.sleep(0.2) -- now yields to the loop
    except KeyboardInterrupt:
        print("Interrupted -- closing session cleanly.")
    finally:
        # FIRST, before anything below can be interrupted. From here on the
        # interview is ending regardless, and the only thing a further stop
        # signal can change is whether the transcript reaches disk.
        #
        # This is not belt-and-braces: uvicorn RE-RAISES the signal it captured
        # when its own context manager exits (`uvicorn/server.py:340`), and
        # that re-raise is delivered at an arbitrary bytecode boundary inside
        # this block. It killed the process before `store.close()`, which is
        # why every overlay recording ever made lacked `closed_at` -- including
        # after `7fcfe05` installed the SIGBREAK handler, which was necessary
        # and not sufficient. See `app/interrupt.py::ignore_stop_signals`.
        ignore_stop_signals()
        control.stop()
        # Stopped before the streams it reads from, mirroring the same
        # ordered-shutdown discipline as everything else in this block.
        if status_poll_task is not None:
            await _cancel_and_await(status_poll_task, task_name="status-poll-task")
        for pair in pairs:
            pair.supervisor.stop()
        for pair in pairs:
            pair.worker.stop()
        # Phase 4 (04-01-PLAN.md §C3, 04-02-PLAN.md §D9): stopped once no
        # worker can submit another segment, before capture -- alongside the
        # other stops here, not as a separate shutdown mechanism.
        if coverage_detector is not None:
            coverage_detector.stop()
        if suggestion_pipeline is not None:
            suggestion_pipeline.stop()
        for pair in pairs:
            pair.capture.stop()
        for pair in pairs:
            if pair.audio_writer is not None:
                pair.audio_writer.close()
        if keepalive is not None:
            keepalive.stop()
        pa.terminate()
        # Ordered alongside the other stops above, not as a separate
        # mechanism (03-RESEARCH.md §1). A fatal ASGI error becomes a named,
        # visible store event -- never a silent death (OPS-05). Skipped
        # entirely under `--no-serve`, where neither was ever constructed.
        if server is not None and serve_task is not None:
            server.should_exit = True
            try:
                await serve_task
            except Exception as exc:  # noqa: BLE001 -- never let a server failure kill the process silently
                logger.error("embedded server failed error_type=%s", type(exc).__name__)
                store.append_event(
                    event="server-status",
                    detail={"error_type": type(exc).__name__},
                )
            except KeyboardInterrupt:
                # UVICORN RE-RAISES THE SIGNAL, AND THAT USED TO END THE
                # INTERVIEW HERE. Measured 2026-08-12 by reproducing a real
                # stop with a real child: `uvicorn.Server.capture_signals`
                # installs its OWN SIGBREAK/SIGINT handler for as long as the
                # server runs, so `app/interrupt.py`'s handler is not the one
                # in force. Uvicorn records the signal, shuts down gracefully,
                # and then on leaving its context manager does
                # `signal.raise_signal(captured_signal)` (server.py:340) to
                # "trigger the expected behaviour now".
                #
                # That re-raise lands HERE, inside the ordered shutdown, and
                # `except Exception` above does not catch it because
                # `KeyboardInterrupt` is a `BaseException`. So it propagated
                # out of the `finally`, and `store.close()` -- three lines
                # below -- never ran. Every overlay recording ever made lacked
                # `closed_at` and `latency_summary` for this reason, including
                # after `7fcfe05` installed the SIGBREAK handler: that fix was
                # real and necessary, and it was not sufficient.
                #
                # Swallowed deliberately and ONLY here. We are already inside
                # the shutdown this signal asked for; the interview is ending
                # either way, and the only question left is whether the
                # transcript gets flushed. Re-raising would answer "no".
                logger.info("the server re-raised the stop signal -- continuing the shutdown")
                store.append_event(event="server-status", detail={"stopped_by": "signal"})
        for pair in pairs:
            print(f"{pair.speaker} segment count: {store.segment_count(pair.speaker)}")
        summary = store.close()
        print(f"Transcript written to: {store.transcript_path}")
        print(f"Latency summary: {summary}")
        # Phase 4 (04-02-PLAN.md §D1/COACH-05): printed beside the transcript
        # latency figures above AND persisted into session metadata -- the
        # same gap STATE.md already records for STT latency (a number only
        # ever visible in a terminal that has since scrolled). The first
        # cycle is reported separately per §D1's schema-compilation note.
        if suggestion_pipeline is not None:
            coaching_summary = suggestion_pipeline.latency_summary()
            store.update_metadata({"coaching_latency_summary": coaching_summary})
            print(f"Coaching latency summary: {coaching_summary}")

    return 0


def run_start(args: argparse.Namespace) -> int:
    # BEFORE the interview, not inside it: the overlay stops this process with
    # CTRL_BREAK_EVENT, and without this the break is a KILL rather than the
    # KeyboardInterrupt `_async_main`'s `finally` needs to flush the store.
    # See `app/interrupt.py` for the measurement, and
    # `app/tests/test_overlay_stop_signal.py` for the test that spawns a real
    # child both ways.
    if not install_stop_handler() and os.name == "nt":
        # Said out loud, never swallowed: without the handler a CTRL_BREAK from
        # the overlay is a KILL, so this session would end with no `closed_at`
        # and a tail that was mid-write. Reached when `start` is driven off the
        # main thread (some tests do); an operator should never see it.
        print(
            "WARNING: the interrupt handler could not be installed -- stopping this "
            "interview may not close it cleanly.",
            file=sys.stderr,
        )
    return asyncio.run(_async_main(args))


def run_render(args: argparse.Namespace) -> int:
    cfg = _apply_project(config_mod.load_config(), args)
    candidate = Path(args.session)
    session_dir = candidate if candidate.is_dir() else cfg.sessions_root / args.session
    if not session_dir.is_dir():
        print(f"ERROR: no session directory found at {session_dir}", file=sys.stderr)
        return 1
    # 06-03-PLAN.md §H4: the only call-site change -- no new flag, no new
    # option. `cfg.context_dir` is the same directory `start` already reads
    # the brief/survey/hypotheses from.
    doc_path = write_rendered_transcript(session_dir, context_dir=cfg.context_dir)
    print(f"Rendered transcript written to: {doc_path}")
    return 0


def run_precheck(args: argparse.Namespace) -> int:
    cfg = config_mod.load_config(engine_override=args.engine)
    pa = construct_pyaudio()
    # The same pins `start` accepts, resolved the same way. A pre-call check
    # that measured the DEFAULT endpoints while the interview was pinned
    # elsewhere would be a green light for a rig nobody is going to run -- the
    # exact shape of harness drift this repo keeps rediscovering, and worse
    # here than useless, because this is the check an operator trusts instead
    # of looking.
    side_defs = tuple(
        (speaker, label, _side_resolver(speaker, pin=_device_pin(args, speaker)))
        for speaker, label in (
            (SPEAKER_INTERVIEWEE, "Interviewee device (loopback)"),
            (SPEAKER_INTERVIEWER, "Interviewer device (microphone)"),
        )
    )
    try:
        report = build_precheck_report(
            pa=pa,
            pyaudio_module=pyaudio,
            cfg=cfg,
            seconds=args.seconds if args.seconds is not None else cfg.precheck_seconds,
            skip_stt=args.skip_stt,
            phone=args.phone,
            side_defs=side_defs,
        )
    finally:
        pa.terminate()
    print_precheck_report(report)
    return report.exit_code


def run_author(args: argparse.Namespace) -> int:
    """`python -m app author` -- 04-03-PLAN.md §E4, extended by 05-03-PLAN.md
    §K13 to a third artifact. A fourth subcommand, following
    `start`/`render`/`precheck` exactly, that opens a loopback-only browser
    page for writing the brief, survey and hypotheses BEFORE a call. Shares
    its process with no capture or STT threads, so unlike `run_start` it runs
    its server directly -- one coroutine, nothing else on the loop -- rather
    than needing the main-thread event-loop arrangement `start` uses (see
    `app/web/server.py`'s own module docstring for why `start` needs that)."""
    cfg = _apply_project(config_mod.load_config(), args)
    app = web_author.create_author_app(context_dir=cfg.context_dir, static_dir=STATIC_AUTHOR_DIR)
    port = cfg.author_port
    url = f"http://127.0.0.1:{port}/"

    @app.on_event("startup")
    async def _print_and_open_browser() -> None:
        # Fired only once the server is actually listening (a FastAPI/
        # Starlette startup event), which is what lets the browser be opened
        # with no race against a not-yet-bound socket -- no sleep, no retry.
        print(f"Authoring page: {url}")
        print("Write the brief, the survey and any hypotheses, then Ctrl+C here when done.")
        if cfg.open_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:  # noqa: BLE001 -- a browser-open failure must never abort authoring
                logger.warning("browser auto-open failed error_type=%s", type(exc).__name__)

    server = web_author.build_uvicorn_server(app, port=port)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("Interrupted -- stopping the authoring server.")
    return 0


def run_hub(args: argparse.Namespace) -> int:
    """`python -m app hub [session] --project X` -- the post-interview review.

    **Scoped to the PROJECT; the interview is chosen in the page.** The session
    argument only picks which one opens first, and omitting it is the normal
    case. Requiring a hand-typed session id was a real reason the close-out
    never happened: reading a call meant knowing its timestamp before you could
    look at it.

    Everything except the accept/reject card only displays. That card is the
    one action on the page: it applies a coach proposal to
    `context/hypotheses.json` on an explicit click, through
    `app/web/accept.py` -- a module that exists solely for that save, so gate
    19's per-file pin stays meaningful over everything else the hub does."""
    cfg = _apply_project(config_mod.load_config(), args)
    port = cfg.author_port
    url = f"http://127.0.0.1:{port}/"

    app = web_review.create_review_app(
        static_dir=STATIC_REVIEW_DIR, project=args.project, session_id=args.session,
    )

    @app.on_event("startup")
    async def _print_and_open_browser() -> None:
        print(f"Review: {url}")
        print("Ctrl+C here when you are done reading.")
        if cfg.open_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:  # noqa: BLE001 -- a browser failure must not abort the review
                logger.warning("browser auto-open failed error_type=%s", type(exc).__name__)

    server = web_review.build_review_server(app, port=port)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("Closed.")
    return 0


def run_console(args: argparse.Namespace) -> int:
    """`python -m app console` -- pick a project, then run the interview IN
    THIS SAME PROCESS.

    The console serves a loopback-only page, waits for the operator to press
    Start, stops its own server, and then falls through to `run_start` with
    the chosen paths. A web page never spawns a subprocess: that keeps the
    README's "one command, one process, no second launcher" principle, and
    avoids the failure where a launcher outlives what it launched.

    The selection is applied by REWRITING `args`, so the interview runs
    through exactly the same `run_start` path a hand-typed
    `--project X --read-from Y` takes. There is no second start path to keep
    in step, which matters more than it looks: `run_start` is where the
    coach, the survey panel and the grounding strip are wired, and a parallel
    path would drift from it silently."""
    cfg = config_mod.load_config()
    port = cfg.author_port
    url = f"http://127.0.0.1:{port}/"

    # Written by the request handler, read after the server returns. A single
    # assignment from the one thread uvicorn dispatches on -- no lock, and
    # nothing else ever reads it while the server is running.
    chosen: "list[web_console.ConsoleSelection]" = []
    server: "Optional[Any]" = None

    def _on_start(selection: "web_console.ConsoleSelection") -> None:
        chosen.append(selection)
        # `should_exit` is uvicorn's own cooperative stop. Set from inside a
        # request handler it lets THIS response finish before the server
        # comes down, which is what puts the confirmation on the page rather
        # than leaving the operator with a dead connection.
        if server is not None:
            server.should_exit = True

    app = web_console.create_console_app(static_dir=STATIC_CONSOLE_DIR, on_start=_on_start)

    @app.on_event("startup")
    async def _print_and_open_browser() -> None:
        print(f"Console: {url}")
        print("Pick a project and press Start, or Ctrl+C here to cancel.")
        if cfg.open_browser:
            try:
                webbrowser.open(url)
            except Exception as exc:  # noqa: BLE001 -- a browser-open failure must never abort the console
                logger.warning("browser auto-open failed error_type=%s", type(exc).__name__)

    server = web_console.build_console_server(app, port=port)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("Interrupted -- no interview started.")
        return 0

    if not chosen:
        # The operator closed the tab or stopped the server without
        # choosing. Exiting 0 is right: nothing failed, and nothing ran.
        print("Console closed without starting an interview.")
        return 0

    selection = chosen[0]
    print(f"Starting interview -- project: {selection.project}")
    if selection.read_from:
        print(f"Also drawing on: {', '.join(selection.read_from)}")

    # `console` has none of `start`'s own flags, so they take their argparse
    # defaults. Set explicitly rather than relied upon: a new `start` flag
    # whose default is not None would otherwise reach `run_start` here as a
    # missing attribute, and it would fail at the point of use rather than
    # here.
    start_args = build_arg_parser().parse_args(["start"])
    start_args.project = selection.project
    start_args.read_from = list(selection.read_from)
    # The unfiled bucket has no `context/`, so a coached run against it would
    # fail loudly on the missing brief the moment a key is present. Derived
    # from the SAME predicate the overlay's controller uses, so the two
    # pickers cannot disagree about what an unfiled recording is.
    start_args.no_coach = project_meta.is_unfiled(selection.project)
    return run_start(start_args)


def run_export(args: argparse.Namespace) -> int:
    """`python -m app export --to <path.zip>` -- a verifiable copy of everything
    a project cannot regenerate.

    Exists because `.gitignore` excludes `projects/` wholesale and standing
    gates assert none of it ever becomes committable. That is correct, and it
    means "never escapes" and "has no backup" are the same property: at
    interview 30 a dead disk ends this product, including the evidentiary value
    of every pre-registration boundary.

    Refuses before writing if the destination is somewhere git can see, because
    an export carries verbatim customer speech and an untracked copy in the
    working tree is one `git add -A` from publication -- a mistake this repo has
    already made once, with a credential file.

    Never opens an audio device, never makes a model call, never writes into the
    project it is copying."""
    from app.session.export import ExportRefused, export_project, verify_archive

    cfg = _apply_project(config_mod.load_config(), args)
    project_dir = cfg.sessions_root.parent
    try:
        result = export_project(
            project_dir, Path(args.to),
            # `app/main.py` -> `app/` -> the repo. Derived here rather than from
            # the CWD: the guard asks git what it can see, and asking from the
            # wrong root would clear a destination git can see perfectly well.
            repo_root=Path(__file__).resolve().parents[1],
            overwrite=args.overwrite,
        )
    except ExportRefused as exc:
        print(f"Export refused: {exc}", file=sys.stderr)
        return 1

    print(f"Exported {result.files} file(s), {result.bytes_written / 1024:.0f} KiB "
          f"uncompressed, to {result.archive}")
    if result.audio_skipped:
        # Named rather than left to be noticed. A restorer must be told what is
        # deliberately absent.
        print(f"  {result.audio_skipped} audio file(s) excluded by policy.")

    if args.verify:
        failures = verify_archive(result.archive)
        if failures:
            print(f"VERIFY FAILED for {len(failures)} file(s): {failures[:5]}", file=sys.stderr)
            return 1
        print("  verified: every file hashes to what the manifest recorded.")
    return 0


def run_extract(args: argparse.Namespace) -> int:
    """`python -m app extract <session>` -- 05-01-PLAN.md Task 1, extended by
    05-02-PLAN.md §K7/§K7a/§K7b/§K8 to be safely RE-runnable. A finished
    session's transcript becomes a set of cited, provenance-tagged facts in
    `context/facts.json` (05-CONTEXT.md D-06/D-07/D-08).

    **Deliberate divergence from `run_author`'s precedent, stated here per
    that precedent's own instruction:** a missing `ANTHROPIC_API_KEY` makes
    `run_start` degrade and proceed with the coach disabled, because a
    coachless interview is still an interview. Extraction without the vendor
    cannot do its job at all -- there is no degraded mode for "no model
    call happened" -- so it fails with a named error and returns 1.

    Reaches the transcript ONLY through `app.session.reader.read_session`
    (§K7) and never opens an audio device or creates a session directory --
    this command's whole job is reading an already-closed-or-open session
    and writing a derived artifact, nothing more.

    **Ordering matters (§K7a):** the existing corpus is checked for
    parseability BEFORE the vendor call -- a user whose corpus is broken
    should not pay for a call whose result would be discarded. The
    unclosed-session/truncated-tail warnings (§K7b) are also printed before
    the vendor call, so a user who did not mean to re-run mid-interview can
    still interrupt before spending anything."""
    cfg = _apply_project(config_mod.load_config(), args)
    candidate = Path(args.session)
    session_dir = candidate if candidate.is_dir() else cfg.sessions_root / args.session
    if not session_dir.is_dir():
        print(f"ERROR: no session directory found at {session_dir}", file=sys.stderr)
        return 1

    try:
        client = reasoning_client.build_client()
    except reasoning_client.CoachConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        hypotheses = coach_corpus.load_hypotheses(cfg.context_dir, missing_ok=True)
    except coach_corpus.CorpusValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # §K7a: read-only pre-flight, BEFORE anything reaches the vendor. A
    # missing corpus is D-01's normal first case and is not an error here;
    # an EXISTING corpus that fails to parse aborts the whole run right now,
    # naming the file and the offending item, with nothing written and
    # nothing paid for.
    try:
        coach_corpus_store.preflight_check_corpus_parses(cfg.context_dir)
    except coach_corpus.CorpusValidationError as exc:
        print(
            f"ERROR: {exc} The existing corpus was left untouched -- fix it (or restore a "
            "backup) before re-running extraction.",
            file=sys.stderr,
        )
        return 1

    read_result = read_session(session_dir)
    segments = tuple(SegmentRecord(**record) for record in read_result.records_of_kind("segment"))

    # §K7b: a loud warning, never a refusal -- only the user can tell a
    # deliberate mid-interview sanity check from a forgotten crash. Printed
    # before the vendor call so a user who did not mean it can interrupt.
    if not read_result.closed:
        print(
            f"WARNING: session {session_dir.name} has no closed_at marker -- it may still be "
            "in progress. The fact set extracted now will look final and may not be; that is "
            "fine for a deliberate mid-interview sanity check, but re-run once the interview "
            "truly ends if this was not deliberate.",
            file=sys.stderr,
        )
    if read_result.truncated_tail is not None:
        print(
            f"WARNING: session {session_dir.name}'s transcript has a truncated final line -- "
            "the recording may have crashed mid-write. Extraction proceeds over the records "
            "that did parse; nothing after the truncation point is included.",
            file=sys.stderr,
        )

    try:
        extraction = coach_extract.extract_facts(
            client,
            session_id=session_dir.name,
            transcript=segments,
            hypotheses=hypotheses,
            # `resolve_extract_model()`, not `resolve_model()`: extraction is
            # OFFLINE, so it is not bound by the live coach's ~8s cycle and
            # may buy label quality (A2) with latency nobody is waiting on.
            # It still falls back to the coach's own id, so a `COACH_BASE_URL`
            # pointing at another provider carries extraction with it and
            # never receives a dated Anthropic id it has never heard of.
            model=reasoning_client.resolve_extract_model(),
            max_tokens=cfg.coach_extract_max_tokens,
            timeout_s=cfg.coach_extract_timeout_s,
        )
    except coach_extract.ExtractFailure as exc:
        print(f"ERROR: extraction call failed: reason={exc.reason} stop_reason={exc.stop_reason}", file=sys.stderr)
        return 1

    for rejection in extraction.rejections:
        print(
            f"REJECTED (reason={rejection.reason}, seqs={list(rejection.seqs) if rejection.seqs is not None else None}): "
            f"{rejection.claim!r}" + (f" quoted_text={rejection.quoted_text!r}" if rejection.quoted_text is not None else "")
        )
    # Printed apart from REJECTED, and deliberately so: a demoted fact was
    # KEPT, relabelled `evidence` -> `inference`. Conflating the two lines
    # would read as "this fact was thrown away", which is the opposite of
    # what happened. A demotion is a quality signal about the EXTRACTION
    # MODEL, so it never touches the exit code.
    for demotion in extraction.demotions:
        print(
            f"DEMOTED to inference (reason={demotion.reason}, "
            f"seqs={list(demotion.seqs) if demotion.seqs is not None else None}): "
            f"{demotion.claim!r} quoted_text={demotion.quoted_text!r}"
        )
    for warning in extraction.dropped_hypothesis_warnings:
        print(f"WARNING: {warning}")

    # Rejections used to exist only on this stdout, which is how "~30% reject
    # as unlocatable-quote" became a number nobody could go back and check.
    # Written UNCONDITIONALLY -- unlike the accepted facts below, an empty
    # slice here is a real answer ("extracted, refused nothing") rather than
    # an empty file pretending to be a result.
    rejections_path = coach_rejection_store.replace_session_rejections(
        cfg.context_dir, session_dir.name, extraction.rejections
    )

    # Demotions were in the state rejections were in above: printed, then
    # gone. Two things were unanswerable because of it. (1) "How often does
    # the extraction model call an inference `evidence`?" is a rate over the
    # corpus, and the only record of it was a terminal. (2) STATE COLLAPSE:
    # `facts.json` stores a demoted fact as a plain `inference`, and nothing
    # in `corpus.Fact` forbids a quote on an inference -- so an overridden
    # label and an honest one can be identical in every stored field, and the
    # hub's evidence/inference panel cannot separate them. This file is where
    # that distinction survives. Written UNCONDITIONALLY, for the same reason
    # as the rejections above: "extracted, overrode no label" is a real
    # answer, and a file that was never created cannot give it.
    demotions_path = coach_demotion_store.replace_session_demotions(
        cfg.context_dir, session_dir.name, extraction.demotions
    )

    # §K7: a session-slice replace, not a whole-file overwrite -- re-running
    # on the same session supersedes only ITS facts, leaving every other
    # session's facts untouched. Guarded on `extraction.accepted` (unchanged
    # from 05-01): a run that accepted nothing writes nothing at all, so a
    # first-ever, fully-rejected extraction never fabricates an empty
    # facts.json (matching the pre-05-02 behaviour this project's own tests
    # already pin).
    if extraction.accepted:
        written_path = coach_corpus_store.replace_session_facts(
            cfg.context_dir, session_dir.name, extraction.accepted
        )
        print(f"Facts written to: {written_path}")
    else:
        print(f"No facts written -- nothing accepted this run (context dir: {cfg.context_dir}).")

    # §K8: 05-01-PLAN.md §K10's per-interview cost is projected arithmetic,
    # never a measurement -- this is how it gets checked against reality, on
    # every real run. Read defensively: the offline stub carries no usage at
    # all, and every offline test in this suite drives this exact line.
    if extraction.usage_input_tokens is not None or extraction.usage_output_tokens is not None:
        usage_str = (
            f" usage_input_tokens={extraction.usage_input_tokens} "
            f"usage_output_tokens={extraction.usage_output_tokens}"
        )
    else:
        usage_str = " usage=not reported by this response"
    print(
        f"Extraction summary: accepted={len(extraction.accepted)} "
        f"rejected={len(extraction.rejections)} "
        f"demoted={len(extraction.demotions)}{usage_str}"
    )
    # The standing tally across every session extracted so far, not just this
    # run's. One session's rejection rate says very little; the reason this
    # file is kept is to make the question answerable over the corpus.
    tally = coach_rejection_store.count_by_reason(cfg.context_dir)
    if tally:
        total = sum(tally.values())
        breakdown = ", ".join(
            f"{reason}={count} ({count / total:.0%})" for reason, count in tally.items()
        )
        print(f"Rejections on record ({total} across all sessions): {breakdown}")
    print(f"Rejections written to: {rejections_path}")
    # The same standing tally for demotions, and for the same reason: one
    # session's demotion count is noise, and the question worth asking of the
    # extraction model ("how often does it mislabel?") is a rate over every
    # session ever extracted.
    demotion_tally = coach_demotion_store.count_by_reason(cfg.context_dir)
    if demotion_tally:
        demoted_total = sum(demotion_tally.values())
        demotion_breakdown = ", ".join(
            f"{reason}={count} ({count / demoted_total:.0%})"
            for reason, count in demotion_tally.items()
        )
        print(f"Demotions on record ({demoted_total} across all sessions): {demotion_breakdown}")
    print(f"Demotions written to: {demotions_path}")
    # KEYED ON REJECTIONS ALONE, and it must stay that way. A rejection
    # DELETED a fact; a demotion KEPT one with a corrected label. Folding
    # demotions in here would fail a run over a session the pipeline handled
    # exactly as designed, and the pressure that creates is to stop demoting
    # -- i.e. to accept the bad label. `test_demotion_store.py` pins this.
    return 1 if extraction.rejections else 0


CRITIQUE_DOC_FILENAME = coach_asked_store.CRITIQUE_DOC_FILENAME


def run_critique_questions(args: argparse.Namespace) -> int:
    """`python -m app critique-questions <session>` -- bands what the
    INTERVIEWER asked, against the rubric the coach itself is held to.

    A separate command rather than a stage of `render` on purpose: `render` has
    never made a network call, and every surface that reads a transcript reads
    it through `render`. Giving it a vendor dependency would put a paid call
    behind opening a session. This writes a durable artifact instead, which
    `render` can read like any other file.

    Ordering follows `run_extract` §K7a/§K7b: everything that can fail for free
    fails before anything reaches the vendor, and the unclosed/truncated
    warnings print before the call so a user who did not mean to re-run
    mid-interview can still interrupt."""
    cfg = _apply_project(config_mod.load_config(), args)
    candidate = Path(args.session)
    session_dir = candidate if candidate.is_dir() else cfg.sessions_root / args.session
    if not session_dir.is_dir():
        print(f"ERROR: no session directory found at {session_dir}", file=sys.stderr)
        return 1

    read_result = read_session(session_dir)
    turns = coach_asked.interviewer_turns(
        read_result.records, min_words=args.min_words
    )
    if not turns:
        print(
            f"No interviewer turns worth judging in {session_dir.name} -- nothing written, "
            "nothing called.",
            file=sys.stderr,
        )
        return 1

    if not read_result.closed:
        print(
            f"WARNING: session {session_dir.name} has no closed_at marker -- it may still be "
            "in progress. The critique written now will look final and may not be.",
            file=sys.stderr,
        )
    if read_result.truncated_tail is not None:
        print(
            f"WARNING: session {session_dir.name}'s transcript has a truncated final line -- "
            "the critique covers the turns that did parse, and nothing after that point.",
            file=sys.stderr,
        )

    try:
        client = reasoning_client.build_client()
    except reasoning_client.CoachConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        hypotheses = coach_corpus.load_hypotheses(cfg.context_dir, missing_ok=True)
    except coach_corpus.CorpusValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Judging {len(turns)} interviewer turns from {session_dir.name}...")
    # NOTE: `is_generic` is always 0 in this report -- `critique_turns` cannot
    # supply the trigger utterance a batch of turns would each need. See its
    # own comment; the flag is honestly absent here rather than invented.
    critique = coach_asked_critique.critique_turns(
        client,
        turns,
        hypotheses=hypotheses,
        model=reasoning_client.resolve_model(),
        # NOT `cfg.coach_critique_*`: those are tuned for the LIVE path, where
        # the whole budget is 8 seconds and a 1000-token cap is a deliberate
        # latency choice. Here there is no latency budget at all, and that cap
        # across a batch of four rationales is a truncation -- which the critic
        # treats as an untrusted response, failing the entire batch. The
        # offline harness's values are the ones that match this workload.
        max_tokens=coach_asked_critique.DEFAULT_MAX_TOKENS,
        timeout_s=coach_asked_critique.DEFAULT_TIMEOUT_S,
    )

    payload = coach_asked_critique.to_json_dict(critique, session_id=session_dir.name)
    # The write lives in `app/coach/asked_store.py`, not here -- see that
    # module's docstring: holding both a write primitive and the string
    # `hypotheses.json` in one file joins gate 19's pinned set of hypothesis
    # writers, and admitting `main.py` to it would retire that gate.
    written_path = coach_asked_store.write_critique(session_dir, payload)

    summary = payload["summary"]
    rate = summary["clean_rate"]
    rate_str = "n/a (nothing could be judged)" if rate is None else f"{rate:.0%}"
    print(f"Critique written to: {written_path}")
    print(
        f"Asked-question summary: turns={summary['turns']} judged={summary['judged']} "
        f"unjudged={summary['unjudged']} clean={summary['clean']} clean_rate={rate_str}"
    )

    def _rate(value: "float | None") -> str:
        return "n/a" if value is None else f"{value:.0%}"

    # The combined rate above answers two questions at once, which is what
    # made it hard to read. Printed with the labels spelled out because
    # "unmarked" is NOT "statements" -- it holds the leading statements AND
    # the roughly half of real questions the transcript gave no '?'.
    print(
        f"  by punctuation: marked questions={summary['marked_questions']} "
        f"clean={_rate(summary['marked_question_clean_rate'])} | "
        f"unmarked (leading statements + unpunctuated questions) "
        f"clean={_rate(summary['unmarked_clean_rate'])}"
    )
    print(
        f"  flags: leading={summary['leading']} "
        f"hypothetical={summary['hypothetical_or_future_tense']} "
        f"pitch={summary['pitch']} opinion_seeking={summary['opinion_seeking']} "
        f"confirmation_only={summary['confirmation_only']}"
    )
    # A7: the rubric was calibrated on questions a model PROPOSED, not on real
    # asked speech. Printed on every run rather than buried in a docstring,
    # because a percentage is exactly the kind of number that gets quoted
    # without its caveat.
    print(
        "  NOTE: the rubric was calibrated on PROPOSED questions, not on real asked "
        "speech. Read these bands; do not report them as calibrated."
    )
    return 1 if critique.unjudged_count else 0


# --- `summary` and `standing`: the post-interview arithmetic, on a terminal ---------
#
# Every number these two print has existed since `app/coach/interview_summary.py`
# did, and the hub and the MCP server have both been serving it. The CLI reached
# none of it: an operator who had just run `extract` could see that N facts were
# written and nothing else without opening a browser. This is that surface, and
# it is only a surface -- nothing below computes a standing, a count or a delta.
#
# **These commands never print verbatim customer speech.** Facts are projected
# onto `interview_summary.FactRef` the moment they are loaded and the `Fact`
# objects are dropped, so a quote is not merely unprinted here, it is not held.
# `FactRef` has no field one could sit in. Claims, hypothesis statements and
# criteria ARE printed: a claim is the abstraction extraction produced, and a
# statement is something the operator wrote.


def _session_dir_for_reading(
    cfg: "config_mod.AppConfig", requested: str
) -> "Optional[Path]":
    """A session id or a full path, resolved the way `extract` and
    `critique-questions` resolve theirs. `None` when nothing is there."""
    candidate = Path(requested)
    session_dir = candidate if candidate.is_dir() else cfg.sessions_root / requested
    return session_dir if session_dir.is_dir() else None


def _read_corpus_refs(
    cfg: "config_mod.AppConfig", *, project_label: str
) -> "tuple[tuple, tuple, bool]":
    """This node's hypotheses and its facts as quote-free references.

    Returns `(hypotheses, refs, corpus_exists)`. `corpus_exists` is captured
    rather than re-derived from `refs` being empty, exactly as the hub captures
    it: a corpus that was never written and a corpus holding nothing are
    different answers with different remedies, and an empty tuple cannot tell
    them apart.

    Raises `CorpusValidationError`; the callers turn that into a message."""
    hypotheses = coach_corpus.load_hypotheses(cfg.context_dir, missing_ok=True)
    corpus_exists = (cfg.context_dir / coach_corpus.FACTS_FILENAME).is_file()
    facts = (
        coach_corpus.load_facts(cfg.context_dir, sessions_root=None) if corpus_exists else ()
    )
    return (
        hypotheses,
        tuple(summary_mod.fact_ref(fact, project=project_label) for fact in facts),
        corpus_exists,
    )


def _print_table(headers: "Sequence[str]", rows: "Sequence[Sequence[str]]", aligns: str) -> None:
    """One aligned table. `aligns` is one character per column, `l` or `r`.

    Widths come from the data as well as the headings, because a count that
    outgrows its heading and pushes the row out of line is exactly how a
    terminal table stops being scannable."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def _line(cells: "Sequence[str]") -> str:
        return "  " + "  ".join(
            cell.rjust(width) if align == "r" else cell.ljust(width)
            for cell, width, align in zip(cells, widths, aligns)
        ).rstrip()

    print(_line(headers))
    for row in rows:
        print(_line(row))


def _node_label(args: argparse.Namespace) -> str:
    return getattr(args, "project", None) or "the root context/ and sessions/"


def run_summary(args: argparse.Namespace) -> int:
    """`python -m app summary <session>` -- what one interview produced.

    Deliberately answers only what the corpus can answer deterministically. A
    paragraph about what the interviewee meant would be a model call, and the
    claims it would restate are already listed below with their hypothesis
    bearings; a generated summary of them would be a second, unfalsifiable
    account of the same evidence.

    An interview that produced nothing is a real result and is reported as one,
    with exit 0. Only a session id that names nothing on disk fails: that is the
    operator's typo, and returning 0 for it would let a scripted close-out
    report success for an interview it never read."""
    cfg = _apply_project(config_mod.load_config(), args)
    session_dir = _session_dir_for_reading(cfg, args.session)
    if session_dir is None:
        print(
            f"ERROR: no session directory found for {args.session!r} under {cfg.sessions_root}",
            file=sys.stderr,
        )
        return 1
    session_id = session_dir.name

    try:
        hypotheses, refs, corpus_exists = _read_corpus_refs(
            cfg, project_label=getattr(args, "project", None) or ""
        )
    except coach_corpus.CorpusValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Interview {session_id} -- {_node_label(args)}")

    if not corpus_exists:
        # Named as ABSENCE, not as zero. "0 claims extracted" is a finding about
        # the interview; a corpus nobody has written yet is a step not taken.
        print(f"  No fact corpus at {cfg.context_dir / coach_corpus.FACTS_FILENAME} yet.")
        print(f"  Run: python -m app extract {session_id}")
        return 0

    summary = summary_mod.summarize_interview(session_id, refs)
    if summary.fact_count == 0:
        # The corpus EXISTS and holds nothing from this interview. What is not
        # knowable is which of two reasons that is -- this session was never
        # extracted, or it was and the model found nothing worth recording.
        # Extraction leaves no record of having run, so nothing on disk
        # separates them, and the wording says so rather than picking one.
        print(
            f"  Nothing from this interview is in the corpus, which holds "
            f"{len(refs)} claim(s) from {summary_mod.count_interviews(refs)} interview(s)."
        )
        print(
            f"  Either extraction has not been run on {session_id}, or it ran and found "
            "nothing worth recording -- nothing on disk distinguishes those."
        )
        return 0

    known_ids = {hypothesis.id for hypothesis in hypotheses}
    known_touches = tuple(t for t in summary.touches if t.hypothesis_id in known_ids)
    # A bearing on an id since renamed or deleted is not a connection the
    # operator can act on. Counted separately rather than folded in, so the
    # headline number never reports the interview as better connected than the
    # project is -- and never silently loses the claims either.
    orphaned = tuple(t for t in summary.touches if t.hypothesis_id not in known_ids)

    print(f"  claims extracted from this interview   {summary.fact_count}")
    # The split the fact count alone hides: an interview whose claims are mostly
    # the model's inference is a weaker result than the same count of things the
    # interviewee actually reported.
    print(f"    reported by the interviewee            {summary.evidence_facts}")
    print(f"    inferred by the model                  {summary.inference_facts}")
    print(
        f"  hypotheses this interview bore on      {len(known_touches)}"
        f"   (of {len(hypotheses)} in this project)"
    )
    print()

    if not known_touches:
        print("  This interview's claims bear on none of this project's hypotheses.")
        print(
            "  That can be a real finding -- an interview about something else genuinely "
            "bears on nothing -- or a sign the hypotheses were not loaded when extraction ran."
        )
    else:
        # Said above the table, because these are the one set of numbers on
        # either command that count CLAIMS. Inside a single interview the
        # interview count is 1 by definition, so a claim count is the only
        # meaningful one -- but `standing` prints interview counts in columns
        # that look identical, and a reader must never have to remember which.
        print("  Bearings, counted in CLAIMS (within one interview, the interview count is 1):")
        _print_table(
            ("HYPOTHESIS", "SUPPORTING", "CONTRARY", "UNCLEAR"),
            [
                (
                    touch.hypothesis_id,
                    str(touch.supporting_facts),
                    str(touch.contrary_facts),
                    str(touch.unclear_facts),
                )
                for touch in known_touches
            ],
            "lrrr",
        )

    if orphaned:
        print()
        print(
            "  Claims bearing on hypotheses this project no longer has "
            f"(renamed or deleted since extraction): {', '.join(t.hypothesis_id for t in orphaned)}"
        )
    return 0


def _delta_cell(before: int, now: int) -> str:
    return f"{before} -> {now}"


def _contribution_phrase(touch: "summary_mod.HypothesisTouch") -> str:
    """This interview's contribution in words rather than three more columns.

    Words because these are CLAIM counts sitting on a row of INTERVIEW counts.
    Three more bare integers there would read as more of the same number and be
    wrong by a factor nobody could see."""
    parts = [
        f"{count} {name}"
        for count, name in (
            (touch.supporting_facts, "supporting"),
            (touch.contrary_facts, "contrary"),
            (touch.unclear_facts, "unclear"),
        )
        if count
    ]
    return ", ".join(parts) if parts else "-"


def run_standing(args: argparse.Namespace) -> int:
    """`python -m app standing [<session>]` -- where each hypothesis stands.

    **Across the project NODE, not interview by interview.** Every column here
    collapses each interview into one vote whichever way it argued, and
    `total_interviews` counts interviews that touched the hypothesis at all --
    never the sum of the three columns, which would count an interview that both
    supported and undermined it twice.

    With a session named, each row also shows what the hypothesis stood at
    BEFORE that interview. That is `standing_delta`'s answer, the same one the
    hub renders -- re-derived here it would be a second confident integer with
    nothing to say which was right."""
    cfg = _apply_project(config_mod.load_config(), args)

    session_id: "Optional[str]" = None
    if args.session is not None:
        session_dir = _session_dir_for_reading(cfg, args.session)
        if session_dir is None:
            print(
                f"ERROR: no session directory found for {args.session!r} under "
                f"{cfg.sessions_root}",
                file=sys.stderr,
            )
            return 1
        session_id = session_dir.name

    try:
        hypotheses, refs, corpus_exists = _read_corpus_refs(
            cfg, project_label=getattr(args, "project", None) or ""
        )
    except coach_corpus.CorpusValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Hypothesis standing -- {_node_label(args)}")
    if not hypotheses:
        # Not an empty table. Zero rows and a corpus full of facts look the
        # same on a terminal, and only one of them is a step not taken.
        print(f"  No hypotheses in {cfg.context_dir} -- there is nothing to stand.")
        print("  Write some first: python -m app author")
        return 0

    print(
        f"  {len(hypotheses)} hypotheses; {len(refs)} extracted claim(s) from "
        f"{summary_mod.count_interviews(refs)} interview(s)."
    )
    if not corpus_exists:
        print(f"  No fact corpus at {cfg.context_dir / coach_corpus.FACTS_FILENAME} yet -- "
              "every count below is zero because nothing has been extracted.")
    # Said on every run, not left to a docstring. The whole reason this counting
    # rule lives in one module is that a per-claim tally lets one talkative
    # interviewee outweigh four separate people, and a column heading cannot
    # carry that on its own.
    print("  Columns count INTERVIEWS, not claims: one interviewee repeating a point counts once.")
    print("  INTERVIEWS is how many touched the hypothesis at all -- never the sum of the three.")
    if session_id is not None:
        print(f"  'before -> now' is the corpus without {session_id}, then with it.")
    print()

    # ONE pass over the corpus for the contribution, computed here and handed to
    # every row -- `summarize_interview` reads every fact on disk, and its answer
    # cannot differ between hypotheses. Computed inside the loop it would be one
    # identical pass per hypothesis, and every one of them correct.
    interview = None if session_id is None else summary_mod.summarize_interview(session_id, refs)
    deltas = [
        (
            hypothesis,
            summary_mod.standing_delta(
                hypothesis,
                refs,
                session_id=session_id,
                project=getattr(args, "project", None) or "",
                summary=interview,
            ),
        )
        for hypothesis in hypotheses
    ]

    if session_id is None:
        headers = ("ID", "STATE", "SCORABLE", "SUPPORTING", "CONTRARY", "UNCLEAR", "INTERVIEWS")
        aligns = "lllrrrr"
        rows = [
            (
                h.id,
                delta.now.state,
                "yes" if delta.now.is_scorable else "no",
                str(delta.now.supporting_interviews),
                str(delta.now.contrary_interviews),
                str(delta.now.unclear_interviews),
                str(delta.now.total_interviews),
            )
            for h, delta in deltas
        ]
    else:
        headers = (
            "ID", "STATE", "SCORABLE", "SUPPORTING", "CONTRARY", "UNCLEAR", "INTERVIEWS",
            "THIS INTERVIEW (claims)",
        )
        aligns = "lllrrrrl"
        rows = [
            (
                h.id,
                delta.now.state,
                "yes" if delta.now.is_scorable else "no",
                _delta_cell(delta.before.supporting_interviews, delta.now.supporting_interviews),
                _delta_cell(delta.before.contrary_interviews, delta.now.contrary_interviews),
                _delta_cell(delta.before.unclear_interviews, delta.now.unclear_interviews),
                str(delta.now.total_interviews),
                # MOVED is the stronger claim and it is `StandingDelta`'s to
                # make: a claim the model could not direct is the absence of
                # direction, so `unclear` alone must never light it.
                _contribution_phrase(delta.contributed) + ("  MOVED" if delta.moved else ""),
            )
            for h, delta in deltas
        ]

    _print_table(headers, rows, aligns)
    print()

    # The statements, because an id is not a hypothesis. Printed once below the
    # table rather than in a column, so the numbers stay aligned and readable --
    # a statement is a sentence and would blow every column width out.
    id_width = max(len(h.id) for h, _ in deltas)
    for hypothesis, delta in deltas:
        print(f"  {hypothesis.id.ljust(id_width)}  {hypothesis.statement}")
        if not delta.now.is_scorable:
            # Not a footnote: a tally against a hypothesis with no
            # pre-registered rule is the target painted round the arrow.
            print(
                " " * (id_width + 4)
                + "no pre-registered criterion -- these counts cannot honestly score it"
            )

    untouched = tuple(h.id for h, delta in deltas if delta.now.total_interviews == 0)
    if untouched:
        print()
        print(
            f"  No interview has borne on: {', '.join(untouched)}. "
            "A zero row is an unasked question, not a refuted belief."
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            return run_start(args)
        if args.command == "render":
            return run_render(args)
        if args.command == "precheck":
            return run_precheck(args)
        if args.command == "author":
            return run_author(args)
        if args.command == "extract":
            return run_extract(args)
        if args.command == "critique-questions":
            return run_critique_questions(args)
        if args.command == "summary":
            return run_summary(args)
        if args.command == "standing":
            return run_standing(args)
        if args.command == "export":
            return run_export(args)
        if args.command == "console":
            return run_console(args)
        if args.command == "hub":
            return run_hub(args)
        if args.command == "overlay":
            from app.overlay.__main__ import run as run_overlay

            # Imported HERE, not at module scope: `app/main.py` is imported by
            # every command and by the test suite, and tkinter must not be a
            # module-scope import on a machine or CI runner with no display.
            return run_overlay()
    except projects_mod.ProjectPathError as exc:
        # Caught at the dispatch boundary rather than in each `run_*`, so
        # there is exactly one place a rejected project path is turned into
        # an operator-facing message. A traceback here would be the wrong
        # shape twice over: `--project` is user input, and the exception's
        # own message already names the path and the rule it broke.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
