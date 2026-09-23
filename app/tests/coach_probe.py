"""Drives the REAL coach from a scripted English interview, on localhost.

Run as `python -m app.tests.coach_probe` -- a subprocess entry point, not
something pytest should collect, following the `rtl_probe.py`/`crash_writer.py`
precedent.

WHY THIS EXISTS. Nothing in this repository has ever driven the coach against
a real interview. `rtl_probe.py` builds the real pipeline but wires no coach;
`test_coach_audit.py` drives a real `SuggestionPipeline` but against
`stub.py`'s canned responses with no surface; `app/stt/replay.py` is not a
transcript replayer at all (it emits one hardcoded segment). So the coach's
SUBSTANCE -- does it propose a question worth asking? -- has never been
observed end to end. This rig is that observation.

WHAT IS REAL HERE. The `SessionStore`, both `StreamWorker`s, the
`CoverageDetector`, the `SuggestionPipeline`, the `ConnectionHub`, the uvicorn
server, the overlay page, and the Anthropic calls are all the production
objects on the production code path -- the coach is attached through exactly
the `segment_hook` wiring `app.main.run_start` uses, and the suggestions reach
the page through the same thread->asyncio bridge and the same wire payload.

WHAT IS NOT REAL, stated plainly so no one over-reads a result:
1. There is no audio. Each interview turn is injected as ONE segment via
   `FakeStreamingEngine`. Real Soniox emits ~5-word segments (median 5.0 words
   across both 2026-08-09 calls), so the coach here sees whole tidy turns
   rather than a stream of fragments. That flatters the coach's INPUT. It does
   not flatter its OUTPUT, which is what this rig is for.
2. Turn timing is synthetic, so the latency figures printed at the end measure
   the coach's own cycle cost and nothing about capture.
3. The transcript is a real interview, but a research-methods one rather than
   a product-discovery call -- the shape is right, the domain is borrowed.

BIND. Localhost only (`_HOST`), so the overlay opens in a browser on this
machine and nothing is exposed to the LAN. The per-run token is still
required on every route -- loopback stops the network, not other pages open
in this browser, which is the same reasoning the authoring server already
follows.

TRANSCRIPT. Read from `_SCRIPT_PATH`, which is gitignored: the source is
CC BY-SA 4.0 and is deliberately not vendored into this repository. See the
`_source`/`_licence` keys inside that file.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import queue
import sys
import time
from pathlib import Path

from app import config as config_mod
from app import main as main_mod
from app import projects as projects_mod
from app.capture.stream import CaptureFrame
from app.coach import reasoning_client
from app.coach.coverage import CoverageDetector
from app.coach.pipeline import SuggestionPipeline
from app.session.store import SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.stt.live import StreamWorker
from app.web import server as web_server
from app.web.hub import ConnectionHub
from app.web.join import generate_join_token

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "benchmark" / "data" / "real" / "coach-probe-script.json"
)

_HOST = "127.0.0.1"
_CADENCE_BYTES = 3200  # one push of exactly this many bytes triggers exactly one emission
_SILENT_FRAME = bytes(_CADENCE_BYTES)

# Seconds between turns generally...
_TURN_INTERVAL_S = 2.0
# ...and the floor between two INTERVIEWEE turns. The suggestion pipeline's
# queue is maxsize=1 with drop-oldest, and one cycle is two sequential vendor
# calls (2.5s + 1.5s timeouts). Push interviewee turns closer together than
# this and cycles are silently dropped as `coach-cycle-skipped` -- the coach
# would look quiet for a reason that has nothing to do with the coach.
_MIN_INTERVIEWEE_GAP_S = 7.0

logger = logging.getLogger(__name__)


def _load_turns(limit: "int | None") -> "list[dict]":
    if not _SCRIPT_PATH.is_file():
        raise SystemExit(
            f"No interview script at {_SCRIPT_PATH}.\n"
            "It is gitignored on purpose (the source transcript is CC BY-SA 4.0 and is not "
            "vendored here). Regenerate it before running this probe."
        )
    payload = json.loads(_SCRIPT_PATH.read_text(encoding="utf-8"))
    turns = payload["turns"]
    if limit is not None:
        turns = turns[:limit]
    return turns


def _echoing_callback(inner, turns_seen: "dict[str, int]"):
    """Wraps `main._suggestion_broadcast_callback` so the run is readable in
    the terminal as well as on the overlay. The broadcast is called FIRST and
    unconditionally -- printing must never be able to suppress the thing being
    tested."""

    def _callback(ranked) -> None:
        try:
            inner(ranked)
        finally:
            turns_seen["suggestions"] += 1
            print(f"\n  >> COACH ({turns_seen['suggestions']}):")
            for i, candidate in enumerate(ranked):
                label = "primary" if i == 0 else f"alt {i}"
                move = getattr(candidate, "move", "?")
                target = getattr(candidate, "target_hypothesis_id", None)
                suffix = f"  [move={move}" + (f" target={target}" if target else "") + "]"
                print(f"     {label}: {candidate.text}{suffix}")

    return _callback


async def _run_probe(
    *, limit: "int | None", interval_s: float, wait_for_overlay: bool, project: "str | None"
) -> int:
    turns = _load_turns(limit)
    cfg = config_mod.load_config()

    # Phase A. This probe reads a context directory, so it has to make the
    # same project choice every other reader does -- and it broke when the
    # eval data moved out of the root, because it was silently reading
    # `cfg.context_dir` and nothing pointed that at a project.
    #
    # Gate 30 did NOT catch that: it scans `app/main.py` only, which is right
    # for a gate about the CLI's own commands and is exactly why this file
    # slipped through. Noted here rather than widened -- a probe is a
    # developer tool, and making the gate scan every module that loads config
    # would pull test rigs and one-off scripts into a rule written about
    # product commands.
    if project is not None:
        paths = projects_mod.resolve(project)
        cfg = dataclasses.replace(
            cfg, context_dir=paths.context_dir, sessions_root=paths.sessions_root
        )
    print(f"Project: {project if project is not None else '(root context/ and sessions/)'}")

    # Fail before a session directory exists if the coach cannot run at all --
    # this rig has no non-coach purpose, so degrading to a silent transcript
    # would be a worse outcome than refusing.
    try:
        client = reasoning_client.build_client()
    except reasoning_client.CoachConfigError as exc:
        raise SystemExit(f"This probe exists to exercise the coach, and the coach cannot start: {exc}")

    context_load_started = time.perf_counter()
    # `extra_fact_dirs` is named explicitly and empty: this rig grounds the
    # coach in ONE node, so a cross-hierarchy read would change what the run
    # measures without saying so.
    brief, survey, hypotheses = main_mod._load_coach_context(
        cfg.context_dir, extra_fact_dirs=()
    )
    grounding_s = time.perf_counter() - context_load_started
    print(
        f"Coach context: brief {len(brief)} item(s), survey {len(survey)} item(s), "
        f"hypotheses {len(hypotheses)} item(s), loaded in {grounding_s:.2f}s"
    )
    if not hypotheses:
        print(
            "WARNING: zero hypotheses loaded. The `test_hypothesis` move and the "
            "`is_confirmation_only` critic flag both judge against an empty set, so this run "
            "cannot say anything about the coach's grounded behaviour."
        )

    session_dir = create_session_dir(cfg.sessions_root)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake", "probe": "coach_probe", "script": _SCRIPT_PATH.name})
    print(f"Session path: {store.session_dir}")

    loop = asyncio.get_running_loop()
    hub = ConnectionHub()

    coverage_detector = CoverageDetector(
        client=client,
        store=store,
        survey=survey,
        model=reasoning_client.DEFAULT_MODEL,
        timeout_s=cfg.coach_coverage_timeout_s,
        queue_maxsize=cfg.coach_queue_maxsize,
    )
    coverage_detector.start()

    counters = {"suggestions": 0}
    suggestion_pipeline = SuggestionPipeline(
        client=client,
        store=store,
        brief=brief,
        survey=survey,
        hypotheses=hypotheses,
        session_start_grounding_s=grounding_s,
        coverage_state_provider=lambda: coverage_detector.state,
        model=reasoning_client.DEFAULT_MODEL,
        generate_timeout_s=cfg.coach_generate_timeout_s,
        critique_timeout_s=cfg.coach_critique_timeout_s,
        generate_max_tokens=cfg.coach_generate_max_tokens,
        critique_max_tokens=cfg.coach_critique_max_tokens,
        max_shown=cfg.coach_max_shown,
        failure_threshold=cfg.coach_failure_threshold,
        language_window_s=cfg.coach_language_window_s,
        language_flip_margin=cfg.coach_language_flip_margin,
        language_min_chars=cfg.coach_language_min_chars,
        queue_maxsize=cfg.coach_pipeline_queue_maxsize,
        on_ranked=_echoing_callback(
            main_mod._suggestion_broadcast_callback(loop, hub, store, max_shown=cfg.coach_max_shown),
            counters,
        ),
    )
    suggestion_pipeline.start()

    # One engine per speaker, each cycling ONLY its own speaker's lines in
    # order -- the interleaving lives in the drive order below, exactly as in
    # rtl_probe. `interviewee` is the loopback side, i.e. the CUSTOMER being
    # interviewed; `interviewer` is the microphone, i.e. us. That reads
    # backwards and is worth re-checking whenever this file is edited.
    texts = {
        speaker: tuple(t["text"] for t in turns if t["speaker"] == speaker)
        for speaker in (main_mod.SPEAKER_INTERVIEWEE, main_mod.SPEAKER_INTERVIEWER)
    }
    queues: "dict[str, queue.Queue[CaptureFrame]]" = {
        speaker: queue.Queue() for speaker in texts
    }
    workers = {
        speaker: StreamWorker(
            engine=FakeStreamingEngine(
                texts=texts[speaker], cadence_bytes=_CADENCE_BYTES, detected_language="en",
            ),
            frame_queue=queues[speaker],
            store=store,
            speaker=speaker,
            sample_rate=cfg.sample_rate,
            loop=loop,
            broadcast=hub.broadcast,
            segment_hook=(
                coverage_detector.submit
                if speaker == main_mod.SPEAKER_INTERVIEWER
                else suggestion_pipeline.submit
            ),
        )
        for speaker in texts
    }
    for worker in workers.values():
        worker.start()

    token = generate_join_token()
    app = web_server.create_app(
        token=token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id=session_dir.name, server_started_at="coach-probe",
    )
    server = web_server.build_uvicorn_server(app, host=_HOST, port=cfg.app_port)
    serve_task = asyncio.create_task(server.serve())

    overlay_url = f"http://{_HOST}:{cfg.app_port}/s/{token}"
    print("\n" + "=" * 78)
    print("OVERLAY (localhost only -- nothing is exposed to the LAN):")
    print(f"  {overlay_url}")
    print("=" * 78)
    print(
        f"\n{len(turns)} turns "
        f"({len(texts[main_mod.SPEAKER_INTERVIEWER])} interviewer / "
        f"{len(texts[main_mod.SPEAKER_INTERVIEWEE])} interviewee). "
        f"Every interviewee turn triggers one real coaching cycle "
        f"(2 Anthropic calls, {reasoning_client.DEFAULT_MODEL})."
    )

    # D-01: the surface never replays missed transcript, so anything emitted
    # before the browser is on the socket is lost for this run. Wait for a real
    # connection rather than racing the human opening a tab.
    if wait_for_overlay:
        print("\nOpen the URL above, then this will start. Waiting for the overlay to connect...")
        while hub.connection_count == 0:
            await asyncio.sleep(0.25)
        print("Overlay connected. Starting in 2s.\n")
    else:
        # --no-wait exists so the coach can be exercised without a human at a
        # browser. The suggestions still go through the real broadcast path;
        # they simply reach zero clients, and the terminal echo is the record.
        print("\n--no-wait: starting without an overlay client. Suggestions print here only.\n")
    await asyncio.sleep(2.0)

    last_interviewee_at = 0.0
    try:
        for i, turn in enumerate(turns, start=1):
            speaker = turn["speaker"]
            if speaker == main_mod.SPEAKER_INTERVIEWEE:
                gap = _MIN_INTERVIEWEE_GAP_S - (time.monotonic() - last_interviewee_at)
                await asyncio.sleep(max(interval_s, gap))
                last_interviewee_at = time.monotonic()
            else:
                await asyncio.sleep(interval_s)
            queues[speaker].put(
                CaptureFrame(
                    seq=0, pcm16=_SILENT_FRAME, kind="synthesized",
                    captured_monotonic=time.monotonic(),
                )
            )
            preview = turn["text"][:88].replace("\n", " ")
            print(f"[{i}/{len(turns)}] {speaker:11s} {preview}...")
        if wait_for_overlay:
            print("\nScript finished. Leaving the server up so you can read the overlay -- Ctrl-C to stop.")
            while not server.should_exit:
                await asyncio.sleep(0.5)
        else:
            # Let the last cycle land rather than tearing down mid-flight and
            # recording it as a failure.
            print("\nScript finished. Draining the last coaching cycle...")
            await asyncio.sleep(_MIN_INTERVIEWEE_GAP_S)
    except KeyboardInterrupt:
        print("\nStopping probe.")
    finally:
        for worker in workers.values():
            worker.stop()
        # Both coach threads MUST stop before the store closes: the pipeline
        # re-reads the session and appends suggestion records from its own
        # thread, and a cycle landing after `store.close()` is recorded as a
        # failure rather than a result.
        coverage_detector.stop()
        suggestion_pipeline.stop()
        server.should_exit = True
        try:
            await serve_task
        except Exception as exc:  # noqa: BLE001 -- a teardown failure must never hide behind a hang
            logger.error("probe server failed error_type=%s", type(exc).__name__)

        summary = suggestion_pipeline.latency_summary()
        health = suggestion_pipeline.health
        store.close()

        print("\n" + "=" * 78)
        print(f"Suggestions broadcast : {counters['suggestions']}")
        print(f"Coach health          : {health}")
        print(f"Latency summary       : {summary}")
        print(f"Transcript            : {store.transcript_path}")
        print(
            "\nRead the transcript for what the terminal cannot show: `suggestion` records carry "
            "EVERY candidate with its band, flags and rationale (including the ones dropped and "
            "never shown), and `reasoning-timeout` / `coach-cycle-skipped` events explain any gap."
        )
        print("=" * 78)

    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Drive the real coach from a scripted interview.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only play the first N turns (a full run is ~90 turns and ~45 coaching cycles)",
    )
    parser.add_argument(
        "--interval", type=float, default=_TURN_INTERVAL_S,
        help=f"seconds between turns (default {_TURN_INTERVAL_S}); interviewee turns are additionally "
             f"spaced at least {_MIN_INTERVIEWEE_GAP_S}s apart so no cycle is dropped",
    )
    parser.add_argument(
        "--project", default=None,
        help="project path to read the brief, survey and hypotheses from (e.g. _eval); "
             "omit to use the root context/ directory",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="do not wait for an overlay client; run headless and print suggestions to the terminal",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return asyncio.run(
        _run_probe(
            limit=args.limit, interval_s=args.interval,
            wait_for_overlay=not args.no_wait, project=args.project,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
