"""D-04's real-phone verification rig (03-01-PLAN.md Task 3). Run as
`python -m app.tests.rtl_probe` -- a subprocess entry point, not something
pytest should collect, following the `crash_writer.py`/`playback.py`
precedent.

Builds the SAME real objects `app.main.run_start` builds -- a real join
surface (token, LAN IP, QR), a real `ConnectionHub`, a real embedded uvicorn
server, a real `SessionStore`, and two real `StreamWorker` instances -- but
drives each worker from an in-memory frame queue instead of a capture device,
so this runs with no audio hardware, no STT credentials, and no call in
progress while a human stands there holding their phone.

Every string goes through `StreamWorker._persist` -> `SessionStore.append_segment`
-> the real thread -> asyncio bridge -> a real WebSocket broadcast -- never
around that path. The point is to prove the REAL pipeline renders correctly
on a real phone, not that a hand-written string can be pushed down a socket
some other way.

The seven strings are the D-04 verification set from `03-RESEARCH.md` §3,
used verbatim. They are split across the two speakers so alternating them
exercises both per-speaker attribution AND the two-region layout, not just
one speaker's blocks: `_INTERVIEWEE_STRINGS` holds table rows 1/3/5/7,
`_INTERVIEWER_STRINGS` holds rows 2/4/6, so driving them in table order
(`_SPEAKER_ORDER` below) alternates speakers naturally.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sys
import time
from pathlib import Path

from app import config as config_mod
from app.capture.stream import CaptureFrame
from app.session.store import SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.stt.live import StreamWorker
from app.web import server as web_server
from app.web.hub import ConnectionHub
from app.web.join import create_join

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

# Verbatim from 03-RESEARCH.md §3 -- table rows 1, 3, 5, 7.
_INTERVIEWEE_STRINGS = (
    "אז מה דעתך על ה-Slack שלנו?",
    "זה עלה לנו 1,200 ש\"ח בחודש שעבר, לא יותר מ-15%.",
    "Okay so, בעצם הבעיה המרכזית היא ה-onboarding, זה לוקח יותר מדי זמן.",
    "זה היה ROI נמוך, בערך 2%.",
)
# Verbatim from 03-RESEARCH.md §3 -- table rows 2, 4, 6.
_INTERVIEWER_STRINGS = (
    "אנחנו משתמשים ב-API הזה כבר שנה, זה ה-MVP שלנו.",
    "התהליך כולו מנוהל דרך ה-CRM.",
    "יש לנו 3 משתמשים (power users) שמשתמשים בזה כל יום.",
)

# Table order 1-7, alternating speakers by construction (see module docstring).
_SPEAKER_ORDER = (
    "interviewee", "interviewer", "interviewee", "interviewer",
    "interviewee", "interviewer", "interviewee",
)

_CADENCE_BYTES = 3200  # arbitrary -- one push of exactly this many bytes triggers exactly one emission
_SILENT_FRAME = bytes(_CADENCE_BYTES)
_EMIT_INTERVAL_S = 3.0  # a realistic pause between utterances, easy to watch arrive one at a time

logger = logging.getLogger(__name__)


async def _run_probe() -> int:
    cfg = config_mod.load_config()
    session_dir = create_session_dir(cfg.sessions_root)
    store = SessionStore(session_dir)
    store.write_metadata({"engine": "fake", "probe": "rtl_probe"})
    print(f"Session path: {store.session_dir}")

    loop = asyncio.get_running_loop()
    hub = ConnectionHub()

    interviewee_queue: "queue.Queue[CaptureFrame]" = queue.Queue()
    interviewer_queue: "queue.Queue[CaptureFrame]" = queue.Queue()

    interviewee_worker = StreamWorker(
        engine=FakeStreamingEngine(
            texts=_INTERVIEWEE_STRINGS, cadence_bytes=_CADENCE_BYTES, detected_language="he",
        ),
        frame_queue=interviewee_queue, store=store, speaker="interviewee",
        sample_rate=cfg.sample_rate, loop=loop, broadcast=hub.broadcast,
    )
    interviewer_worker = StreamWorker(
        engine=FakeStreamingEngine(
            texts=_INTERVIEWER_STRINGS, cadence_bytes=_CADENCE_BYTES, detected_language="he",
        ),
        frame_queue=interviewer_queue, store=store, speaker="interviewer",
        sample_rate=cfg.sample_rate, loop=loop, broadcast=hub.broadcast,
    )
    interviewee_worker.start()
    interviewer_worker.start()

    join_info = create_join(
        port=cfg.app_port, session_dir=session_dir, store=store, emit_qr=cfg.emit_qr
    )
    app = web_server.create_app(
        token=join_info.token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id=session_dir.name, server_started_at="rtl-probe",
    )
    server = web_server.build_uvicorn_server(app, host="0.0.0.0", port=cfg.app_port)
    serve_task = asyncio.create_task(server.serve())

    print("Scan the QR above (or the PNG that just opened) with the phone you will use during the interview.")
    print(f"Join URL: {join_info.join_url}")
    print(f"Emitting {len(_SPEAKER_ORDER)} code-switched utterances, alternating speakers, "
          f"one every {_EMIT_INTERVAL_S:.1f}s...")

    queues = {"interviewee": interviewee_queue, "interviewer": interviewer_queue}

    # D-01 means the surface never replays missed transcript, so a string
    # emitted before the phone is on the socket is lost for this run. Wait for
    # the real connection rather than racing a human scanning a QR code.
    print("Waiting for the phone to connect...")
    while hub.connection_count == 0:
        await asyncio.sleep(0.25)
    print("Phone connected. Starting in 2s -- look at the phone now.")
    await asyncio.sleep(2.0)

    try:
        for i, speaker in enumerate(_SPEAKER_ORDER, start=1):
            await asyncio.sleep(_EMIT_INTERVAL_S)
            queues[speaker].put(
                CaptureFrame(seq=0, pcm16=_SILENT_FRAME, kind="synthesized", captured_monotonic=time.monotonic())
            )
            print(f"  [{i}/{len(_SPEAKER_ORDER)}] sent ({speaker})")
        print("All seven strings sent. Leave this running while you inspect the phone -- Ctrl-C to stop.")
        while not server.should_exit:
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping probe.")
    finally:
        interviewee_worker.stop()
        interviewer_worker.stop()
        server.should_exit = True
        try:
            await serve_task
        except Exception as exc:  # noqa: BLE001 -- a probe teardown failure must never hide behind a hang
            logger.error("probe server failed error_type=%s", type(exc).__name__)
        store.close()
        print(f"Transcript written to: {store.transcript_path}")

    return 0


def main(argv: "list[str] | None" = None) -> int:
    del argv  # no arguments -- this rig is fixed to the D-04 seven strings
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return asyncio.run(_run_probe())


if __name__ == "__main__":
    sys.exit(main())
