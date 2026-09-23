"""The phone surface (phase_specification 03-01, D-05). Everything the live
transcript needs to reach a browser lives under this package: the join/token/QR
helper (`join.py`), the connection fan-out (`hub.py`), and the ASGI app +
uvicorn embedding (`server.py`).

`app.stt.live.StreamWorker` must never import anything from this package --
the dependency runs one way only (`app/web` depends on `app.session`/`app.stt`
shapes, never the reverse), so the STT worker stays engine-agnostic and
hub-agnostic and remains unit-testable with a plain callable standing in for
the broadcast (phase_specification §P3, STT-03's posture extended to Phase 3).
"""

from __future__ import annotations
