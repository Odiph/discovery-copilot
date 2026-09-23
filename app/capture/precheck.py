"""`python -m app precheck`: go or no-go before the interview starts.

A pre-call check that passes on a muted device is worse than no pre-call
check -- it converts an obvious failure into a confident one
(phase_specification T3). Every row asserts an OBSERVATION, never a
successful open: liveness is measured from the frames themselves (the share
tagged `real` plus their level), and the engine row proves the vendor
actually answers, not merely that a key is present.

Built on the exact modules `start` uses -- the same device resolution, the
same `CaptureStream`, the same engine registry -- never a parallel code
path, and every device/engine dependency is injectable so the pass path and
every FAIL path are testable without hardware or credentials.
"""

from __future__ import annotations

import asyncio
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from app import config as config_mod
from app.capture.devices import (
    DeviceInfo,
    DeviceOpenError,
    resolve_default_input_device,
    resolve_default_loopback_device,
)
from app.capture.levels import BAND_CLIPPING, LevelMeasurement, measure_level
from app.capture.stream import CaptureStream
from app.session.store import SessionStore
from app.stt.live import get_engine
from app.web import join as join_mod
from app.web import server as web_server
from app.web.hub import ConnectionHub
from benchmark.engines.base import StreamingEngine

RESULT_PASS = "PASS"
RESULT_FAIL = "FAIL"
RESULT_SKIPPED = "SKIPPED"

# app/capture/precheck.py -> parent is app/capture, parent.parent is app/ -- same static
# assets `app/main.py`'s embedded server serves, reused here so the reachability row
# exercises the REAL page/asset routes, never a parallel code path.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

PREFLIGHT_RULES = (
    'Fix your call output device BEFORE pressing start -- do NOT change it mid-call.',
    'Disable sleep / set the power plan to "never sleep" for the duration of the call.',
    "Loopback captures EVERYTHING the machine plays -- close other audio sources first.",
    "Confirm the printed LAN address matches your Wi-Fi adapter -- not a VPN or virtual "
    "adapter -- before trusting the QR (compare leading octets with your phone's own Wi-Fi IP).",
    'If a second monitor is in use, set your screen-share selection to that single window '
    'or display NOW -- never "entire desktop" or "all displays".',
)

# §R3's one-line triage rule, verbatim -- the remedy for a phone that never connects within
# the reachability window. Memorable under pressure is the whole point (03-RESEARCH §7).
_PHONE_TRIAGE_REMEDY = (
    "No phone connected within the window. Triage: slow or no response at all -> assume Wi-Fi "
    "client isolation, try the phone's personal hotspot first; fast refusal -> Windows Firewall "
    "blocked the port, accept the prompt or add an allow rule; loads but the printed IP looks "
    "wrong -> a VPN or virtual adapter owns the route, disconnect it and re-run precheck; loads "
    "fine but frozen -> not a network problem, look at the laptop's own status banner instead."
)

_LIVENESS_REMEDY = (
    "Play the call app's test audio (for the loopback row) or speak into the microphone "
    "(for the microphone row), then retry. A device that opens but stays silent still FAILS here."
)

_CLIPPING_REMEDY = (
    "The loudest half-second is at or above -1 dBFS -- the signal is clipping, and clipped "
    "audio transcribes worse than quiet audio. Turn the device's input level DOWN (Windows "
    "Sound settings -> the device -> Properties -> Levels) or move further from the "
    "microphone, then retry."
)

_SIDE_DEFS: tuple[tuple[str, str, Callable], ...] = (
    ("interviewee", "Interviewee device (loopback)", resolve_default_loopback_device),
    ("interviewer", "Interviewer device (microphone)", resolve_default_input_device),
)


@dataclass(frozen=True)
class PrecheckRow:
    name: str
    result: str
    measured: Optional[str] = None
    remedy: Optional[str] = None


@dataclass(frozen=True)
class PrecheckReport:
    rows: tuple[PrecheckRow, ...]

    @property
    def exit_code(self) -> int:
        return 1 if any(row.result == RESULT_FAIL for row in self.rows) else 0


@dataclass(frozen=True)
class LivenessResult:
    real_fraction: float
    level: LevelMeasurement
    frames_seen: int
    pcm16: bytes


def _run_liveness_window(
    *, pyaudio_module, pa, devices: dict[str, DeviceInfo], seconds: float, sample_rate: int, frame_ms: int,
) -> dict[str, LivenessResult]:
    """Runs every resolved side's `CaptureStream` concurrently for one
    bounded window -- the SAME capture worker `start` uses -- so the user
    plays test audio and speaks into the mic once, not once per side."""
    captures: dict[str, CaptureStream] = {
        key: CaptureStream(
            pyaudio_module, pa, device,
            target_sample_rate=sample_rate, frame_ms=frame_ms,
            queue_max_seconds=seconds + 5.0, label=f"precheck-{key}",
        )
        for key, device in devices.items()
    }
    for capture in captures.values():
        capture.start()
    time.sleep(max(0.0, seconds))
    for capture in captures.values():
        capture.stop()

    results: dict[str, LivenessResult] = {}
    for key, capture in captures.items():
        total = 0
        real = 0
        chunks: list[bytes] = []
        while True:
            try:
                frame = capture.queue.get_nowait()
            except queue.Empty:
                break
            total += 1
            if frame.kind == "real":
                real += 1
                chunks.append(frame.pcm16)
        pcm16 = b"".join(chunks)
        results[key] = LivenessResult(
            real_fraction=(real / total) if total else 0.0,
            level=measure_level(pcm16, sample_rate=sample_rate),
            frames_seen=total,
            pcm16=pcm16,
        )
    return results


def _liveness_row(
    name: str, result: LivenessResult, *, real_fraction_min: float, level_dbfs_min: float
) -> PrecheckRow:
    """Judges the LOUDEST half-second, never the window mean -- see
    `app/capture/levels.py` for the 127-at-8s / 34-at-30s measurement that
    forced the change. Clipping FAILS with its own remedy: turning the gain
    down is the opposite instruction to speaking up, and handing a user the
    wrong one is how a fixable level problem survives the pre-call check."""
    level = result.level
    clipping = level.band == BAND_CLIPPING
    passed = (
        result.real_fraction >= real_fraction_min
        and level.peak_dbfs >= level_dbfs_min
        and not clipping
    )
    measured = (
        f"real_fraction={result.real_fraction:.2f} peak_500ms={level.describe()}"
        + (f" over {level.window_s:.2f}s captured" if level.window_s < 0.5 else "")
    )
    return PrecheckRow(
        name=name,
        result=RESULT_PASS if passed else RESULT_FAIL,
        measured=measured,
        remedy=None if passed else (_CLIPPING_REMEDY if clipping else _LIVENESS_REMEDY),
    )


_SELF_TEST_LABELS = {
    "interviewee": "Interviewee audio (system sound)",
    "interviewer": "Interviewer audio (microphone)",
}


def run_self_test(
    *,
    pyaudio_module,
    pa,
    devices: dict[str, DeviceInfo],
    cfg: config_mod.AppConfig,
    seconds: Optional[float] = None,
) -> PrecheckReport:
    """Is each side of THIS interview actually carrying sound? Run by `start`
    immediately before the session begins.

    `devices` arrives ALREADY RESOLVED, and that is the point. `precheck`
    resolves its own sides because it runs standalone; a self-test that did the
    same would measure the default endpoint while a `--interviewee-device` pin
    sent the interview to a different one, and would then certify a device the
    interview never opens. Handing the resolved devices in is what makes this a
    gate on the real run rather than a second opinion about a similar one.

    Judged by `_liveness_row` against the SAME `cfg` thresholds `precheck`
    uses -- deliberately, because two definitions of "silent" would mean the
    laxer one silently becomes the real one.

    Returns a `PrecheckReport` so `exit_code`, row shape and printing are the
    ones the operator has already read once in `precheck` output."""
    window_seconds = seconds if seconds is not None else cfg.self_test_seconds
    liveness = _run_liveness_window(
        pyaudio_module=pyaudio_module, pa=pa, devices=devices,
        seconds=window_seconds, sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms,
    )
    rows = tuple(
        _liveness_row(
            _SELF_TEST_LABELS.get(key, f"{key} audio"),
            liveness[key],
            real_fraction_min=cfg.precheck_real_fraction_min,
            level_dbfs_min=cfg.precheck_level_dbfs_min,
        )
        for key in devices
    )
    return PrecheckReport(rows=rows)


def print_self_test_report(report: PrecheckReport) -> None:
    """The self-test's rows, without `precheck`'s pre-flight rules.

    Those rules are advice for someone setting a room up minutes before a call.
    Reprinting them here, at the instant the operator has already pressed
    record, is noise on top of the one line they need to read."""
    print("Self-test -- is each side carrying sound?")
    for row in report.rows:
        line = f"[{row.result:^7}] {row.name}"
        if row.measured:
            line += f" -- measured: {row.measured}"
        print(line)
        if row.result == RESULT_FAIL and row.remedy:
            print(f"            remedy: {row.remedy}")


def _engine_row(
    *, engine_factory: Callable[[], StreamingEngine], sample_rate: int, pcm16: bytes, timeout_s: float,
) -> PrecheckRow:
    name = "Engine reachability"
    deadline = time.monotonic() + timeout_s
    try:
        engine = engine_factory()
        session = engine.open_session(sample_rate=sample_rate)
    except Exception as exc:  # noqa: BLE001 -- reuse the adapter's own typed, key-material-free message
        return PrecheckRow(name=name, result=RESULT_FAIL, remedy=str(exc))

    sample_for_probe = pcm16[: sample_rate * 2] or (b"\x00\x00" * int(sample_rate * 0.5))
    try:
        session.send_audio(sample_for_probe)
        remaining = max(0.0, deadline - time.monotonic())
        session.poll(timeout_s=min(timeout_s, remaining))
    except TimeoutError as exc:
        return PrecheckRow(
            name=name, result=RESULT_FAIL,
            remedy=(
                f"No response from the STT engine within {timeout_s:.0f}s ({exc}). Check network "
                "connectivity and the engine vendor's status page."
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- reuse the adapter's own typed, key-material-free message
        return PrecheckRow(name=name, result=RESULT_FAIL, remedy=str(exc))
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 -- best-effort close; the verdict above is already decided
            pass

    if time.monotonic() > deadline:
        return PrecheckRow(
            name=name, result=RESULT_FAIL,
            remedy=(
                f"No response from the STT engine within {timeout_s:.0f}s. Check network "
                "connectivity and the engine vendor's status page."
            ),
        )
    return PrecheckRow(name=name, result=RESULT_PASS, measured="session opened, audio accepted, responded in time")


def _port_row(port: int, *, port_checker: Callable[[int], None]) -> PrecheckRow:
    """PASSES when `port` is free to bind, FAILS -- with the port number and
    the environment variable that changes it -- when something already holds
    it (e.g. a previous session that did not exit cleanly). Reuses
    `port_checker`'s own already-actionable message rather than composing a
    second one (matches the engine row's "reuse the adapter's own message"
    precedent)."""
    name = "Web port available"
    try:
        port_checker(port)
    except web_server.PortInUseError as exc:
        return PrecheckRow(name=name, result=RESULT_FAIL, remedy=str(exc))
    return PrecheckRow(name=name, result=RESULT_PASS, measured=f"port {port} is free")


def _lan_address_row(*, lan_ip_resolver: Callable[[], str]) -> PrecheckRow:
    """PASSES carrying the measured address plus the confirm-your-adapter
    instruction (03-RESEARCH assumption A3, MEDIUM confidence -- mitigated by
    a human-readable line, not eliminated). FAILS, rather than reporting a
    wrong answer confidently, when resolution raises."""
    name = "LAN address"
    try:
        lan_ip = lan_ip_resolver()
    except OSError as exc:
        return PrecheckRow(
            name=name, result=RESULT_FAIL,
            remedy=(
                f"Could not resolve a LAN IP ({exc}). Check ipconfig for your Wi-Fi adapter's "
                "IPv4 address and share the join URL manually."
            ),
        )
    return PrecheckRow(
        name=name, result=RESULT_PASS,
        measured=(
            f"{lan_ip} -- confirm this matches your Wi-Fi adapter's address (compare leading "
            "octets with your phone's own Wi-Fi IP over ipconfig), not a VPN or virtual adapter"
        ),
    )


def _start_phone_reachability_server(
    *, port: int, session_dir: Path, store: SessionStore, emit_qr: bool, host: str = "0.0.0.0",
):
    """Starts the real embedded server -- the same `create_app`/
    `build_uvicorn_server` factory `app/main.py` uses -- on its own background
    thread driving its own event loop, exactly like
    `app/tests/test_web_serving.py`'s own `_running_server` fixture. Prints the
    real join URL -- and, when `emit_qr`, both QR forms -- via `create_join`
    so a real phone can actually attempt to connect. Returns `(hub, actual_port, join_info, stop)`;
    callers MUST call `stop()` exactly once, in a `finally` block."""
    hub = ConnectionHub()
    join_info = join_mod.create_join(
        port=port, session_dir=session_dir, store=store, emit_qr=emit_qr
    )
    server_started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
    app = web_server.create_app(
        token=join_info.token, hub=hub, static_dir=STATIC_DIR, store=store,
        session_id=session_dir.name, server_started_at=server_started_at,
    )
    loop = asyncio.new_event_loop()
    server = web_server.build_uvicorn_server(app, host=host, port=port)

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    start_deadline = time.monotonic() + 10.0
    while not server.started and time.monotonic() < start_deadline:
        time.sleep(0.02)

    def _stop() -> None:
        server.should_exit = True
        thread.join(timeout=10.0)

    if not server.started:
        _stop()
        raise RuntimeError("the embedded server failed to start for the reachability check")

    actual_port = server.servers[0].sockets[0].getsockname()[1]
    return hub, actual_port, join_info, _stop


def _wait_for_a_connection(hub: ConnectionHub, *, timeout_s: float, poll_interval_s: float = 0.1) -> bool:
    """Polls `hub.connection_count` for a bounded window -- never longer than
    `timeout_s` -- returning as soon as a connection appears. A pre-call check
    that hangs one minute before an interview is worse than one that fails
    (T-03-27)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hub.connection_count > 0:
            return True
        time.sleep(poll_interval_s)
    return hub.connection_count > 0


def run_phone_reachability_check(
    *,
    port: int,
    session_dir: Path,
    store: SessionStore,
    timeout_s: float,
    emit_qr: bool,
    host: str = "0.0.0.0",
    on_ready: Optional[Callable[[int, str], None]] = None,
) -> PrecheckRow:
    """§R4 row 3, opt-in via `--phone`: starts the real server, shows the join
    URL (and the QR forms when `emit_qr`), waits up to `timeout_s` for a real client to
    connect, then stops the server cleanly whether or not one arrived. FAILs
    with the one-line triage rule (§R3) rather than hanging. `on_ready`, if
    given, is called once with `(actual_port, token)` the moment the server is
    up -- the seam a test uses to connect a real (non-phone) client inside the
    window, with no hardware and no credentials involved."""
    name = "Phone reachability"
    try:
        hub, actual_port, join_info, stop = _start_phone_reachability_server(
            port=port, session_dir=session_dir, store=store, host=host, emit_qr=emit_qr,
        )
    except RuntimeError as exc:
        return PrecheckRow(name=name, result=RESULT_FAIL, remedy=str(exc))

    try:
        if on_ready is not None:
            on_ready(actual_port, join_info.token)
        connected = _wait_for_a_connection(hub, timeout_s=timeout_s)
    finally:
        stop()

    if connected:
        return PrecheckRow(name=name, result=RESULT_PASS, measured="a client connected within the window")
    return PrecheckRow(name=name, result=RESULT_FAIL, remedy=_PHONE_TRIAGE_REMEDY)


def build_precheck_report(
    *,
    pa,
    pyaudio_module,
    cfg: config_mod.AppConfig,
    seconds: Optional[float] = None,
    skip_stt: bool = False,
    side_defs: tuple = _SIDE_DEFS,
    engine_factory: Optional[Callable[[], StreamingEngine]] = None,
    phone: bool = False,
    port_checker: Optional[Callable[[int], None]] = None,
    lan_ip_resolver: Optional[Callable[[], str]] = None,
    phone_check: Optional[Callable[[], PrecheckRow]] = None,
) -> PrecheckReport:
    """Assembles the full report: the port and LAN-address rows (§R4),
    device-resolution rows, a concurrent liveness window across every resolved
    side, the engine-reachability row (or a SKIPPED row if `skip_stt`), and --
    only with `phone=True` -- the bounded real-phone reachability row. Every
    dependency (`side_defs`, `engine_factory`, `port_checker`,
    `lan_ip_resolver`, `phone_check`) is injectable so this whole function is
    testable without hardware, credentials or a phone."""
    window_seconds = seconds if seconds is not None else cfg.precheck_seconds
    rows: list[PrecheckRow] = []
    rows.append(_port_row(cfg.app_port, port_checker=port_checker or web_server.check_port_available))
    rows.append(_lan_address_row(lan_ip_resolver=lan_ip_resolver or join_mod.get_lan_ip))
    devices: dict[str, DeviceInfo] = {}

    for key, label, resolver in side_defs:
        try:
            device = resolver(pa)
        except DeviceOpenError as exc:
            rows.append(PrecheckRow(name=f"{label} resolves", result=RESULT_FAIL, remedy=str(exc)))
            continue
        rows.append(PrecheckRow(
            name=f"{label} resolves", result=RESULT_PASS,
            measured=f"{device.name} ({device.native_sample_rate} Hz, {device.channels} ch)",
        ))
        devices[key] = device

    liveness_results: dict[str, LivenessResult] = {}
    if devices:
        liveness_results = _run_liveness_window(
            pyaudio_module=pyaudio_module, pa=pa, devices=devices,
            seconds=window_seconds, sample_rate=cfg.sample_rate, frame_ms=cfg.frame_ms,
        )
    for key, label, _resolver in side_defs:
        if key not in liveness_results:
            continue
        rows.append(_liveness_row(
            f"{label.split(' (')[0]} liveness", liveness_results[key],
            real_fraction_min=cfg.precheck_real_fraction_min, level_dbfs_min=cfg.precheck_level_dbfs_min,
        ))

    if skip_stt:
        rows.append(PrecheckRow(name="Engine reachability", result=RESULT_SKIPPED))
    else:
        factory = engine_factory or (lambda: get_engine(cfg.engine_name))
        mic_pcm16 = liveness_results.get("interviewer")
        probe_pcm16 = mic_pcm16.pcm16 if mic_pcm16 is not None else b""
        rows.append(_engine_row(
            engine_factory=factory, sample_rate=cfg.sample_rate, pcm16=probe_pcm16,
            timeout_s=cfg.precheck_engine_timeout_s,
        ))

    # Opt-in, off by default (§R4 row 3) -- the everyday precheck stays fast;
    # only `--phone` pays the cost of a bounded wait for a real connection.
    if phone:
        if phone_check is not None:
            rows.append(phone_check())
        else:
            with tempfile.TemporaryDirectory(prefix="precheck-phone-") as tmp_dir_name:
                tmp_dir = Path(tmp_dir_name)
                phone_store = SessionStore(tmp_dir)
                try:
                    phone_store.write_metadata({"purpose": "precheck-phone-reachability-check"})
                    rows.append(run_phone_reachability_check(
                        emit_qr=cfg.emit_qr,
                        port=cfg.app_port, session_dir=tmp_dir, store=phone_store,
                        timeout_s=cfg.precheck_phone_wait_s,
                    ))
                finally:
                    # Windows refuses to delete a file that is still open, so the
                    # temp dir's own cleanup raises PermissionError and takes the
                    # whole report down with it -- after every row has passed.
                    # The store owns an append handle on transcript.jsonl; close
                    # it here, before TemporaryDirectory.__exit__ tries to unlink.
                    phone_store.close()

    return PrecheckReport(rows=tuple(rows))


def print_precheck_report(report: PrecheckReport) -> None:
    print("What to do first (pre-flight rules -- these cannot be fixed in code):")
    for rule in PREFLIGHT_RULES:
        print(f"  - {rule}")
    print()
    for row in report.rows:
        line = f"[{row.result:^7}] {row.name}"
        if row.measured:
            line += f" -- measured: {row.measured}"
        print(line)
        if row.result == RESULT_FAIL and row.remedy:
            print(f"            remedy: {row.remedy}")
    print()
    if report.exit_code == 0:
        print("ALL CHECKS PASSED -- go.")
    else:
        print("AT LEAST ONE CHECK FAILED -- fix the FAIL row(s) above before starting.")
