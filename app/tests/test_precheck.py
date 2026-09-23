"""Task 3 done criteria: `python -m app precheck` asserts OBSERVATIONS, never
a successful open -- a check that passes on a muted device is worse than no
check (phase_specification T3). Every dependency (device resolution, the
engine) is injected here, so the pass path, each FAIL path, SKIPPED, and the
exit code are all provable without hardware or credentials.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import pytest

from app import config as config_mod
from app import main as app_main
from app.capture import precheck as precheck_mod
from app.capture.devices import DeviceInfo, DeviceOpenError
from app.capture.levels import LevelMeasurement, classify_dbfs
from app.capture.precheck import (
    PrecheckReport,
    PrecheckRow,
    build_precheck_report,
    print_precheck_report,
)
from app.session.store import SessionStore, create_session_dir
from app.stt.fake import FakeStreamingEngine
from app.tests.stub_audio import StubPyAudio, StubPyAudioModule, StubStream
from app.web import server as web_server

LOOPBACK_DEVICE = DeviceInfo(
    index=1, name="Stub Loopback", native_sample_rate=16000, channels=1,
    is_loopback=True, identity="loopback:stub",
)
MIC_DEVICE = DeviceInfo(
    index=2, name="Stub Mic", native_sample_rate=16000, channels=1,
    is_loopback=False, identity="input:stub",
)
SIDE_DEFS = (
    ("interviewee", "Interviewee device (loopback)", lambda pa: LOOPBACK_DEVICE),
    ("interviewer", "Interviewer device (microphone)", lambda pa: MIC_DEVICE),
)


def _make_pa(amplitude_by_index: dict[int, int]) -> StubPyAudio:
    def factory(**kwargs) -> StubStream:
        idx = kwargs.get("input_device_index")
        amplitude = amplitude_by_index.get(idx, 0)
        return StubStream(
            frames_per_buffer=kwargs["frames_per_buffer"], channels=kwargs["channels"], amplitude=amplitude,
        )

    return StubPyAudio(stream_factory=factory)


def _cfg(**overrides):
    cfg = config_mod.load_config(engine_override="fake")
    if overrides:
        cfg = config_mod.AppConfig(**{**cfg.__dict__, **overrides})
    return cfg


# --- All PASS: both sides loud, engine answers -----------------------------------------


def test_all_pass_when_both_sides_loud_and_engine_answers():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.3, skip_stt=False,
        side_defs=SIDE_DEFS, engine_factory=lambda: FakeStreamingEngine(cadence_bytes=1),
    )
    assert report.exit_code == 0
    assert all(row.result == "PASS" for row in report.rows), report.rows
    liveness_rows = [r for r in report.rows if "liveness" in r.name]
    assert liveness_rows
    for row in liveness_rows:
        assert row.measured is not None


# --- Silent room fails closed, with a remedy on every FAIL ------------------------------


def test_silent_room_fails_closed_with_remedy_on_every_fail_row():
    pa = _make_pa({1: 0, 2: 0})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
    )
    assert report.exit_code != 0
    fails = [row for row in report.rows if row.result == "FAIL"]
    assert fails
    for row in fails:
        assert row.remedy, f"FAIL row {row.name!r} has no remedy"


def test_a_device_that_opens_but_stays_silent_still_fails_liveness():
    """Opening is not liveness -- exactly the T3 distinction this check exists for."""
    pa = _make_pa({1: 0, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
    )
    loopback_liveness = next(r for r in report.rows if r.name == "Interviewee device liveness")
    mic_liveness = next(r for r in report.rows if r.name == "Interviewer device liveness")
    assert loopback_liveness.result == "FAIL"
    assert mic_liveness.result == "PASS"
    assert report.exit_code != 0


# --- The liveness row judges the loudest 500ms, in dBFS, with a band ---------------------
#
# `_liveness_row` is the seam: the level itself is measured by
# `app.capture.levels` (covered in test_levels.py against signals with real
# silence in them), and these tests pin what the ROW does with the verdict.


def _liveness(peak_dbfs: float, *, real_fraction: float = 1.0) -> precheck_mod.LivenessResult:
    return precheck_mod.LivenessResult(
        real_fraction=real_fraction,
        level=LevelMeasurement(
            peak_dbfs=peak_dbfs, band=classify_dbfs(peak_dbfs), window_s=0.5
        ),
        frames_seen=100,
        pcm16=b"",
    )


def test_liveness_row_reports_dbfs_and_a_band_never_a_raw_int16_rms():
    row = precheck_mod._liveness_row(
        "Mic liveness", _liveness(-12.0), real_fraction_min=0.5, level_dbfs_min=-36.0,
    )
    assert row.result == "PASS"
    assert "dBFS" in row.measured
    assert "good" in row.measured
    assert "rms_level" not in row.measured


def test_liveness_row_passes_a_quiet_but_workable_level_and_fails_below_the_floor():
    workable = precheck_mod._liveness_row(
        "Mic liveness", _liveness(-30.0), real_fraction_min=0.5, level_dbfs_min=-36.0,
    )
    below = precheck_mod._liveness_row(
        "Mic liveness", _liveness(-42.0), real_fraction_min=0.5, level_dbfs_min=-36.0,
    )
    assert workable.result == "PASS"
    assert below.result == "FAIL"
    assert "too quiet" in below.measured


def test_liveness_row_fails_clipping_with_the_opposite_instruction_to_the_quiet_remedy():
    """Loud enough is not the same as correct: a clipped signal must not be
    told to speak up, which is what a bare `>= floor` test would have done."""
    row = precheck_mod._liveness_row(
        "Mic liveness", _liveness(-0.5), real_fraction_min=0.5, level_dbfs_min=-36.0,
    )
    assert row.result == "FAIL"
    assert "clipping" in row.measured
    assert row.remedy == precheck_mod._CLIPPING_REMEDY
    assert row.remedy != precheck_mod._LIVENESS_REMEDY
    assert "down" in row.remedy.lower()


def test_a_digitally_silent_capture_never_prints_infinity_to_a_human():
    """The failure this whole check exists to catch reads as a hard zero. It
    must FAIL, and it must FAIL in words -- `log10(0)` is a crash and `-inf`
    is not a level a human can act on."""
    pa = _make_pa({1: 0, 2: 0})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
    )
    liveness_rows = [r for r in report.rows if "liveness" in r.name]
    assert liveness_rows
    for row in liveness_rows:
        assert row.result == "FAIL"
        assert "silence" in row.measured
        assert "inf" not in row.measured.lower()
        assert "nan" not in row.measured.lower()


# --- Missing device: a FAIL row with remedy, no crash ------------------------------------


def test_missing_device_produces_a_fail_row_with_remedy():
    def _raise(pa):
        raise DeviceOpenError("No default WASAPI input (microphone) device is available.")

    side_defs = (
        ("interviewee", "Interviewee device (loopback)", lambda pa: LOOPBACK_DEVICE),
        ("interviewer", "Interviewer device (microphone)", _raise),
    )
    pa = _make_pa({1: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=side_defs,
    )
    resolve_rows = [r for r in report.rows if "resolves" in r.name]
    interviewer_row = next(r for r in resolve_rows if "Interviewer" in r.name)
    assert interviewer_row.result == "FAIL"
    assert "microphone" in interviewer_row.remedy.lower()
    assert report.exit_code != 0


# --- Skip flag: SKIPPED, not a silent PASS ------------------------------------------------


def test_skip_stt_reports_skipped_not_pass():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
    )
    engine_row = next(r for r in report.rows if r.name == "Engine reachability")
    assert engine_row.result == "SKIPPED"
    assert report.exit_code == 0  # SKIPPED must not itself force a non-zero exit


# --- Engine reachability FAIL paths: distinct remedies, never a silent pass -------------


class _MissingKeyEngine:
    name = "stub"

    def open_session(self, *, sample_rate: int, label: str = ""):
        raise RuntimeError("STUB_API_KEY is not set.")


class _RejectedKeySession:
    def send_audio(self, pcm16: bytes) -> None:
        raise RuntimeError("Stub vendor error 401: invalid api key")

    def poll(self, timeout_s: float = 0.0) -> tuple:
        return ()

    def close(self) -> tuple:
        return ()


class _RejectedKeyEngine:
    name = "stub"

    def open_session(self, *, sample_rate: int, label: str = ""):
        return _RejectedKeySession()


class _TimeoutSession:
    def send_audio(self, pcm16: bytes) -> None:
        pass

    def poll(self, timeout_s: float = 0.0) -> tuple:
        time.sleep(timeout_s + 0.3)
        raise TimeoutError("stub vendor never responded")

    def close(self) -> tuple:
        return ()


class _TimeoutEngine:
    name = "stub"

    def open_session(self, *, sample_rate: int, label: str = ""):
        return _TimeoutSession()


def test_missing_key_fails_with_a_distinct_actionable_remedy():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.1, skip_stt=False,
        side_defs=SIDE_DEFS, engine_factory=lambda: _MissingKeyEngine(),
    )
    row = next(r for r in report.rows if r.name == "Engine reachability")
    assert row.result == "FAIL"
    assert "STUB_API_KEY" in row.remedy
    assert report.exit_code != 0


def test_rejected_key_fails_with_a_distinct_actionable_remedy():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.1, skip_stt=False,
        side_defs=SIDE_DEFS, engine_factory=lambda: _RejectedKeyEngine(),
    )
    row = next(r for r in report.rows if r.name == "Engine reachability")
    assert row.result == "FAIL"
    assert "401" in row.remedy
    assert report.exit_code != 0


def test_timeout_fails_with_a_distinct_actionable_remedy():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg(precheck_engine_timeout_s=0.2)
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.1, skip_stt=False,
        side_defs=SIDE_DEFS, engine_factory=lambda: _TimeoutEngine(),
    )
    row = next(r for r in report.rows if r.name == "Engine reachability")
    assert row.result == "FAIL"
    assert row.remedy, "a timeout FAIL must still carry an actionable remedy"
    assert report.exit_code != 0


def test_all_three_engine_fail_remedies_are_distinct_from_each_other():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg(precheck_engine_timeout_s=0.2)
    remedies = set()
    for factory in (_MissingKeyEngine, _RejectedKeyEngine, _TimeoutEngine):
        report = build_precheck_report(
            pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.1, skip_stt=False,
            side_defs=SIDE_DEFS, engine_factory=factory,
        )
        row = next(r for r in report.rows if r.name == "Engine reachability")
        remedies.add(row.remedy)
    assert len(remedies) == 3, remedies


def test_engine_row_remedy_never_includes_key_material():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()

    class _Engine:
        name = "stub"

        def open_session(self, *, sample_rate: int, label: str = ""):
            raise RuntimeError("SONIOX_API_KEY is not set. Read from the environment only.")

    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.1, skip_stt=False,
        side_defs=SIDE_DEFS, engine_factory=lambda: _Engine(),
    )
    row = next(r for r in report.rows if r.name == "Engine reachability")
    # The remedy is exactly the adapter's own already-safe message -- precheck
    # never appends key material, an env dump, or anything else to it.
    assert row.remedy == "SONIOX_API_KEY is not set. Read from the environment only."


# --- Printing: pre-flight rules and a clear summary --------------------------------------


def test_print_report_includes_preflight_rules_and_summary(capsys):
    report = PrecheckReport(rows=(PrecheckRow(name="X", result="PASS", measured="ok"),))
    print_precheck_report(report)
    out = capsys.readouterr().out
    assert "Loopback captures EVERYTHING" in out
    assert "ALL CHECKS PASSED" in out


def test_print_report_shows_fail_summary_and_remedy(capsys):
    report = PrecheckReport(rows=(PrecheckRow(name="X", result="FAIL", remedy="do the thing"),))
    print_precheck_report(report)
    out = capsys.readouterr().out
    assert "AT LEAST ONE CHECK FAILED" in out
    assert "do the thing" in out


# --- CLI wiring: --seconds and --skip-stt are threaded through main.py -------------------


def test_cli_precheck_subcommand_parses_seconds_and_skip_stt():
    from app import main as app_main

    args = app_main.build_arg_parser().parse_args(["precheck", "--seconds", "2", "--skip-stt"])
    assert args.command == "precheck"
    assert args.seconds == 2.0
    assert args.skip_stt is True


# --- 03-03 Task 1: the port row -- FAILS on a REAL held port, not a mock ----------------


@pytest.fixture(autouse=True)
def _no_gui_side_effects(monkeypatch):
    """`create_join` (invoked by the phone-reachability tests below) writes a
    real QR PNG and opens it via `os.startfile` on Windows -- the exact
    GUI-process-leak class caught in 03-01 Task 2. Neutralized for every test
    in this module, matching `test_web_join.py`'s own autouse fixture."""
    monkeypatch.setattr(os, "startfile", lambda path: None, raising=False)


def test_port_row_passes_when_free_and_fails_when_actually_held():
    """Tested by actually holding the port with a real socket, not by mocking
    the answer, per this task's own instruction."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    def _checker(p: int) -> None:
        web_server.check_port_available(p, host="127.0.0.1")

    free_row = precheck_mod._port_row(port, port_checker=_checker)
    assert free_row.result == "PASS"
    assert str(port) in free_row.measured

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", port))
    holder.listen(1)
    try:
        held_row = precheck_mod._port_row(port, port_checker=_checker)
        assert held_row.result == "FAIL"
        assert str(port) in held_row.remedy
        assert "APP_PORT" in held_row.remedy
    finally:
        holder.close()


# --- 03-03 Task 1: the LAN-address row -- carries the measured address, FAILs on error --


def test_lan_address_row_passes_with_confirm_your_adapter_instruction():
    row = precheck_mod._lan_address_row(lan_ip_resolver=lambda: "192.168.1.50")
    assert row.result == "PASS"
    assert "192.168.1.50" in row.measured
    assert "wi-fi adapter" in row.measured.lower()
    assert "vpn" in row.measured.lower()


def test_lan_address_row_fails_rather_than_reporting_a_wrong_answer_on_resolution_error():
    def _raise() -> str:
        raise OSError("no route to host")

    row = precheck_mod._lan_address_row(lan_ip_resolver=_raise)
    assert row.result == "FAIL"
    assert row.remedy


# --- 03-03 Task 1: the report assembles port + LAN rows unconditionally, phone opt-in ---


def test_without_phone_no_reachability_row_appears_and_port_lan_rows_are_present():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
    )
    names = [r.name for r in report.rows]
    assert "Phone reachability" not in names
    assert "Web port available" in names
    assert "LAN address" in names
    assert report.exit_code == 0


def test_phone_flag_appends_exactly_one_reachability_row_via_injection():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
        phone=True,
        phone_check=lambda: PrecheckRow(name="Phone reachability", result="PASS", measured="stub"),
    )
    names = [r.name for r in report.rows]
    assert names.count("Phone reachability") == 1
    assert report.exit_code == 0


def test_phone_flag_reachability_fail_makes_the_existing_exit_code_property_nonzero():
    pa = _make_pa({1: 20000, 2: 20000})
    cfg = _cfg()
    report = build_precheck_report(
        pa=pa, pyaudio_module=StubPyAudioModule(), cfg=cfg, seconds=0.2, skip_stt=True, side_defs=SIDE_DEFS,
        phone=True,
        phone_check=lambda: PrecheckRow(name="Phone reachability", result="FAIL", remedy="x"),
    )
    assert report.exit_code != 0


# --- 03-03 Task 1: the real bounded phone-reachability check -- no hardware, no phone ---


def test_phone_reachability_passes_when_a_real_client_connects_inside_the_window(tmp_path):
    from websockets.sync.client import connect as ws_connect

    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "precheck-phone-reachability-test"})

    connected = threading.Event()

    def _on_ready(port: int, token: str) -> None:
        def _client() -> None:
            try:
                with ws_connect(f"ws://127.0.0.1:{port}/s/{token}/ws", open_timeout=5) as ws:
                    connected.set()
                    time.sleep(0.4)
            except Exception:  # noqa: BLE001 -- a failed test client must not hang the test
                pass

        threading.Thread(target=_client, daemon=True).start()

    row = precheck_mod.run_phone_reachability_check(
        emit_qr=False,
        port=0, session_dir=session_dir, store=store, timeout_s=3.0, host="127.0.0.1", on_ready=_on_ready,
    )
    assert connected.is_set(), "the test's own client never connected -- test infrastructure issue"
    assert row.result == "PASS"


def test_phone_reachability_fails_within_the_bounded_window_when_no_client_connects(tmp_path):
    session_dir = create_session_dir(tmp_path)
    store = SessionStore(session_dir)
    store.write_metadata({"purpose": "precheck-phone-reachability-test"})

    start = time.monotonic()
    row = precheck_mod.run_phone_reachability_check(
        emit_qr=False,
        port=0, session_dir=session_dir, store=store, timeout_s=0.3, host="127.0.0.1",
    )
    elapsed = time.monotonic() - start

    assert row.result == "FAIL"
    assert elapsed < 8.0, "the reachability row hung rather than failing within its bounded window"
    assert row.remedy == precheck_mod._PHONE_TRIAGE_REMEDY
    assert "hotspot" in row.remedy.lower()


# --- 03-03 Task 1: --phone / --no-serve CLI wiring --------------------------------------


def test_cli_precheck_subcommand_parses_phone_flag():
    args = app_main.build_arg_parser().parse_args(["precheck"])
    assert args.phone is False
    args_with_flag = app_main.build_arg_parser().parse_args(["precheck", "--phone"])
    assert args_with_flag.phone is True


def test_cli_start_subcommand_parses_no_serve_flag():
    args = app_main.build_arg_parser().parse_args(["start"])
    assert args.no_serve is False
    args_with_flag = app_main.build_arg_parser().parse_args(["start", "--no-serve"])
    assert args_with_flag.no_serve is True


# --- 03-03 Task 1: `start --no-serve` binds no port and opens no browser ----------------


def test_no_serve_runs_full_capture_with_no_port_bound_and_no_browser_opened(tmp_path, monkeypatch):
    opened_urls: list[str] = []
    monkeypatch.setattr(app_main.webbrowser, "open", lambda url: opened_urls.append(url) or True)
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(tmp_path))
    monkeypatch.setenv("STT_ENGINE", "fake")
    # This test is about the PORT under `--no-serve` and has nothing to do
    # with the coach, but `run_start` loads the brief before it gets there.
    # It used to reach the repo's own `context/brief.json`; Phase A moved the
    # real brief into `projects/_eval/`, and the root is now empty.
    #
    # It kept passing in a full run and failed when this file ran alone --
    # some earlier test leaves a brief behind in the repo's `context/`. That
    # ordering dependency was always there; the migration only made it
    # visible. Fixed by pointing BOTH roots at tmp_path and turning the coach
    # off, so this test no longer reads repository state at all.
    monkeypatch.setenv("APP_CONTEXT_DIR", str(tmp_path / "context"))
    # EMPTY, never absent. `_load_dotenv_once` fills in any variable "not
    # already present", so deleting this hands the real key straight back --
    # the exact defect that made a credential-absent test pass for the wrong
    # reason since Phase 4 (commit e4c883c).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    args = app_main.build_arg_parser().parse_args(["start", "--seconds", "3", "--no-serve"])
    result_holder: dict = {}

    def _run() -> None:
        result_holder["code"] = app_main.run_start(args)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    cfg = config_mod.load_config(engine_override="fake")
    deadline = time.monotonic() + 2.5
    observed_free_mid_run = False
    while time.monotonic() < deadline and thread.is_alive():
        try:
            web_server.check_port_available(cfg.app_port)
            observed_free_mid_run = True
        except web_server.PortInUseError:
            observed_free_mid_run = False
            break
        time.sleep(0.2)

    thread.join(timeout=15.0)

    assert result_holder.get("code") == 0
    assert observed_free_mid_run, "the pinned port was not observed free while --no-serve was running"
    assert opened_urls == [], "no browser should ever be opened under --no-serve"

    transcripts = list(tmp_path.glob("*/transcript.jsonl"))
    assert transcripts, "no session was created -- --no-serve must still run the full capture path"
