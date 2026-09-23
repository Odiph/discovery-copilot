"""The self-test `start` runs before an interview (`app/capture/precheck.py`).

WHY THIS EXISTS. `precheck` could already prove a channel was silent -- on
2026-08-17 it did, reading `peak_500ms=-69.5 dBFS (silence)` on the interviewee
side -- but `start` never ran it, so a two-minute session recorded 24
interviewer segments and zero interviewee segments and only said so afterwards,
in a latency summary. The measurement was not missing. The GATE was.

Two properties carry the whole feature:

- It measures THE DEVICES THE INTERVIEW WILL USE, handed in already resolved,
  rather than resolving its own. A self-test that re-resolved would check the
  default endpoint while a `--interviewee-device` pin sent the interview
  somewhere else -- a harness measuring a system that is not the one running,
  which is the recurring failure in this repo.
- It judges with the SAME thresholds as `precheck`. Two definitions of "silent"
  is one definition too many; whichever was laxer would become the real one.
"""

from __future__ import annotations

from app import config as config_mod
from app.capture.devices import DeviceInfo
from app.capture.precheck import RESULT_FAIL, RESULT_PASS, run_self_test
from app.tests.stub_audio import StubPyAudio, StubPyAudioModule, StubStream

LOOPBACK_DEVICE = DeviceInfo(
    index=1, name="Speakers (High Definition Audio Device) [Loopback]",
    native_sample_rate=16000, channels=1, is_loopback=True, identity="loopback:stub",
)
MIC_DEVICE = DeviceInfo(
    index=2, name="Microphone (High Definition Audio Device)",
    native_sample_rate=16000, channels=1, is_loopback=False, identity="input:stub",
)
DEVICES = {"interviewee": LOOPBACK_DEVICE, "interviewer": MIC_DEVICE}


def _make_pa(amplitude_by_index: "dict[int, int]") -> StubPyAudio:
    def factory(**kwargs) -> StubStream:
        return StubStream(
            frames_per_buffer=kwargs["frames_per_buffer"],
            channels=kwargs["channels"],
            amplitude=amplitude_by_index.get(kwargs.get("input_device_index"), 0),
        )

    return StubPyAudio(stream_factory=factory)


def _run(amplitudes, *, cfg=None):
    return run_self_test(
        pyaudio_module=StubPyAudioModule(),
        pa=_make_pa(amplitudes),
        devices=DEVICES,
        cfg=cfg or config_mod.load_config(engine_override="fake"),
        seconds=0.3,
    )


def _row(report, needle):
    return next(row for row in report.rows if needle in row.name)


# --- the pass path ----------------------------------------------------------------------


def test_both_sides_live_passes_and_does_not_block_the_interview():
    report = _run({1: 20000, 2: 20000})
    assert report.exit_code == 0
    assert all(row.result == RESULT_PASS for row in report.rows)


def test_every_resolved_side_is_measured_and_named():
    report = _run({1: 20000, 2: 20000})
    assert {row.name for row in report.rows} == {
        "Interviewee audio (system sound)",
        "Interviewer audio (microphone)",
    }


# --- the failure this was written for ----------------------------------------------------


def test_a_silent_interviewee_channel_fails_and_names_that_side():
    """The 2026-08-17 shape exactly: the microphone is fine, the loopback is
    reading a Bluetooth endpoint Windows has stopped playing to."""
    report = _run({1: 0, 2: 20000})
    assert report.exit_code == 1
    assert _row(report, "Interviewee").result == RESULT_FAIL
    assert _row(report, "Interviewer").result == RESULT_PASS


def test_a_failing_side_carries_a_remedy():
    """A gate that blocks without saying what to do gets bypassed once and then
    always."""
    report = _run({1: 0, 2: 20000})
    assert _row(report, "Interviewee").remedy


def test_the_measured_value_is_reported_so_a_borderline_level_is_visible():
    report = _run({1: 0, 2: 20000})
    assert "dBFS" in (_row(report, "Interviewee").measured or "")


def test_a_silent_interviewer_channel_fails_too():
    report = _run({1: 20000, 2: 0})
    assert _row(report, "Interviewer").result == RESULT_FAIL


def test_both_silent_fails_both_rather_than_stopping_at_the_first():
    """One window, every side judged -- the operator fixes both in one pass
    instead of discovering the second after fixing the first."""
    report = _run({1: 0, 2: 0})
    assert [row.result for row in report.rows] == [RESULT_FAIL, RESULT_FAIL]


# --- one definition of "silent" -----------------------------------------------------------


def test_the_threshold_is_the_configured_precheck_threshold():
    """Raising the bar in config must move this gate too. Two thresholds would
    mean a channel that `precheck` calls dead and `start` calls fine."""
    cfg = config_mod.load_config(engine_override="fake")
    strict = config_mod.AppConfig(**{**cfg.__dict__, "precheck_level_dbfs_min": 0.0})
    report = _run({1: 20000, 2: 20000}, cfg=strict)
    assert report.exit_code == 1


# --- it measures what the interview will use ----------------------------------------------


def test_it_measures_the_devices_handed_to_it_and_resolves_nothing_itself():
    """`run_self_test` takes resolved devices. If it re-resolved, a pinned run
    would be gated on a device the interview is not going to open."""
    only_interviewee = run_self_test(
        pyaudio_module=StubPyAudioModule(),
        pa=_make_pa({1: 20000, 2: 0}),
        devices={"interviewee": LOOPBACK_DEVICE},
        cfg=config_mod.load_config(engine_override="fake"),
        seconds=0.3,
    )
    # The silent microphone is not opened at all, so it cannot fail the gate --
    # which is what `--only interviewee` needs.
    assert [row.name for row in only_interviewee.rows] == ["Interviewee audio (system sound)"]
    assert only_interviewee.exit_code == 0
