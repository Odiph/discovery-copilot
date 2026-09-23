"""Tests for WASAPI enumeration and identity resolution (`app/capture/devices.py`).

The fake below reproduces the shape measured on the maintainer's machine 2026-08-15, and
the detail that matters is the DUPLICATION: one physical microphone is listed
once per host API, and MME truncates its name to 31 characters. An enumeration
that walked every API would offer three entries for one microphone -- two of
them useless, since loopback is a WASAPI feature -- and `identity`, which is
built from the name, would collide across APIs while naming a different device.

The other rule under test is that a saved device which has vanished falls back
to the default but is never SILENT about it. `devices.py`'s own docstring calls
following the wrong device "the most likely quiet failure in this phase"; a
silent fallback is that failure with a friendly face.
"""

from __future__ import annotations

import pytest

from app.capture.devices import (
    DeviceInfo,
    display_name,
    list_capture_devices,
    resolve_device_by_identity,
)

WASAPI = 2

# Verbatim names, including MME's 31-character truncation of the Logitech.
_MME = [
    {"index": 1, "name": "Microphone (Logitech Webcam C92", "defaultSampleRate": 44100,
     "maxInputChannels": 2, "isLoopbackDevice": False},
]
_WASAPI = [
    {"index": 22, "name": "Microphone (High Definition Audio Device)",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": False},
    {"index": 23, "name": "Microphone (Logitech Webcam C925e)",
     "defaultSampleRate": 16000, "maxInputChannels": 2, "isLoopbackDevice": False},
    {"index": 27, "name": "Headphones (Bose QC45 Stereo) [Loopback]",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": True},
    # A render-only endpoint: no input channels, so not capture-worthy.
    {"index": 30, "name": "Headphones (Bose QC45 Stereo)",
     "defaultSampleRate": 44100, "maxInputChannels": 0, "isLoopbackDevice": False},
]


class _FakePyAudioModule:
    paWASAPI = WASAPI


class _FakePa:
    """Indexes devices per host API, the way PortAudio really does."""

    def __init__(self, *, apis=None, raise_on=()):
        self._apis = apis if apis is not None else {0: _MME, WASAPI: _WASAPI}
        self._raise_on = raise_on

    def get_host_api_info_by_type(self, api_type):
        if api_type not in self._apis:
            raise OSError("no such host API")
        return {"index": api_type, "deviceCount": len(self._apis[api_type])}

    def get_device_info_by_host_api_device_index(self, api_index, offset):
        if (api_index, offset) in self._raise_on:
            raise OSError("device vanished mid-enumeration")
        return self._apis[api_index][offset]


def _default(_pa):
    return DeviceInfo(
        index=27, name="Headphones (Bose QC45 Stereo) [Loopback]",
        native_sample_rate=44100, channels=2, is_loopback=True,
        identity="loopback:Headphones (Bose QC45 Stereo) [Loopback]",
    )


# --- WASAPI only, and why ---------------------------------------------------------


def test_only_wasapi_devices_are_offered():
    found = list_capture_devices(_FakePyAudioModule(), _FakePa())
    assert [d.index for d in found] == [22, 23, 27]


def test_the_mme_truncated_duplicate_never_appears():
    """'Microphone (Logitech Webcam C92' is the SAME microphone as index 23,
    truncated by MME to 31 characters. Offering both would ask the operator to
    choose between one device and a misspelling of it."""
    names = [d.name for d in list_capture_devices(_FakePyAudioModule(), _FakePa())]
    assert "Microphone (Logitech Webcam C92" not in names
    assert "Microphone (Logitech Webcam C925e)" in names


def test_a_render_only_endpoint_is_not_capture_worthy():
    """Index 30 is the output endpoint itself. It cannot be captured FROM --
    only its `[Loopback]` analogue can."""
    assert 30 not in [d.index for d in list_capture_devices(_FakePyAudioModule(), _FakePa())]


def test_loopback_and_input_are_returned_together_and_distinguishable():
    found = list_capture_devices(_FakePyAudioModule(), _FakePa())
    assert [d.is_loopback for d in found] == [False, False, True]


def test_one_unreadable_device_does_not_hide_the_others():
    """The tree of devices is the OS's, and one endpoint mid-teardown must not
    empty the whole picker."""
    pa = _FakePa(raise_on={(WASAPI, 1)})
    # 23 is the one that raised; 30 is render-only and excluded regardless.
    assert [d.index for d in list_capture_devices(_FakePyAudioModule(), pa)] == [22, 27]


def test_a_machine_with_no_wasapi_enumerates_nothing_rather_than_raising():
    """A setup panel that raised here would take the overlay down on a machine
    whose audio stack is mid-restart."""
    pa = _FakePa(apis={0: _MME})
    assert list_capture_devices(_FakePyAudioModule(), pa) == ()


# --- display name -------------------------------------------------------------------


def test_the_loopback_marker_is_stripped_for_display():
    device = [d for d in list_capture_devices(_FakePyAudioModule(), _FakePa()) if d.is_loopback][0]
    assert display_name(device) == "Headphones (Bose QC45 Stereo)"
    assert "[Loopback]" in device.identity


def test_an_input_name_is_shown_unchanged():
    device = list_capture_devices(_FakePyAudioModule(), _FakePa())[1]
    assert display_name(device) == "Microphone (Logitech Webcam C925e)"


# --- resolution by identity, and the LOUD fallback ------------------------------------


def test_a_saved_device_that_is_present_is_used_and_reports_no_fallback():
    choice = resolve_device_by_identity(
        _FakePyAudioModule(), _FakePa(),
        "input:Microphone (Logitech Webcam C925e)", default_resolver=_default,
    )
    assert choice.device.index == 23
    assert choice.fell_back is False


def test_a_saved_device_that_is_gone_falls_back_AND_SAYS_SO():
    """The whole point. Recording still starts -- an interview must not be
    blocked because a headset was switched off -- but the caller is handed the
    fact so it can be said out loud."""
    choice = resolve_device_by_identity(
        _FakePyAudioModule(), _FakePa(),
        "input:Microphone (Sennheiser Momentum)", default_resolver=_default,
    )
    assert choice.fell_back is True
    assert choice.requested_identity == "input:Microphone (Sennheiser Momentum)"
    assert choice.device.index == 27


def test_no_saved_device_is_not_a_fallback():
    """A first run has nothing saved. That is the normal case, not a device
    that vanished, and it must not raise a warning at the operator."""
    choice = resolve_device_by_identity(
        _FakePyAudioModule(), _FakePa(), None, default_resolver=_default,
    )
    assert choice.fell_back is False
    assert choice.requested_identity is None


def test_an_empty_identity_is_treated_as_nothing_saved():
    choice = resolve_device_by_identity(
        _FakePyAudioModule(), _FakePa(), "", default_resolver=_default,
    )
    assert choice.fell_back is False


def test_identity_matching_never_falls_through_to_index():
    """An index is not stable across a device change; matching on one would
    eventually select somebody else's microphone."""
    choice = resolve_device_by_identity(
        _FakePyAudioModule(), _FakePa(), "23", default_resolver=_default,
    )
    assert choice.fell_back is True
