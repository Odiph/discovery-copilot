"""Pinning a capture device by name (`app/capture/devices.py`).

WHY THIS EXISTS. On 2026-08-17 a two-minute session recorded 24 interviewer
segments and ZERO interviewee segments. The cause was not a dead device: the
Bose QC45 publishes an A2DP endpoint (`Headphones (... Stereo)`) and an HFP
endpoint (`Headset (... Hands-Free AG Audio)`) AT THE SAME TIME, and Windows
moves playback from the first to the second the moment a call app opens the
microphone. `resolve_default_loopback_device` had latched the A2DP endpoint at
session start, so the capture read a live-but-idle endpoint -- real frames,
every sample zero -- for the whole interview. `levels.py`'s own docstring
already described this failure; nothing let the operator route around it.

Pinning is that route. The rules under test are the ones that keep a pin from
becoming a NEW quiet failure:

- A pin that matches nothing RAISES. It never falls back to the default, which
  is the behaviour `resolve_device_by_identity` is right to have for a
  REMEMBERED device and wrong to have for one typed on the command line
  seconds ago: the operator who names a device has said which one they mean.
- A pin that matches two devices RAISES rather than picking the first. On this
  machine the string "Bose QC45" matches three endpoints, two of which are the
  wrong one.
- A pin is SIDE-ISOLATED. The interviewee side may only ever land on a
  loopback endpoint and the interviewer side only on an input, so a pin cannot
  do what `_resolve_devices`' own docstring forbids -- silently substitute one
  side for the other.
"""

from __future__ import annotations

import pytest

from app.capture.devices import (
    DeviceOpenError,
    resolve_capture_device_by_name,
)

WASAPI = 2

# The real enumeration measured on the maintainer's machine 2026-08-17, with both Bluetooth
# endpoints present at once -- which is the whole point.
_WASAPI = [
    {"index": 22, "name": "Microphone (High Definition Audio Device)",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": False},
    {"index": 23, "name": "Headset (Bose QC45 Hands-Free AG Audio)",
     "defaultSampleRate": 16000, "maxInputChannels": 1, "isLoopbackDevice": False},
    {"index": 24, "name": "Headset (Bose QC45 Hands-Free AG Audio) [Loopback]",
     "defaultSampleRate": 16000, "maxInputChannels": 1, "isLoopbackDevice": True},
    {"index": 25, "name": "Headphones (Bose QC45 Stereo) [Loopback]",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": True},
    {"index": 26, "name": "Speakers (High Definition Audio Device) [Loopback]",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": True},
]


class _FakePyAudioModule:
    paWASAPI = WASAPI


class _FakePa:
    def __init__(self, *, apis=None):
        self._apis = apis if apis is not None else {WASAPI: _WASAPI}

    def get_host_api_info_by_type(self, api_type):
        if api_type not in self._apis:
            raise OSError("no such host API")
        return {"index": api_type, "deviceCount": len(self._apis[api_type])}

    def get_device_info_by_host_api_device_index(self, api_index, offset):
        return self._apis[api_index][offset]


def _resolve(name, *, is_loopback):
    return resolve_capture_device_by_name(
        _FakePyAudioModule(), _FakePa(), name, is_loopback=is_loopback
    )


# --- the match itself -----------------------------------------------------------------


def test_an_exact_display_name_resolves():
    """The name the operator reads on the panel is the name they can type --
    `display_name` strips ` [Loopback]`, so requiring the raw name would mean
    typing a suffix that is never shown."""
    device = _resolve("Speakers (High Definition Audio Device)", is_loopback=True)
    assert device.index == 26
    assert device.is_loopback


def test_matching_ignores_case():
    device = _resolve("speakers (high definition audio device)", is_loopback=True)
    assert device.index == 26


def test_an_unambiguous_fragment_resolves():
    """Device names are long and parenthesised; retyping one exactly, under
    time pressure, before an interview, is a way to get it wrong."""
    assert _resolve("Speakers", is_loopback=True).index == 26


def test_the_raw_loopback_suffix_is_also_accepted():
    """`session.json` records the raw name. Pasting back what the last session
    recorded must work, or the record of what went wrong cannot be used to fix
    it."""
    device = _resolve("Speakers (High Definition Audio Device) [Loopback]", is_loopback=True)
    assert device.index == 26


# --- the two ways a pin must REFUSE ----------------------------------------------------


def test_an_ambiguous_fragment_raises_and_names_every_candidate():
    """"Bose" is two loopback endpoints, and picking the first is exactly the
    A2DP-instead-of-HFP mistake this whole module exists to make impossible."""
    with pytest.raises(DeviceOpenError) as exc:
        _resolve("Bose", is_loopback=True)
    message = str(exc.value)
    assert "Headphones (Bose QC45 Stereo)" in message
    assert "Headset (Bose QC45 Hands-Free AG Audio)" in message


def test_a_pin_that_matches_nothing_raises_and_lists_what_is_available():
    """It never falls back to the default. A pin is an instruction, not a
    preference -- and a silent fallback here would reproduce the failure that
    motivated the flag, while looking like it had been fixed."""
    with pytest.raises(DeviceOpenError) as exc:
        _resolve("Yeti", is_loopback=True)
    message = str(exc.value)
    assert "Yeti" in message
    assert "Speakers (High Definition Audio Device)" in message


# --- side isolation --------------------------------------------------------------------


def test_the_interviewee_side_cannot_be_pinned_to_a_microphone():
    """`_resolve_devices`' docstring: neither side is ever silently substituted
    for the other. A loopback pin sees only loopback endpoints."""
    with pytest.raises(DeviceOpenError):
        _resolve("Microphone (High Definition Audio Device)", is_loopback=True)


def test_the_interviewer_side_cannot_be_pinned_to_a_loopback():
    with pytest.raises(DeviceOpenError):
        _resolve("Speakers (High Definition Audio Device)", is_loopback=False)


def test_one_name_on_two_sides_resolves_to_the_side_that_was_asked_for():
    """"Headset (Bose QC45 Hands-Free AG Audio)" exists as BOTH an input and a
    loopback. The same string must select a different device per side, which is
    only true if the side filter runs before the match."""
    assert _resolve("Hands-Free", is_loopback=False).index == 23
    assert _resolve("Hands-Free", is_loopback=True).index == 24


# --- the failure this was written for ---------------------------------------------------


def test_the_2026_08_17_workaround_resolves():
    """The operator's actual escape from the Bluetooth endpoint swap: pin the
    interviewee side to the wired output's loopback and stop depending on
    whatever Windows calls the default render endpoint mid-call."""
    device = _resolve("Speakers (High Definition Audio Device)", is_loopback=True)
    assert device.name == "Speakers (High Definition Audio Device) [Loopback]"
    assert device.native_sample_rate == 44100
