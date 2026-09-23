"""Every seam that chooses a capture device honours the pin (`app/main.py`).

WHY A SEPARATE FILE. `test_capture_device_pinning.py` proves the MATCHING
rule; this file proves the WIRING, and the wiring is where this class of bug
actually lives. `app/main.py` chose a device in THREE places:

  1. `_resolve_devices`      -- at session start
  2. the supervisor's `device_resolver` -- when the default device changes mid-call
  3. the lazy microphone build -- when `--only interviewee` is later given a mic

A pin honoured at (1) alone looks completely correct for the first minute of
an interview and then reverts to the default the moment Windows renegotiates
an endpoint -- which, with a Bluetooth headset, is exactly when a call starts.
That is the failure mode the standing rule ("two seams must honour a device
choice; patching one silently reverts mid-call") was written about, and it had
three seams, not two.

So the invariant is structural rather than behavioural: the default resolvers
may be NAMED IN ONLY ONE PLACE. A test that called `_resolve_devices` and
asserted the right device came back would pass just as happily with seams (2)
and (3) still reverting, because it exercises one side of the seam and proves
intent rather than behaviour. The AST check below is the one that fails when a
fourth seam is added without thought.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app import main as app_main
from app.capture.devices import DeviceInfo, DeviceOpenError

WASAPI = 2

_WASAPI = [
    {"index": 22, "name": "Microphone (High Definition Audio Device)",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": False},
    {"index": 25, "name": "Headphones (Bose QC45 Stereo) [Loopback]",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": True},
    {"index": 26, "name": "Speakers (High Definition Audio Device) [Loopback]",
     "defaultSampleRate": 44100, "maxInputChannels": 2, "isLoopbackDevice": True},
]


class _FakePyAudioModule:
    paWASAPI = WASAPI


class _FakePa:
    def get_host_api_info_by_type(self, api_type):
        if api_type != WASAPI:
            raise OSError("no such host API")
        return {"index": api_type, "deviceCount": len(_WASAPI)}

    def get_device_info_by_host_api_device_index(self, api_index, offset):
        return _WASAPI[offset]


@pytest.fixture
def fake_wasapi(monkeypatch):
    monkeypatch.setattr(app_main, "pyaudio", _FakePyAudioModule())
    return _FakePa()


# --- the resolver itself ---------------------------------------------------------------


def test_a_pinned_interviewee_side_resolves_to_the_named_loopback(fake_wasapi):
    resolve = app_main._side_resolver(
        app_main.SPEAKER_INTERVIEWEE, pin="Speakers (High Definition Audio Device)"
    )
    assert resolve(fake_wasapi).index == 26


def test_a_pinned_interviewer_side_resolves_to_the_named_input(fake_wasapi):
    resolve = app_main._side_resolver(
        app_main.SPEAKER_INTERVIEWER, pin="Microphone (High Definition Audio Device)"
    )
    assert resolve(fake_wasapi).index == 22


def test_an_unpinned_side_still_uses_the_default(monkeypatch, fake_wasapi):
    """Pinning is opt-in. The overwhelming majority of runs pass no flag and
    must behave exactly as they did before it existed."""
    sentinel = DeviceInfo(
        index=99, name="Default [Loopback]", native_sample_rate=48000,
        channels=2, is_loopback=True, identity="loopback:Default [Loopback]",
    )
    monkeypatch.setattr(app_main, "resolve_default_loopback_device", lambda pa: sentinel)
    resolve = app_main._side_resolver(app_main.SPEAKER_INTERVIEWEE, pin=None)
    assert resolve(fake_wasapi) is sentinel


def test_a_bad_pin_raises_at_every_seam_rather_than_falling_back(fake_wasapi):
    """The resolver is the SAME callable at all three seams, so a pin that has
    become unresolvable mid-call raises there too rather than quietly reverting
    to the endpoint the operator was steering away from."""
    resolve = app_main._side_resolver(app_main.SPEAKER_INTERVIEWEE, pin="Yeti")
    with pytest.raises(DeviceOpenError):
        resolve(fake_wasapi)


# --- the wiring ------------------------------------------------------------------------


def test_resolve_devices_routes_through_the_given_resolvers(fake_wasapi):
    resolvers = {
        app_main.SPEAKER_INTERVIEWEE: app_main._side_resolver(
            app_main.SPEAKER_INTERVIEWEE, pin="Speakers"
        ),
        app_main.SPEAKER_INTERVIEWER: app_main._side_resolver(
            app_main.SPEAKER_INTERVIEWER, pin="Microphone"
        ),
    }
    devices = app_main._resolve_devices(
        fake_wasapi, app_main.ALL_SPEAKERS, resolvers=resolvers
    )
    assert devices[app_main.SPEAKER_INTERVIEWEE].index == 26
    assert devices[app_main.SPEAKER_INTERVIEWER].index == 22


def _names_referenced_outside(function_name: str, wanted: "set[str]") -> "dict[str, list[str]]":
    """Every reference to `wanted` in `app/main.py`, grouped by the top-level
    function it sits in, excluding `function_name` itself. Imports are
    `alias` nodes rather than `Name` nodes and so never count."""
    tree = ast.parse(Path(app_main.__file__).read_text(encoding="utf-8"))
    found: "dict[str, list[str]]" = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == function_name:
            continue
        hits = [
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in wanted
        ]
        if hits:
            found[node.name] = hits
    return found


def test_the_default_resolvers_are_named_in_exactly_one_place():
    """The structural guard. Three seams reverted to the default independently;
    the fix is not "patch all three" but "leave one place where the default can
    be named". A fourth seam added later fails HERE, at import time of the test
    suite, instead of in an interview."""
    stragglers = _names_referenced_outside(
        "_side_resolver",
        {"resolve_default_loopback_device", "resolve_default_input_device"},
    )
    assert stragglers == {}, (
        "these functions choose a capture device without going through "
        f"_side_resolver, so a --interviewee-device/--interviewer-device pin "
        f"will not reach them: {stragglers}"
    )
