"""Resolves capture devices by ROLE, never by index -- device indices are not
stable across a device change (phase_specification S3/S5). Loopback follows
the DEFAULT render endpoint at resolution time; there can be more than one
render endpoint on a machine (measured on this laptop: two), and following
anything but the default is the most likely quiet failure in this phase --
it would silently record silence from the interviewee.

Confirmed against the installed PyAudioWPatch version at implementation time,
per phase_specification: `get_default_wasapi_loopback()` returns the same
"info dict" shape as PortAudio's own `get_device_info_by_index` (keys
`index`, `name`, `defaultSampleRate`, `maxInputChannels`, ...), measured on
this machine 2026-08-05.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Every enumeration below is WASAPI-ONLY, and that is not a preference.
# Measured on the maintainer's machine 2026-08-15: the same physical microphone is listed
# once per host API -- MME (0), DirectSound (1) and WASAPI (2) -- and **MME
# truncates the name to 31 characters**, so the Logitech appears as
# 'Microphone (Logitech Webcam C92'. Enumerating every API would therefore
# offer the operator three entries for one microphone, two of them unusable
# for loopback, and `identity` (which is built from the name) would collide
# across APIs while silently naming a different device. Loopback is a WASAPI
# feature in the first place, so the other APIs can offer nothing this program
# can use.
_LOOPBACK_SUFFIX = " [Loopback]"


class DeviceOpenError(Exception):
    """Raised when a capture device cannot be resolved or opened. The message
    always names the device and an actionable remedy -- this is a device the
    user can fix from Windows Sound settings, not a bug report."""


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable description of one capture-worthy WASAPI device. `identity`
    is a stable string used to detect a default-device change later (plan
    02-02 hardens this) -- it is NOT the PortAudio `index`, which is not
    guaranteed stable across a device change."""

    index: int
    name: str
    native_sample_rate: int
    channels: int
    is_loopback: bool
    identity: str


def _to_device_info(info: dict, *, is_loopback: bool) -> DeviceInfo:
    name = info["name"]
    return DeviceInfo(
        index=int(info["index"]),
        name=name,
        native_sample_rate=int(info["defaultSampleRate"]),
        channels=int(info["maxInputChannels"]),
        is_loopback=is_loopback,
        identity=f"{'loopback' if is_loopback else 'input'}:{name}",
    )


def resolve_default_loopback_device(pa) -> DeviceInfo:
    """The interviewee side: the loopback analogue of whatever the DEFAULT
    render endpoint is right now (T-02-01, S3). `pa` is a `pyaudiowpatch.PyAudio`
    instance -- passed in rather than constructed here so the caller controls
    its lifetime."""
    try:
        info = pa.get_default_wasapi_loopback()
    except OSError as exc:
        raise DeviceOpenError(
            "No WASAPI loopback device is available. Confirm the WASAPI audio "
            "driver is present (this is a Windows-only feature) and that a "
            "default output device is selected at Settings -> System -> Sound "
            "-> Output."
        ) from exc
    except LookupError as exc:
        raise DeviceOpenError(
            "No loopback analogue was found for the default output device. "
            "Set an output device as default at Settings -> System -> Sound -> "
            "Output, then retry."
        ) from exc
    return _to_device_info(info, is_loopback=True)


def resolve_default_render_device(pa) -> DeviceInfo:
    """The keep-alive stream's target: the actual DEFAULT render (output)
    endpoint, resolved directly via `get_default_wasapi_device(d_out=True)`
    -- NOT the fork's `get_wasapi_loopback_analogue_by_*` helpers. Those
    helpers only map an OUTPUT device's info dict -> its LOOPBACK analogue;
    called the other way round (passing a loopback dict in) they just return
    the same dict unchanged, per their own `if not info_dict["isLoopbackDevice"]:
    ... else: return info_dict` branch (confirmed against the installed
    0.2.12.8 API, measured 2026-08-05). `get_default_wasapi_loopback()`
    itself resolves through this exact same call internally, so this is the
    real render endpoint the loopback device already follows -- with no lossy
    round-trip through the analogue helper."""
    try:
        info = pa.get_default_wasapi_device(d_out=True)
    except OSError as exc:
        raise DeviceOpenError(
            "No default WASAPI output device is available for the keep-alive "
            "stream. Confirm a default output device is selected at Settings "
            "-> System -> Sound -> Output."
        ) from exc
    return DeviceInfo(
        index=int(info["index"]),
        name=info["name"],
        native_sample_rate=int(info["defaultSampleRate"]),
        channels=int(info["maxOutputChannels"]),
        is_loopback=False,
        identity=f"render:{info['name']}",
    )


def resolve_default_input_device(pa) -> DeviceInfo:
    """The interviewer side: the default microphone. Not used by this plan's
    single-stream tracer (loopback only) -- resolved the same way plan 02-02
    will use it for the mic stream."""
    try:
        info = pa.get_default_wasapi_device(d_in=True)
    except OSError as exc:
        raise DeviceOpenError(
            "No default WASAPI input (microphone) device is available. "
            "Confirm a microphone is connected and selected at Settings -> "
            "System -> Sound -> Input."
        ) from exc
    return _to_device_info(info, is_loopback=False)


@dataclass(frozen=True)
class DeviceChoice:
    """A resolved device plus HOW it was resolved.

    `fell_back` is the whole reason this type exists. This module's own opening
    docstring warns that following anything but the default render endpoint is
    "the most likely quiet failure in this phase -- it would silently record
    silence from the interviewee". A saved device that has since vanished (the
    headset powered off, the webcam unplugged) is exactly that failure wearing
    a friendly face: recording continues, the operator believes their chosen
    setup is live, and the transcript is silent.

    So the fallback is allowed -- an interview must not be blocked because a
    device changed -- but it is never SILENT. The caller is handed the fact and
    is expected to say it out loud."""

    device: DeviceInfo
    fell_back: bool = False
    requested_identity: Optional[str] = None


def _wasapi_devices(pyaudio_module, pa) -> "list[dict]":
    """Every WASAPI device info dict, or an empty list where WASAPI is absent.

    A machine with no WASAPI host API is not an error to raise here -- the
    callers that need one already raise `DeviceOpenError` with a remedy, and an
    enumeration that exploded would take the whole setup panel down with it."""
    try:
        api = pa.get_host_api_info_by_type(pyaudio_module.paWASAPI)
    except (OSError, LookupError, ValueError):
        return []
    found: "list[dict]" = []
    for offset in range(int(api.get("deviceCount", 0))):
        try:
            found.append(pa.get_device_info_by_host_api_device_index(api["index"], offset))
        except (OSError, LookupError, ValueError):
            # One unreadable device must not hide the other eleven.
            continue
    return found


def list_capture_devices(pyaudio_module, pa) -> "tuple[DeviceInfo, ...]":
    """Every WASAPI device this program could capture FROM, loopback and
    microphone alike, in enumeration order.

    Loopback entries are what the interviewee side can be pointed at; the rest
    are what the interviewer side can be pointed at. Both are returned together
    because they come from one enumeration and the caller already has to tell
    them apart by `is_loopback` -- splitting them here would mean walking the
    device list twice and inventing a second place for the WASAPI rule to live."""
    devices: "list[DeviceInfo]" = []
    for info in _wasapi_devices(pyaudio_module, pa):
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        devices.append(
            _to_device_info(info, is_loopback=bool(info.get("isLoopbackDevice", False)))
        )
    return tuple(devices)


def display_name(device: DeviceInfo) -> str:
    """The device name without the enumeration's own ` [Loopback]` marker.

    The suffix is real and belongs in `identity`, but on a panel that already
    labels the row "system sound" it is noise repeated on every entry."""
    if device.is_loopback and device.name.endswith(_LOOPBACK_SUFFIX):
        return device.name[: -len(_LOOPBACK_SUFFIX)]
    return device.name


def resolve_device_by_identity(
    pyaudio_module,
    pa,
    identity: "Optional[str]",
    *,
    default_resolver,
) -> DeviceChoice:
    """The saved device if it is still here, otherwise the default -- SAYING
    which of the two happened.

    Matched on `identity`, never on `index`: this module's whole premise is
    that PortAudio indices are not stable across a device change, and a saved
    index would eventually name somebody else's microphone. Measured on this
    machine, unplugging a device renumbers the ones after it.

    `default_resolver` is passed in rather than chosen here so one function
    serves both sides -- the interviewee's default is the loopback analogue of
    the default render endpoint, the interviewer's is the default input -- and
    so neither side can be silently substituted for the other."""
    if identity:
        for device in list_capture_devices(pyaudio_module, pa):
            if device.identity == identity:
                return DeviceChoice(device=device, requested_identity=identity)
    fallback = default_resolver(pa)
    return DeviceChoice(
        device=fallback,
        fell_back=bool(identity),
        requested_identity=identity,
    )


def resolve_capture_device_by_name(
    pyaudio_module,
    pa,
    name: str,
    *,
    is_loopback: bool,
) -> DeviceInfo:
    """The one capture device on `is_loopback`'s side whose name matches
    `name` -- raising rather than guessing, and never falling back.

    This is the OPERATOR-TYPED counterpart to `resolve_device_by_identity`, and
    the difference in fallback behaviour between the two is deliberate. A saved
    identity describes a device chosen in some previous session, which may
    legitimately be gone, so falling back to the default (loudly) is right. A
    name typed on the command line seconds ago is an instruction about THIS
    interview; falling back would silently record the very device the operator
    was steering away from. On 2026-08-17 that device was a Bluetooth A2DP
    endpoint that Windows had already stopped playing to, and the interview
    captured two minutes of digital silence on the interviewee channel.

    Matched against `display_name`, so the string on the panel is the string
    that works; the raw ` [Loopback]` form is accepted too, because that is
    what `session.json` records and the record of a bad session is the obvious
    thing to paste back. Case-insensitive substring, then narrowed: an exact
    match wins outright, which is what lets a name that is a prefix of another
    device's name still be selectable at all.

    `is_loopback` filters BEFORE matching, never after. One physical headset
    publishes its hands-free endpoint as both an input and a loopback under one
    name, so a filter applied afterwards would make the same string ambiguous
    that the side already disambiguates -- and `_resolve_devices`' rule that
    neither side is ever silently substituted for the other would hold only by
    luck."""
    wanted = name.strip().casefold()
    if not wanted:
        raise DeviceOpenError("A device name was requested but it is empty.")

    side = [d for d in list_capture_devices(pyaudio_module, pa) if d.is_loopback == is_loopback]
    side_label = "system sound (loopback)" if is_loopback else "microphone (input)"

    def _names(device: DeviceInfo) -> "tuple[str, str]":
        return (display_name(device).casefold(), device.name.casefold())

    exact = [d for d in side if wanted in _names(d)]
    matches = exact or [d for d in side if any(wanted in n for n in _names(d))]

    if not matches:
        raise DeviceOpenError(
            f"No {side_label} device matches {name!r}. "
            f"Available: {_offer(side)}."
        )
    if len(matches) > 1:
        raise DeviceOpenError(
            f"{name!r} matches {len(matches)} {side_label} devices and is ambiguous -- "
            f"name one of them exactly: {_offer(matches)}."
        )
    return matches[0]


def _offer(devices: "Sequence[DeviceInfo]") -> str:
    """The candidate list carried in a `DeviceOpenError`. Shown by
    `display_name`, because the suffix the operator never sees is not a name
    they can be asked to choose between."""
    if not devices:
        return "(none -- check Windows Sound settings)"
    return ", ".join(repr(display_name(d)) for d in devices)


def open_capture_stream(pyaudio_module, pa, device: DeviceInfo, *, frames_per_buffer: int):
    """Open a blocking-mode input stream for `device`. Wraps every open
    failure in `DeviceOpenError`, naming the device and an actionable remedy.
    The exclusive-mode case names the specific Sound-settings checkbox to
    clear, because "a client can enable loopback mode only for a shared-mode
    stream" (Microsoft Learn) is an OS rule this program cannot engineer
    around."""
    try:
        return pa.open(
            format=pyaudio_module.paInt16,
            channels=device.channels,
            rate=device.native_sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=frames_per_buffer,
        )
    except OSError as exc:
        message = str(exc)
        if "Invalid device" in message or "-9996" in message or "-9998" in message:
            raise DeviceOpenError(
                f"Could not open '{device.name}' (index {device.index}): {exc}. "
                "The device may have been unplugged or changed since it was "
                "resolved; retry from the start."
            ) from exc
        raise DeviceOpenError(
            f"Could not open '{device.name}' (index {device.index}) in shared "
            f"mode: {exc}. If this device has exclusive mode enabled, clear "
            "'Allow applications to take exclusive control of this device' at "
            "Settings -> System -> Sound -> Output -> Device properties -> "
            "Additional device properties -> Advanced tab, then retry -- "
            "WASAPI loopback only works on a shared-mode stream."
        ) from exc
