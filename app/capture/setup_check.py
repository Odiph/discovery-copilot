"""Whether a chosen pair of devices can actually work TOGETHER.

Separate from `devices.py` because that module answers "what is there and can I
open it", one device at a time. Every rule here is about the RELATIONSHIP
between the two sides, which is where the failures that cost a whole interview
live -- both of the ones below produce a recording that looks healthy and is
worthless.

Pure: no I/O, no PyAudio, no clock. Takes `DeviceInfo`s and returns findings.

**These are warnings, never refusals.** Both rules are name heuristics over
strings a driver author chose, and a heuristic that blocks recording will
eventually block a setup that works, in front of a waiting interviewee. The
operator is told; the operator decides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The Bluetooth hands-free profile, as Windows spells it in an endpoint name.
# Windows names a Bluetooth headset's two endpoints "<Device> Hands-Free AG
# Audio" (HFP) and "<Device> Stereo" (A2DP); matching the profile words is what
# lets the pairing below be recognised without a device database.
_HFP_MARKERS = ("hands-free", "handsfree", "hands free")
_A2DP_MARKERS = ("stereo",)

# Words that appear in an endpoint name because of its ROLE or its PROFILE
# rather than because of the hardware. Stripped before two names are compared,
# so "Headphones (Bose QC45 Stereo)" and "Headset (Bose QC45 Hands-Free AG
# Audio)" reduce to the same "bose qc45" and are recognised as one headset.
_ROLE_WORDS = frozenset({
    "headphones", "headset", "speakers", "microphone", "mic", "loopback",
    "hands-free", "handsfree", "free", "hands", "ag", "audio", "stereo",
    "device", "input", "output", "wireless", "bluetooth",
})

FINDING_BLUETOOTH_PROFILE_CLASH = "bluetooth-profile-clash"
FINDING_SHARED_ENDPOINT = "shared-endpoint"


@dataclass(frozen=True)
class SetupFinding:
    """One reason a pair of devices may not work together. `remedy` is always
    an action the operator can take from where they are sitting."""

    code: str
    message: str
    remedy: str


def _hardware_token(name: str) -> str:
    """A device name reduced to the words that identify the HARDWARE.

    Punctuation becomes spaces, role and profile words are dropped, and what is
    left is the model. Deliberately crude: it only has to be stable across the
    two profile names Windows gives ONE headset, and being wrong costs a
    warning, never a recording."""
    lowered = re.sub(r"[^a-z0-9]+", " ", name.lower())
    words = [word for word in lowered.split() if word and word not in _ROLE_WORDS]
    return " ".join(words)


def _has(name: str, markers) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in markers)


def check_pair(interviewee, interviewer) -> "tuple[SetupFinding, ...]":
    """Findings for capturing `interviewee` (loopback) alongside `interviewer`
    (microphone) at the same time.

    Either side may be `None` -- a loopback-only run has no microphone, and a
    pair with a missing side cannot clash with itself."""
    if interviewee is None or interviewer is None:
        return ()

    findings: "list[SetupFinding]" = []

    if interviewee.identity == interviewer.identity:
        findings.append(SetupFinding(
            code=FINDING_SHARED_ENDPOINT,
            message=(
                "Both sides are pointed at the same endpoint, so both tracks "
                "would record identical audio and no turn could be attributed "
                "to either speaker."
            ),
            remedy="Point the microphone at a different device from system sound.",
        ))
        return tuple(findings)

    # The A2DP/HFP clash. One physical Bluetooth headset exposes a stereo
    # render endpoint (A2DP) and a mono hands-free endpoint with a microphone
    # (HFP), and Windows cannot run both at once: opening the headset's
    # microphone switches the headset into HFP, at which point the stereo
    # endpoint the interviewee side is capturing goes silent or drops to ~16kHz
    # mono. Measured on the maintainer's machine 2026-08-15, where the loopback track
    # recorded a hard zero -- digital silence, not quiet -- while the operator
    # also could not hear anything through the headphones.
    same_hardware = (
        _hardware_token(interviewee.name)
        and _hardware_token(interviewee.name) == _hardware_token(interviewer.name)
    )
    if same_hardware and _has(interviewer.name, _HFP_MARKERS):
        stereo_side = _has(interviewee.name, _A2DP_MARKERS)
        findings.append(SetupFinding(
            code=FINDING_BLUETOOTH_PROFILE_CLASH,
            message=(
                "The microphone and system sound are two profiles of the same "
                "Bluetooth headset. Opening its microphone forces the headset "
                "into hands-free mode, which "
                + (
                    "silences the stereo endpoint system sound is captured from"
                    if stereo_side
                    else "drops both sides to about 16kHz mono"
                )
                + "."
            ),
            remedy=(
                "Use a separate microphone (a webcam or USB mic) and leave the "
                "headset for sound only -- that keeps it in stereo."
            ),
        ))

    return tuple(findings)
