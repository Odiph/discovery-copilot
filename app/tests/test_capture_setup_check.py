"""Tests for the device-pair checks (`app/capture/setup_check.py`).

Every name below is VERBATIM from the WASAPI enumeration on the maintainer's machine
(2026-08-15). That matters more than usual here: both rules are heuristics over
strings a driver author chose, so a fixture with tidied-up names would test the
tidying, not the rule. The Bose pair in particular is the exact pair that
silenced a real recording -- loopback measured a hard zero, digital silence,
while the operator could hear nothing through the headphones.
"""

from __future__ import annotations

from app.capture.devices import DeviceInfo
from app.capture.setup_check import (
    FINDING_BLUETOOTH_PROFILE_CLASH,
    FINDING_SHARED_ENDPOINT,
    check_pair,
)

# --- verbatim from `get_device_info_by_host_api_device_index`, WASAPI --------

BOSE_A2DP_LOOPBACK = DeviceInfo(
    index=27, name="Headphones (Bose QC45 Stereo) [Loopback]",
    native_sample_rate=44100, channels=2, is_loopback=True,
    identity="loopback:Headphones (Bose QC45 Stereo) [Loopback]",
)
BOSE_HFP_INPUT = DeviceInfo(
    index=24, name="Headset (Bose QC45 Hands-Free AG Audio)",
    native_sample_rate=16000, channels=1, is_loopback=False,
    identity="input:Headset (Bose QC45 Hands-Free AG Audio)",
)
BOSE_HFP_LOOPBACK = DeviceInfo(
    index=26, name="Headset (Bose QC45 Hands-Free AG Audio) [Loopback]",
    native_sample_rate=16000, channels=1, is_loopback=True,
    identity="loopback:Headset (Bose QC45 Hands-Free AG Audio) [Loopback]",
)
LOGITECH_INPUT = DeviceInfo(
    index=23, name="Microphone (Logitech Webcam C925e)",
    native_sample_rate=16000, channels=2, is_loopback=False,
    identity="input:Microphone (Logitech Webcam C925e)",
)
SPEAKERS_LOOPBACK = DeviceInfo(
    index=28, name="Speakers (High Definition Audio Device) [Loopback]",
    native_sample_rate=48000, channels=2, is_loopback=True,
    identity="loopback:Speakers (High Definition Audio Device) [Loopback]",
)


def _codes(findings):
    return [finding.code for finding in findings]


# --- the clash that actually happened -------------------------------------------


def test_the_bose_pair_that_silenced_a_real_recording_is_flagged():
    """A2DP loopback + HFP microphone on ONE headset. Opening the microphone
    forces hands-free mode and the stereo endpoint goes silent."""
    findings = check_pair(BOSE_A2DP_LOOPBACK, BOSE_HFP_INPUT)
    assert _codes(findings) == [FINDING_BLUETOOTH_PROFILE_CLASH]
    assert "silences the stereo endpoint" in findings[0].message


def test_the_clash_finding_names_a_remedy_the_operator_can_act_on():
    finding = check_pair(BOSE_A2DP_LOOPBACK, BOSE_HFP_INPUT)[0]
    assert "separate microphone" in finding.remedy


def test_both_hands_free_endpoints_still_clash_but_report_the_quality_cost():
    """Pointing BOTH sides at the hands-free profile does not silence anything
    -- it is already in hands-free mode -- but it does mean the whole interview
    is captured at about 16kHz mono."""
    findings = check_pair(BOSE_HFP_LOOPBACK, BOSE_HFP_INPUT)
    assert _codes(findings) == [FINDING_BLUETOOTH_PROFILE_CLASH]
    assert "16kHz mono" in findings[0].message


# --- the pairs that must stay quiet ----------------------------------------------


def test_the_working_setup_produces_no_findings():
    """Bose for sound, Logitech for the microphone -- different hardware, so
    the headset stays in stereo. This is the pair the machine defaults to and
    it must never be warned about."""
    assert check_pair(BOSE_A2DP_LOOPBACK, LOGITECH_INPUT) == ()


def test_speakers_plus_a_separate_microphone_is_not_a_profile_clash():
    """Different hardware entirely. (Speakers carry a real bleed risk, but that
    is a different finding and is deliberately not claimed here.)"""
    assert check_pair(SPEAKERS_LOOPBACK, LOGITECH_INPUT) == ()


def test_a_stereo_only_bluetooth_pairing_is_not_flagged():
    """Same headset, but the microphone side is NOT the hands-free profile, so
    nothing forces a profile switch. Flagging on shared hardware alone would
    warn about a setup that works."""
    stereo_named_input = DeviceInfo(
        index=99, name="Microphone (Bose QC45 Stereo)", native_sample_rate=44100,
        channels=2, is_loopback=False, identity="input:Microphone (Bose QC45 Stereo)",
    )
    assert check_pair(BOSE_A2DP_LOOPBACK, stereo_named_input) == ()


# --- both sides pointed at one endpoint -------------------------------------------


def test_the_same_endpoint_on_both_sides_is_flagged_first():
    findings = check_pair(BOSE_A2DP_LOOPBACK, BOSE_A2DP_LOOPBACK)
    assert _codes(findings) == [FINDING_SHARED_ENDPOINT]
    assert "no turn could be attributed" in findings[0].message


def test_a_shared_endpoint_short_circuits_the_profile_check():
    """One finding, not two. The operator gets the thing to fix, not a list."""
    assert len(check_pair(BOSE_HFP_LOOPBACK, BOSE_HFP_LOOPBACK)) == 1


# --- a missing side ----------------------------------------------------------------


def test_a_loopback_only_run_has_no_pair_to_check():
    """`--only interviewee` opens no microphone, so there is nothing to clash
    with and nothing to warn about."""
    assert check_pair(BOSE_A2DP_LOOPBACK, None) == ()
    assert check_pair(None, LOGITECH_INPUT) == ()
    assert check_pair(None, None) == ()
