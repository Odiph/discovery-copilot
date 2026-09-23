"""Tests for the microphone clock-skew correction (`app/session/mic_skew.py`).

The bug being corrected is dangerous because it is invisible: a released
microphone freezes the interviewer's audio timeline while the wall clock keeps
running, so every later interviewer latency is inflated by the whole
off-duration and still looks like a plausible float. These tests are mostly
about the correction not becoming a second way to get a wrong number -- it must
not touch the interviewee, must not fire on a session that never toggled, and
must not over-correct into a negative.
"""

from __future__ import annotations

from app.session.mic_skew import (
    OffWindow, correct_interviewer_latencies, off_windows, paused_before,
)


def _mic(at_s, enabled):
    return {"kind": "event", "event": "microphone", "detail": {"enabled": enabled},
            "at_s": at_s}


def _segment(speaker, latency_s, ready_wall_s):
    return {"kind": "segment", "speaker": speaker,
            "latency_s": latency_s, "ready_wall_s": ready_wall_s}


# --- reading the windows ---------------------------------------------------------


def test_a_closed_toggle_is_one_window():
    windows = off_windows([_mic(10.0, False), _mic(70.0, True)])
    assert windows == (OffWindow(began_at_s=10.0, ended_at_s=70.0),)


def test_a_session_that_ended_with_the_microphone_off_leaves_the_window_open():
    """Left open rather than given an invented end. An invented end would
    silently under- or over-correct everything after it."""
    windows = off_windows([_mic(10.0, False)])
    assert windows == (OffWindow(began_at_s=10.0, ended_at_s=None),)


def test_a_repeated_off_does_not_open_a_second_window():
    """The control channel and the overlay can both issue a toggle. A duplicate
    is a UI artefact, not a second release of the device, and counting it twice
    would double-correct a pause that happened once."""
    windows = off_windows([_mic(10.0, False), _mic(12.0, False), _mic(70.0, True)])
    assert len(windows) == 1
    assert windows[0].ended_at_s == 70.0
    # The FIRST off is when the device was released. Asserted because dropping
    # the `open_at is None` guard leaves the count and the end untouched and
    # only moves the START -- a mutation this test passed until it checked
    # this line, and one that would under-correct every later segment by the
    # gap between the two toggles.
    assert windows[0].began_at_s == 10.0


def test_a_repeated_on_does_not_close_a_window_that_is_not_open():
    windows = off_windows([_mic(5.0, True), _mic(10.0, False), _mic(70.0, True)])
    assert len(windows) == 1


def test_events_other_than_microphone_are_ignored():
    records = [
        {"kind": "event", "event": "keepalive-warning", "detail": {}, "at_s": 5.0},
        _mic(10.0, False), _mic(20.0, True),
    ]
    assert len(off_windows(records)) == 1


def test_an_event_with_no_usable_timestamp_is_skipped_not_guessed():
    assert off_windows([_mic(None, False), _mic(20.0, True)]) == ()


# --- the arithmetic --------------------------------------------------------------


def test_pause_before_a_moment_inside_the_window_counts_only_what_elapsed():
    window = OffWindow(began_at_s=10.0, ended_at_s=70.0)
    assert window.paused_before(5.0) == 0.0
    assert window.paused_before(40.0) == 30.0
    assert window.paused_before(200.0) == 60.0


def test_pauses_accumulate_across_windows():
    windows = (OffWindow(10.0, 20.0), OffWindow(50.0, 80.0))
    assert paused_before(windows, 100.0) == 40.0
    assert paused_before(windows, 60.0) == 20.0


# --- the correction --------------------------------------------------------------


def test_the_inflated_latency_is_restored():
    """A 60-second pause makes a healthy 0.9s pipeline report 60.9s. That is the
    whole defect, in one assertion."""
    records = [
        _segment("interviewer", 0.9, 5.0),
        _mic(10.0, False), _mic(70.0, True),
        _segment("interviewer", 60.9, 75.0),
    ]
    result = correct_interviewer_latencies(records)
    assert result.original == (0.9, 60.9)
    assert result.corrected[0] == 0.9, "a segment before the pause must not move"
    assert abs(result.corrected[1] - 0.9) < 1e-9
    assert result.applied is True


def test_the_interviewee_is_never_corrected():
    """The microphone is the interviewer's device; the interviewee arrives over
    loopback and is untouched. Correcting both would move a number that was
    never wrong."""
    records = [
        _mic(10.0, False), _mic(70.0, True),
        _segment("interviewee", 0.8, 75.0),
        _segment("interviewer", 60.8, 75.0),
    ]
    result = correct_interviewer_latencies(records)
    assert result.original == (60.8,), "only interviewer segments are in scope"


def test_a_session_that_never_toggled_reports_no_correction():
    """Not 'corrected by zero'. A report claiming a correction on a session that
    needed none invites the reader to distrust numbers that were always right."""
    records = [_segment("interviewer", 0.9, 5.0), _segment("interviewer", 1.1, 9.0)]
    result = correct_interviewer_latencies(records)
    assert result.corrected == result.original
    assert result.applied is False
    assert result.windows == 0


def test_a_segment_finalized_inside_the_off_window_is_only_partly_corrected():
    """Rare -- the engine finalized an utterance just after the device was
    released. Crediting the whole window would over-correct into a negative."""
    records = [_mic(10.0, False), _segment("interviewer", 5.0, 12.0), _mic(70.0, True)]
    result = correct_interviewer_latencies(records)
    assert abs(result.corrected[0] - 3.0) < 1e-9


def test_the_correction_never_produces_a_negative():
    records = [_mic(10.0, False), _mic(70.0, True), _segment("interviewer", 0.5, 75.0)]
    assert correct_interviewer_latencies(records).corrected == (0.0,)


def test_a_segment_with_no_wall_position_is_carried_through_unchanged():
    """Dropping it would quietly shrink the sample the median is taken over,
    which moves the reported figure without saying so."""
    records = [
        _mic(10.0, False), _mic(70.0, True),
        {"kind": "segment", "speaker": "interviewer", "latency_s": 4.0},
    ]
    result = correct_interviewer_latencies(records)
    assert result.corrected == (4.0,)
    assert len(result.corrected) == len(result.original) == 1


def test_the_total_off_time_is_reported():
    records = [_mic(10.0, False), _mic(70.0, True), _mic(100.0, False), _mic(130.0, True)]
    result = correct_interviewer_latencies(records)
    assert result.windows == 2
    assert result.total_off_s == 90.0


def test_two_pauses_both_count_against_a_later_segment():
    records = [
        _mic(10.0, False), _mic(70.0, True),
        _mic(100.0, False), _mic(130.0, True),
        _segment("interviewer", 91.0, 140.0),
    ]
    result = correct_interviewer_latencies(records)
    assert abs(result.corrected[0] - 1.0) < 1e-9
