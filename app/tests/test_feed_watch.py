"""One feed stops being transcribed while the other keeps going
(`app/session/feed_watch.py`).

WHY THE EXISTING DETECTORS DID NOT CATCH THIS. `StreamSupervisor` already
watches for a stream going quiet, but it watches FRAME ARRIVAL: `idle` means
the device stopped handing over buffers. On 2026-08-17 the device handed over
buffers for two solid minutes and every sample in them was zero -- a Bluetooth
headset had moved playback to its hands-free endpoint and the capture was
still bound to the stereo one. Frames arrived. Nothing was said. The session
recorded 24 interviewer segments and 0 interviewee segments and reported
itself healthy throughout.

So this watches the OUTPUT instead: segments. The rule it encodes is
comparative, not absolute, and that is the whole design. "This side has
produced nothing" is not evidence of a fault -- for the first minute of an
interview it is just the interviewer asking a question. "This side has
produced nothing WHILE THE OTHER SIDE HAS BEEN TALKING" is evidence, because
one working feed proves the interview is under way and the machine is
transcribing.

An alarm that fires during ordinary silence gets ignored within one interview,
and an ignored alarm is worse than none: it is the same green banner, wearing
a warning's clothes. Every threshold below exists to keep the first alert
true.
"""

from __future__ import annotations

from app.session.feed_watch import FeedWatch, FeedWatchConfig

CONFIG = FeedWatchConfig(silence_s=45.0, repeat_s=120.0, min_other_segments=3)


def _watch(config=CONFIG):
    return FeedWatch(speakers=("interviewee", "interviewer"), config=config)


def _talk(watch, speaker, times):
    for at_s in times:
        watch.note_segment(speaker, at_s=at_s)


# --- the quiet start is NOT a fault -------------------------------------------------------


def test_nothing_is_reported_before_anyone_has_spoken():
    """Both sides silent means the interview has not started, not that a feed
    is broken."""
    watch = _watch()
    assert watch.poll(elapsed_s=300.0) == ()


def test_one_side_silent_is_not_reported_while_the_other_has_barely_spoken():
    """Two interviewer segments is an opening question. It is not proof that
    the machine is transcribing well enough to indict the other side."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 2.0])
    assert watch.poll(elapsed_s=300.0) == ()


def test_one_side_silent_is_not_reported_before_the_silence_window():
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 2.0, 3.0, 4.0])
    assert watch.poll(elapsed_s=44.0) == ()


# --- the failure this was written for ------------------------------------------------------


def test_a_side_that_never_transcribes_is_reported_once_the_other_is_clearly_working():
    """The 2026-08-17 shape: interviewer flowing, interviewee at zero."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0, 40.0])
    alerts = watch.poll(elapsed_s=46.0)
    assert [(a.speaker, a.kind) for a in alerts] == [("interviewee", "not-transcribing")]


def test_the_alert_carries_how_long_the_side_has_been_silent():
    """A number the operator can act on -- "nothing for 46s" is checkable
    against what they just heard in the room."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    alert = watch.poll(elapsed_s=46.0)[0]
    assert alert.since_s == 46.0


def test_a_feed_that_dies_MID_interview_is_reported_too():
    """The Bluetooth endpoint swap happens when the CALL starts, which can be
    minutes in. A check that only looked at "zero for the whole session" would
    miss every one of those."""
    watch = _watch()
    _talk(watch, "interviewee", [10.0, 20.0, 30.0])
    _talk(watch, "interviewer", [11.0, 40.0, 60.0, 70.0])
    assert watch.poll(elapsed_s=60.0) == ()
    alerts = watch.poll(elapsed_s=80.0)
    assert [a.speaker for a in alerts] == ["interviewee"]


# --- edge-triggered, and it says when it comes back -----------------------------------------


def test_it_reports_once_rather_than_on_every_poll():
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    assert len(watch.poll(elapsed_s=46.0)) == 1
    assert watch.poll(elapsed_s=47.0) == ()
    assert watch.poll(elapsed_s=60.0) == ()


def test_it_repeats_after_the_repeat_window_because_the_problem_is_still_live():
    """Said once at minute one and never again, for a fault that lasts the
    whole interview, is how an operator ends up believing it resolved."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    assert len(watch.poll(elapsed_s=46.0)) == 1
    assert watch.poll(elapsed_s=150.0) == ()
    assert len(watch.poll(elapsed_s=167.0)) == 1


def test_recovery_is_reported_so_a_fix_is_visibly_confirmed():
    """The operator switches an output device mid-call and needs to know it
    worked without waiting for the transcript afterwards."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    assert len(watch.poll(elapsed_s=46.0)) == 1
    watch.note_segment("interviewee", at_s=50.0)
    alerts = watch.poll(elapsed_s=51.0)
    assert [(a.speaker, a.kind) for a in alerts] == [("interviewee", "recovered")]


def test_recovery_is_reported_once_and_then_the_watch_is_quiet_again():
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    watch.poll(elapsed_s=46.0)
    watch.note_segment("interviewee", at_s=50.0)
    watch.poll(elapsed_s=51.0)
    assert watch.poll(elapsed_s=52.0) == ()


def test_a_side_can_be_reported_again_after_recovering_and_dying_again():
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    watch.poll(elapsed_s=46.0)
    watch.note_segment("interviewee", at_s=50.0)
    watch.poll(elapsed_s=51.0)
    _talk(watch, "interviewer", [60.0, 70.0, 80.0])
    alerts = watch.poll(elapsed_s=100.0)
    assert [(a.speaker, a.kind) for a in alerts] == [("interviewee", "not-transcribing")]


# --- the interviewer side is watched too ------------------------------------------------------


def test_an_interviewer_listening_while_the_interviewee_answers_is_NOT_a_fault():
    """Measured on 2026-08-18, and the reason the two sides stopped sharing one
    window. A real 10-minute interview produced 186 interviewee segments
    against 105 interviewer ones and fired this alert TWICE -- at 6:04 and
    7:41 -- both times on the microphone, both times because the interviewer
    had spent 45s listening while the interviewee talked.

    Under the comparative rule those alerts were correct: the other side WAS
    producing. They were still false, because the premise was wrong. An
    interviewer quiet for 45s while the interviewee answers is not a degraded
    interview, it is a well-run one -- the Mom Test's whole posture is that
    the interviewer should be the side producing less. A detector that fires
    on the target behaviour trains the operator to ignore it, which is exactly
    what this module's own docstring says is worse than having no alarm."""
    watch = _watch()
    _talk(watch, "interviewee", [1.0, 5.0, 20.0])
    assert watch.poll(elapsed_s=46.0) == ()


def test_the_interviewer_side_is_still_watched_on_a_LONGER_rule():
    """Not exempt -- a dead microphone still costs the coach its entire
    coverage signal, and that has to be caught. It is held to a window long
    enough that only a genuinely absent interviewer reaches it."""
    watch = _watch()
    _talk(watch, "interviewee", [1.0, 5.0, 20.0])
    alerts = watch.poll(elapsed_s=181.0)
    assert [a.speaker for a in alerts] == ["interviewer"]


def test_the_interviewee_side_keeps_the_SHORT_window():
    """The asymmetry only makes sense in one direction. An interviewee silent
    for 45s while the interviewer talks is the 2026-08-17 fault this module
    was built for, and nothing about the microphone's longer window may
    lengthen it."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    alerts = watch.poll(elapsed_s=46.0)
    assert [a.speaker for a in alerts] == ["interviewee"]


def test_is_silent_exposes_the_same_judgement_for_the_status_banner():
    """The poller folds this into `StreamHealth` rather than keeping a second
    opinion about the same fact."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    watch.poll(elapsed_s=46.0)
    assert watch.is_silent("interviewee") is True
    assert watch.is_silent("interviewer") is False


# --- the level probe: is the feed dead, or is nobody talking? ----------------------------


def test_a_quiet_microphone_is_not_reported_because_quiet_means_nobody_talking():
    """The distinction the transcript alone cannot draw. A microphone carrying
    digital silence has nothing to transcribe, so "no segments" is the correct
    output, not a fault. Alerting there says "your mic is broken" to somebody
    who is simply listening."""
    watch = _watch()
    _talk(watch, "interviewee", [1.0, 5.0, 20.0])
    assert watch.poll(elapsed_s=181.0, levels={"interviewer": -80.0}) == ()


def test_a_LOUD_microphone_with_no_segments_IS_reported_because_that_is_a_fault():
    """The other half, and the one that earns the probe. Real sound going in
    and no segments coming out cannot be explained by silence in the room --
    something between the device and the engine has stopped."""
    watch = _watch()
    _talk(watch, "interviewee", [1.0, 5.0, 20.0])
    alerts = watch.poll(elapsed_s=181.0, levels={"interviewer": -12.0})
    assert [a.speaker for a in alerts] == ["interviewer"]


def test_a_quiet_INTERVIEWEE_is_still_reported_because_silence_there_IS_the_fault():
    """The probe is deliberately one-sided. On 2026-08-17 the loopback carried
    two solid minutes of zeros -- digital silence WAS the fault, because the
    call was audible in the room and the capture was bound to an endpoint that
    had stopped receiving it. Suppressing on quiet there would blind this
    module to the exact case it was written for."""
    watch = _watch()
    _talk(watch, "interviewer", [1.0, 5.0, 20.0])
    alerts = watch.poll(elapsed_s=46.0, levels={"interviewee": -90.0})
    assert [a.speaker for a in alerts] == ["interviewee"]


def test_no_level_reading_falls_back_to_reporting_rather_than_suppressing():
    """An absent measurement must never be read as "quiet". A probe that fails
    open would silently disable the whole detector the moment level plumbing
    broke, which is the failure mode hardest to notice."""
    watch = _watch()
    _talk(watch, "interviewee", [1.0, 5.0, 20.0])
    alerts = watch.poll(elapsed_s=181.0, levels={"interviewer": None})
    assert [a.speaker for a in alerts] == ["interviewer"]
    watch2 = _watch()
    _talk(watch2, "interviewee", [1.0, 5.0, 20.0])
    assert [a.speaker for a in watch2.poll(elapsed_s=181.0)] == ["interviewer"]
