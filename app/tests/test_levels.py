"""The pre-call level check must judge what a human judges.

The measurement this replaces was `sqrt(mean(samples**2))` over the WHOLE
capture window, which mostly measured the pauses: on the real machine the same
microphone read 127 over an 8-second window and 34 over a 30-second one while
the user talked throughout, and the conclusion drawn was that the mic was
broken. Every fixture below therefore contains REAL silence -- a fixture whose
mean and whose loudest half-second coincide (a continuous tone) cannot tell
this change apart from its opposite, so the two headline tests assert both
that the new measure passes AND that the old one would have failed on the very
same samples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.capture.levels import (
    BAND_CLIPPING,
    BAND_GOOD,
    BAND_HOT,
    BAND_LOW,
    BAND_SILENCE,
    BAND_TOO_QUIET,
    FULL_SCALE,
    SILENCE_FLOOR_DBFS,
    classify_dbfs,
    measure_level,
)

SAMPLE_RATE = 16000

# The floor `precheck` scores against -- imported here in value, not by name,
# so this module states the property (passes / would have failed) in the same
# units a reader of the report sees.
LEVEL_FLOOR_DBFS = -36.0


def _tone(seconds: float, amplitude: int, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)))
    return (amplitude * np.sin(2 * np.pi * 440.0 * t / sample_rate)).astype(np.int16)


def _silence(seconds: float, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.int16)


def _speech_with_pauses(
    *, total_s: float, burst_s: float, bursts: int, amplitude: int, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """`bursts` spans of tone, evenly spread through `total_s` of otherwise
    silent capture -- the intermittent shape real speech has, and the shape
    the old whole-window mean punished."""
    signal = _silence(total_s, sample_rate=sample_rate)
    gap = len(signal) // bursts
    burst = _tone(burst_s, amplitude, sample_rate=sample_rate)
    for i in range(bursts):
        start = i * gap
        signal[start:start + len(burst)] = burst
    return signal


def _whole_window_rms_dbfs(samples: np.ndarray) -> float:
    """The measure this change removed, kept alive HERE so the tests can prove
    the difference rather than assert it."""
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return 20.0 * math.log10(rms / FULL_SCALE) if rms > 0 else SILENCE_FLOOR_DBFS


# --- The headline property: peaks, not the mean -----------------------------------------


def test_intermittent_speech_passes_the_floor_the_old_whole_window_mean_would_have_failed():
    # 30 seconds of capture containing 0.7s of speech: exactly the case that
    # was misdiagnosed as a broken microphone.
    samples = _speech_with_pauses(total_s=30.0, burst_s=0.7, bursts=1, amplitude=4000)

    old_dbfs = _whole_window_rms_dbfs(samples)
    new = measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE)

    assert old_dbfs < LEVEL_FLOOR_DBFS, (
        f"fixture cannot exhibit the bug: the old measure ({old_dbfs:.1f} dBFS) already passed"
    )
    assert new.peak_dbfs >= LEVEL_FLOOR_DBFS
    assert new.is_usable and new.band == BAND_LOW
    # ~15 dB apart -- the gap is the pauses, and it is not a rounding argument.
    assert new.peak_dbfs - old_dbfs > 10.0


def test_a_longer_window_of_the_same_speech_scores_the_same_while_the_old_measure_collapses():
    # Same voice, same loudness, two window lengths. The old measure fell with
    # window length (127 at 8s, 34 at 30s on the real machine); the new one
    # must not move at all.
    short = _speech_with_pauses(total_s=8.0, burst_s=0.7, bursts=1, amplitude=4000)
    long = np.concatenate([short, _silence(22.0)])

    old_short = _whole_window_rms_dbfs(short)
    old_long = _whole_window_rms_dbfs(long)
    new_short = measure_level(short.tobytes(), sample_rate=SAMPLE_RATE)
    new_long = measure_level(long.tobytes(), sample_rate=SAMPLE_RATE)

    assert old_long < old_short - 5.0, "fixture cannot exhibit the bug: the old measure held steady"
    assert old_short >= LEVEL_FLOOR_DBFS > old_long, (
        "fixture cannot exhibit the VERDICT flip: the old measure must pass at 8s and fail at 30s"
    )
    assert new_long.peak_dbfs == pytest.approx(new_short.peak_dbfs, abs=0.01)
    assert new_long.band == new_short.band == BAND_LOW


def test_the_loudest_half_second_wins_even_when_it_sits_at_the_very_end():
    # Guards the search, not just the arithmetic: a peak finder that only
    # looked at the head of the buffer would pass every other test here.
    samples = np.concatenate([_silence(5.0), _tone(0.6, 9000)])
    assert measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE).band == BAND_GOOD


def test_a_single_loud_click_cannot_pass_a_dead_microphone():
    # The other side of "score the peak": one sample at full scale is a pop,
    # not a level. Averaged over the 500ms window it cannot reach the floor,
    # which is why the peak is a windowed RMS and not `max(abs(samples))`.
    samples = _silence(3.0)
    samples[1000] = 32767
    measurement = measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE)
    assert measurement.band == BAND_TOO_QUIET
    assert not measurement.is_usable


# --- dBFS arithmetic, checked against a value computed by hand ---------------------------


def test_a_constant_half_scale_signal_measures_minus_six_dbfs():
    # A square wave at 16384 has RMS exactly 16384 = half of full scale, so
    # 20*log10(0.5) = -6.02 dBFS. Chosen because the answer cannot coincide
    # with a mistaken formula (mean-of-abs, peak-sample, or a 10*log10).
    samples = np.full(SAMPLE_RATE, 16384, dtype=np.int16)
    measurement = measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE)
    assert measurement.peak_dbfs == pytest.approx(-6.02, abs=0.02)
    # -6.02 sits a hundredth of a dB BELOW the good/hot boundary at -6.0, so
    # this also pins which side of an exact boundary the classifier takes.
    assert measurement.band == BAND_GOOD


def test_a_full_scale_signal_reads_as_clipping():
    samples = np.full(SAMPLE_RATE, 32767, dtype=np.int16)
    assert measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE).band == BAND_CLIPPING


# --- Bands ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dbfs,expected",
    [
        (-90.0, BAND_SILENCE),
        (-50.1, BAND_SILENCE),
        (-50.0, BAND_TOO_QUIET),
        (-36.1, BAND_TOO_QUIET),
        (-36.0, BAND_LOW),
        (-18.1, BAND_LOW),
        (-18.0, BAND_GOOD),
        (-6.1, BAND_GOOD),
        (-6.0, BAND_HOT),
        (-1.1, BAND_HOT),
        (-1.0, BAND_CLIPPING),
        (0.0, BAND_CLIPPING),
    ],
)
def test_band_boundaries_are_exactly_where_the_report_claims(dbfs, expected):
    assert classify_dbfs(dbfs) == expected


def test_only_the_middle_bands_count_as_usable():
    assert not measure_level(_silence(1.0).tobytes(), sample_rate=SAMPLE_RATE).is_usable
    assert measure_level(_tone(1.0, 5000).tobytes(), sample_rate=SAMPLE_RATE).is_usable
    assert not measure_level(
        np.full(SAMPLE_RATE, 32767, dtype=np.int16).tobytes(), sample_rate=SAMPLE_RATE
    ).is_usable


# --- Degenerate inputs: no crash, no -inf, no confident wrong answer ---------------------


def test_all_zero_input_reports_a_finite_floor_never_negative_infinity():
    measurement = measure_level(_silence(2.0).tobytes(), sample_rate=SAMPLE_RATE)
    assert math.isfinite(measurement.peak_dbfs)
    assert measurement.peak_dbfs == SILENCE_FLOOR_DBFS
    assert measurement.band == BAND_SILENCE
    # The whole point: a human reads this string off a pre-call report.
    assert "inf" not in measurement.describe().lower()
    assert "nan" not in measurement.describe().lower()


def test_an_empty_capture_is_silence_not_a_crash():
    measurement = measure_level(b"", sample_rate=SAMPLE_RATE)
    assert measurement.band == BAND_SILENCE
    assert measurement.peak_dbfs == SILENCE_FLOOR_DBFS
    assert measurement.window_s == 0.0


def test_a_window_shorter_than_500ms_is_scored_whole_and_says_how_short_it_was():
    samples = _tone(0.2, 16384)
    measurement = measure_level(samples.tobytes(), sample_rate=SAMPLE_RATE)
    assert measurement.window_s == pytest.approx(0.2, abs=0.001)
    # A 0.2s capture of a half-scale sine: RMS = 16384/sqrt(2) -> -9.03 dBFS.
    # Scored whole rather than discarded, and NOT diluted by the 300ms of
    # window that was never captured.
    assert measurement.peak_dbfs == pytest.approx(-9.03, abs=0.05)
    assert measurement.band == BAND_GOOD


def test_an_odd_length_buffer_of_one_sample_still_measures():
    measurement = measure_level(np.array([16384], dtype=np.int16).tobytes(), sample_rate=SAMPLE_RATE)
    assert measurement.peak_dbfs == pytest.approx(-6.02, abs=0.02)
    assert measurement.window_s == pytest.approx(1 / SAMPLE_RATE)


def test_a_numpy_array_and_its_bytes_measure_identically():
    samples = _speech_with_pauses(total_s=4.0, burst_s=0.6, bursts=2, amplitude=6000)
    assert measure_level(samples, sample_rate=SAMPLE_RATE) == measure_level(
        samples.tobytes(), sample_rate=SAMPLE_RATE
    )


# --- RecentLevel: the live counterpart of measure_level ---------------------------------


def _tone_pcm(amplitude: int, samples: int = 320) -> bytes:
    import numpy as np
    return (np.full(samples, amplitude, dtype=np.int16)).tobytes()


def test_recent_level_reports_nothing_before_any_frame_arrives():
    """`None` means "not measured". Every caller has to tell that apart from
    "quiet", because reading an absent measurement as silence is what would
    let a stalled pusher pass for a quiet room."""
    from app.capture.levels import RecentLevel

    assert RecentLevel().peak_dbfs(at_s=10.0) is None


def test_recent_level_scores_a_loud_stream_loud():
    from app.capture.levels import RecentLevel

    level = RecentLevel()
    level.push(_tone_pcm(8000), at_s=1.0)
    assert level.peak_dbfs(at_s=1.5) > -20.0


def test_recent_level_scores_digital_silence_on_the_floor():
    from app.capture.levels import RecentLevel, SILENCE_FLOOR_DBFS

    level = RecentLevel()
    level.push(_tone_pcm(0), at_s=1.0)
    assert level.peak_dbfs(at_s=1.5) == SILENCE_FLOOR_DBFS


def test_recent_level_forgets_sound_older_than_its_window():
    """Sound from a minute ago must not vouch for a feed that has since died,
    which is the entire reason this is a window and not a running maximum."""
    from app.capture.levels import RecentLevel

    level = RecentLevel(window_s=15.0)
    level.push(_tone_pcm(8000), at_s=1.0)
    assert level.peak_dbfs(at_s=10.0) > -20.0
    assert level.peak_dbfs(at_s=30.0) is None


def test_recent_level_keeps_the_loudest_frame_not_the_latest():
    """A peak, not a mean: one sentence has to keep the stream "loud" across
    the pause that follows it, or an ordinary gap between sentences reads as a
    dead microphone."""
    from app.capture.levels import RecentLevel

    level = RecentLevel()
    level.push(_tone_pcm(8000), at_s=1.0)
    level.push(_tone_pcm(0), at_s=2.0)
    assert level.peak_dbfs(at_s=3.0) > -20.0
