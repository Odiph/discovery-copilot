"""A session's median latency mostly reports how much the speakers paused.

**The investigation this file exists to prevent.** On 2026-08-12 a real session
measured a ~4.8s median on both sides, against a ~1.83s figure treated as the
baseline and a 1.5-2.0s target. It was written up as an unexplained 2.6x
regression and carried forward as an open bug.

There was no regression. Per-segment `latency_s` is BIMODAL -- a fast cluster
near 0.65-1.0s and a slow one near 4.7-5.6s, with almost nothing between them --
and the two ALTERNATE inside a single minute of the same session on the same
code. Sessions recorded LATER, on that same code, came back at 0.89s and 1.77s.
The "baseline" had been measured on a 25-second benchmark clip with natural
turn-taking pauses; the slow session was continuous playback. One number had
been asked to describe two populations, and it answered about whichever one
happened to dominate.

So this is a REPORTING fix, and nothing here makes anything faster. What it
changes is that the next person to read a latency summary is told which
population they are looking at.
"""

from __future__ import annotations

import statistics

from app.session.store import (
    FINALIZATION_BOUNDARY_S,
    latency_stats,
    latency_stats_with_modes,
)

# Real per-segment values, in order, from `20260812T232519` -- the session that
# triggered the investigation. The alternation is the entire point: a queue
# backlog cannot recover from 4.71s to 0.70s between consecutive segments, so
# whatever produces the slow mode is not congestion.
_REAL_BIMODAL = [0.78, 5.64, 4.85, 5.64, 4.71, 0.70, 0.75, 0.68, 0.75, 1.31, 1.08, 0.53]


def test_the_single_median_is_still_reported_unchanged():
    """Every existing reader keeps working, and the headline number keeps
    meaning what it meant. The split is added beside it, never instead of it --
    the median is still the right answer to "what did this session feel like"."""
    values = list(_REAL_BIMODAL)
    plain = latency_stats(values)
    split = latency_stats_with_modes(values)
    for key in ("count", "median_s", "p95_s", "min_s", "max_s"):
        assert split[key] == plain[key], key


def test_the_real_session_is_reported_as_two_populations():
    stats = latency_stats_with_modes(list(_REAL_BIMODAL))
    assert stats["is_bimodal"] is True
    assert stats["endpoint_final_count"] == 8
    assert stats["window_final_count"] == 4
    # The two medians must be far apart -- that separation IS the finding. If a
    # future boundary change collapsed them toward each other this would be
    # reporting a distinction that no longer exists.
    assert stats["window_final_median_s"] - stats["endpoint_final_median_s"] > 3.0


def test_the_conflated_median_sits_in_neither_population():
    """The precise reason the single number misled a reader for a day.

    Its median lands in the fast cluster here, but the session was WRITTEN UP
    from the interviewee/interviewer medians of ~4.8s -- and on a continuous
    stretch the median lands in the slow cluster instead. Either way it
    describes one mode while being read as a property of the pipeline."""
    stats = latency_stats_with_modes(list(_REAL_BIMODAL))
    fast = [v for v in _REAL_BIMODAL if v < FINALIZATION_BOUNDARY_S]
    slow = [v for v in _REAL_BIMODAL if v >= FINALIZATION_BOUNDARY_S]
    assert stats["endpoint_final_median_s"] == statistics.median(fast)
    assert stats["window_final_median_s"] == statistics.median(slow)


def test_a_single_mode_session_is_not_labelled_bimodal():
    """An ordinary turn-taking interview is all one mode, and must read exactly
    as it always has. Labelling it bimodal on the strength of a stray sample
    would put a paragraph of explanation under every clean recording -- which is
    how a warning becomes something to scroll past."""
    stats = latency_stats_with_modes([0.6, 0.7, 0.8, 0.9, 0.75])
    assert stats["is_bimodal"] is False
    assert stats["window_final_count"] == 0
    assert "window_final_median_s" not in stats, (
        "a mode with no samples must report no median -- a 0.0 there would read as 'instant', "
        "the most flattering possible lie about a mode that never occurred"
    )


def test_an_all_slow_session_is_not_labelled_bimodal_either():
    stats = latency_stats_with_modes([4.7, 5.1, 4.9])
    assert stats["is_bimodal"] is False
    assert stats["endpoint_final_count"] == 0
    assert "endpoint_final_median_s" not in stats


def test_an_empty_session_gains_no_mode_keys():
    """`latency_stats` answers `{"count": 0}` for no samples, and this must not
    quietly grow that contract -- `render.py` branches on `count < 2` before it
    ever looks at a mode key."""
    assert latency_stats_with_modes([]) == {"count": 0}


def test_the_boundary_sits_in_the_empty_band_between_the_two_clusters():
    """The constant is measured rather than chosen, and this pins that claim.

    Across the real session above, nothing sits within 0.6s of the boundary on
    the slow side and the fast cluster tops out well below it. A boundary landing
    INSIDE either cluster would split one population in half and report the
    halves as two modes -- which looks exactly like the finding and is noise."""
    for value in _REAL_BIMODAL:
        assert abs(value - FINALIZATION_BOUNDARY_S) > 0.5, (
            f"{value}s sits on the boundary, so the split is guessing about it"
        )
