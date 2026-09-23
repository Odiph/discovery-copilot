"""`CoachHealth` -- the consecutive-failure counter that tells a single bad
cycle apart from a dead coach (04-CONTEXT.md §D7; 04-02-PLAN.md §D7).

Shaped like `app/web/status.py::StreamHealth`: a small frozen value, no
concrete dependency imported, fully unit-testable with no reasoning stack at
all. A CONSECUTIVE-failure COUNTER, not STT's immediate-fail boolean -- a
single slow round trip is normal here (the next utterance brings another
chance within seconds) and a single dead stream is not. The threshold is a
config constant (`app/config.py::DEFAULT_COACH_FAILURE_THRESHOLD`), passed to
`is_unhealthy()` at call time rather than stored on the value itself, mirroring
`derive_status()`'s own "reads state, decides externally" shape.

This module deliberately does NOT extend `ServerStatus`/`derive_status` in
`app/web/status.py` -- that is 04-03's edit. Keeping it here is what lets this
plan stay fully autonomous and lets 04-03's status tests run without a
reasoning stack.

Must NEVER import `anthropic`, `app.coach.pipeline`, or any other concrete
coach module -- a plain value, constructible and assertable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoachHealth:
    """Immutable. `consecutive_failures` counts timeouts, parse failures,
    refusals, truncations and any other exception from EITHER stage --
    routed through exactly one place (`app/coach/pipeline.py`) so the counter
    can never be updated from three call sites with three different
    opinions. A cycle that was SKIPPED (one already in flight) is neither a
    success nor a failure and must never touch this counter at all."""

    consecutive_failures: int = 0

    def with_failure(self) -> "CoachHealth":
        return CoachHealth(consecutive_failures=self.consecutive_failures + 1)

    def with_success(self) -> "CoachHealth":
        """ANY completed cycle resets to zero -- a perfectly healthy coach
        must never accumulate toward the threshold across turns separated by
        a success, only across a genuinely unbroken run of failures."""
        return CoachHealth(consecutive_failures=0)

    def is_unhealthy(self, *, threshold: int) -> bool:
        return self.consecutive_failures >= threshold
