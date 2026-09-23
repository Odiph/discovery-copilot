"""Which triggering rule puts a suggestion on screen before the interviewer
has to speak -- simulated over a real transcript, cheaply enough to sweep.

**The question this exists to answer.** The coach currently fires one cycle per
interviewee SEGMENT, and segments are ~5-word engine finalizations, so a
60-word answer produces ~12 cycles. The measured 26-minute interview ran 184
cycles and showed 401 suggestions, about 15 a minute, which the design backlog
calls unusable. The obvious fix -- fire once when the interviewee stops talking
-- is probably WRONG, and that is the whole reason for this file: a cycle takes
~8 seconds, so a suggestion triggered at the end of a turn lands well into the
interviewer's next question, after the moment it was built for has passed.

Firing on every fragment is not carelessness. It is a latency workaround: fire
early and often so that something is on screen by the time it is needed.
Removing it without addressing latency breaks the thing it was compensating
for, and no cost model can see that -- a cost model would happily recommend the
cheapest rule, which is the one that never helps anybody.

So the metric that decides this is not cycles and not dollars. It is **the share
of interviewer turns that had a FRESH suggestion ready before the turn began**,
where fresh means triggered by the interviewee speech that preceded that turn
rather than left over from earlier.

**This is a SIMULATION and its assumptions are load-bearing.** It replays a real
transcript's timings against a measured cycle-latency figure. It assumes cycle
latency is independent of what triggered it, that a completed cycle is equally
useful whenever it lands, and that a suggestion on screen is a suggestion read.
None of those is exactly true. What it is good for is ORDERING the rules, which
is the decision at hand; the winner should then be confirmed with a live
rehearsal, which is the only thing that can measure the parts this cannot.

The queue is modelled faithfully rather than approximately: one worker, one
slot, and a full slot drops the OLDEST waiting trigger and keeps the newest --
`pipeline.submit`'s actual behaviour. That detail is what makes the current
rule's numbers come out saturated rather than merely high, and a simpler model
would have made every rule look better than it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from benchmark.console import use_utf8_console  # noqa: E402

INTERVIEWER = "interviewer"
INTERVIEWEE = "interviewee"

# Measured by `rehearse.py` against google/gemini-2.5-flash: median 7.9s,
# p95 9.8s. Haiku direct came out at 7.3s median. The default is the gemini
# median because that is what `.env` currently selects; `--cycle-s` sweeps it,
# and the sweep matters more than the default -- if the ordering of the rules
# changes with latency, the choice of rule is really a choice about latency.
DEFAULT_CYCLE_S = 7.9


@dataclass(frozen=True)
class Turn:
    speaker: str
    start_s: float
    end_s: float
    segments: "tuple[int, ...]"


@dataclass(frozen=True)
class SimResult:
    rule: str
    cycles: int
    dropped: int
    turns: int
    covered: int
    stale_only: int

    @property
    def coverage(self) -> float:
        return self.covered / self.turns if self.turns else 0.0


def turns_of(records: "Sequence[Mapping[str, Any]]") -> "list[Turn]":
    """Maximal runs of consecutive segments by one speaker.

    A turn is the conversational unit; segments are an engine artefact. Built
    from speaker CHANGES rather than from pauses.

    **A turn's start is the END of the previous segment, never that segment's
    own `audio_start_s`.** `app/stt/live.py` fabricates `audio_start_s` as the
    end of the SAME SPEAKER's previous segment, so for the first segment of a
    turn it points back past the whole of the other speaker's turn -- turns
    built from it overlap, orderings invert, and every "was it ready in time"
    comparison silently answers no. The first version of this file did exactly
    that and reported 0% coverage for every rule at every latency, which is
    what a broken denominator looks like when it is polite about it. Only
    `audio_end_s` is trustworthy here, so the timeline is reconstructed from
    end-to-end deltas alone."""
    segments = [r for r in records if r.get("kind") == "segment"]
    segments.sort(key=lambda r: r.get("audio_end_s") or 0.0)
    turns: "list[Turn]" = []
    previous_end = 0.0
    for record in segments:
        speaker = record.get("speaker")
        end = float(record.get("audio_end_s") or 0.0)
        seq = record.get("seq")
        if turns and turns[-1].speaker == speaker:
            previous = turns[-1]
            turns[-1] = Turn(speaker, previous.start_s, end, previous.segments + (seq,))
        else:
            turns.append(Turn(speaker, previous_end, end, (seq,)))
        previous_end = end
    return turns


# --- the rules -------------------------------------------------------------------
# Each returns the audio-clock moments at which a cycle is TRIGGERED.


def rule_every_segment(records, turns) -> "list[float]":
    """Today's behaviour: every interviewee segment."""
    return [float(r.get("audio_end_s") or 0.0) for r in records
            if r.get("kind") == "segment" and r.get("speaker") == INTERVIEWEE]


def rule_turn_end(records, turns) -> "list[float]":
    """Once, when the interviewee stops talking. Semantically clean, and the
    one this file exists to be sceptical about."""
    return [turn.end_s for turn in turns if turn.speaker == INTERVIEWEE]


def rule_turn_plus(delay_s: float) -> "Callable[[Any, Any], list[float]]":
    """Once per interviewee turn, `delay_s` INTO it -- so the cycle completes
    while they are still talking. Skips turns too short to contain the delay:
    a four-second answer usually does not need a coached follow-up, and firing
    anyway would spend a cycle that cannot land in time."""
    def rule(records, turns) -> "list[float]":
        return [turn.start_s + delay_s for turn in turns
                if turn.speaker == INTERVIEWEE and turn.end_s - turn.start_s >= delay_s]
    return rule


def rule_min_interval(interval_s: float) -> "Callable[[Any, Any], list[float]]":
    """Today's triggers, rate-limited. The arbitrary-number option, included so
    the comparison is fair to it rather than assuming it loses."""
    def rule(records, turns) -> "list[float]":
        kept: "list[float]" = []
        for at in rule_every_segment(records, turns):
            if not kept or at - kept[-1] >= interval_s:
                kept.append(at)
        return kept
    return rule


def simulate(
    records: "Sequence[Mapping[str, Any]]",
    trigger_at: "Callable[[Any, Any], list[float]]",
    *,
    name: str,
    cycle_s: float,
) -> SimResult:
    """Run one rule and score it on turns covered.

    The queue is `pipeline.submit`'s: one worker, one waiting slot, and a full
    slot drops the OLDEST waiting trigger. A cycle triggered at T finishes at
    the later of T and the worker's free time, plus `cycle_s`."""
    turns = turns_of(records)
    triggers = sorted(trigger_at(records, turns))

    free_at = 0.0
    waiting: "Optional[float]" = None
    completions: "list[tuple[float, float]]" = []  # (trigger_at, finished_at)
    dropped = 0

    for at in triggers:
        if at >= free_at:
            # Anything already waiting would have started first.
            if waiting is not None:
                start = max(waiting, free_at)
                completions.append((waiting, start + cycle_s))
                free_at = start + cycle_s
                waiting = None
            if at >= free_at:
                completions.append((at, at + cycle_s))
                free_at = at + cycle_s
                continue
        if waiting is not None:
            dropped += 1  # the OLDEST waiting trigger is discarded
        waiting = at
    if waiting is not None:
        start = max(waiting, free_at)
        completions.append((waiting, start + cycle_s))

    # Scoring: for each interviewer turn, was there a cycle that FINISHED before
    # the turn began and was TRIGGERED after the previous interviewer turn ended?
    # The second half is what makes it "fresh" -- a suggestion computed from
    # speech the interviewer has already responded to is not help, it is
    # scrollback.
    interviewer_turns = [t for t in turns if t.speaker == INTERVIEWER]
    covered = stale_only = 0
    previous_end = 0.0
    for turn in interviewer_turns:
        ready = [c for c in completions if c[1] <= turn.start_s]
        fresh = [c for c in ready if c[0] >= previous_end]
        if fresh:
            covered += 1
        elif ready:
            stale_only += 1
        previous_end = turn.end_s

    return SimResult(
        rule=name, cycles=len(completions), dropped=dropped,
        turns=len(interviewer_turns), covered=covered, stale_only=stale_only,
    )


def load(session_dir: Path) -> "list[dict[str, Any]]":
    path = session_dir / "transcript.jsonl"
    if not path.is_file():
        raise SystemExit(f"no transcript at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--cycle-s", type=float, action="append", default=None,
                        help="cycle latency to simulate; repeatable to sweep")
    args = parser.parse_args(argv)

    records = load(args.session_dir)
    latencies = args.cycle_s or [DEFAULT_CYCLE_S]

    rules = [
        ("every-segment (today)", rule_every_segment),
        ("turn-end", rule_turn_end),
        ("turn+3s", rule_turn_plus(3.0)),
        ("turn+5s", rule_turn_plus(5.0)),
        ("min-interval 20s", rule_min_interval(20.0)),
        ("min-interval 45s", rule_min_interval(45.0)),
    ]

    turns = turns_of(records)
    spoken_min = max((t.end_s for t in turns), default=0.0) / 60.0
    print(f"session   : {args.session_dir.name}")
    print(f"turns     : {sum(1 for t in turns if t.speaker == INTERVIEWER)} interviewer, "
          f"{sum(1 for t in turns if t.speaker == INTERVIEWEE)} interviewee, "
          f"{spoken_min:.1f} min\n")

    for cycle_s in latencies:
        print(f"--- cycle latency {cycle_s:.1f}s " + "-" * 44)
        print(f"  {'rule':24} {'cycles':>7} {'dropped':>8} {'cyc/min':>8} "
              f"{'turns covered':>14} {'stale-only':>11}")
        for name, rule in rules:
            result = simulate(records, rule, name=name, cycle_s=cycle_s)
            per_min = result.cycles / spoken_min if spoken_min else 0.0
            print(f"  {name:24} {result.cycles:>7} {result.dropped:>8} {per_min:>8.1f} "
                  f"{result.covered:>6}/{result.turns:<4} {result.coverage:>4.0%} "
                  f"{result.stale_only:>8}")
        print()

    print("turns covered = an interviewer turn that began with a FRESH suggestion")
    print("already finished -- triggered by the interviewee speech just before it.")
    print("stale-only    = something was on screen, but computed from speech the")
    print("interviewer had already responded to. Scrollback, not help.")
    print("\nSIMULATED. Assumes cycle latency is independent of the trigger, that a")
    print("finished cycle is equally useful whenever it lands, and that a suggestion")
    print("shown is a suggestion read. Good for ORDERING the rules; confirm the")
    print("winner with a live rehearsal.")
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
