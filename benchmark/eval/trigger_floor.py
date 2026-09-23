"""What a minimum-turn-words floor would have removed, over a real transcript.

`trigger_sim.py` chose `turn+3s` by asking which rule puts a FRESH suggestion
in front of the interviewer before they have to speak. That question is about
timing and it is settled. This asks the different question the first live
coached interview raised: of the cycles `turn+3s` fires, how many fire on a
turn that has not yet said anything?

The live run's answer was "enough to matter" -- the coach fired on turns whose
text so far was `'Um, I guess'` and `'So, uh, I'`, and the critic then flagged
the resulting questions `is_generic`, correctly, because a question cannot
engage with an utterance carrying no content.

**This replays the real `TurnTrigger`, not a copy of its rule.** The whole
point of the 2026-08-15 finding was that a simulation of a rule can disagree
with the rule; so every floor below is measured by driving the shipping object
with the recorded segments in seq order.

**What it CANNOT tell you.** Whether a removed cycle would have produced a
suggestion the interviewer wanted. It reports how many cycles a floor removes
and what the triggering turns looked like; whether that is a good trade is a
judgement about the interview, and the honest place to make it is by reading
the sample it prints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.coach.trigger import DEFAULT_TURN_DELAY_S, TurnTrigger  # noqa: E402
from app.session.store import SegmentRecord  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

FLOORS = (0, 4, 6, 8, 10, 12, 15, 20)


def _segments(path: Path) -> "list[SegmentRecord]":
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("kind") != "segment":
                continue
            rows.append(record)
    rows.sort(key=lambda r: r["seq"])
    return [
        SegmentRecord(
            kind="segment", seq=r["seq"], speaker=r["speaker"], text=r.get("text") or "",
            audio_start_s=r.get("audio_start_s") or 0.0,
            audio_end_s=r.get("audio_end_s") or 0.0,
            ready_wall_s=r.get("ready_wall_s") or 0.0,
            latency_s=r.get("latency_s") or 0.0,
            detected_language=r.get("detected_language"),
            engine=r.get("engine") or "replay", written_at=r.get("written_at") or "",
        )
        for r in rows
    ]


def fire_at(segments: "Sequence[SegmentRecord]", floor: int, delay_s: float):
    """The seqs the SHIPPING trigger fires on at this floor, and the words the
    turn carried when it did."""
    fired: "list[tuple[int, int, str]]" = []
    turn_words = {"n": 0, "speaker": None}

    def downstream(record: SegmentRecord) -> None:
        fired.append((record.seq, turn_words["n"], record.text))

    trigger = TurnTrigger(downstream=downstream, delay_s=delay_s, min_turn_words=floor)
    for segment in segments:
        # Mirrors the trigger's own accounting so the printed word count is the
        # turn's, not the segment's. Kept OUT of the trigger to avoid growing
        # its public surface for a reporting need.
        if segment.speaker != turn_words["speaker"]:
            turn_words["speaker"] = segment.speaker
            turn_words["n"] = 0
        turn_words["n"] += len((segment.text or "").split())
        trigger.submit(segment)
    return fired


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="A session directory holding transcript.jsonl")
    parser.add_argument("--delay-s", type=float, default=DEFAULT_TURN_DELAY_S)
    parser.add_argument("--show", type=int, default=0,
                        help="Print the turns a floor removes, at this floor")
    args = parser.parse_args(argv)

    segments = _segments(Path(args.session) / "transcript.jsonl")
    interviewee = [s for s in segments if s.speaker == "interviewee"]
    print(f"segments: {len(segments)} ({len(interviewee)} interviewee)")
    print()
    print(f"{'floor':>6}  {'cycles':>7}  {'removed':>8}  {'median turn words at fire':>26}")

    baseline = fire_at(segments, 0, args.delay_s)
    for floor in FLOORS:
        fired = fire_at(segments, floor, args.delay_s)
        words = sorted(w for _seq, w, _t in fired)
        median = words[len(words) // 2] if words else 0
        removed = len(baseline) - len(fired)
        print(f"{floor:>6}  {len(fired):>7}  {removed:>8}  {median:>26}")

    if args.show:
        # A floor does TWO different things and reporting them as one number
        # is wrong. It removes a turn's cycle entirely when the whole turn
        # stays under the floor -- and it DELAYS the cycle to a later segment
        # of the same turn when the turn goes on to say something, because
        # `_fired` stays False until the floor is cleared. The second is the
        # behaviour worth having, and counting it as a removal would badly
        # understate what the floor does and overstate what it costs.
        turn_of = _turn_index(segments)
        after = fire_at(segments, args.show, args.delay_s)
        after_by_turn = {turn_of[seq]: (seq, words, text) for seq, words, text in after}
        print()
        print(f"--- what a floor of {args.show} changes ---")
        dropped = moved = 0
        for seq, words, text in baseline:
            landing = after_by_turn.get(turn_of[seq])
            if landing is None:
                dropped += 1
                print(f"  DROPPED  seq={seq:>4} turn_words={words:>3}  {text!r}")
            elif landing[0] != seq:
                moved += 1
                print(f"  DELAYED  seq={seq:>4} ({words}w) -> seq={landing[0]} ({landing[1]}w)"
                      f"  {text!r} -> {landing[2]!r}")
        print()
        print(f"  {dropped} turn(s) lose their cycle · {moved} fire later in the same turn")
    return 0


def _turn_index(segments: "Sequence[SegmentRecord]") -> "dict[int, int]":
    """Which turn each segment belongs to. A turn is a speaker run -- the same
    definition `TurnTrigger` uses, so the two cannot disagree about what a
    turn is."""
    index: "dict[int, int]" = {}
    turn = -1
    speaker = None
    for segment in segments:
        if segment.speaker != speaker:
            speaker = segment.speaker
            turn += 1
        index[segment.seq] = turn
    return index


if __name__ == "__main__":
    use_utf8_console()
    raise SystemExit(main())
