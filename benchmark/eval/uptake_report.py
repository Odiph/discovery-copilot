"""P1's number for one interview, with the pairs a human can adjudicate.

`.planning/PRODUCT-HYPOTHESES.md` P1: "at least 1 in 5 shown suggestions is
followed, within 90 seconds, by an interviewer segment scoring >=0.6 similarity
to it", across the first three coached interviews. This prints that fraction for
one session, and prints the matches so the fraction can be disbelieved.

**Printing the pairs is not decoration.** The metric cannot tell "the coach
suggested it and the interviewer took it" from "both arrived at the obvious
next question independently" -- so a rate on its own is exactly as trustworthy
as its threshold, which was a guess. A reader who can see the pairs can say
"that one is a coincidence" and recompute; a reader handed 0.24 cannot.

`app/coach/uptake.py` returns sequence numbers and never text, deliberately.
This module is the reporting layer that joins them back to the transcript it
already has open -- the same split `interview_summary`/`review.py` use for
facts and quotes.

Usage:

    .venv/Scripts/python.exe benchmark/eval/uptake_report.py <session-dir>
    .venv/Scripts/python.exe benchmark/eval/uptake_report.py --project demo-claudictive
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from app.coach.uptake import (  # noqa: E402
    DEFAULT_THRESHOLD, DEFAULT_WINDOW_S, Uptake, uptake,
)
from benchmark.console import use_utf8_console  # noqa: E402

# The pre-registered bar, named so a reader is told what they are looking at
# rather than left to remember. PRODUCT-HYPOTHESES.md P1.
PRE_REGISTERED_RATE = 0.20


def read_records(session_dir: Path) -> "list[dict[str, Any]]":
    path = session_dir / "transcript.jsonl"
    if not path.is_file():
        raise SystemExit(f"no transcript at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def report(records: "Sequence[dict[str, Any]]", result: Uptake, *, show_misses: bool) -> None:
    by_seq = {r.get("seq"): r for r in records}

    print(f"shown suggestions : {result.shown}")
    if result.rate is None:
        # An interview where the coach proposed nothing is not one where the
        # interviewer ignored it, and P1 says nothing about it either way.
        print("taken             : n/a -- nothing was shown, so P1 is not measured here")
        return

    print(f"taken             : {result.taken}  ({result.rate:.0%})")
    print(f"window            : {result.window_s:.0f}s   threshold: {result.threshold}")
    verdict = "MEETS" if result.rate >= PRE_REGISTERED_RATE else "BELOW"
    print(f"pre-registered bar: {PRE_REGISTERED_RATE:.0%} -- this interview {verdict} it")
    print()
    print("A LOW rate is evidence against P1. A high rate is NOT evidence for it:")
    print("nothing here distinguishes the coach being taken from the coach and the")
    print("interviewer independently reaching the same obvious question.")
    print()

    for match in result.matches:
        if not match.taken and not show_misses:
            continue
        suggestion = by_seq.get(match.suggestion_seq) or {}
        label = "TAKEN " if match.taken else "missed"
        print(f"[{label} {match.score:.2f} @ {match.at_s / 60:5.1f}m] "
              f"{(suggestion.get('text') or '')[:88]}")
        for seq in match.matched_seqs:
            spoken = by_seq.get(seq) or {}
            print(f"           #{seq}: {(spoken.get('text') or '')[:88]}")
        print()


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", nargs="?", type=Path)
    parser.add_argument("--project", default=None,
                        help="take the NEWEST session under projects/<name>/sessions")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-delay", type=float, default=0.0,
                        help="ignore speech within N seconds of the trigger -- a suggestion "
                             "is written after its cycle completes, so the earliest part of "
                             "the window can predate the interviewer seeing it")
    parser.add_argument("--misses", action="store_true",
                        help="print the suggestions that were NOT taken, with their best score")
    args = parser.parse_args(argv)

    if args.project:
        root = REPO / "projects" / args.project / "sessions"
        candidates = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if not candidates:
            raise SystemExit(f"no sessions under {root}")
        session_dir = candidates[-1]
    elif args.session_dir:
        session_dir = args.session_dir
    else:
        parser.error("give a session directory or --project")

    print(f"session: {session_dir.name}\n")
    records = read_records(session_dir)
    result = uptake(
        records, window_s=args.window, threshold=args.threshold, min_delay_s=args.min_delay,
    )
    report(records, result, show_misses=args.misses)
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
