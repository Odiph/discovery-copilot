"""Scores the mock interview against what was actually played into it.

**This file is a REPORT FOR A HUMAN, not a pass/fail gate.** It has no
thresholds and always exits 0, deliberately: every number below is one sample
of a nondeterministic system, and a threshold over one sample would either be
so loose it never fires or so tight it fires on noise. What it does instead is
put the numbers next to the thing they are supposed to measure, and say out
loud where it cannot measure at all.

Three questions, kept apart because they fail independently:

1. Did the PIPELINE hear it -- per-speaker word similarity against the script,
   and whether the tail arrived at all.
2. Did COVERAGE fire -- which survey questions matched, and for any match on a
   question the script never asked, WHICH utterance triggered it, so a human
   can tell a hallucinated match from a defensible paraphrase match.
3. Did the COACH earn its cost -- cycle counts, and what it actually proposed
   at the four moments the script was built to provoke.

**Speaker attribution is NOT measured here, and cannot be.** Labels come from
which file the frames arrived on, and the two files have zero bleed, so
attribution is correct by construction. An earlier version of this file
"checked" it against each speaker's whole-script vocabulary, which every
segment containing "the" or "game" passes -- a metric a total channel swap
would still have scored perfect. It was removed rather than weakened.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from benchmark.console import use_utf8_console  # noqa: E402

# Imported, never re-spelled. `coverage_report.py`'s own docstring makes the
# argument: a reader that greps for its own spelling of the event name goes
# quietly empty the day the writer is renamed, and "nothing was asked" is a
# perfectly plausible-looking answer. Using the real reader also puts that
# module under test, which it otherwise would not be.
from app.coach.coverage import EVENT_COVERAGE_ASKED, EVENT_COVERAGE_CONSIDERED  # noqa: E402
from app.coach.coverage_report import asked_question_ids  # noqa: E402
from benchmark.eval import critic_gold  # noqa: E402

PROJECT = REPO / "projects" / "demo-claudictive"

# Asked close to verbatim in the script. s3 and s7 are never asked.
EXPECTED_ASKED = {"s1", "s2", "s4", "s5", "s6"}
EXPECTED_NEVER_ASKED = {"s3", "s7"}

# The moments the script was built to provoke, found by a phrase from the turn
# rather than by index, so editing the script cannot silently mis-align them.
PLANTED = [
    ("the skipped survey question (s3)", "let me skip ahead"),
    ("an unfollowed generalisation", "it's a bit grim when you say it out loud"),
    ("a retraction of his own numbers", "that thursday wasn't typical"),
    ("the run of five bad questions", "so the overwolf install was the frustrating part"),
]


def normalise(text: str) -> "list[str]":
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def words_of(texts) -> "list[str]":
    out: "list[str]" = []
    for text in texts:
        out.extend(normalise(text))
    return out


def session_dir() -> Path:
    """The newest session under the demo project.

    Not a date glob. An earlier version hardcoded one day's prefix, which is
    correct for exactly as long as it takes to run this the following morning."""
    candidates = sorted(p for p in (PROJECT / "sessions").iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit("no sessions under demo-claudictive")
    return candidates[-1]


def main() -> int:
    directory = session_dir()
    records = [json.loads(line) for line in
               (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    truth = json.loads((HERE / "claudictive_truth.json").read_text(encoding="utf-8"))

    segments = [r for r in records if r.get("kind") == "segment"]
    events = [r for r in records if r.get("kind") == "event"]
    suggestions = [r for r in records if r.get("kind") == "suggestion"]
    by_seq = {s.get("seq"): s for s in segments}

    print(f"session: {directory.name}\n")
    print("=" * 68)
    print("1. THE PIPELINE")
    print("=" * 68)
    closed = metadata.get("closed_at")
    print(f"closed_at : {closed or 'MISSING -- this session never closed cleanly'}")
    print(f"records   : {len(segments)} segments, {len(events)} events, "
          f"{len(suggestions)} suggestions")

    for speaker in ("interviewer", "interviewee"):
        said = words_of(t["text"] for t in truth if t["speaker"] == speaker)
        heard = words_of(s["text"] for s in segments if s["speaker"] == speaker)
        ratio = SequenceMatcher(None, said, heard).ratio() if said else 0.0
        print(f"  {speaker:12} played {len(said):5} words | heard {len(heard):5} "
              f"| similarity {ratio:.1%}")

    # Did the tail arrive? The planted material sits near the END of the
    # script, so a run cut short by a mis-set --seconds would silently lose
    # precisely what the eval was built for, and every other number here would
    # still look fine.
    last_turn = truth[-1]
    heard_all = " ".join(words_of(s["text"] for s in segments))
    tail_words = normalise(last_turn["text"])
    tail_found = " ".join(tail_words[:6]) in heard_all
    print(f"  tail check : last scripted turn {'FOUND' if tail_found else 'NOT FOUND'} "
          f"-- {'the whole interview played' if tail_found else 'THE RUN WAS CUT SHORT'}")
    print(f"               (looked for: {' '.join(tail_words[:6])!r})")

    # --- 2. coverage ---------------------------------------------------------------
    print()
    print("=" * 68)
    print("2. SURVEY COVERAGE  (the deliberately skipped question)")
    print("=" * 68)
    asked = list(asked_question_ids(records))
    considered = sum(1 for r in events if r.get("event") == EVENT_COVERAGE_CONSIDERED)
    print(f"matched as asked        : {asked or 'NONE'}")
    print(f"considered, unmatched   : {considered}")
    print(f"expected asked          : {sorted(EXPECTED_ASKED)}")
    print(f"  asked but not detected: {sorted(EXPECTED_ASKED - set(asked)) or 'none'}")

    unexpected = sorted(set(asked) & EXPECTED_NEVER_ASKED)
    if not unexpected:
        print(f"  s3/s7 left unmatched  : YES")
    else:
        # NOT branded a false positive. The interviewer probes the substance of
        # s3 shortly after skipping it ("Was that a decision, or just how it
        # worked out?"), and the matcher sees one segment with no context and
        # is told to match paraphrases. A match there may be the detector
        # complying with its spec. Print the trigger so a human can decide.
        print(f"  s3/s7 MATCHED         : {unexpected} -- adjudicate below, do not assume a bug")
        for record in events:
            if record.get("event") != EVENT_COVERAGE_ASKED:
                continue
            detail = record.get("detail") or {}
            if detail.get("question_id") in unexpected:
                trigger = by_seq.get(detail.get("matched_seq"))
                print(f"      {detail.get('question_id')} matched on seq "
                      f"{detail.get('matched_seq')}: "
                      f"{(trigger or {}).get('text', '<segment not found>')!r}")

    print("\n  NOTE: s1/s2/s4/s5/s6 are asked close to VERBATIM in the script, so a")
    print("  detector that regressed to exact-string matching would still score them")
    print("  all. This measures the plumbing more than the paraphrase judgement.")

    # --- 3. the coach --------------------------------------------------------------
    print()
    print("=" * 68)
    print("3. THE COACH")
    print("=" * 68)
    for name, count in sorted(Counter(r.get("event") for r in events).items(),
                              key=lambda kv: -kv[1]):
        print(f"  {name:28} {count}")

    shown = [s for s in suggestions if s.get("shown")]
    print(f"\nsuggestions: {len(suggestions)} generated, {len(shown)} shown")
    print("  move: " + ", ".join(f"{k}={v}" for k, v in
                                 Counter(s.get("move") for s in suggestions).items()))
    print("  band: " + ", ".join(f"{k}={v}" for k, v in
                                 Counter(s.get("band") for s in suggestions).items()))
    # Imported rather than re-typed: a hand-copied list here stays well-formed
    # with a flag missing, so the report would silently under-count forever.
    flags = list(critic_gold.ALL_FLAGS)
    print("  critic flags: " + ", ".join(
        f"{f.replace('is_', '')}={sum(1 for s in suggestions if s.get(f))}" for f in flags))
    print("  ^ THE CRITIC IS NOT UNDER TEST HERE. It bands the coach's OWN candidates,")
    print("    never the interviewer's questions, so the script's five bad questions")
    print("    provoke nothing measurable. A spread of bands is evidence the critic is")
    print("    LIVE; it is not evidence any individual banding is CORRECT, and an")
    print("    all-clean run would be indistinguishable from a critic stuck on 'clean'.")
    print("    Judging the bandings needs a harness that forces known-bad candidates in.")

    targeted = Counter(s.get("target_hypothesis_id") for s in suggestions
                       if s.get("target_hypothesis_id"))
    print(f"\n  aimed at a hypothesis: {sum(targeted.values())} {dict(targeted) or ''}")
    proposals = [s for s in suggestions if s.get("proposed_state")]
    print(f"  state proposals: {len(proposals)} "
          f"{Counter((p.get('target_hypothesis_id'), p.get('proposed_state')) for p in proposals) if proposals else ''}")

    # --- what it did at the planted moments ------------------------------------------
    print()
    print("=" * 68)
    print("4. WHAT IT PROPOSED AT THE PLANTED MOMENTS")
    print("=" * 68)
    for label, phrase in PLANTED:
        turn = next((t for t in truth if phrase in t["text"].lower()), None)
        if turn is None:
            print(f"\n{label}: PHRASE NOT IN SCRIPT ({phrase!r}) -- the script moved")
            continue
        print(f"\n{label}  (played at {turn['start_s'] / 60:.1f} min)")
        window = []
        for suggestion in suggestions:
            trigger = by_seq.get(suggestion.get("trigger_seq"))
            if trigger is None:
                continue
            # Anchored on the TRIGGER segment's audio position, and bounded on
            # both sides. The unbounded version of this selected five sixths of
            # the interview and reported it as a window.
            at = trigger.get("audio_end_s", 0.0)
            if turn["start_s"] - 10 <= at <= turn["end_s"] + 90:
                window.append((at, suggestion))
        if not window:
            print("  nothing suggested within 90s of it")
        for at, suggestion in window[:6]:
            print(f"  [{at / 60:5.1f}m {suggestion.get('move')}/{suggestion.get('band')}] "
                  f"{suggestion.get('text')}")
        if len(window) > 6:
            print(f"  ... and {len(window) - 6} more")

    print()
    print("=" * 68)
    print("5. LATENCY")
    print("=" * 68)
    print(json.dumps(metadata.get("latency_summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
