"""The same interview, the same moments, a different model.

**The question.** The 2026-08-15 live interview produced suggestions the
operator judged too generic even where the critic suppressed correctly. That is
a claim about the GENERATE stage, not the critic -- so the experiment has to
hold everything except the model still, and read the questions themselves.

**The design, and why it is a controlled comparison rather than another
rehearsal.** `rehearse.py` drives a live interviewee agent, so the conversation
changes when the coach changes and no two runs are comparable.
`run_eval_interview.py` replays audio through the real STT, which re-exercises
capture but costs a full interview's wall clock. Neither can answer "would a
better model have said something better HERE".

This replays a RECORDED session and holds fixed:

- the transcript, exactly as it was recorded;
- the brief, survey and hypotheses the live coach was given;
- **the trigger points -- the actual `trigger_seq` values the live coach fired
  on**, read off the recorded suggestion records rather than re-derived.

Reusing the recorded triggers is the load-bearing choice. Re-simulating the
`turn+3s` rule would let trigger-timing differences leak into a comparison that
is supposed to be about wording, and any difference found would then be
unattributable. Here the only variable is the model.

**What is faithfully reconstructed, and what is approximated.** Stated plainly
because the approximations are where a wrong conclusion would come from.

Faithful: the transcript prefix visible at each trigger (every segment with
`seq <= trigger_seq`, in seq order); the trigger segment itself; the brief,
survey and hypotheses; the generate and critique prompts, which are the
shipping `assemble_*` functions, not copies.

Approximated:
- `recently_shown` is rebuilt from the recorded suggestions that were SHOWN
  before this trigger, capped at `DEFAULT_PROMPT_RECENT` like the live path.
  The live `recent` store also carries a dedup window this does not model, so a
  replayed model may repeat something the live one would have suppressed.
  Repeats are therefore counted and reported, never silently dropped.
- `asked` is rebuilt from `coverage-asked` events before the trigger. The live
  path additionally filters it through the CURRENT coverage state, which needs
  the coverage detector to have run; that is a model call per segment and is
  deliberately not paid here.
- `detected_language` is taken from the trigger segment's own value rather than
  from the rolling window vote.

**This measures wording, not latency.** A model that answers in 30 seconds
would fail the live product for reasons this harness cannot see -- it records
elapsed time per cycle so the comparison is at least visible, but nothing here
should be read as evidence a model is fast enough to ship.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import config as app_config  # noqa: E402
from app.coach import corpus as coach_corpus  # noqa: E402
from app.coach import critic as critic_mod  # noqa: E402
from app.coach import generate as generate_mod  # noqa: E402
from app.coach import reasoning_client  # noqa: E402
from app.coach.context import load_brief, load_survey  # noqa: E402
from app.coach.repetition import DEFAULT_PROMPT_RECENT, normalise  # noqa: E402
from app.session.store import SegmentRecord  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

FLAGS = (
    "is_leading",
    "is_hypothetical_or_future_tense",
    "is_pitch",
    "is_opinion_seeking",
    "is_confirmation_only",
    "is_generic",
    "is_off_topic",
)


def _read_session(path: Path) -> "tuple[list[dict], list[dict], list[dict]]":
    """Segments, suggestions and events, each in file order."""
    segments: "list[dict]" = []
    suggestions: "list[dict]" = []
    events: "list[dict]" = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # A truncated final line is a real recorded shape (a crashed
                # writer). Skipping it is what `read_session` does too.
                continue
            kind = record.get("kind")
            if kind == "segment":
                segments.append(record)
            elif kind == "suggestion":
                suggestions.append(record)
            elif kind == "event":
                events.append(record)
    return segments, suggestions, events


def _as_segment_record(row: dict) -> SegmentRecord:
    return SegmentRecord(
        kind="segment",
        seq=row["seq"],
        speaker=row["speaker"],
        text=row.get("text") or "",
        audio_start_s=row.get("audio_start_s") or 0.0,
        audio_end_s=row.get("audio_end_s") or 0.0,
        ready_wall_s=row.get("ready_wall_s") or 0.0,
        latency_s=row.get("latency_s") or 0.0,
        detected_language=row.get("detected_language"),
        engine=row.get("engine") or "replay",
        written_at=row.get("written_at") or "",
    )


def _asked_before(events: "Sequence[dict]", trigger_seq: int) -> "dict[str, Any]":
    """Survey questions marked asked by a `coverage-asked` event that landed
    before this trigger. See the module docstring: the live path also filters
    these through the live coverage state, which is not reconstructed."""
    asked: "dict[str, Any]" = {}
    for event in events:
        if event.get("event") != "coverage-asked":
            continue
        if (event.get("seq") or 0) > trigger_seq:
            continue
        detail = event.get("detail") or {}
        qid = detail.get("question_id")
        seq = detail.get("matched_seq")
        audio_end_s = detail.get("matched_audio_end_s")
        # Built as the real `AskedAt`, not a dict: `prompts.survey_block` reads
        # `.seq` and `.audio_end_s` off it, and a dict here fails at prompt
        # assembly rather than at the seam that produced it.
        if qid and seq is not None and audio_end_s is not None:
            asked[qid] = generate_mod.AskedAt(seq=seq, audio_end_s=audio_end_s)
    return asked


def replay_trigger(
    client: Any,
    *,
    trigger_seq: int,
    segments: "Sequence[dict]",
    events: "Sequence[dict]",
    shown_before: "Sequence[str]",
    brief: Any,
    survey: Any,
    hypotheses: Any,
    model: str,
    generate_max_tokens: int,
    critique_max_tokens: int,
    timeout_s: float,
) -> dict:
    """One cycle: generate, then critique, at one recorded trigger."""
    prefix = [row for row in segments if row["seq"] <= trigger_seq]
    trigger_row = next((row for row in prefix if row["seq"] == trigger_seq), None)
    if trigger_row is None:
        return {"trigger_seq": trigger_seq, "error": "trigger segment not in transcript"}

    transcript = [_as_segment_record(row) for row in prefix]
    trigger = _as_segment_record(trigger_row)

    started = time.monotonic()
    try:
        candidates = generate_mod.run_generate(
            client,
            brief=brief,
            survey=survey,
            asked=_asked_before(events, trigger_seq),
            transcript=transcript,
            trigger=trigger,
            detected_language=trigger_row.get("detected_language") or "en",
            hypotheses=hypotheses,
            recently_shown=tuple(shown_before[-DEFAULT_PROMPT_RECENT:]),
            model=model,
            max_tokens=generate_max_tokens,
            timeout_s=timeout_s,
        )
    except generate_mod.GenerateFailure as exc:
        return {
            "trigger_seq": trigger_seq,
            "trigger_text": trigger.text,
            "error": f"generate: {exc.reason}",
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    # An empty candidate list is a parse failure by construction; a DECLINE
    # arrives as the `no_question` move and must not be confused with one.
    judgments: "list[Any]" = []
    if candidates:
        try:
            judgments = list(
                critic_mod.run_critique(
                    client,
                    candidates=candidates,
                    hypotheses=hypotheses,
                    trigger_text=trigger.text,
                    model=model,
                    max_tokens=critique_max_tokens,
                    timeout_s=timeout_s,
                )
            )
        except critic_mod.CritiqueFailure as exc:
            return {
                "trigger_seq": trigger_seq,
                "trigger_text": trigger.text,
                "error": f"critique: {exc.reason}",
                "elapsed_s": round(time.monotonic() - started, 3),
            }

    by_index = {j.index: j for j in judgments}
    rows = []
    for candidate in candidates:
        judgment = by_index.get(candidate.index)
        row = {
            "text": candidate.text,
            "move": candidate.move,
            "band": getattr(judgment, "band", None),
            "rationale": getattr(judgment, "rationale", None),
        }
        for flag in FLAGS:
            row[flag] = bool(getattr(judgment, flag, False))
        rows.append(row)

    return {
        "trigger_seq": trigger_seq,
        "trigger_text": trigger.text,
        "trigger_words": len(trigger.text.split()),
        "elapsed_s": round(time.monotonic() - started, 3),
        "candidates": rows,
    }


def summarise(results: "Sequence[dict]", label: str) -> dict:
    """The numbers that decide the question, kept separate from the reading."""
    cycles = [r for r in results if "candidates" in r]
    all_rows = [row for r in cycles for row in r["candidates"]]
    non_decline = [row for row in all_rows if row["move"] != "no_question"]
    elapsed = [r["elapsed_s"] for r in cycles if r.get("elapsed_s")]
    flags = {flag: sum(1 for row in all_rows if row[flag]) for flag in FLAGS}
    return {
        "label": label,
        "cycles": len(cycles),
        "errors": sum(1 for r in results if r.get("error")),
        "candidates": len(all_rows),
        "declines": sum(1 for row in all_rows if row["move"] == "no_question"),
        "bands": dict(Counter(row["band"] for row in all_rows)),
        "moves": dict(Counter(row["move"] for row in all_rows)),
        "flags": flags,
        "generic_rate": (flags["is_generic"] / len(non_decline)) if non_decline else None,
        "clear_violation_rate": (
            sum(1 for row in all_rows if row["band"] == "clear_violation") / len(all_rows)
            if all_rows else None
        ),
        "distinct_texts": len({normalise(row["text"]) for row in non_decline}),
        "median_cycle_s": round(statistics.median(elapsed), 2) if elapsed else None,
        "p95_cycle_s": (
            round(sorted(elapsed)[max(0, int(len(elapsed) * 0.95) - 1)], 2) if elapsed else None
        ),
    }


def baseline_from_session(suggestions: "Sequence[dict]") -> dict:
    """What the LIVE run actually produced, in the same shape, so the recorded
    model is a row in the same table rather than a story told separately."""
    rows = []
    for row in suggestions:
        entry = {
            "text": row.get("text"),
            "move": row.get("move"),
            "band": row.get("band"),
            "rationale": row.get("rationale"),
        }
        for flag in FLAGS:
            entry[flag] = bool(row.get(flag))
        rows.append(entry)
    triggers = sorted({row.get("trigger_seq") for row in suggestions if row.get("trigger_seq")})
    return summarise([{"candidates": rows, "trigger_seq": t} for t in triggers[:1]] +
                     [{"candidates": [], "trigger_seq": t} for t in triggers[1:]], "recorded")


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="Path to a session directory holding transcript.jsonl")
    parser.add_argument("--project", required=True, help="Project path whose context/ was used")
    parser.add_argument("--model", required=True, help="Model id to replay with")
    parser.add_argument("--limit", type=int, default=0, help="Replay only the first N triggers")
    parser.add_argument("--out", default="", help="Write per-cycle JSONL here")
    args = parser.parse_args(argv)

    app_config.load_config()
    from app import projects as projects_mod

    paths = projects_mod.resolve(args.project)
    session_dir = Path(args.session)
    segments, suggestions, events = _read_session(session_dir / "transcript.jsonl")

    # These two raise rather than degrade -- correctly, since a replay against
    # a brief the live coach did not have would compare two different setups
    # and call the difference a model effect.
    brief = load_brief(paths.context_dir)
    survey = load_survey(paths.context_dir)
    hypotheses = coach_corpus.load_hypotheses(paths.context_dir, missing_ok=True)

    # The ACTUAL moments the live coach fired -- see the module docstring on
    # why these are read off the record rather than re-simulated.
    triggers = sorted({row["trigger_seq"] for row in suggestions if row.get("trigger_seq")})
    if args.limit:
        triggers = triggers[: args.limit]

    client = reasoning_client.build_client()
    cfg = app_config.load_config()

    print(f"replaying {len(triggers)} recorded triggers through {args.model}", file=sys.stderr)
    results = []
    shown_texts: "list[str]" = []
    for index, trigger_seq in enumerate(triggers, start=1):
        result = replay_trigger(
            client,
            trigger_seq=trigger_seq,
            segments=segments,
            events=events,
            shown_before=shown_texts,
            brief=brief,
            survey=survey,
            hypotheses=hypotheses,
            model=args.model,
            generate_max_tokens=cfg.coach_generate_max_tokens,
            critique_max_tokens=cfg.coach_critique_max_tokens,
            timeout_s=30.0,
        )
        results.append(result)
        # Everything not banded `clear_violation` and not a decline would have
        # reached the panel, so it is what the NEXT cycle must avoid repeating.
        for row in result.get("candidates", []):
            if row["band"] != "clear_violation" and row["move"] != "no_question":
                shown_texts.append(row["text"])
        print(f"  [{index}/{len(triggers)}] seq={trigger_seq} "
              f"{'ERROR ' + result['error'] if result.get('error') else str(len(result.get('candidates', []))) + ' candidates'}",
              file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps(summarise(results, args.model), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    use_utf8_console()
    raise SystemExit(main())
