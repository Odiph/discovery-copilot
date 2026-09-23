"""Forces known-bad and known-good questions through the real critique call
and scores what comes back against `critic_gold.py`'s labels.

This is the harness `analyse_eval.py` says is missing:

    A spread of bands is evidence the critic is LIVE; it is not evidence any
    individual banding is CORRECT [...] Judging the bandings needs a harness
    that forces known-bad candidates in.

**It drives the real path.** `app.coach.critic.run_critique` is called
directly -- the same assembly, the same forced tool call, the same parse and
the same two-tier failure discipline the live pipeline uses. Nothing about
the critique is reimplemented here, so a change to the rubric or the schema
shows up in these numbers rather than being shadowed by a parallel copy.

**Two outcomes are singled out, because they are not symmetric.**

- A GOOD question banded `clear_violation` is DROPPED by `rank_candidates`
  and never reaches the interviewer. Nothing in the product surfaces this,
  and no number produced so far could have detected it: the measured run
  banded 26% clear_violation with no labels to check them against.
- A BAD question banded `clean` is handed to the interviewer as advice.

Everything else in the report is context for those two rows.

**Run it against any endpoint.** The client comes from
`reasoning_client.build_client()`, so an OpenAI-compatible runtime is one env
var away and costs nothing:

    $env:COACH_BASE_URL="http://localhost:11434/v1"
    $env:COACH_MODEL="qwen2.5:7b"
    .venv/Scripts/python.exe benchmark/eval/judge_critic.py

**MEASURED: `qwen2.5:3b` CANNOT run this, and the reason is not the one the
previous session expected.** Session 8 ruled local models out for the LIVE
coach on two gates -- tool fidelity on the generate stage's nested schema,
and two calls per cycle against an 8.5s gap -- and recorded that local was
"still right for the offline judge", on the reasoning that the judge has no
latency budget and uses the simpler critique schema. The latency half of that
holds. The fidelity half does not, and it fails for a different reason:

    schema        prompt        result
    simple        334 chars     well-formed judgments
    real          334 chars     well-formed judgments
    simple      20,989 chars    echoed the INPUT back under a 'candidates' key
    real        20,989 chars    echoed the INPUT back under a 'candidates' key

The schema is innocent. **Prompt LENGTH alone collapses it**, which is a
different failure from the generate stage's, where session 8 found schema and
prompt only failed in combination. Sweeping the length (truncating the real
rubric, so content varies too -- a characterisation, not a proof):

    1k / 2k / 4k / 6k / 8k chars   -> well-formed, 4 of 4 judgments
    12k chars                      -> no tool block at all
    16k chars                      -> well-formed, but only 3 of 4 judgments
    20,989 chars (the real rubric) -> wrong shape entirely

The 16k row is the one to be afraid of. A short judgment list PARSES: the
missing candidate simply has no matching index, and `rank_candidates` drops
it as unvetted. So a model degrading this way does not error -- it silently
withholds verdicts, and both this harness and the live pipeline record a
successful cycle. `judge_all` below records a failed batch as `None` for that
exact reason, and the report prints UNJUDGED rather than counting it clean.

The rubric grew past 4,096 tokens deliberately in session 8, to clear Haiku
4.5's minimum cacheable prefix. That was correct for cost and is precisely
what puts it out of a 3B model's reach. Do not shrink it to fit a local
model: a condensed rubric would make this harness measure a critic that does
not exist in production, which is worse than not measuring at all. Use a
larger local model, or a cheap hosted one.

**One run is one sample of a nondeterministic system.** `--repeat N` re-runs
the whole set and reports per-case band stability. A critic that returns a
different band for the same sentence across runs is a distinct defect from
one that is consistently wrong, and only the repeat flag can tell them apart.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from app import config as config_mod  # noqa: E402
from app.coach import corpus, critic, reasoning_client  # noqa: E402
from app.coach.generate import Candidate  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402
from benchmark.eval.critic_gold import (  # noqa: E402
    ALL_FLAGS,
    ALL_LANGUAGES,
    DEFAULT_LANGUAGES,
    GOLD_CASES,
    GOLD_SURVEY,
    TIER_BAD,
    TIER_CONTESTED,
    TIER_GOOD,
    GoldCase,
    cases_by_tier,
    scored_cases,
)

PROJECT = REPO / "projects" / "demo-claudictive"

# Mirrors the live pipeline's own critique bound rather than inventing a new
# one. Nothing here is latency-sensitive, but a call that hangs forever in a
# report is still a bad call.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_TOKENS = 2048
# Production sends the critic whatever ONE generate cycle proposed -- a
# handful, not the whole corpus. Batching all sixteen cases into a single call
# would measure the critic on a workload it never sees, and would let one case
# prime the judgement of the next fifteen.
DEFAULT_BATCH_SIZE = 4


def interleaved(cases: "Sequence[GoldCase]") -> "list[GoldCase]":
    """Round-robins the tiers so no batch is all-bad or all-good.

    The five bad cases are consecutive in the script and would otherwise stay
    consecutive here. A model shown five violations in a row has been told
    something about the sixth, and a run ordered that way measures priming as
    much as judgement."""
    buckets = [
        [c for c in cases if c.tier == TIER_BAD],
        [c for c in cases if c.tier == TIER_GOOD],
        [c for c in cases if c.tier == TIER_CONTESTED],
    ]
    out: "list[GoldCase]" = []
    while any(buckets):
        for bucket in buckets:
            if bucket:
                out.append(bucket.pop(0))
    return out


def to_candidate(case: GoldCase, index: int) -> Candidate:
    """The gold case as the critic will see it -- indistinguishable from a
    candidate the generate stage produced. `index` is per-BATCH, matching the
    live path, where it is the emission order within one cycle."""
    return Candidate(
        index=index,
        text=case.text,
        move=case.move,
        target_hypothesis_id=case.target_hypothesis_id,
    )


def judge_all(
    client: Any,
    cases: "Sequence[GoldCase]",
    *,
    hypotheses: Sequence[Any],
    model: str,
    batch_size: int,
    max_tokens: int,
    timeout_s: float,
) -> "dict[str, Optional[critic.Judgment]]":
    """Returns a judgment per case id. A case whose whole BATCH failed maps to
    `None` -- recorded as unjudged rather than silently dropped, because "the
    critic could not be reached" and "the critic said clean" must never look
    the same in the report."""
    results: "dict[str, Optional[critic.Judgment]]" = {}
    # Batched BY TRIGGER, not by position (2026-08-15). One request carries one
    # trigger utterance, because `is_generic` asks whether a candidate engages
    # with what the interviewee just said -- so cases responding to different
    # utterances cannot share a request without making that flag meaningless.
    # Cases with no trigger group together and are judged with an explicitly
    # empty one, which the prompt handles deterministically (set is_generic
    # false, never guess) rather than inviting the model to invent a verdict.
    # Grouping rather than one-call-per-case keeps the request shape, and the
    # cost, close to the live path's.
    for _trigger, group in _grouped_by_trigger(cases):
        for start in range(0, len(group), batch_size):
            batch = list(group[start:start + batch_size])
            candidates = [to_candidate(case, i) for i, case in enumerate(batch)]
            _judge_one_batch(
                client, batch, candidates, results,
                hypotheses=hypotheses, trigger_text=_trigger, model=model,
                max_tokens=max_tokens, timeout_s=timeout_s,
            )
    return results


def _grouped_by_trigger(
    cases: "Sequence[GoldCase]",
) -> "list[tuple[str, list[GoldCase]]]":
    """Stable grouping: first-seen trigger order, and each group keeps the
    cases' own order. Deterministic so a re-run compares against the last one
    rather than reshuffling which cases shared a request."""
    order: "list[str]" = []
    groups: "dict[str, list[GoldCase]]" = {}
    for case in cases:
        trigger = getattr(case, "trigger", "") or ""
        if trigger not in groups:
            groups[trigger] = []
            order.append(trigger)
        groups[trigger].append(case)
    return [(trigger, groups[trigger]) for trigger in order]


def _judge_one_batch(
    client: Any,
    batch: "Sequence[GoldCase]",
    candidates: "Sequence[Candidate]",
    results: "dict[str, Optional[critic.Judgment]]",
    *,
    hypotheses: Sequence[Any],
    trigger_text: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
) -> None:
        try:
            judgments = critic.run_critique(
                client,
                candidates=candidates,
                hypotheses=hypotheses,
                # The gold set's own plan (2026-08-19). `is_off_topic` cannot
                # be judged without one, and this is the same survey every
                # case in the set is labelled against -- see `GOLD_SURVEY`'s
                # own comment for why it is one shared plan and not a
                # per-case one.
                survey=GOLD_SURVEY,
                trigger_text=trigger_text,
                model=model,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        except critic.CritiqueFailure as exc:
            print(f"  batch of {len(batch)}: CRITIQUE FAILED "
                  f"({exc.reason}, stop_reason={exc.stop_reason}) -- "
                  f"{len(batch)} cases unjudged")
            for case in batch:
                results[case.case_id] = None
            return
        by_index = {j.index: j for j in judgments}
        for i, case in enumerate(batch):
            results[case.case_id] = by_index.get(i)


def _flag_report(cases: "Sequence[GoldCase]", judged: "dict[str, Optional[critic.Judgment]]") -> None:
    """Per-flag counts over the cases where that flag is actually constrained.

    A flag is constrained on a BAD case only if the case names it in
    `must_flag`; a bad case that is also leading is not a false positive for
    leading just because the label happened to name a different rule. On a
    GOOD case every flag is constrained to False. Cases where a flag is
    unconstrained are excluded from its row entirely rather than counted as
    negatives, which would inflate every rate with free correct answers."""
    print("\nPER-FLAG (only cases where the flag is constrained)")
    print(f"  {'flag':38} {'caught':>6} {'missed':>6} {'over-flagged':>13}")
    for flag in ALL_FLAGS:
        caught = missed = over = 0
        for case in cases:
            judgment = judged.get(case.case_id)
            if judgment is None:
                continue
            said = bool(getattr(judgment, flag))
            if flag in case.must_flag:
                caught += said
                missed += not said
            elif case.no_other_flags:
                over += said
        print(f"  {flag:38} {caught:>6} {missed:>6} {over:>13}")


def report(
    judged: "dict[str, Optional[critic.Judgment]]",
    *,
    band_history: "dict[str, Counter]",
    runs: int,
    cases: "Optional[Sequence[GoldCase]]" = None,
) -> int:
    """Prints the report and returns the number of DANGEROUS outcomes -- the
    two asymmetric ones named in the module docstring. Everything else is
    printed but does not count."""
    # The cases actually RUN, which since 2026-08-14 is English alone by
    # default. Every denominator below is computed from this rather than from
    # `GOLD_CASES`, because a coverage line reading "13 of 32" for a run that
    # deliberately judged 13 would report a decision as a failure -- and the
    # section directly under it is the one that exists to stop a partial run
    # being read as a whole one.
    universe = tuple(cases) if cases is not None else GOLD_CASES
    in_scope = {case.case_id for case in universe}
    scored = tuple(case for case in scored_cases() if case.case_id in in_scope)
    tier_of = lambda tier: tuple(c for c in universe if c.tier == tier)  # noqa: E731

    excluded = [c for c in GOLD_CASES if c.case_id not in in_scope]
    if excluded:
        # Named, never silent. An excluded register is a decision; an excluded
        # register nobody mentions is indistinguishable from one nobody built.
        languages = sorted({c.language for c in excluded})
        print(f"\nNOT JUDGED: {len(excluded)} case(s) in {languages} — excluded by "
              f"selection, not by failure. Pass --all-languages to include them.")

    # Coverage FIRST, and deliberately above everything else. A measured run
    # judged 10 of 17 cases -- seven calls failed -- and every table below it
    # still rendered as a complete-looking report with a clean bill of health.
    # The numbers were not wrong; they were computed over whatever survived,
    # which is exactly how a partial run comes to be read as a whole one. A
    # reader who sees this line first cannot make that mistake.
    judged_count = sum(1 for case in universe if judged.get(case.case_id) is not None)
    total = len(universe)
    print()
    print("=" * 72)
    print(f"0. COVERAGE -- {judged_count} of {total} cases judged")
    print("=" * 72)
    if judged_count < total:
        print(f"  {total - judged_count} case(s) were NEVER JUDGED: the critique call failed.")
        print("  Every number below is computed over the survivors only. This is a")
        print("  PARTIAL run and must not be quoted as a score for the whole set.")
    else:
        print("  every case was judged")

    print()
    print("=" * 72)
    print("1. THE TWO OUTCOMES THAT MATTER")
    print("=" * 72)

    dropped_good: "list[GoldCase]" = []
    passed_bad: "list[GoldCase]" = []
    unjudged: "list[GoldCase]" = []
    for case in scored:
        judgment = judged.get(case.case_id)
        if judgment is None:
            unjudged.append(case)
            continue
        if case.tier == TIER_GOOD and judgment.band == critic.BAND_CLEAR_VIOLATION:
            dropped_good.append(case)
        if case.tier == TIER_BAD and judgment.band == critic.BAND_CLEAN:
            passed_bad.append(case)

    print(f"\nGOOD questions banded clear_violation (SILENTLY DROPPED): "
          f"{len(dropped_good)} of {len(tier_of(TIER_GOOD))}")
    for case in dropped_good:
        judgment = judged[case.case_id]
        flags = [f for f in ALL_FLAGS if getattr(judgment, f)]
        print(f"  {case.case_id:28} {case.text[:58]!r}")
        print(f"  {'':28} flagged {flags or ['(none -- band with no flag)']}")
        print(f"  {'':28} rationale: {judgment.rationale[:90]}")

    print(f"\nBAD questions banded clean (SHOWN AS ADVICE): "
          f"{len(passed_bad)} of {len(tier_of(TIER_BAD))}")
    for case in passed_bad:
        judgment = judged[case.case_id]
        print(f"  {case.case_id:28} {case.text[:58]!r}")
        print(f"  {'':28} should have flagged {sorted(case.must_flag)}")
        print(f"  {'':28} rationale: {judgment.rationale[:90]}")

    if unjudged:
        print(f"\nUNJUDGED (the call failed -- NOT a pass): {[c.case_id for c in unjudged]}")

    print()
    print("=" * 72)
    print("2. FLAG ACCURACY")
    print("=" * 72)
    _flag_report(scored, judged)

    # --- the paired comparison, which is the reason the translations exist ------
    print("\nBY LANGUAGE  (same questions, carried across with the same labels --")
    print("so a gap between these rows is the LANGUAGE, not the questions)")
    print(f"  {'lang':6} {'judged':>8} {'caught':>10} {'over-flagged':>13} {'dangerous':>10}")
    for language in ALL_LANGUAGES:
        cases = [c for c in scored if c.language == language]
        if not cases:
            continue
        seen = [c for c in cases if judged.get(c.case_id)]
        bad = [c for c in cases if c.tier == TIER_BAD and judged.get(c.case_id)]
        good = [c for c in cases if c.tier == TIER_GOOD and judged.get(c.case_id)]
        caught = sum(1 for c in bad
                     if all(getattr(judged[c.case_id], f) for f in c.must_flag))
        over = sum(1 for c in good
                   if any(getattr(judged[c.case_id], f) for f in ALL_FLAGS))
        danger = sum(1 for c in good
                     if judged[c.case_id].band == critic.BAND_CLEAR_VIOLATION)
        danger += sum(1 for c in bad if judged[c.case_id].band == critic.BAND_CLEAN)
        print(f"  {language:6} {len(seen):>4}/{len(cases):<3} {caught:>7}/{len(bad):<2} "
              f"{over:>10}/{len(good):<2} {danger:>10}")
    print("  English is the reference row. A model that scores well there and badly")
    print("  below is not multilingual, whatever its marketing says -- and every")
    print("  quality number this project produced before 2026-08-14 was the top row")
    print("  alone, on a product whose defining constraint is the rows under it.")

    print("\nBANDS")
    for tier in (TIER_BAD, TIER_GOOD):
        counts = Counter(
            judged[c.case_id].band if judged.get(c.case_id) else "UNJUDGED"
            for c in tier_of(tier)
        )
        print(f"  {tier:10} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print()
    print("=" * 72)
    print("3. CONTESTED -- reported, deliberately NOT scored")
    print("=" * 72)
    print("These violate the letter of the rubric while being defensible interviewing.")
    print("A clear_violation here is a decision about the RUBRIC, not a proven bug.\n")
    for case in tier_of(TIER_CONTESTED):
        judgment = judged.get(case.case_id)
        if judgment is None:
            print(f"  {case.case_id:28} UNJUDGED")
            continue
        flags = [f.replace("is_", "") for f in ALL_FLAGS if getattr(judgment, f)]
        print(f"  {case.case_id:28} {judgment.band:16} {flags or '-'}")
        print(f"  {'':28} {case.text[:66]!r}")

    if runs > 1:
        print()
        print("=" * 72)
        print(f"4. STABILITY ACROSS {runs} RUNS")
        print("=" * 72)
        print("A case that changes band between identical runs is a different defect")
        print("from one that is consistently wrong. Only these rows separate them.\n")
        unstable = {cid: counts for cid, counts in band_history.items() if len(counts) > 1}
        if not unstable:
            print("  every case returned the same band on every run")
        for cid, counts in sorted(unstable.items()):
            print(f"  {cid:28} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print()
    print("=" * 72)
    print("WHAT THIS DOES NOT MEASURE")
    print("=" * 72)
    print("  These questions were not written by the coach and are judged outside the")
    print("  conversation they came from. Scoring well here does not establish that the")
    print("  critic bands the coach's OWN in-flight candidates correctly -- assumption")
    print("  A7 in app/coach/critic.py is narrowed by this harness, not closed.")

    return len(dropped_good) + len(passed_bad)


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the whole set N times and report band stability")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"candidates per critique call (default {DEFAULT_BATCH_SIZE}, "
                             "matching one generate cycle's shape)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--model", default=None,
                        help="overrides COACH_MODEL / the default for this run only")
    parser.add_argument("--dry-run", action="store_true",
                        help="assemble and describe the requests without calling anything")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any dangerous outcome occurred (for CI)")
    parser.add_argument("--language", action="append", choices=ALL_LANGUAGES, default=None,
                        help="which registers to judge; repeatable. Defaults to English "
                             "alone, by the 2026-08-14 decision that interviews start "
                             "100%% English. Pass --language he --language mixed to run "
                             "the translated cases when that changes.")
    parser.add_argument("--all-languages", action="store_true",
                        help="judge every register, including the retained translations")
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    # `.env`, explicitly, and by ABSOLUTE path. This is a standalone entry point
    # -- nothing here goes through `load_config()`, which is what loads the file
    # for the app -- so without this the harness cannot see a key the app can,
    # and the first thing it prints is that no credential exists. That is
    # exactly what it did the first time it was pointed at a freshly configured
    # provider, which is the worst possible moment to be wrong about a key.
    # `app/tests/test_tracer_capture_to_disk.py:183` does the same for the same
    # reason. The absolute path matters because the loader's default is relative
    # to the CWD, and this script is routinely run from elsewhere.
    config_mod._load_dotenv_once(REPO / ".env")

    languages = (ALL_LANGUAGES if args.all_languages
                 else tuple(args.language or DEFAULT_LANGUAGES))
    selected = tuple(case for case in GOLD_CASES if case.language in languages)
    if not selected:
        raise SystemExit(f"no gold cases in {languages}")

    hypotheses = corpus.load_hypotheses(PROJECT / "context", missing_ok=False)
    ordered = interleaved(selected)

    # Every case that names a hypothesis must find it, or `is_confirmation_only`
    # is judged against a falsifier that is not there. Checked BEFORE any call,
    # because the failure is silent otherwise: the critic would simply judge a
    # targetless candidate and the report would look complete.
    known = {h.id for h in hypotheses}
    missing = sorted({c.target_hypothesis_id for c in GOLD_CASES
                      if c.target_hypothesis_id and c.target_hypothesis_id not in known})
    if missing:
        raise SystemExit(f"gold cases target hypotheses that do not exist: {missing}")

    model = args.model or reasoning_client.resolve_model()
    print(f"cases : {len(ordered)} ({sum(1 for c in selected if c.tier == TIER_BAD)} bad, "
          f"{sum(1 for c in selected if c.tier == TIER_GOOD)} good, "
          f"{sum(1 for c in selected if c.tier == TIER_CONTESTED)} contested)  langs={list(languages)}")
    print(f"model : {model}")
    print(f"batch : {args.batch_size} candidates per call, "
          f"{-(-len(ordered) // args.batch_size)} calls per run, {args.repeat} run(s)")

    if args.dry_run:
        print("\n--dry-run: assembling requests, calling nothing\n")
        for start in range(0, len(ordered), args.batch_size):
            batch = ordered[start:start + args.batch_size]
            request = critic.assemble_critique_request(
                candidates=[to_candidate(c, i) for i, c in enumerate(batch)],
                hypotheses=hypotheses,
                # Same pre-existing break as `cost_model.py`: both of these
                # are required keywords this dry-run path had stopped passing,
                # so it has raised TypeError rather than printing sizes.
                survey=(),
                trigger_text="the interviewee just said something specific",
                model=model,
                max_tokens=args.max_tokens,
            )
            system_chars = sum(len(block["text"]) for block in request["system"])
            user_chars = len(request["messages"][0]["content"])
            print(f"  batch {start // args.batch_size}: {len(batch)} candidates | "
                  f"system {system_chars} chars | user {user_chars} chars | "
                  f"cached prefix: {'yes' if request['system'][0].get('cache_control') else 'NO'}")
            print(f"    {[c.case_id for c in batch]}")
        return 0

    try:
        client = reasoning_client.build_client()
    except reasoning_client.CoachConfigError as exc:
        print(f"\nno client: {exc}")
        return 2

    band_history: "dict[str, Counter]" = defaultdict(Counter)
    judged: "dict[str, Optional[critic.Judgment]]" = {}
    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\nrun {run + 1}/{args.repeat}")
        judged = judge_all(
            client, ordered,
            hypotheses=hypotheses,
            model=model,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
        )
        for case_id, judgment in judged.items():
            band_history[case_id][judgment.band if judgment else "UNJUDGED"] += 1

    # The report describes the LAST run; `band_history` carries all of them,
    # which is why the stability section exists rather than an averaged band
    # that would describe no run that actually happened.
    dangerous = report(judged, band_history=band_history, runs=args.repeat, cases=selected)

    print(f"\ndangerous outcomes: {dangerous}")
    if args.strict and dangerous:
        return 1
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
