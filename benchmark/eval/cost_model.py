"""What one interview costs the coach, replayed from a real session.

**Estimated, not measured.** Token counts here are `len(text)/4`. The real
figure needs `/v1/messages/count_tokens`, which needs API credit -- the same
thing this script exists to help budget. Treat every number as +/-15% and the
COMPARISON between layouts as the reliable part: both sides use the identical
estimator, so the ratio survives the estimator being wrong.

It replays the actual assembled requests (`assemble_generate_request`,
`assemble_critique_request`) against a real session's transcript, cycle by
cycle, and applies the cache rules that decide the bill:

- a cached prefix is only READ when its bytes match the previous cycle's;
- a prefix shorter than the model's minimum never caches at all, silently;
- a write costs 1.25x base, a read 0.1x.

The old single-block layout is replayed alongside the current split one, so
the saving is derived rather than asserted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from app.coach import prompts, repetition  # noqa: E402
from app.coach.context import load_brief, load_survey  # noqa: E402
from app.coach.corpus import load_hypotheses  # noqa: E402
from app.coach.critic import assemble_critique_request  # noqa: E402
from app.coach.generate import Candidate, assemble_generate_request  # noqa: E402
from app.session.store import SegmentRecord  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

# Haiku 4.5 list prices, per million tokens.
IN_PER_M = 1.00
OUT_PER_M = 5.00
CACHE_WRITE_MULT = 1.25   # 5-minute TTL
CACHE_READ_MULT = 0.10
# Haiku 4.5's minimum cacheable prefix. A prefix below this silently does not
# cache -- no error, `cache_creation_input_tokens: 0`.
CACHE_MIN_TOKENS = 4096

CHARS_PER_TOKEN = 4.0


def tok(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def blocks_tokens(blocks) -> float:
    return sum(tok(b["text"]) for b in blocks)


def price(session_dir: Path, project: Path, *, chunk: int, legacy: bool = False) -> dict:
    """`legacy=True` reproduces the pre-fix layout: ONE transcript block with
    the breakpoint at its end. That prefix clears the floor easily but grows
    every cycle, so it is written every time and read never -- the 1.25x
    premium with none of the discount."""
    records = [json.loads(l) for l in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    segments = [SegmentRecord(**r) for r in records if r.get("kind") == "segment"]
    suggestions = [r for r in records if r.get("kind") == "suggestion"]

    ctx = project / "context"
    brief = load_brief(ctx)
    survey = load_survey(ctx)
    hyps = load_hypotheses(ctx, missing_ok=True)

    by_trigger: "dict[int, list]" = {}
    for s in suggestions:
        by_trigger.setdefault(s["trigger_seq"], []).append(s)

    in_full = in_write = in_read = out_tokens = 0.0
    critique_in = critique_out = gen_tail = 0.0
    critique_prefix_seen = False
    previous_prefix: "tuple[str, ...] | None" = None
    reads = writes = uncacheable = 0

    for trigger_seq in sorted(by_trigger):
        upto = [s for s in segments if s.seq <= trigger_seq]
        if not upto:
            continue
        trigger = upto[-1]
        cycle_suggestions = by_trigger[trigger_seq]
        recent = [s["text"] for s in cycle_suggestions][:repetition.DEFAULT_PROMPT_RECENT]

        request = assemble_generate_request(
            brief=brief, survey=survey, asked={}, transcript=upto, trigger=trigger,
            detected_language="en", hypotheses=hyps, recently_shown=recent,
            transcript_cache_chunk=chunk, model="m", max_tokens=2000,
        )
        if legacy:
            cached = [
                {"text": prompts.static_block(brief=brief, hypotheses=hyps)},
                {"text": prompts.survey_block(survey, {})},
                {"text": prompts.transcript_block(upto)},
            ]
            uncached = []
        else:
            cached = [b for b in request["system"] if "cache_control" in b]
            uncached = [b for b in request["system"] if "cache_control" not in b]

        # `tools` renders BEFORE `system`, so it sits inside every cached
        # prefix -- pricing it separately at full rate understates caching on
        # both layouts equally, but it is still wrong.
        tools_text = json.dumps(request["tools"])
        prefix_tokens = blocks_tokens(cached) + tok(tools_text)
        tail_tokens = blocks_tokens(uncached)
        user_tokens = tok(request["messages"][-1]["content"])
        tools_tokens = 0.0

        prefix_key = (tools_text,) + tuple(b["text"] for b in cached)
        if prefix_tokens < CACHE_MIN_TOKENS:
            in_full += prefix_tokens          # below the floor: never caches
            uncacheable += 1
        elif prefix_key == previous_prefix:
            in_read += prefix_tokens
            reads += 1
        else:
            in_write += prefix_tokens
            writes += 1
        previous_prefix = prefix_key
        in_full += tail_tokens + user_tokens + tools_tokens

        # Output: the candidate text actually produced, plus tool-call JSON.
        out_tokens += sum(tok(s["text"]) for s in cycle_suggestions) + 30 * len(cycle_suggestions)

        # --- critique, same cycle -----------------------------------------------
        candidates = [
            Candidate(index=i, text=s["text"], move=s["move"])
            for i, s in enumerate(cycle_suggestions)
        ]
        crit = assemble_critique_request(
            candidates=candidates,
            hypotheses=hyps,
            # The session's REAL survey, and a placeholder trigger. This
            # script measures TOKEN COUNTS, never judgement quality, so the
            # trigger's wording is irrelevant while its presence is not --
            # both fields are required keywords this call had silently
            # stopped satisfying, and it has raised TypeError since
            # `trigger_text` landed. The survey is passed for real because a
            # system block assembled without it would under-report the very
            # cached prefix this script exists to size.
            survey=survey,
            trigger_text="the interviewee just said something specific",
            model="m",
            max_tokens=1000,
        )
        # The critique system block is byte-identical for the whole session,
        # so it is written once and read thereafter -- but only if it clears
        # the floor. Below the floor it silently costs full price every call,
        # which is what it did before the rubric was enlarged.
        crit_prefix = blocks_tokens(crit["system"]) + tok(json.dumps(crit["tools"]))
        crit_volatile = tok(crit["messages"][-1]["content"])
        # `legacy` also reverts the critique to its pre-fix form (a bare
        # string system block, no breakpoint), so the OLD row is genuinely the
        # old cost and not the old generate layout wearing the new critique.
        crit_cached = (not legacy) and all("cache_control" in b for b in crit["system"])
        if not crit_cached or crit_prefix < CACHE_MIN_TOKENS:
            in_full += crit_prefix
            critique_in += crit_prefix                      # billed at 1.0x
        elif critique_prefix_seen:
            in_read += crit_prefix
            critique_in += crit_prefix * CACHE_READ_MULT
        else:
            in_write += crit_prefix
            critique_in += crit_prefix * CACHE_WRITE_MULT
        critique_prefix_seen = True
        in_full += crit_volatile
        critique_in += crit_volatile
        crit_out = sum(tok(s.get("rationale") or "") + 40 for s in cycle_suggestions)
        out_tokens += crit_out
        critique_out += crit_out
        gen_tail += tail_tokens + user_tokens

    cost = (
        in_full * IN_PER_M
        + in_write * IN_PER_M * CACHE_WRITE_MULT
        + in_read * IN_PER_M * CACHE_READ_MULT
        + out_tokens * OUT_PER_M
    ) / 1_000_000
    return {
        "cost": cost, "cycles": len(by_trigger),
        "in_full": in_full, "in_write": in_write, "in_read": in_read, "out": out_tokens,
        "reads": reads, "writes": writes, "below_floor": uncacheable,
        "critique_in": critique_in, "critique_out": critique_out, "gen_tail": gen_tail,
    }


# --- local models -------------------------------------------------------------------
#
# For a local model the bill is not dollars, it is THROUGHPUT and CONTENTION.
# Two facts from this project decide it:
#
#   1. The interview machine is already busy. During a run it holds two Soniox
#      WebSocket sessions, a PortAudio capture stream per speaker, the resampler,
#      the coach's own worker thread and the session writer. `run_eval_interview.py`
#      paces its feed by the wall clock precisely because starving that pipeline
#      invalidates every latency number. Local inference competes for the same CPU.
#   2. There is no cross-request prompt cache to lean on. The 62% saving measured
#      above is almost entirely cache READS; a local runtime without prefix caching
#      re-prefills the whole prompt every cycle. vLLM's automatic prefix caching
#      recovers most of it, llama.cpp's prompt reuse recovers some, Ollama's
#      behaviour depends on the version -- so the honest range is wide and both
#      ends are reported below.
INTERVIEW_MINUTES = 26.0


def local_estimate(r: dict, *, no_cache_input_tokens: float) -> None:
    cycles = r["cycles"]
    decode = r["out"]
    # Every cycle prefills its whole prompt when nothing is cached.
    prefill_uncached = no_cache_input_tokens
    # With prefix caching, only the volatile part is re-prefilled: the tail, the
    # user message, and the critique's candidate list.
    prefill_cached = r["in_full"] + r["in_write"]

    per_cycle_uncached = prefill_uncached / cycles
    per_cycle_cached = prefill_cached / cycles
    per_cycle_decode = decode / cycles
    seconds_between = INTERVIEW_MINUTES * 60 / cycles

    print("\n" + "=" * 60)
    print("LOCAL MODEL FEASIBILITY (throughput, not dollars)")
    print("=" * 60)
    print(f"  cycles                     {cycles} over {INTERVIEW_MINUTES:.0f} min "
          f"-> one every {seconds_between:.1f}s")
    print(f"  prefill/cycle, NO caching  {per_cycle_uncached:8,.0f} tok")
    print(f"  prefill/cycle, prefix cache{per_cycle_cached:8,.0f} tok")
    print(f"  decode/cycle               {per_cycle_decode:8,.0f} tok")
    print(f"  totals per interview       {prefill_uncached:,.0f} prefill (uncached) / "
          f"{decode:,.0f} decode")

    # Prefill and decode are sequential but NOT symmetric: prefill processes the
    # prompt in parallel and is fast, decode emits one token at a time and is the
    # bottleneck. So the useful question is not "what throughput do I need" in the
    # abstract -- it is: given a plausible prefill rate, how fast must DECODE be?
    print("\n  Decode speed required, given a prefill rate (both stages per cycle):")
    print(f"  {'budget':>7} {'prefill@':>9} {'prefill time':>13} {'decode tok/s needed':>21}")
    for budget, budget_label in ((2.0, "COACH-05"), (8.0, "user-OK"), (seconds_between, "cycle gap")):
        for prefill_rate in (500, 1000, 2000):
            for name, prefill in (("no cache", per_cycle_uncached),
                                  ("prefix cache", per_cycle_cached)):
                prefill_s = prefill / prefill_rate
                left = budget - prefill_s
                if left <= 0:
                    verdict = "IMPOSSIBLE (prefill alone exceeds budget)"
                    print(f"  {budget:6.1f}s {prefill_rate:8,} {prefill_s:12.1f}s  "
                          f"{verdict}  ({name})")
                    continue
                need = per_cycle_decode / left
                print(f"  {budget:6.1f}s {prefill_rate:8,} {prefill_s:12.1f}s "
                      f"{need:20,.0f}  ({name})")
        print(f"           ^ budget = {budget_label}")


def main() -> int:
    project = REPO / "projects" / "demo-claudictive"
    session = project / "sessions" / "20260813T213559"
    print(f"session: {session.name}   (token counts ESTIMATED at {CHARS_PER_TOKEN} chars/token)\n")

    rows = [
        ("OLD (one transcript block, breakpoint at its end)", 0, True),
        ("NO CACHING AT ALL (control)", 0, False),
        ("NEW (split transcript, chunk=25)", 25, False),
    ]
    results = {}
    for label, chunk, legacy in rows:
        r = price(session, project, chunk=chunk, legacy=legacy)
        results[label] = r
        print(f"{label}")
        print(f"  cycles              {r['cycles']}")
        print(f"  input, full price   {r['in_full']:12,.0f} tok")
        print(f"  input, cache WRITE  {r['in_write']:12,.0f} tok  ({r['writes']} calls)")
        print(f"  input, cache READ   {r['in_read']:12,.0f} tok  ({r['reads']} calls)")
        print(f"  prefix below floor  {r['below_floor']} calls (never cached)")
        print(f"  output              {r['out']:12,.0f} tok")
        print(f"  >>> COST PER INTERVIEW  ${r['cost']:.2f}\n")

    old = results[rows[0][0]]["cost"]
    new = results[rows[-1][0]]["cost"]
    print("=" * 60)
    print(f"saving: ${old - new:.2f}  ({(old - new) / old:.0%})")
    print(f"under $1 target: {'YES' if new < 1.0 else 'NO'}")

    r = results[rows[-1][0]]
    print("\nWhere the remaining cost sits (NEW layout):")
    # `critique_in` already carries its cache multipliers, so the generate
    # side is the total minus it -- otherwise cached critique tokens get
    # counted twice (once in the prefix buckets, once here) and the shares
    # sum past 100%.
    effective_in = (
        r["in_full"] + r["in_write"] * CACHE_WRITE_MULT + r["in_read"] * CACHE_READ_MULT
    )
    parts = [
        ("generate input (net of cache)", (effective_in - r["critique_in"]) * IN_PER_M / 1e6),
        ("critique input (net of cache)", r["critique_in"] * IN_PER_M / 1e6),
        ("generate output", (r["out"] - r["critique_out"]) * OUT_PER_M / 1e6),
        ("critique output", r["critique_out"] * OUT_PER_M / 1e6),
    ]
    for name, amount in sorted(parts, key=lambda kv: -kv[1]):
        print(f"  {name:28} ${amount:5.2f}   {amount / r['cost']:4.0%}")

    print("\nIf the cycle count were reduced (everything scales with it):")
    for cycles, label in ((109, "4.0s debounce"), (71, "6.0s debounce")):
        print(f"  {cycles:3} cycles ({label:14}) -> ${r['cost'] * cycles / r['cycles']:.2f}")

    control = results[rows[1][0]]
    local_estimate(
        r,
        no_cache_input_tokens=control["in_full"] + control["in_write"] + control["in_read"],
    )
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
