"""A dress rehearsal: the real coach against an interviewee that answers back.

**What this is for.** Before 2026-08-14 the coach had never produced usable
output in a real interview, and the one scored run was a REPLAY -- a fixed
script that said the same thing no matter what was asked, so nothing the coach
did could change what it heard next. This drives the real `SuggestionPipeline`
against `interviewee_agent.py`, which responds to the question it was actually
asked. The conversation is novel every time, which is the point: it is the first
setting in which the coach can be wrong in a way that matters.

**What it does NOT exercise, stated first because the gap is the dangerous
part.** No audio is captured, no microphone is opened, no loopback device is
touched, and the STT engine never runs. So this rehearsal CANNOT close "nobody
has ever watched a full record→stop cycle" -- that needs real capture and is a
separate exercise. What it does close is everything downstream of a segment
landing: the generate call, the critique call, dedup, ranking, the durable
records, and the volume the operator would actually face.

**Segments are FRAGMENTED, and that is not cosmetic.** The pipeline runs one
cycle per submitted segment, and the real STT emits roughly five-word
finalizations rather than whole turns. Submitting a turn as one segment would
under-count cycles by an order of magnitude and would report a comfortable
suggestion rate that the product does not have -- the measured 26-minute
interview produced 401 shown suggestions precisely because every fragment is a
trigger. `--words-per-segment` exists so that assumption can be varied rather
than trusted.

**Time is synthetic and says so.** With no audio there is no audio clock, so
segment times advance at `--speech-rate` words per second. Every latency figure
this prints is therefore a MODEL-call latency measured against a simulated
speech timeline; it is a fair measure of how long the coach takes and NOT a
measure of whether that is fast enough against real speech.

Usage:

    .venv/Scripts/python.exe benchmark/eval/rehearse.py --auto --turns 12
    .venv/Scripts/python.exe benchmark/eval/rehearse.py            # you ask, it answers
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from app import config as config_mod  # noqa: E402
from benchmark.console import use_utf8_console  # noqa: E402

config_mod._load_dotenv_once(REPO / ".env")

from app.coach import corpus, reasoning_client  # noqa: E402
from app.coach.context import SURVEY_FILENAME, SurveyItem, load_survey  # noqa: E402
from app.coach.coverage import CoverageState  # noqa: E402
from app.coach.pipeline import SuggestionPipeline  # noqa: E402
from app.coach.trigger import TurnTrigger  # noqa: E402
from app.coach.uptake import uptake  # noqa: E402
from app.session.store import SessionStore, create_session_dir  # noqa: E402
from app.session.talk_share import talk_share  # noqa: E402
from benchmark.eval import claudictive_script  # noqa: E402
from benchmark.eval.interviewee_agent import (  # noqa: E402
    IntervieweeAgent, IntervieweeFailure, build_interviewee_client,
    resolve_interviewee_model,
)

DEFAULT_PROJECT = "demo-claudictive"
# ~150 words/minute is unhurried conversational English. Only used to advance a
# synthetic clock; nothing here is a claim about real speech.
DEFAULT_SPEECH_RATE = 2.5
DEFAULT_WORDS_PER_SEGMENT = 5


def scripted_questions() -> "list[str]":
    """The interviewer's turns from the synthetic script, reused as a stand-in
    operator. Reused rather than invented so an unattended run asks questions
    somebody already judged competent -- and so `--auto` measures the coach
    rather than measuring my ability to improvise an interview."""
    return [text for speaker, text in claudictive_script.TURNS if speaker == "interviewer"]


def fragments(text: str, words_per_segment: int) -> "list[str]":
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + words_per_segment])
            for i in range(0, len(words), words_per_segment)]


class Clock:
    """A synthetic audio clock. Real capture gets these from the engine."""

    def __init__(self, rate: float) -> None:
        self.at = 0.0
        self._rate = rate

    def advance(self, text: str) -> "tuple[float, float]":
        start = self.at
        self.at += max(len(text.split()) / self._rate, 0.4)
        return start, self.at


def main(argv: "Optional[Sequence[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--auto", action="store_true",
                        help="use the script's interviewer turns instead of prompting you")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--words-per-segment", type=int, default=DEFAULT_WORDS_PER_SEGMENT)
    parser.add_argument("--speech-rate", type=float, default=DEFAULT_SPEECH_RATE)
    parser.add_argument("--max-shown", type=int, default=config_mod.DEFAULT_COACH_MAX_SHOWN)
    parser.add_argument("--settle", type=float, default=6.0,
                        help="seconds to wait after each turn for the coach to answer")
    args = parser.parse_args(argv)

    project_dir = REPO / "projects" / args.project
    context_dir = project_dir / "context"
    # `rehearsals/`, NEVER `sessions/`, and this is a correctness decision rather
    # than tidiness. `analyse_eval.py` scores "the newest session under
    # demo-claudictive", the hub's picker lists everything in `sessions/`, and
    # `_coverage` walks all of them. A rehearsal landing there would displace the
    # scored eval run, appear in the operator's interview list as though a
    # customer had been spoken to, and contribute invented speech to the project
    # standing. The first smoke test of this file did exactly that.
    sessions_root = project_dir / "rehearsals"
    sessions_root.mkdir(parents=True, exist_ok=True)

    hypotheses = corpus.load_hypotheses(context_dir, missing_ok=True)
    # The project's REAL survey, not an empty list.
    #
    # This was `[]`, and that quietly disabled a third of the coach. Two of the
    # five moves -- `advance` and `fill_gap` -- exist to move the interviewer
    # through the survey and reach for items nothing has covered; with no
    # survey they have nowhere to go, so a rehearsal could only ever exercise
    # `deepen` and `test_hypothesis` while reporting itself as a full run of
    # "the real coach". It also meant the coverage-driven half of the prompt
    # was empty on every cycle.
    #
    # Tolerant of absence (a project may genuinely have no survey) but not of
    # a malformed one, which raises -- the same split `load_hypotheses` above
    # already makes.
    survey: "Sequence[SurveyItem]" = ()
    survey_path = context_dir / SURVEY_FILENAME
    if survey_path.is_file():
        survey = load_survey(context_dir)

    session_dir = create_session_dir(sessions_root)
    store = SessionStore(session_dir)
    store.write_metadata({
        "purpose": "rehearsal",
        # Recorded IN the session, because a rehearsal transcript sitting beside
        # real interviews is a corpus-poisoning hazard: extraction would happily
        # mine invented customer speech into `facts.json` and every later
        # standing would be evidenced partly by a machine talking to itself.
        "synthetic": True,
        "rehearsal_note": "NOT a real interview. Interviewee text is model-generated.",
        "coach_enabled": True,
    })

    coach_client = reasoning_client.build_client()
    model = reasoning_client.resolve_model()
    shown_log: "list[str]" = []

    def on_ranked(candidates) -> None:
        for candidate in candidates:
            shown_log.append(candidate.text)
            print(f"    \033[36mcoach:\033[0m {candidate.text}")

    pipeline = SuggestionPipeline(
        client=coach_client,
        store=store,
        brief=(),
        survey=survey,
        hypotheses=hypotheses,
        session_start_grounding_s=time.monotonic(),
        coverage_state_provider=lambda: CoverageState(asked=frozenset()),
        model=model,
        generate_timeout_s=config_mod.DEFAULT_COACH_GENERATE_TIMEOUT_S,
        critique_timeout_s=config_mod.DEFAULT_COACH_CRITIQUE_TIMEOUT_S,
        generate_max_tokens=config_mod.DEFAULT_COACH_GENERATE_MAX_TOKENS,
        critique_max_tokens=config_mod.DEFAULT_COACH_CRITIQUE_MAX_TOKENS,
        max_shown=args.max_shown,
        failure_threshold=config_mod.DEFAULT_COACH_FAILURE_THRESHOLD,
        language_window_s=config_mod.DEFAULT_COACH_LANGUAGE_WINDOW_S,
        language_flip_margin=config_mod.DEFAULT_COACH_LANGUAGE_FLIP_MARGIN,
        language_min_chars=config_mod.DEFAULT_COACH_LANGUAGE_MIN_CHARS,
        on_ranked=on_ranked,
    )

    interviewee_client = build_interviewee_client()
    interviewee_model = resolve_interviewee_model()
    agent = IntervieweeAgent(model=interviewee_model)
    clock = Clock(args.speech_rate)

    print(f"session : {session_dir.name}")
    print(f"coach   : {model}")
    print(f"guest   : {interviewee_model}")
    print(f"NOT a real interview: no audio, no STT, no record→stop.\n")

    # The SAME gate `app/main.py` wires between the STT workers and the
    # pipeline, wired here for the same reason. This harness used to submit
    # every fragment straight to `pipeline.submit`, which was faithful to the
    # coach as it existed when the file was written and stopped being faithful
    # the moment `turn+3s` shipped -- and stopped again when the word floor
    # did. A rehearsal that skips the gate measures a coach nobody runs, and
    # would have reported a suggestion rate several times the real one while
    # looking perfectly healthy.
    #
    # Constructed with its own defaults rather than a copy of them, so a change
    # to the shipped delay or floor reaches this harness without anybody
    # remembering to update it.
    trigger = TurnTrigger(downstream=pipeline.submit)

    def say(speaker: str, text: str) -> None:
        for fragment in fragments(text, args.words_per_segment):
            start, end = clock.advance(fragment)
            # `ready_wall_s` sits on the SYNTHETIC clock, not on `monotonic()`.
            # The store derives per-segment `latency_s` from the gap between
            # these two, so a real monotonic value against a synthetic audio
            # position produced a recorded STT latency of 181,767 seconds in the
            # first smoke test -- a number that would have been written into
            # `session.json`'s latency summary and read later as a measurement.
            # The 0.5s stand-in is invented, and per-segment latency from a
            # rehearsal is therefore meaningless BY CONSTRUCTION; the real
            # figure this run produces is the coach's own cycle latency, which
            # `pipeline.latency_summary()` measures against the wall clock.
            record = store.append_segment(
                speaker=speaker, text=fragment,
                audio_start_s=start, audio_end_s=end,
                ready_wall_s=end + 0.5, detected_language="en", engine="rehearsal",
            )
            trigger.submit(record)

    pipeline.start()
    started = time.monotonic()
    questions = scripted_questions() if args.auto else []
    asked = 0
    try:
        while asked < args.turns:
            if args.auto:
                if asked >= len(questions):
                    break
                question = questions[asked]
            else:
                try:
                    question = input("you> ").strip()
                except EOFError:
                    break
                if not question:
                    break
            print(f"\n\033[33myou:\033[0m {question}")
            say("interviewer", question)

            try:
                reply, agent = agent.answer(interviewee_client, question)
            except IntervieweeFailure as exc:
                print(f"  guest failed ({exc.reason}) -- ending the rehearsal")
                break
            print(f"\033[32mguest:\033[0m {reply.answer}")
            if reply.caved:
                print("  (the guest was pressed and cooperated anyway)")
            say("interviewee", reply.answer)

            asked += 1
            # The coach answers on its own thread; without a pause the whole
            # rehearsal would finish before a single cycle returned and the run
            # would report a coach that said nothing.
            time.sleep(args.settle)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        pipeline.stop()
        store.close()

    wall = time.monotonic() - started
    records = [
        __import__("json").loads(line)
        for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suggestions = [r for r in records if r.get("kind") == "suggestion"]
    shown = [s for s in suggestions if s.get("shown")]
    segments = [r for r in records if r.get("kind") == "segment"]
    spoken_minutes = clock.at / 60.0

    print("\n" + "=" * 68)
    print("REHEARSAL")
    print("=" * 68)
    print(f"turns            : {asked}")
    print(f"segments          : {len(segments)}  (one coach cycle each)")
    print(f"suggestions       : {len(suggestions)} produced, {len(shown)} shown")
    if spoken_minutes > 0:
        print(f"rate              : {len(shown) / spoken_minutes:.1f} shown per SPOKEN minute "
              f"({spoken_minutes:.1f} min of simulated speech)")
    print(f"wall clock        : {wall / 60:.1f} min")
    print(f"coach health      : {pipeline.health}")
    # The one latency figure this run is entitled to report. Per-segment
    # `latency_s` in the transcript is synthetic (see `say`); this is the real
    # wall-clock cost of a generate+critique cycle against the chosen model.
    print(f"coach cycle       : {pipeline.latency_summary()}")

    share = talk_share([
        {"speaker": r.get("speaker"), "text": r.get("text")} for r in segments
    ])
    if share["measurable"]:
        mine = share["speakers"].get("interviewer")
        if mine:
            print(f"talk share        : you {mine['share']:.0%} of {share['total_words']} words")

    took = uptake(records)
    if took.rate is not None:
        print(f"uptake            : {took.taken}/{took.shown} ({took.rate:.0%}) "
              f"-- noise floor on the replay corpus was 4%")

    print(f"\nsession written to: {session_dir}")
    print("It is marked synthetic:true. Do NOT run `extract` on it -- the")
    print("interviewee's words are model-generated and would enter the corpus")
    print("as though a customer had said them.")
    return 0


if __name__ == "__main__":
    use_utf8_console()
    sys.exit(main())
