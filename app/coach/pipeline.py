"""The suggestion pipeline: owns the bounded, single-slot trigger queue, the
dedicated worker thread, the two chained calls in order, the ranking, the
durable write and the latency measurement (04-CONTEXT.md §D2-§D9;
04-02-PLAN.md §D2, §D7, §D8, §D9).

**Sequential, not pipelined -- a stated tradeoff, not a quiet default (§D2).**
One generate call returning ALL candidates, then one critique call
classifying all of them. `04-RESEARCH.md` §4/§7 documents a pipelined design
(stream generate internally, fire critique per-candidate as its block closes)
that lands more reliably inside the ~1.5-2.0s budget; this plan builds the
simpler sequential design first and wires per-cycle latency measurement from
its own first commit specifically so the upgrade decision is made from a
measured miss rate, not an impression -- `04-RESEARCH.md`'s own Pitfall 2, and
a mistake already made once on this project (STATE.md: every latency figure
on record before this phase came from a fake engine making no network call).
If a rehearsal shows the sequential design missing COACH-05 on a meaningful
fraction of turns, the pipelined design is already researched and is the
documented next step -- do not build it speculatively.

**Never on a capture or STT thread (non-negotiable #7, §D9).** `submit()`
enqueues onto a bounded (maxsize=1) queue and returns -- no network, no
blocking, no raising, ever. The dedicated worker thread here is the ONLY
place a reasoning call for a suggestion happens.

**Replace, never queue, per §D9.** A maxsize-1 queue with drop-oldest-on-full
IS "one cycle at a time" plus "a new trigger replaces any queued-but-unstarted
one": while a cycle is running, the FIRST arriving segment queues normally
(the queue was empty, since the worker already pulled its own item); a
SECOND arrival before that cycle finishes drops the first queued one and
records a named event (`EVENT_CYCLE_SKIPPED`) -- never counted as a failure
(§D7/Task 3), since skipping under load is correct behaviour, not a broken
coach.

**D-03/D-04, structural, not convention.** This module only ever READS
coverage state via `coverage_state_provider` (a plain callable, injected --
never a concrete `CoverageDetector` import) -- it has no write path into
coverage at all, and it never triggers or waits on detection. The generate
call reads whatever coverage state happens to be current at cycle start;
detection itself runs entirely on `app/coach/coverage.py`'s own thread.

**D-08's two-tier rule, applied here.** The durable store
(`SessionStore.append_suggestion`) gets every candidate the cycle ever
produced -- shown, ranked down, or never vetted at all -- with its full text.
The logger and every event `detail` dict get counts, bands and error-type
names only, NEVER candidate or transcript text (T-04-27).

Must NEVER import `anthropic` (T-04-12, gate 11) and must NEVER call
`.result()` on anything from a capture/STT thread -- `submit()` is the only
surface those threads ever touch, and it never blocks.

**06-01-PLAN.md §F4: `session_start_grounding_s` is a keyword-only
constructor argument with NO DEFAULT, and that absence is load-bearing, not
an oversight.** `_cycle_latency_s` below measures from the trigger segment's
own `written_at` -- a real wall-clock field that does not exist until AFTER
a session has started and an interviewee has spoken. A cost paid once at
session start, before any trigger exists, is therefore structurally OUTSIDE
what that measurement can ever see. Bringing retrieval inside the per-cycle
window is not available here (it would need its own cache breakpoint and
break `prompts.static_block`'s byte-stability contract -- see
06-RESEARCH.md's Design B, deliberately not adopted). So the cost is
measured once by the caller (`app/main.py`, around its own
`_load_coach_context` call -- never inside this module, which gains no
second clock) and REPORTED inseparably from the per-cycle numbers via
`pipeline_latency_summary`. Giving this argument a default would let a
pipeline be built without ever having measured that cost, silently reporting
it as whatever the default happened to be -- gate 22
(`app/tests/test_phase_gates.py`) asserts the absence of a default is what
keeps ROADMAP criterion 6 from being judged against a number that
structurally excludes this phase's own addition.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from typing import Any, Callable, Optional, Sequence

from app.coach import critic, generate, language, repetition
from app.coach.context import ContextItem, SurveyItem
from app.coach.coverage import EVENT_COVERAGE_ASKED, CoverageState
from app.coach.health import CoachHealth
from app.coach import reasoning_client
from app.session.reader import read_session
from app.session.store import SegmentRecord, SessionStore, latency_stats

logger = logging.getLogger(__name__)

SPEAKER_INTERVIEWEE = "interviewee"

# Reused verbatim from app/coach/coverage.py -- the SAME event name, a stage
# label inside `detail` is what tells "coverage-match failed" and
# "generate/critique failed" apart, per §D7's "a stage label" instruction.
EVENT_REASONING_FAILURE = "reasoning-timeout"
EVENT_CYCLE_SKIPPED = "coach-cycle-skipped"
# A candidate the critic passed, withheld because the operator has already been
# shown the same question. Named separately from the two above: it is neither a
# failure nor load-shedding, and counting it as either would misreport a healthy
# coach doing exactly what it was asked to do.
EVENT_SUGGESTION_REPEATED = "coach-suggestion-repeated"

# The cycle ran, succeeded, and judged that nothing needed asking (2026-08-15).
# Named separately for the same reason as the event above: it is neither a
# failure nor load-shedding nor a repeat, and filing it as any of those would
# report a coach exercising judgement as a coach malfunctioning. Without it a
# declining cycle leaves no trace at all, and "nothing worth asking" becomes
# indistinguishable from "the coach died quietly" -- which is the single
# question a quiet panel makes an interviewer ask.
EVENT_SUGGESTION_DECLINED = "coach-suggestion-declined"


def _parse_iso(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)


def pipeline_latency_summary(
    cycle_latencies: Sequence[float], *, session_start_grounding_s: float
) -> dict:
    """count/median/p95 over completed cycles, with the FIRST cycle reported
    separately (§D1: a schema the API has not seen before pays a one-time
    compilation cost on its first use, landing on the very first suggestion
    of the first interview of the day). Built on the SAME shared
    `latency_stats` the transcript-latency summary uses (Task 3), so the two
    numbers can never drift into two implementations. Never raises when
    `cycle_latencies` is empty -- a session where the coach never produced a
    suggestion still paid the session-start grounding cost, and that is
    exactly the case a reader needs to see (06-01-PLAN.md §F4).

    `session_start_grounding_s` is carried through UNCHANGED, never
    recomputed or defaulted here -- the caller (`SuggestionPipeline`, whose
    own constructor argument of the same name has no default; see the module
    docstring) is the one place this number is allowed to originate. The
    returned dict's key set is exactly `{"first_cycle_s", "stats",
    "session_start_grounding_s"}` -- gate 22 pins this by equality."""
    if not cycle_latencies:
        return {
            "first_cycle_s": None,
            "stats": latency_stats([]),
            "session_start_grounding_s": session_start_grounding_s,
        }
    first_cycle_s = cycle_latencies[0]
    rest = list(cycle_latencies[1:])
    return {
        "first_cycle_s": first_cycle_s,
        "stats": latency_stats(rest),
        "session_start_grounding_s": session_start_grounding_s,
    }


class SuggestionPipeline:
    """Owns the bounded single-slot trigger queue and the dedicated worker
    thread (§D9). Public surface is deliberately `submit`, `start`, `stop`,
    `health` and `latency_summary` -- mirroring `CoverageDetector`'s own
    narrow public surface from 04-01."""

    def __init__(
        self,
        *,
        client: Any,
        store: SessionStore,
        brief: Sequence[ContextItem],
        survey: Sequence[SurveyItem],
        hypotheses: Sequence[Any] = (),
        session_start_grounding_s: float,
        coverage_state_provider: Callable[[], CoverageState],
        model: str = reasoning_client.DEFAULT_MODEL,
        generate_timeout_s: float,
        critique_timeout_s: float,
        generate_max_tokens: int,
        critique_max_tokens: int,
        max_shown: int,
        failure_threshold: int,
        language_window_s: float,
        language_flip_margin: float,
        language_min_chars: int,
        repeat_window: int = repetition.DEFAULT_WINDOW,
        repeat_threshold: float = repetition.DEFAULT_THRESHOLD,
        repeat_prompt_recent: int = repetition.DEFAULT_PROMPT_RECENT,
        queue_maxsize: int = 1,
        on_ranked: Optional[Callable[["list[generate.Candidate]"], None]] = None,
    ) -> None:
        self._client = client
        self._store = store
        self._brief = tuple(brief)
        self._survey = tuple(survey)
        # 06-02-PLAN.md §G3/Task 3: injected at construction, never read from
        # a file on the worker thread -- `_load_coach_context` (06-01 Task 3)
        # already loads these once, at session start, exactly like brief and
        # survey. Defaults to `()` (unlike `session_start_grounding_s`, which
        # deliberately has no default as gate 22's forcing function) so an
        # unmodified caller degrades to zero hypotheses -- the normal D-02
        # empty-corpus case, never a construction-time failure.
        self._hypotheses = tuple(hypotheses)
        self._session_start_grounding_s = session_start_grounding_s
        self._coverage_state_provider = coverage_state_provider
        self._model = model
        self._generate_timeout_s = generate_timeout_s
        self._critique_timeout_s = critique_timeout_s
        self._generate_max_tokens = generate_max_tokens
        self._critique_max_tokens = critique_max_tokens
        self._max_shown = max_shown
        self._failure_threshold = failure_threshold
        self._language_window_s = language_window_s
        self._language_flip_margin = language_flip_margin
        self._language_min_chars = language_min_chars
        # A callback for "here is a ranked set" that 04-03 attaches the
        # broadcast to -- this plan passes nothing, so the pipeline is fully
        # testable and fully useful with no surface at all (§D9).
        self._on_ranked = on_ranked

        self._queue: "queue.Queue[SegmentRecord]" = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._repeat_prompt_recent = repeat_prompt_recent
        # Touched ONLY on this class's own worker thread (`_process`), exactly
        # like `_interviewee_segments` and `_current_language` below, so it
        # needs no lock. Rebound, never mutated -- `RecentSuggestions` is frozen.
        self._recent = repetition.RecentSuggestions(
            window=repeat_window, threshold=repeat_threshold
        )

        self._interviewee_segments: "list[SegmentRecord]" = []
        self._current_language = language.LANGUAGE_ENGLISH
        self._health = CoachHealth()
        self._cycle_latencies: "list[float]" = []

    @property
    def health(self) -> CoachHealth:
        return self._health

    def latency_summary(self) -> dict:
        return pipeline_latency_summary(
            self._cycle_latencies, session_start_grounding_s=self._session_start_grounding_s
        )

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="coach-suggestion", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=15.0)
            self._thread = None

    def submit(self, record: SegmentRecord) -> None:
        """The ONLY entry point a capture/STT thread ever touches (§D9).
        Never blocks, never raises, never touches the network. A full queue
        (a cycle already has one queued-but-unstarted trigger) drops the
        OLDEST queued entry and records a named event -- never a failure --
        before enqueuing the new one."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._store.append_event(
                    event=EVENT_CYCLE_SKIPPED,
                    speaker=SPEAKER_INTERVIEWEE,
                    detail={"reason": "cycle-in-flight"},
                )
            try:
                self._queue.put_nowait(record)
            except queue.Full:  # pragma: no cover -- only under a concurrent racing drain
                pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process_guarded(record)
        while True:
            try:
                record = self._queue.get_nowait()
            except queue.Empty:
                break
            self._process_guarded(record)

    def _process_guarded(self, record: SegmentRecord) -> None:
        """Degrade and continue, never die.

        Only the two vendor calls inside `_process` are wrapped. Everything
        else can raise and unwind this loop: reading and parsing the session
        file, constructing records from a resumed session, `append_suggestion`
        against a full disk or a store closed by a shutdown race, timestamp
        parsing, language detection. If any of them does, the thread exits and
        no suggestion is ever produced again.

        The dangerous part is not the death, it is that `self._health` keeps
        whatever value it held at the moment of death -- typically zero
        failures -- so `is_unhealthy()` stays False, `derive_status` keeps
        reporting a healthy coach, and the phone never says suggestions are
        unavailable. A dead coach that looks alive is precisely the state the
        coach-failure banner exists to prevent, and it is worse than a coach
        that admits it is down.

        So: count it as a failure (which is what drives the banner), record it
        with the error TYPE only -- never transcript text -- and keep going.
        """
        try:
            self._process(record)
        except Exception as exc:  # noqa: BLE001 -- a dead pipeline is worse than any single failure
            logger.warning("coach cycle failed error_type=%s", type(exc).__name__)
            self._health = self._health.with_failure()
            try:
                self._store.append_event(
                    event=EVENT_REASONING_FAILURE,
                    speaker=SPEAKER_INTERVIEWEE,
                    detail={"error_type": type(exc).__name__, "stage": "cycle"},
                )
            except Exception:  # noqa: BLE001 -- the store itself may be gone during shutdown
                pass

    def _read_transcript_and_asked(self) -> "tuple[list[SegmentRecord], dict[str, generate.AskedAt]]":
        """Reads the durable record for the whole transcript (both speakers,
        untruncated -- D4) and for the seq/time of every `coverage-asked`
        event `app/coach/coverage.py` has already written. This module never
        reconstructs coverage STATE from these events (that would duplicate
        `CoverageDetector`'s own replay logic, and D-04 keeps coverage
        write-side entirely inside `coverage.py`) -- it only enriches the
        already-authoritative `asked` id set from `coverage_state_provider`
        with the display metadata the generate prompt wants (§D3/§D4)."""
        result = read_session(self._store.session_dir)
        transcript = [SegmentRecord(**record) for record in result.records_of_kind("segment")]
        asked_at: "dict[str, generate.AskedAt]" = {}
        for event in result.records_of_kind("event"):
            if event.get("event") != EVENT_COVERAGE_ASKED:
                continue
            detail = event.get("detail") or {}
            question_id = detail.get("question_id")
            seq = detail.get("matched_seq")
            audio_end_s = detail.get("matched_audio_end_s")
            if question_id is not None and seq is not None and audio_end_s is not None:
                asked_at[question_id] = generate.AskedAt(seq=seq, audio_end_s=audio_end_s)
        return transcript, asked_at

    def _cycle_latency_s(self, trigger: SegmentRecord) -> float:
        """From the trigger segment's own ready time (its `written_at`, the
        real wall-clock moment it was durably persisted -- essentially the
        same moment this cycle was triggered) to right now. `SegmentRecord`
        carries no other real wall-clock field: `ready_wall_s`/`audio_end_s`
        are in the STT engine's own session-relative clock domain, not
        comparable to `datetime.now()` (see benchmark/engines/base.py)."""
        started = _parse_iso(trigger.written_at)
        elapsed = (datetime.now(started.tzinfo) - started).total_seconds()
        return max(0.0, elapsed)

    def _write_suggestion_records(
        self,
        candidates: "Sequence[generate.Candidate]",
        *,
        judgments: "Sequence[critic.Judgment]",
        trigger: SegmentRecord,
        shown_indices: "frozenset[int]" = frozenset(),
    ) -> None:
        """Every candidate the cycle ever produced -- shown, ranked down, or
        dropped, or never vetted at all -- is recorded here (COACH-04/§D8).
        A candidate with no matching judgment (critique failed entirely) is
        recorded with every judgment field `None` and `shown=False`.

        06-03-PLAN.md §H1: `grounding_ids`/`target_hypothesis_id`/
        `proposed_state` are passed straight through from the CANDIDATE,
        never recomputed here -- the validation against known ids/states
        already happened once, in `generate.py`'s own parse step
        (06-02-PLAN.md §G5); doing it twice would create a second place for
        the rule to drift. They are independent of whether the candidate was
        ever vetted -- a self-report the model made survives a critique
        failure exactly like the candidate's own text does. `is_confirmation_only`
        follows the SAME "only real when judged" shape the existing four
        flags already use."""
        judgment_by_index: "dict[int, critic.Judgment]" = {j.index: j for j in judgments}
        for candidate in candidates:
            judgment = judgment_by_index.get(candidate.index)
            self._store.append_suggestion(
                trigger_seq=trigger.seq,
                text=candidate.text,
                move=candidate.move,
                band=(judgment.band if judgment is not None else None),
                is_leading=(judgment.is_leading if judgment is not None else None),
                is_hypothetical_or_future_tense=(
                    judgment.is_hypothetical_or_future_tense if judgment is not None else None
                ),
                is_pitch=(judgment.is_pitch if judgment is not None else None),
                is_opinion_seeking=(judgment.is_opinion_seeking if judgment is not None else None),
                is_confirmation_only=(judgment.is_confirmation_only if judgment is not None else None),
                is_generic=(judgment.is_generic if judgment is not None else None),
                is_off_topic=(judgment.is_off_topic if judgment is not None else None),
                rationale=(judgment.rationale if judgment is not None else None),
                shown=(candidate.index in shown_indices),
                grounding_ids=candidate.grounding_ids,
                target_hypothesis_id=candidate.target_hypothesis_id,
                proposed_state=candidate.proposed_state,
            )

    def _process(self, trigger: SegmentRecord) -> None:
        self._interviewee_segments.append(trigger)
        detected_language = language.detect_language(
            self._interviewee_segments,
            window_s=self._language_window_s,
            flip_margin=self._language_flip_margin,
            min_chars=self._language_min_chars,
            previous_language=self._current_language,
        )
        self._current_language = detected_language

        transcript, asked_at = self._read_transcript_and_asked()
        coverage_state = self._coverage_state_provider()
        # Defensive: only ids the CURRENT coverage state actually considers
        # asked are ever passed on -- a stale event for a question id no
        # longer in the survey (coverage.py's own replay already logs and
        # drops these) must never resurface here.
        asked = {qid: info for qid, info in asked_at.items() if coverage_state.is_asked(qid)}

        try:
            candidates = generate.run_generate(
                self._client,
                brief=self._brief,
                survey=self._survey,
                asked=asked,
                transcript=transcript,
                trigger=trigger,
                detected_language=detected_language,
                # 06-02-PLAN.md §G1/§G3 Task 3: threaded from the
                # constructor, never read from a file on this worker thread.
                hypotheses=self._hypotheses,
                # What the operator has already been shown, so this call stops
                # re-proposing it. The measured cause of a 32%-repeat run.
                recently_shown=self._recent.for_prompt(self._repeat_prompt_recent),
                model=self._model,
                max_tokens=self._generate_max_tokens,
                timeout_s=self._generate_timeout_s,
            )
        except generate.GenerateFailure as exc:
            self._store.append_event(
                event=EVENT_REASONING_FAILURE,
                speaker=SPEAKER_INTERVIEWEE,
                detail={"stage": "generate", "error_type": type(exc).__name__, "stop_reason": exc.stop_reason},
            )
            self._health = self._health.with_failure()
            return

        try:
            judgments = critic.run_critique(
                self._client,
                candidates=candidates,
                # 06-02-PLAN.md §G4 Task 3: the SAME hypotheses passed to
                # generate above -- the critique stage judges is_confirmation_only
                # against the identical falsifier text the candidate was
                # constructed against.
                hypotheses=self._hypotheses,
                # The SAME survey passed to generate above (2026-08-19), for
                # the same reason the hypotheses are: `is_off_topic` asks
                # whether a candidate serves the interviewer's own plan, and
                # judging it against a different rendering of that plan than
                # the one the candidate was generated against would let the
                # two stages disagree about what this interview is for.
                survey=self._survey,
                # The trigger utterance the whole cycle was fired by
                # (2026-08-15). `is_generic` asks whether a candidate engages
                # with what THIS interviewee just said, and this is that
                # utterance -- the same record the generate stage was given,
                # never a re-read or a summary of it.
                trigger_text=trigger.text,
                model=self._model,
                max_tokens=self._critique_max_tokens,
                timeout_s=self._critique_timeout_s,
            )
        except critic.CritiqueFailure as exc:
            self._store.append_event(
                event=EVENT_REASONING_FAILURE,
                speaker=SPEAKER_INTERVIEWEE,
                detail={"stage": "critique", "error_type": type(exc).__name__, "stop_reason": exc.stop_reason},
            )
            self._health = self._health.with_failure()
            # Still durably recorded -- the audit trail shows what was
            # proposed even when nothing was vetted (Task 3). Never shown.
            self._write_suggestion_records(candidates, judgments=(), trigger=trigger)
            return

        # Ranked over the FULL survivor set, then de-duplicated, then capped --
        # in that order, and the order matters. Capping first would let a
        # repeat consume one of the `max_shown` slots and then be suppressed,
        # so a cycle with three good candidates behind one repeat would show
        # two. Ranking is pure and deterministic, so asking it for every
        # survivor costs nothing (`critic.rank_candidates`).
        ordered = critic.rank_candidates(candidates, judgments, max_shown=len(candidates))
        ranked: "list[generate.Candidate]" = []
        suppressed = 0
        recent = self._recent
        declined = 0
        for candidate in ordered:
            if len(ranked) >= self._max_shown:
                break
            # A declining candidate is a considered ANSWER, not a question
            # (2026-08-15). It must not reach the phone, must not enter the
            # repetition window (its text would poison near-duplicate
            # matching for every later cycle), and must not join
            # `shown_indices` below -- that set feeds `SuggestionRecord.shown`,
            # which is `app/coach/uptake.py`'s P1 denominator, so counting a
            # decline as shown would quietly deflate the measure of whether
            # the interviewer uses what the coach offers. It IS still written
            # to the durable record with every other candidate.
            if candidate.move == generate.MOVE_NO_QUESTION:
                declined += 1
                continue
            if recent.contains_near(candidate.text):
                suppressed += 1
                continue
            ranked.append(candidate)
            # Remembered as it is ACCEPTED, so two near-identical candidates
            # inside the SAME cycle cannot both be shown -- the x67 offender
            # was repeated across cycles, but nothing made within-cycle
            # duplication impossible either.
            recent = recent.remember(candidate.text)
        self._recent = recent
        if suppressed:
            # Counts only, never candidate text (D-08's two-tier rule, T-04-27).
            # Recorded because a silent guard is indistinguishable from a coach
            # that simply stopped proposing -- and if the threshold is ever set
            # too tight, this event is the only way anyone would find out.
            self._store.append_event(
                event=EVENT_SUGGESTION_REPEATED,
                speaker=SPEAKER_INTERVIEWEE,
                detail={"suppressed": suppressed, "shown": len(ranked)},
            )
        if declined:
            # Counts only, never text -- same two-tier rule as the repetition
            # event above, and here for the same reason. A cycle that
            # deliberately declined and a cycle that never ran are otherwise
            # indistinguishable in the durable record: both leave a quiet
            # panel and a healthy coach. This event is what separates "the
            # coach judged there was nothing to ask" from "the coach stopped
            # working", which is exactly the confusion a quiet panel invites.
            self._store.append_event(
                event=EVENT_SUGGESTION_DECLINED,
                speaker=SPEAKER_INTERVIEWEE,
                detail={"declined": declined, "shown": len(ranked)},
            )
        shown_indices = frozenset(c.index for c in ranked)
        self._write_suggestion_records(candidates, judgments=judgments, trigger=trigger, shown_indices=shown_indices)

        self._cycle_latencies.append(self._cycle_latency_s(trigger))
        self._health = self._health.with_success()

        if self._on_ranked is not None:
            try:
                self._on_ranked(ranked)
            except Exception as exc:  # noqa: BLE001 -- a broadcast-callback failure must never
                # take down the pipeline's own worker thread.
                logger.error("on_ranked callback failed error_type=%s", type(exc).__name__)
