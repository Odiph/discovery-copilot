"""Bands the interviewer's OWN turns with the same rubric the coach is held to.

The critic has always judged questions the coach PROPOSED. This points it at
the questions that were actually asked -- which are the only ones the
interview's findings actually rest on. It runs after the call, where there is
no latency budget to respect and no reason to hurry.

**Reachability and cleanliness must never look the same.** A turn whose batch
failed is recorded as `None`, never dropped and never defaulted to clean. A
dropped row is invisible in a report where a null is not, and silently removing
the turns a call could not reach would flatter the interviewer for free. This
is the same rule `benchmark/eval/judge_critic.judge_all` follows.

**A7 applies here and is not discharged by anything in this module.** The
rubric and its worked examples were written for questions a model PROPOSED, and
were calibrated on that distribution. Real interviewer speech is a different
distribution -- disfluent, mid-conversation, often a follow-up whose subject
lives in the previous answer. The bands this produces are worth reading and are
not worth reporting as calibrated until a gold set of REAL asked questions
exists. Nothing here measures that; `benchmark/eval/judge_critic.py` measures
the proposed case only, and says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from app.coach import critic as critic_mod
from app.coach.asked import InterviewerTurn
from app.coach.critic import BAND_CLEAN, CritiqueFailure, Judgment
from app.coach.generate import Candidate

logger = logging.getLogger(__name__)

# Deliberately NOT a member of `generate.ALL_MOVES`. A move is a statement
# about why a question was chosen, and the interviewer never made one --
# borrowing `deepen` or `advance` here would render intent into the prompt
# (`prompts.candidates_block` writes `move=` verbatim) that nobody expressed.
MOVE_ASKED = "asked"

# Matches the offline harness. Production sends the critic one cycle's handful,
# so a batch far larger would measure it on a workload it never sees and would
# let one turn prime the judgement of the rest.
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class JudgedTurn:
    """One turn and its verdict. `judgment is None` means the critic could not
    be reached for it -- NOT that it was clean."""

    turn: InterviewerTurn
    judgment: Optional[Judgment]
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class AskedCritique:
    judged: "tuple[JudgedTurn, ...]"

    @property
    def judged_count(self) -> int:
        return sum(1 for item in self.judged if item.judgment is not None)

    @property
    def unjudged_count(self) -> int:
        return sum(1 for item in self.judged if item.judgment is None)

    @property
    def clean_count(self) -> int:
        return sum(
            1
            for item in self.judged
            if item.judgment is not None and item.judgment.band == BAND_CLEAN
        )

    def flag_count(self, field: str) -> int:
        return sum(
            1
            for item in self.judged
            if item.judgment is not None and getattr(item.judgment, field)
        )

    @property
    def clean_rate(self) -> Optional[float]:
        """Over the JUDGED turns only. `None` when nothing could be judged: an
        interview the critic could not reach is not an interview that scored
        perfectly, and a 0.0 or 1.0 would file those together."""
        if self.judged_count == 0:
            return None
        return self.clean_count / self.judged_count

    def _rate_over(self, marked: bool) -> Optional[float]:
        judged = [
            item for item in self.judged
            if item.judgment is not None and item.turn.is_marked_question is marked
        ]
        if not judged:
            return None
        return sum(1 for item in judged if item.judgment.band == BAND_CLEAN) / len(judged)

    @property
    def marked_question_clean_rate(self) -> Optional[float]:
        """The clean rate over turns PUNCTUATED as questions.

        This is the number to read as "how good were my questions". The
        combined `clean_rate` above mixes it with unmarked turns, which is
        what made the headline figure hard to interpret -- not because the
        unmarked turns are noise, but because they are a DIFFERENT question
        about the interview and were being averaged into one number."""
        return self._rate_over(True)

    @property
    def unmarked_clean_rate(self) -> Optional[float]:
        """The clean rate over turns carrying no `?`.

        ⚠️ NOT "the statement clean rate", and the distinction is the whole
        reason both are reported instead of one being filtered away. This
        group holds the interviewer's leading STATEMENTS -- the most damaging
        rubric violation there is, and the thing a `?` filter would have
        silently discarded -- mixed with the roughly half of genuine
        questions that simply were not transcribed with a question mark.
        A low number here is worth reading; it is not worth reporting as a
        statement score."""
        return self._rate_over(False)

    @property
    def marked_question_count(self) -> int:
        return sum(
            1 for item in self.judged
            if item.judgment is not None and item.turn.is_marked_question
        )


def to_candidate(turn: InterviewerTurn, index: int) -> Candidate:
    """The turn as the critic will see it. `index` is per-BATCH, matching the
    live path, where it is the emission order within one cycle."""
    return Candidate(index=index, text=turn.text, move=MOVE_ASKED)


def critique_turns(
    client: Any,
    turns: "Sequence[InterviewerTurn]",
    *,
    hypotheses: "Sequence[Any]",
    model: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_critique: Optional[Callable[..., "list[Judgment]"]] = None,
) -> AskedCritique:
    """Bands every turn, in batches, surviving a failure in any one of them.

    `run_critique` is injectable so the batching, the index mapping and the
    failure bookkeeping are testable without a vendor call -- the seam is the
    critic, not the client."""
    call = run_critique or critic_mod.run_critique
    results: "list[JudgedTurn]" = []

    for start in range(0, len(turns), batch_size):
        batch = list(turns[start:start + batch_size])
        candidates = [to_candidate(turn, i) for i, turn in enumerate(batch)]
        try:
            judgments = call(
                client,
                candidates=candidates,
                hypotheses=hypotheses,
                # Deliberately empty, and this is a KNOWN LIMITATION rather
                # than an oversight. `is_generic` is judged against the
                # utterance a candidate was responding to, but one request
                # here carries a BATCH of interviewer turns from different
                # points in the interview, each with a different preceding
                # utterance -- there is no single trigger that could serve
                # them. Sending one turn per call would make `is_generic`
                # judgeable at the cost of ~70 vendor calls per interview.
                # The prompt handles the empty case deterministically (set
                # the flag false, never guess), so the reported number is
                # honestly absent rather than quietly invented. This sits
                # alongside the A7 caveat this module already carries: the
                # rubric was calibrated on PROPOSED questions, and running it
                # over real speech is the weaker measurement of the two.
                trigger_text="",
                model=model,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        except CritiqueFailure as exc:
            logger.warning(
                "asked-question critique batch failed: reason=%s stop_reason=%s",
                exc.reason,
                exc.stop_reason,
            )
            results.extend(
                JudgedTurn(turn=turn, judgment=None, failure_reason=exc.reason)
                for turn in batch
            )
            continue

        # Keyed by the index the critic answered with, then matched back by
        # position. An index the batch never sent is discarded rather than
        # written onto a neighbour.
        by_index = {judgment.index: judgment for judgment in judgments}
        results.extend(
            JudgedTurn(turn=turn, judgment=by_index.get(i))
            for i, turn in enumerate(batch)
        )

    return AskedCritique(judged=tuple(results))


def to_json_dict(critique: AskedCritique, *, session_id: str) -> dict:
    """The durable artifact. Carries `seqs` rather than relying on the text
    alone, so a reporting layer can find the words again in the transcript it
    already has."""
    return {
        "session_id": session_id,
        "summary": {
            "turns": len(critique.judged),
            "judged": critique.judged_count,
            "unjudged": critique.unjudged_count,
            "clean": critique.clean_count,
            "clean_rate": critique.clean_rate,
            # Split out because the combined rate answers two different
            # questions at once. `marked_question_*` is "how good were my
            # questions"; `unmarked_*` covers leading statements AND the ~half
            # of real questions the transcript gave no `?`, so it is a signal
            # to read, never a statement score. See the properties' own
            # docstrings -- the naming is load-bearing.
            "marked_questions": critique.marked_question_count,
            "marked_question_clean_rate": critique.marked_question_clean_rate,
            "unmarked_clean_rate": critique.unmarked_clean_rate,
            "leading": critique.flag_count("is_leading"),
            "hypothetical_or_future_tense": critique.flag_count(
                "is_hypothetical_or_future_tense"
            ),
            "pitch": critique.flag_count("is_pitch"),
            "opinion_seeking": critique.flag_count("is_opinion_seeking"),
            "confirmation_only": critique.flag_count("is_confirmation_only"),
            # Always 0 by construction here -- see `critique_turns` on why no
            # trigger utterance can be supplied for a batch. Reported anyway
            # so the key set matches the live path's records rather than
            # differing in a way a reader has to discover.
            "generic": critique.flag_count("is_generic"),
        },
        "turns": [
            {
                "index": item.turn.index,
                "seqs": list(item.turn.seqs),
                "at_s": item.turn.at_s,
                "text": item.turn.text,
                "failure_reason": item.failure_reason,
                "judgment": None
                if item.judgment is None
                else {
                    "band": item.judgment.band,
                    "is_leading": item.judgment.is_leading,
                    "is_hypothetical_or_future_tense": item.judgment.is_hypothetical_or_future_tense,
                    "is_pitch": item.judgment.is_pitch,
                    "is_opinion_seeking": item.judgment.is_opinion_seeking,
                    "is_confirmation_only": item.judgment.is_confirmation_only,
                    "is_generic": item.judgment.is_generic,
                    "rationale": item.judgment.rationale,
                },
            }
            for item in critique.judged
        ],
    }
