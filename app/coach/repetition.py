"""What the coach has already proposed, and whether a new candidate is the
same thing again.

**Why this module exists.** The first scored eval run (session 20260813T213559,
26 minutes, 584 candidates) generated one single question -- "walk me through
the last time you removed something from your gaming machine" -- **67 times,
and showed it 64 times**. A second, "what is on your second screen while you
play", ran to 36/33. Across the whole run, **32% of everything shown to the
operator was an exact repeat of something already shown.** No unit test could
have found that: every individual cycle was correct in isolation.

**Two users, ONE definition of "the same question".** The cause is that each
generate call is blind to the previous ones, so `already_proposed_block` feeds
recent texts back into the prompt; the symptom is that a repeat still reaches
the operator, so `contains_near` suppresses it at the point of showing. Those
two must agree on what counts as a repeat or the guard starts contradicting
the instruction the model was given -- suppressing questions the prompt never
told it to avoid, and vice versa. Hence one module, one `normalise`, one
threshold, used by both.

**Similarity, not equality.** The x67 offender had a x17 near-variant differing
by two words ("...from your gaming machine" vs "...from that machine"). An
equality check scores that as two distinct questions, which is exactly the
judgement an operator would not make. Measured against the real run, a rolling
window of 25 at threshold 0.90 suppresses 32% of shown output; 0.80 suppresses
36%. The default is the conservative end -- a false suppression silently
withholds a good question, which is worse than one visible repeat.

**Immutable by construction.** `remember` returns a NEW instance rather than
mutating, so the pipeline's worker thread holds a plain rebind and no caller
can be surprised by a window that changed under it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Sequence

# Measured defaults -- see the module docstring. Both are overridable from
# config; neither is a guess.
DEFAULT_WINDOW = 25
DEFAULT_THRESHOLD = 0.90
# How many of the remembered texts are fed back into the generate prompt. The
# prompt block lands in the UNCACHED user message (it changes every call, so it
# cannot sit behind a cache breakpoint -- see `generate.assemble_generate_request`),
# and is therefore paid for in full on every cycle. Ten recent questions is
# roughly 250 tokens; the whole 25-deep dedup window would be ~625. The window
# and the prompt list are deliberately allowed to differ in size: remembering
# more than you say out loud costs nothing.
DEFAULT_PROMPT_RECENT = 10


def normalise(text: str) -> str:
    """Casefold, drop punctuation, collapse whitespace. The single definition
    of "the same words" for both the prompt and the guard."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def similarity(left: str, right: str) -> float:
    """Word-level ratio, never character-level. Character similarity rates two
    unrelated questions of the same length far higher than a reader would, and
    the ranking here decides whether an operator sees a question at all."""
    return SequenceMatcher(None, normalise(left).split(), normalise(right).split()).ratio()


@dataclass(frozen=True)
class RecentSuggestions:
    """A bounded, immutable memory of what has already been shown."""

    texts: "tuple[str, ...]" = ()
    window: int = DEFAULT_WINDOW
    threshold: float = DEFAULT_THRESHOLD

    def contains_near(self, text: str) -> bool:
        """True when `text` says the same thing as something already shown.

        An empty or whitespace-only candidate is never treated as a duplicate:
        it is a different defect, and swallowing it here would hide it."""
        if not normalise(text):
            return False
        return any(similarity(text, previous) >= self.threshold for previous in self.texts)

    def remember(self, text: str) -> "RecentSuggestions":
        """A NEW instance with `text` appended and the window trimmed from the
        FRONT, so the most recent survive. Never mutates."""
        if not normalise(text):
            return self
        kept = (self.texts + (text,))[-self.window:] if self.window > 0 else ()
        return replace(self, texts=kept)

    def remember_all(self, texts: Sequence[str]) -> "RecentSuggestions":
        current = self
        for text in texts:
            current = current.remember(text)
        return current

    def for_prompt(self, limit: int = DEFAULT_PROMPT_RECENT) -> "tuple[str, ...]":
        """The most recent texts, newest last, capped for the prompt."""
        if limit <= 0:
            return ()
        return self.texts[-limit:]
