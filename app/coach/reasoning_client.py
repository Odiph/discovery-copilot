"""The ONE place `anthropic.Anthropic()` is constructed, and the only module
under `app/coach/` allowed to import the vendor SDK at all (04-CONTEXT.md
§C4, 04-RESEARCH.md §7). `app/coach/coverage.py` and every future coach
module (04-02's `generate.py`/`critic.py`) receive a client by injection;
none of them construct one, and none of them import `anthropic`. A standing
AST-based gate in `app/tests/test_phase_gates.py` walks every module under
`app/coach/` and asserts this module is the only one whose parsed imports
name the vendor -- the same discipline `app/stt/live.py`'s STT-03 gate
already applies to STT vendors, extended here to the reasoning vendor
(T-04-12).

Two invariants this module owns and must never violate:

- **The key is read from `os.environ` explicitly and passed to the
  constructor as a keyword argument.** A bare, zero-argument
  `anthropic.Anthropic()` can silently authenticate from an OAuth profile
  left on disk by the vendor's own CLI -- a real property of this SDK's
  credential-resolution chain, not a hypothetical. That would make OPS-03's
  "one secret, from the environment" false on a machine where that CLI has
  ever been used, while every line of code at the call site still looks
  correct (T-04-04).
- **`max_retries=0`.** The SDK's default is 2 retries, and it retries
  timeouts among other things, serially. A call configured with a 2.5s
  timeout can therefore occupy up to 7.5s of wall clock before raising --
  silently tripling the bound non-negotiable #9 promises. Retrying is also
  the wrong behaviour here: a coverage match or a suggestion that arrives
  three timeouts late is worse than none, and the next utterance brings
  another chance within seconds (T-04-06).

Every call site names its OWN timeout explicitly via `with_timeout()` below
-- never the client's 10-minute default -- so no call site can silently
inherit a bound nobody chose.

This module must NEVER be imported for its side effects from a capture or
STT thread, and must NEVER cache module-level API-key state -- `build_client()`
reads the environment fresh on every call, exactly like
`benchmark/engines/soniox.py`'s `_get_api_key()`.
"""

from __future__ import annotations

import os

import anthropic

from app.coach import openai_seam

# Claude Haiku 4.5, both stages -- latency-driven, not cost-driven (locked
# decision, .planning/research/reasoning.md §2; Sonnet's own p95 TTFT alone
# can exceed the whole ~1.5-2s COACH-05 budget). The full dated model id is
# used so a silent point-release swap on the vendor's side never changes
# behaviour underneath this project without a deliberate edit here.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# The values that turn the cache markers OFF on the OpenAI-compatible path.
# Only an explicit opt-out counts: an unset variable, an empty string, or a
# typo all leave caching ON, because the failure mode of a stray value
# silently disabling the cache is a quiet 8x prefill bill nobody would look
# for. See `openai_seam`'s docstring for why the hatch exists at all.
_CACHING_OFF = frozenset({"0", "false", "no", "off"})


def prompt_caching_enabled() -> bool:
    """Whether `cache_control` markers survive translation to the
    OpenAI-compatible dialect. Read from `COACH_PROMPT_CACHING`, environment
    only, like the key and the model above."""
    return os.environ.get("COACH_PROMPT_CACHING", "").strip().lower() not in _CACHING_OFF


class CoachConfigError(Exception):
    """Raised when `ANTHROPIC_API_KEY` is unset. The message names the
    env var and where to put it -- it never contains a key-shaped value,
    matching `SonioxConfigError`'s discipline exactly."""


def build_client() -> "anthropic.Anthropic":
    """The one construction site (§C4). Reads `ANTHROPIC_API_KEY` from the
    environment only -- never from `app/config.py`, whose own docstring
    already states "nothing here reads an API key". Raises `CoachConfigError`
    with an actionable message if the key is absent; the caller (`app/main.py`
    `start`) is expected to catch this and start with the coach disabled
    rather than refuse to run (SURF-06's documented degrade path)."""
    # An OpenAI-compatible endpoint wins when one is named -- Ollama, vLLM,
    # llama.cpp's server, or any cheap hosted provider speaking that dialect.
    # Kept HERE rather than at the call sites so §C4's "one construction site"
    # still holds with two possible clients behind it: everything downstream
    # goes on taking its client by injection and cannot tell the difference.
    base_url = os.environ.get("COACH_BASE_URL")
    if base_url:
        # No key required: a local runtime usually wants none, and sending an
        # empty bearer token is worse than sending no header at all.
        return openai_seam.OpenAICompatibleClient(
            base_url=base_url,
            api_key=os.environ.get("COACH_API_KEY") or None,
            prompt_caching=prompt_caching_enabled(),
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise CoachConfigError(
            "ANTHROPIC_API_KEY is not set. Read from the environment only -- "
            "put it in the repo's .env file (already git-ignored) or export it "
            "in the shell. See README.md. To use an OpenAI-compatible endpoint "
            "instead (Ollama, vLLM, or a hosted provider), set COACH_BASE_URL "
            "and COACH_MODEL."
        )
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


def resolve_model() -> str:
    """The model id every coach stage should use.

    `DEFAULT_MODEL` is a full dated Anthropic id and is meaningless to another
    provider, so pointing `COACH_BASE_URL` somewhere else without also naming
    a model would send a candidate endpoint an id it has never heard of. Read
    from the environment only, like the key above."""
    return os.environ.get("COACH_MODEL") or DEFAULT_MODEL


def resolve_extract_model() -> str:
    """The model id the OFFLINE extraction stage should use.

    Separate from `resolve_model` for one reason: the live coach must answer
    inside the ~8s architectural cycle, so its model is chosen under a latency
    bound. Extraction runs after the interview has ended, with nobody waiting,
    so a slow strong model there costs money and nothing else -- it is the one
    place in this system where quality is free. `A2` is the reason the knob
    exists (`app/coach/extract.py`'s own docstring): the extraction model
    mislabels inference as evidence, and testing a stronger one used to mean
    moving the live coach too.

    **Defaults to `resolve_model()`, never to `DEFAULT_MODEL`.** Unset, this
    changes nothing at all; and pointing the whole rig at another provider
    with `COACH_BASE_URL`/`COACH_MODEL` carries extraction along with it,
    rather than quietly sending that provider a dated Anthropic id -- the
    exact trap `resolve_model` exists to prevent, which a `DEFAULT_MODEL`
    fallback here would reintroduce one layer down.

    ⚠️ There is ONE client construction site (`build_client`), so `EXTRACT_
    MODEL` must name a model THE SAME ENDPOINT serves. It selects a model, it
    does not select a provider."""
    return os.environ.get("EXTRACT_MODEL") or resolve_model()


def with_timeout(client: "anthropic.Anthropic", timeout_s: float) -> "anthropic.Anthropic":
    """Binds an explicit per-call timeout via the SDK's own per-call options
    override (§C4). Every call site in `app/coach/` must go through this --
    it is what forces each call to name its own bound rather than silently
    inheriting the client's 10-minute default."""
    return client.with_options(timeout=timeout_s)
