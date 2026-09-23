"""An OpenAI-compatible endpoint wearing the shape this codebase's coach expects.

**Why this exists.** `run_generate`/`run_critique` take their client by
injection (D1/D2) and `app/coach/stub.py` already proves the seam works with
something that is not the Anthropic SDK. This module is the second
implementation of that seam: it speaks `/v1/chat/completions` to anything
OpenAI-compatible -- Ollama, vLLM, llama.cpp's server, Groq, Together,
OpenRouter -- while presenting the `.with_options(timeout=).messages.create()`
surface and the Anthropic-shaped response the coach parses.

**One dialect, not many.** OpenAI-compatible is the common tongue of both the
local runtimes and the cheap hosted providers, so a single translation reaches
the whole candidate set without a dependency. It uses `httpx`, which is
already installed as the Anthropic SDK's own transitive dependency.

**The translation that actually matters is the tool call.** Everything else
here is plumbing. A coach cycle forces exactly one tool (`tool_choice` names
it) with a strict schema, and `generate.py`/`critic.py` read the result out of
a `tool_use` block. If a provider returns prose instead of a tool call, or
arguments that are not valid JSON, that must surface as a REFUSAL to parse --
never as a silently empty candidate list. `GenerateFailure` is the correct
outcome there and the pipeline already treats it as a health failure.

**`finish_reason` mapping is load-bearing, not cosmetic.** `critic.py` treats
`max_tokens` as an untrusted stop reason and refuses to parse the body. So
OpenAI's `length` MUST map to `max_tokens`, or a truncated tool call gets
parsed as though it were complete -- which is the exact failure the untrusted
list exists to prevent.

**Prompt caching survives the translation, and the reason it did not used to
is worth recording.** This docstring used to say `cache_control` was dropped
because "the OpenAI schema has no equivalent". That was wrong, and being wrong
about the REASON is what kept it unfixed: it framed an accident as a
constraint. OpenAI's own schema takes message `content` as either a string or
an array of content parts, and OpenRouter carries an Anthropic-style
`cache_control` key on those parts straight through to the upstream provider.
Nothing about the dialect forbade caching.

What actually dropped the markers was `_system_content`'s ancestor, which
flattened the system block list into ONE string with `"\n\n".join(...)`. The
markers live on the blocks, so joining the blocks deleted them -- silently,
with no error and no log line, which is why `benchmark/eval/cost_model.py`
went on modelling a cache that was never requested.

So the blocks now leave as an array whenever any of them carries a marker.
A provider that does not understand `cache_control` sees a spec-legal array
of text parts and ignores one extra key; a provider that does understand it
caches. `COACH_PROMPT_CACHING=0` forces the old flattened string back for a
runtime strict enough to reject the unknown key -- the escape hatch exists
because this reaches local servers (Ollama, vLLM, llama.cpp) that the hosted
providers' behaviour says nothing about.

⚠️ **Emitting the markers is not evidence they worked.** Only the response's
`usage.prompt_tokens_details.cached_tokens` says a cache was READ, and a
prefix shorter than the model's minimum cacheable length is accepted and
silently not cached (Gemini 2.5 Flash's floor is 1,024 tokens, and
`prompts.MIN_STATIC_BLOCK_CHARS = 4000` chars is ~1,000 -- right on it).
`cached_tokens` is surfaced on the adapted response for exactly this reason:
so the question is answerable from a real call instead of assumed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

# OpenAI's finish_reason -> the Anthropic stop_reason the coach's parsers read.
# `length` -> `max_tokens` is the load-bearing row; see the module docstring.
_FINISH_REASON = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class ToolUseBlock:
    """The shape `_parse_generate_response` / `_parse_critique_response` read."""

    type = "tool_use"

    def __init__(self, *, name: str, input: dict) -> None:
        self.name = name
        self.input = input


class TextBlock:
    type = "text"

    def __init__(self, *, text: str) -> None:
        self.text = text


class AdaptedResponse:
    def __init__(
        self,
        *,
        content: "list[Any]",
        stop_reason: Optional[str],
        cached_tokens: Optional[int] = None,
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        # The ONLY honest answer to "did the prefix cache actually hit?".
        # None means the provider said nothing about it -- which is NOT the
        # same as zero, and must not be reported as a miss.
        self.cached_tokens = cached_tokens
        # Present so a caller that guards on it (critic.py does) finds the
        # attribute rather than raising AttributeError on the guard itself.
        self.stop_details = None


def _flatten_system(system: Any) -> str:
    """The blocks as ONE string, cache markers gone. This is what the seam
    used to do unconditionally; it survives as the `prompt_caching=False`
    path and as the shape used when no block asked to be cached."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    return "\n\n".join(block["text"] for block in system)


def _system_content(system: Any, *, prompt_caching: bool) -> Any:
    """The `content` for the system message: a plain string, or an array of
    content parts when a block carries `cache_control` and caching is on.

    The string is returned whenever it loses nothing -- no blocks, a system
    that was already a string, no markers present, or caching switched off.
    That keeps the request byte-identical to the pre-caching seam for every
    caller that was never going to cache anyway, so the array form (and
    whatever a strict provider makes of it) is only ever sent on behalf of a
    caller that actually asked for a cache breakpoint."""
    if system is None or isinstance(system, str) or not system:
        return _flatten_system(system)
    if not prompt_caching or not any("cache_control" in block for block in system):
        return _flatten_system(system)
    parts: "list[dict]" = []
    for block in system:
        part: "dict[str, Any]" = {"type": "text", "text": block["text"]}
        marker = block.get("cache_control")
        if marker is not None:
            # Passed through verbatim rather than rebuilt. The marker's shape
            # is the upstream provider's business, not this seam's, and
            # inventing a canonical form here would silently rewrite a caller
            # that had a reason to send something else.
            part["cache_control"] = marker
        parts.append(part)
    return parts


def _translate_tools(tools: Sequence[dict]) -> "list[dict]":
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                # Anthropic calls it `input_schema`; OpenAI calls it
                # `parameters`. Same JSON Schema underneath -- passed through
                # unchanged so a strict schema stays strict.
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


def _translate_tool_choice(tool_choice: Optional[dict]) -> Any:
    if not tool_choice:
        return None
    if tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if tool_choice.get("type") == "any":
        return "required"
    if tool_choice.get("type") == "none":
        return "none"
    return "auto"


def _parse_arguments(raw: Any) -> Optional[dict]:
    """Tool arguments arrive as a JSON *string*. A provider that emits
    malformed JSON, or an object where a string was promised, must not be
    smoothed over -- returning None makes the caller raise its own typed
    failure instead of inventing an empty tool input."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _log_cache_usage(payload: dict, adapted: "AdaptedResponse") -> None:
    """Say, once per call, whether the prefix cache was actually READ.

    This exists because the caching fix was half-finished without it. The
    markers were being emitted and `cached_tokens` was being carried off the
    response, and NOTHING recorded it -- so the value was reachable in code and
    invisible in practice, and a real interview still could not answer "did
    caching work". The gap survived a live session before it was noticed.

    COUNTS ONLY, never a prompt or a candidate: this module sits under
    `app/coach/`, where gate 12 forbids putting transcript-derived text into a
    log line. Nothing here is derived from what anyone said.

    Silent when the provider reports no usage at all -- a line saying
    `cached=None` on every call of a provider that never reports it would train
    the reader to ignore the one place the answer appears. `None` and `0` stay
    distinct everywhere else for the same reason.
    """
    if adapted.cached_tokens is None:
        return
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    logger.info(
        "prefix cache prompt_tokens=%s cached_tokens=%d",
        prompt_tokens if isinstance(prompt_tokens, int) else "unreported",
        adapted.cached_tokens,
    )


def _cached_tokens(payload: dict) -> Optional[int]:
    """`usage.prompt_tokens_details.cached_tokens`, or None if the provider
    did not report it. Anything non-integer reads as "not reported" rather
    than as a count -- a string here would otherwise become a number that
    looks measured."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    # bool is an int subclass; `True` is not a token count.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _translate_response(payload: dict) -> AdaptedResponse:
    cached_tokens = _cached_tokens(payload)
    choices = payload.get("choices") or []
    if not choices:
        return AdaptedResponse(
            content=[], stop_reason=None, cached_tokens=cached_tokens
        )
    choice = choices[0]
    message = choice.get("message") or {}
    finish = choice.get("finish_reason")
    stop_reason = _FINISH_REASON.get(finish, finish)

    content: "list[Any]" = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = _parse_arguments(function.get("arguments"))
        if arguments is None:
            # Unparseable arguments are NOT a tool call. Emitting a block with
            # an empty input would look to the parser like a model that
            # returned no candidates, which is a different and much quieter
            # failure than the one that actually happened.
            continue
        content.append(ToolUseBlock(name=function.get("name", ""), input=arguments))
    text = message.get("content")
    if text:
        content.append(TextBlock(text=text))
    return AdaptedResponse(
        content=content, stop_reason=stop_reason, cached_tokens=cached_tokens
    )


class _Messages:
    def __init__(self, client: "OpenAICompatibleClient", timeout_s: Optional[float]) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def create(self, **kwargs: Any) -> AdaptedResponse:
        system = _system_content(
            kwargs.get("system"), prompt_caching=self._client.prompt_caching
        )
        body: "dict[str, Any]" = {
            "model": kwargs["model"],
            "max_tokens": kwargs.get("max_tokens"),
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [
                    {"role": m["role"], "content": m["content"]}
                    for m in kwargs.get("messages", [])
                ]
            ),
        }
        tools = kwargs.get("tools")
        if tools:
            body["tools"] = _translate_tools(tools)
            choice = _translate_tool_choice(kwargs.get("tool_choice"))
            if choice is not None:
                body["tool_choice"] = choice
        payload = self._client.post(body, timeout_s=self._timeout_s)
        adapted = _translate_response(payload)
        _log_cache_usage(payload, adapted)
        return adapted


class _Bound:
    def __init__(self, client: "OpenAICompatibleClient", timeout_s: Optional[float]) -> None:
        self._client = client
        self._timeout_s = timeout_s

    @property
    def messages(self) -> _Messages:
        return _Messages(self._client, self._timeout_s)


class OpenAICompatibleClient:
    """Drop-in for the Anthropic client at this codebase's coach seam.

    `transport` is injected so the whole translation is testable with no
    network and no running model -- the same discipline `StubAnthropicClient`
    already applies to the Anthropic path."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        transport: Optional[Any] = None,
        prompt_caching: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport
        # Public because `_Messages` reads it through the client it is bound
        # to, and a bound-then-rebound client must not be able to disagree
        # with itself about whether the markers go out.
        self.prompt_caching = prompt_caching

    def with_options(self, *, timeout: float) -> _Bound:
        """Mirrors `reasoning_client.with_timeout`'s contract: every coach call
        site must name its own bound rather than inherit a default."""
        return _Bound(self, timeout)

    @property
    def messages(self) -> _Messages:
        # Unbound use, matching StubAnthropicClient: only a test that
        # deliberately skips `with_timeout()` ever reaches this.
        return _Messages(self, None)

    def post(self, body: dict, *, timeout_s: Optional[float]) -> dict:
        if self._transport is not None:
            return self._transport(body, timeout_s)
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()
