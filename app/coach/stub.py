"""The offline stand-in for `anthropic.Anthropic`, mirroring `app/stt/fake.py`'s
role for STT (04-CONTEXT.md §C9). Every coach component that calls the
vendor takes its client by injection; this is what lets `app/coach/coverage.py`
(and 04-02's `generate.py`/`critic.py`) be exercised end to end -- including
timeout handling, parse-failure handling and thread-boundary timing -- with
no network, no credentials, and no cost.

Exposes the SAME call shape the real client does: `.with_options(timeout=...)`
returns a bound object whose `.messages.create(...)` records the call
(model, the bound timeout, and the tool name it was asked to force -- never
asserting on prompt TEXT in a way that would churn every time the prompt
wording changes) and returns the next queued canned response, or raises the
configured exception, or sleeps first. A test that passes against this stub
is therefore testing the real call path in `app/coach/coverage.py`, not a
parallel implementation of it.

Two ways to configure failure: `raises=<exception instance>` makes EVERY
call raise it (for "the vendor is simply down" tests); a `BaseException`
instance placed inside the `responses` queue itself is raised only when ITS
turn comes up, which is what lets a test prove "one failure, then a second,
successful call still produces a match" against the exact same client.

This module must NEVER import `anthropic` -- it has no need to; a fake
response is a plain, duck-typed stand-in (see `ToolUseResponse` below), not
an instance of any real SDK type.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


class StubExhaustedError(Exception):
    """Raised when `.messages.create(...)` is called more times than the
    stub was given canned responses for -- a test bug (an under-primed
    stub), never a condition the product code itself should ever trigger."""


@dataclass(frozen=True)
class ToolUseBlock:
    """Duck-types the one shape `app/coach/coverage.py` reads off a real
    `anthropic` tool-use response block: `.type`, `.name`, `.input`."""

    name: str
    input: dict
    type: str = "tool_use"


@dataclass(frozen=True)
class ToolUseResponse:
    """Duck-types the one shape `app/coach/coverage.py` reads off a real
    `anthropic.types.Message`: `.content` (a list of blocks) and
    `.stop_reason`."""

    content: "tuple[ToolUseBlock, ...]"
    stop_reason: str = "tool_use"


def tool_use_response(tool_name: str, input_payload: dict) -> ToolUseResponse:
    """Builds one canned response carrying a single forced tool-use block --
    the shape every coverage-match call in this plan expects back."""
    return ToolUseResponse(content=(ToolUseBlock(name=tool_name, input=input_payload),))


@dataclass(frozen=True)
class StubCall:
    """One recorded call. Never carries prompt or segment text -- only what
    a real call site is allowed to assert on without re-testing the prompt's
    own wording every time it changes."""

    model: str
    timeout_s: Optional[float]
    tool_name: Optional[str]


class StubAnthropicClient:
    """Queue of canned responses, replayed in order. Thread-safe: the
    coverage detector's worker thread is the caller in production, and tests
    exercise that exact thread boundary."""

    def __init__(
        self,
        *,
        responses: Optional[Sequence[Any]] = None,
        raises: Optional[BaseException] = None,
        sleep_s: float = 0.0,
    ) -> None:
        self._responses: "list[Any]" = list(responses) if responses else []
        self._raises = raises
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self.calls: "list[StubCall]" = []

    def with_options(self, *, timeout: float) -> "_BoundStubClient":
        return _BoundStubClient(self, timeout_s=timeout)

    @property
    def messages(self) -> "_StubMessages":
        # Unbound use (no explicit timeout) -- exercised only by a test that
        # deliberately skips `with_timeout()`, which every real call site
        # in `app/coach/` must never do.
        return _StubMessages(self, timeout_s=None)

    def _record_and_respond(self, *, model: str, tool_name: Optional[str], timeout_s: Optional[float]) -> Any:
        with self._lock:
            self.calls.append(StubCall(model=model, timeout_s=timeout_s, tool_name=tool_name))
            sleep_s = self._sleep_s
            raises = self._raises
            responses = self._responses
        if sleep_s:
            time.sleep(sleep_s)
        if raises is not None:
            raise raises
        if not responses:
            raise StubExhaustedError(
                "StubAnthropicClient has no queued responses left -- prime it with more "
                "`responses=[...]` than the test drives calls."
            )
        with self._lock:
            next_item = self._responses.pop(0)
        if isinstance(next_item, BaseException):
            raise next_item
        return next_item


@dataclass
class _BoundStubClient:
    """Returned by `.with_options(timeout=...)` -- mirrors the real SDK's
    own "bound client" shape closely enough that `with_timeout(client, s)`
    from `app/coach/reasoning_client.py` works identically against this stub
    and the real `anthropic.Anthropic` instance."""

    _client: StubAnthropicClient
    timeout_s: float
    messages: "_StubMessages" = field(init=False)

    def __post_init__(self) -> None:
        self.messages = _StubMessages(self._client, timeout_s=self.timeout_s)


@dataclass
class _StubMessages:
    _client: StubAnthropicClient
    timeout_s: Optional[float]

    def create(
        self,
        *,
        model: str,
        tool_choice: Optional[dict] = None,
        **_kwargs: Any,
    ) -> Any:
        tool_name = tool_choice.get("name") if tool_choice else None
        return self._client._record_and_respond(model=model, tool_name=tool_name, timeout_s=self.timeout_s)
