"""Which client `build_client()` returns, and which model id goes with it.

The seam only helps if the ONE construction site actually honours the
environment. These pin that, and pin the trap that comes with it: pointing the
coach at another provider while still sending Anthropic's dated model id.
"""

from __future__ import annotations

import pytest

from app.coach import reasoning_client
from app.coach.openai_seam import OpenAICompatibleClient


def test_an_openai_compatible_endpoint_is_used_when_named(monkeypatch):
    monkeypatch.setenv("COACH_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("COACH_API_KEY", raising=False)
    client = reasoning_client.build_client()
    assert isinstance(client, OpenAICompatibleClient)


def test_a_local_endpoint_needs_no_key_and_no_anthropic_key(monkeypatch):
    """The whole point of the local path. If this required ANTHROPIC_API_KEY
    the local option would be unusable exactly when it is most wanted."""
    monkeypatch.setenv("COACH_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("COACH_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = reasoning_client.build_client()
    assert isinstance(client, OpenAICompatibleClient)
    assert client._api_key is None


def test_the_anthropic_path_is_unchanged_when_no_endpoint_is_named(monkeypatch):
    monkeypatch.delenv("COACH_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    client = reasoning_client.build_client()
    assert not isinstance(client, OpenAICompatibleClient)


def test_a_missing_anthropic_key_still_refuses_and_now_mentions_the_alternative(monkeypatch):
    monkeypatch.delenv("COACH_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(reasoning_client.CoachConfigError) as excinfo:
        reasoning_client.build_client()
    assert "COACH_BASE_URL" in str(excinfo.value)


def test_the_model_id_follows_the_endpoint(monkeypatch):
    """The trap this exists to prevent: a candidate endpoint receiving
    `claude-haiku-4-5-20251001`, which it has never heard of."""
    monkeypatch.setenv("COACH_MODEL", "qwen2.5:3b-instruct")
    assert reasoning_client.resolve_model() == "qwen2.5:3b-instruct"


def test_the_model_id_defaults_to_the_pinned_anthropic_one(monkeypatch):
    monkeypatch.delenv("COACH_MODEL", raising=False)
    assert reasoning_client.resolve_model() == reasoning_client.DEFAULT_MODEL


# --- EXTRACT_MODEL: extraction is OFFLINE, so it may buy quality with latency ----------
#
# The live coach must answer inside the ~8s architectural cycle, so its model
# choice is latency-bound. Extraction runs after the interview is over, with
# nobody waiting, so a slow strong model there costs money and NOTHING else.
# Before this seam both read one id and could not be moved independently.


def test_the_extraction_model_defaults_to_the_coachs_own_so_nothing_changes_unset(monkeypatch):
    monkeypatch.delenv("EXTRACT_MODEL", raising=False)
    monkeypatch.delenv("COACH_MODEL", raising=False)
    assert reasoning_client.resolve_extract_model() == reasoning_client.resolve_model()


def test_the_extraction_model_defaults_to_the_coachs_own_even_when_that_is_overridden(monkeypatch):
    # The default that matters: pointing the whole rig at another provider
    # must still carry extraction along with it, or setting COACH_BASE_URL
    # alone would send that provider an Anthropic id -- the exact trap
    # `resolve_model` exists to prevent, reintroduced one layer down.
    monkeypatch.delenv("EXTRACT_MODEL", raising=False)
    monkeypatch.setenv("COACH_MODEL", "qwen2.5:3b-instruct")
    assert reasoning_client.resolve_extract_model() == "qwen2.5:3b-instruct"


def test_the_extraction_model_can_be_moved_without_moving_the_live_coachs(monkeypatch):
    monkeypatch.setenv("COACH_MODEL", "fast-cheap-model")
    monkeypatch.setenv("EXTRACT_MODEL", "slow-strong-model")
    assert reasoning_client.resolve_extract_model() == "slow-strong-model"
    assert reasoning_client.resolve_model() == "fast-cheap-model"


def test_extraction_asks_for_the_extraction_model_not_the_coachs(monkeypatch, tmp_path):
    """A seam nothing calls is a seam that does not exist. This pins the CLI
    call site, not just the resolver -- `app/main.py` read `resolve_model()`
    directly before this change."""
    import ast
    import inspect
    import textwrap

    from app import main as main_mod

    source = textwrap.dedent(inspect.getsource(main_mod.run_extract))
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "resolve_extract_model" in called
    assert "resolve_model" not in called, (
        "run_extract still resolves the LIVE COACH's model -- the two ids are "
        "coupled again and EXTRACT_MODEL is decorative"
    )
