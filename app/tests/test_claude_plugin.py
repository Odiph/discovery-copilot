"""The Claude Code plugin -- what teaches ANOTHER repository to use this corpus.

`.mcp.json` at the root reaches the server from inside this checkout only: its
interpreter path is relative, and a product repository opened in Claude Code
is somewhere else entirely. The plugin under `integrations/claude-code/` is
the bridge. It carries the server registration, a skill that says how to
reason with the evidence, and the commands that bind a repository to one
project node.

Three properties, each one a way the bridge could fail quietly:

1. **It launches from a foreign working directory.** Claude Code copies an
   installed plugin into its own cache, so neither the plugin's directory nor
   the product repository's is this checkout. The registration therefore
   names the checkout through `DISCOVERY_COPILOT_HOME`, expanded by Claude Code,
   and the end-to-end test below launches it from a temp directory to prove
   the server does not lean on its cwd.
2. **The same no-secret, no-machine-path rules as `.mcp.json`.** The plugin
   is committed, and it is the file most likely to be shared.
3. **The skill never names a tool the server does not have.** A skill that
   tells a model to call `get_quotes` is worse than no skill: the model
   improvises when the call fails.

Not numbered as a phase gate, for the reason `test_mcp_harness.py` gives.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from app.mcp.tools import TOOL_DEFINITIONS
from app.tests.test_mcp_harness import _KEY_SHAPED_VALUE, _SECRET_SHAPED_NAME, _strings

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE_ROOT = REPO_ROOT / "integrations" / "claude-code"
PLUGIN_NAME = "discovery"
PLUGIN_ROOT = MARKETPLACE_ROOT / PLUGIN_NAME
SERVER_NAME = "discovery-copilot"
HOME_VARIABLE = "DISCOVERY_COPILOT_HOME"

# Every backticked `list_*` / `get_*` identifier in the teaching text is read as
# a tool name. Narrow on purpose: prose words never take that shape.
_TOOL_REFERENCE = re.compile(r"`((?:list|get)_[a-z_]+)`")
_EXPANSION = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _read_json(path: Path) -> dict:
    assert path.is_file(), f"{path.relative_to(REPO_ROOT)} is missing"
    return json.loads(path.read_text(encoding="utf-8"))


def _expand(text: str, env: "dict[str, str]") -> str:
    """Claude Code's `${VAR}` / `${VAR:-default}` expansion, for the launch test."""
    def replace(match: "re.Match[str]") -> str:
        name, default = match.group(1), match.group(2)
        if name in env:
            return env[name]
        assert default is not None, f"${{{name}}} is unset and has no default"
        return default
    return _EXPANSION.sub(replace, text)


def _teaching_documents() -> "list[Path]":
    documents = sorted(PLUGIN_ROOT.rglob("*.md"))
    assert documents, "the plugin carries no skill or command text at all"
    return documents


@pytest.fixture(scope="module")
def server_entry() -> dict:
    config = _read_json(PLUGIN_ROOT / ".mcp.json")
    servers = config["mcpServers"]
    assert SERVER_NAME in servers, f"expected a {SERVER_NAME!r} server, got {sorted(servers)}"
    return servers[SERVER_NAME]


# --- the plugin is installable ---------------------------------------------------------


def test_the_marketplace_lists_the_plugin_at_a_directory_that_exists():
    marketplace = _read_json(MARKETPLACE_ROOT / ".claude-plugin" / "marketplace.json")
    entries = {entry["name"]: entry for entry in marketplace["plugins"]}
    assert PLUGIN_NAME in entries, f"the marketplace does not list {PLUGIN_NAME!r}"
    source = (MARKETPLACE_ROOT / entries[PLUGIN_NAME]["source"]).resolve()
    assert source == PLUGIN_ROOT.resolve(), source
    manifest = _read_json(PLUGIN_ROOT / ".claude-plugin" / "plugin.json")
    assert manifest["name"] == PLUGIN_NAME


def test_the_plugin_is_not_swallowed_by_gitignore():
    """Checked with real `git check-ignore`, as for `.mcp.json`."""
    for path in (PLUGIN_ROOT / ".mcp.json", *_teaching_documents()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "check-ignore", "-v", relative],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 1, f"{relative} is git-ignored by {result.stdout.strip()!r}"


# --- the registration --------------------------------------------------------------------


def test_the_registration_names_the_checkout_only_through_the_home_variable(server_entry):
    """Relative paths cannot work here (the plugin runs from Claude Code's
    cache), and an absolute one is the `open-console.bat` failure. What is
    left is a variable the operator sets once."""
    placeholder = "${" + HOME_VARIABLE + "}"
    assert server_entry["command"].startswith(placeholder + "/.venv/"), server_entry["command"]
    assert server_entry["args"] == ["-m", "app.mcp"]
    assert server_entry["env"]["PYTHONPATH"] == placeholder, (
        "without the checkout on PYTHONPATH, `-m app.mcp` cannot import `app` from a "
        "foreign cwd -- the server dies at startup and Claude Code only says it failed"
    )


def test_the_registration_carries_no_machine_path_and_no_secret(server_entry):
    offenders = []
    for path, text in _strings(server_entry):
        if re.match(r"\A[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\\\")) or "\\" in text:
            offenders.append((path, text))
        if _KEY_SHAPED_VALUE.search(text):
            offenders.append((path, "key-shaped value"))
        if path.endswith("(key)") and _SECRET_SHAPED_NAME.search(text):
            offenders.append((path, "credential-shaped name"))
    assert offenders == [], offenders


def test_the_registration_really_speaks_the_protocol_from_a_foreign_directory(
    server_entry, tmp_path
):
    """The end-to-end check: expand the entry the way Claude Code will, then
    launch it from a directory that is NOT this checkout -- which is where a
    product repository always is."""
    venv_home = next(
        (root for root in (REPO_ROOT, *REPO_ROOT.parents)
         if (root / ".venv" / "Scripts" / "python.exe").is_file()
         or (root / ".venv" / "bin" / "python").is_file()),
        None,
    )
    if venv_home is None:
        pytest.skip("no virtualenv in this checkout or its parents")

    # The interpreter may live in the main checkout (a worktree shares it);
    # the code under test is always THIS checkout's, via PYTHONPATH.
    command = Path(_expand(server_entry["command"], {HOME_VARIABLE: venv_home.as_posix()}))
    if not command.is_file():
        command = venv_home / ".venv" / "bin" / "python"
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    env.update({
        name: _expand(value, {HOME_VARIABLE: str(REPO_ROOT)})
        for name, value in server_entry["env"].items()
    })

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "plugin-test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "list_projects", "arguments": {}}},
    ]
    completed = subprocess.run(
        [str(command), *server_entry["args"]],
        cwd=tmp_path, env=env,
        input="".join(json.dumps(request) + "\n" for request in requests).encode("utf-8"),
        capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    lines = [line for line in completed.stdout.decode("utf-8").split("\n") if line]
    responses = [json.loads(line) for line in lines]
    assert [response["id"] for response in responses] == [1, 2]
    listing = json.loads(responses[1]["result"]["content"][0]["text"])
    assert "projects" in listing, (
        "list_projects answered, but not with the project listing -- the server found "
        "no corpus from a foreign cwd"
    )


# --- the teaching text -------------------------------------------------------------------


def _unknown_tool_references(text: str) -> "set[str]":
    known = {definition["name"] for definition in TOOL_DEFINITIONS}
    return set(_TOOL_REFERENCE.findall(text)) - known


def test_the_skill_and_commands_name_only_tools_the_server_has():
    unknown = {
        path.relative_to(PLUGIN_ROOT).as_posix(): sorted(_unknown_tool_references(
            path.read_text(encoding="utf-8")))
        for path in _teaching_documents()
    }
    unknown = {path: names for path, names in unknown.items() if names}
    assert unknown == {}, (
        f"the plugin's text names tools the server does not have: {unknown}. A model told "
        "to call a missing tool improvises instead"
    )
    referenced = set().union(*(
        _TOOL_REFERENCE.findall(path.read_text(encoding="utf-8")) for path in _teaching_documents()
    ))
    assert {"list_projects", "get_standing", "get_facts"} <= referenced, (
        "the skill no longer teaches the three calls every evidence question starts with"
    )


def test_the_tool_name_check_is_not_vacuous():
    assert _unknown_tool_references("call `get_quotes`, then `get_standing`") == {"get_quotes"}


def test_the_skill_teaches_the_corpus_distinction():
    """The reason `074e914` exists: demo claims read exactly like customer
    research. The MCP descriptions say so to a model that reads them; the
    skill has to say so to a model deciding what to put in a PRD."""
    skill = (PLUGIN_ROOT / "skills" / "discovery-evidence" / "SKILL.md").read_text(encoding="utf-8")
    for term in ("demo", "undeclared", "unextracted_projects", "is_scorable", "author"):
        assert term in skill, f"the skill no longer mentions {term!r}"
