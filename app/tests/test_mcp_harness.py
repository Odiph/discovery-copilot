"""`.mcp.json` -- the committed registration that makes the server reachable.

A server nobody can reach is not a feature, so the wiring is code and gets
tests like code. Two standing properties, both about a file that is COMMITTED
and therefore travels to every checkout and every clone:

1. **No secret, ever.** An MCP client config is the natural place to park an
   API key, every vendor's documentation shows one there, and this repository
   already holds the most sensitive plaintext it will ever produce. A key in a
   committed config is a key in the git history forever.
2. **No machine-specific absolute path.** `open-console.bat` hardcodes
   `C:\\projects\\Interview Agent 2.0` and is broken on every other checkout,
   including the worktrees this project is developed in. That mistake is not
   repeated here, and the assertion below is what keeps it from being
   repeated later by someone debugging a path problem at 2am.

Deliberately NOT numbered as a phase gate: the `^def test_gate` prefix is
counted by this project's own plan arithmetic, and these are properties of a
harness file rather than of the product. Same precedent as
`test_quoted_text_scan_reaches_modules_outside_the_coach_package`.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_CONFIG_PATH = REPO_ROOT / ".mcp.json"

SERVER_NAME = "discovery-copilot"


@pytest.fixture(scope="module")
def config() -> dict:
    assert MCP_CONFIG_PATH.is_file(), (
        ".mcp.json is missing -- without it no coding agent can reach this server, "
        "and the tool package is unreachable code"
    )
    return json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))


def _strings(value, path="$"):
    """Every string in the document, with the path it was found at -- keys
    included. A sweep over values alone would miss a key literally NAMED
    `sk-ant-...`, and a sweep that only visited the server entry would miss
    anything parked at the top level."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.{key}(key)", str(key)
            yield from _strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


# --- the registration works ------------------------------------------------------------


def test_the_config_registers_this_server_to_run_the_package_entry_point(config):
    servers = config["mcpServers"]
    assert SERVER_NAME in servers, f"expected a {SERVER_NAME!r} server, got {sorted(servers)}"
    entry = servers[SERVER_NAME]
    assert entry["args"] == ["-m", "app.mcp"], (
        "the entry point is `python -m app.mcp` -- a script path would bypass the package "
        "__main__ that configures logging to stderr, and stdout belongs to the protocol"
    )
    assert entry.get("type", "stdio") == "stdio"


def test_the_interpreter_it_names_is_a_relative_venv_python(config):
    """A bare `python` would be the system interpreter with none of this
    project's pins -- `app.mcp` imports `pydantic` on the first line of its
    tool module, so it would die at startup. `open-console.bat` already
    refuses to start on exactly that mistake, with exactly that comment.

    **Relative on purpose, and the mechanics were measured rather than
    assumed.** MCP clients spawn stdio servers through Node/libuv, which DOES
    resolve a relative command against the `cwd` it passes; Python's own
    `subprocess` does NOT (`CreateProcess` resolves against the caller's
    directory instead). So a relative command is correct for the clients this
    file registers for, and any Python-side test that launches the server has
    to resolve the path itself -- which is what the end-to-end test below
    does. The alternative, an absolute path, is the `open-console.bat`
    failure: correct on one machine and broken in every worktree.

    This asserts the SHAPE, not the existence of the file: a git worktree
    shares the main checkout's virtualenv and legitimately has no `.venv` of
    its own, so an existence assertion here would fail in exactly the place
    this project does its development."""
    command = config["mcpServers"][SERVER_NAME]["command"]
    assert command.split("/")[-1].lower().startswith("python"), command
    assert ".venv" in command, (
        "a bare `python` is a system interpreter with none of this project's pins"
    )
    assert not Path(command).is_absolute(), command
    assert Path(command).parent.name in ("Scripts", "bin"), (
        f"{command!r} does not point at a virtualenv interpreter location"
    )


def _resolved_interpreter(config) -> "Path | None":
    """The interpreter the config names, resolved the way libuv would -- and
    falling back to the main checkout, because a git worktree shares its
    virtualenv rather than owning one."""
    command = config["mcpServers"][SERVER_NAME]["command"]
    for root in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = root / command
        if candidate.is_file():
            return candidate
    return None


def test_the_registered_command_really_speaks_the_protocol(config):
    """The end-to-end check the whole file exists for: launch the server as a
    SUBPROCESS, exactly as the committed config says to, and hold a real
    conversation with it over real pipes.

    Every other MCP test in this repository drives `serve()` over
    `io.BytesIO`. That proves the handlers and the framing, and proves nothing
    at all about whether `python -m app.mcp` starts -- a broken `__main__`, a
    stray `print` in the import graph, or an interpreter without `pydantic`
    are all invisible to an in-memory test and all fatal to the feature."""
    interpreter = _resolved_interpreter(config)
    if interpreter is None:
        pytest.skip("no virtualenv in this checkout or its parents")

    entry = config["mcpServers"][SERVER_NAME]
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "harness-test", "version": "0"}}},
        # A notification, mixed in: it must draw NO response, and a server
        # that answered it would desynchronize every id below.
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    completed = subprocess.run(
        [str(interpreter), *entry["args"]],
        cwd=REPO_ROOT,
        input="".join(json.dumps(request) + "\n" for request in requests).encode("utf-8"),
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b"", (
        "something wrote to stderr on a clean run; more importantly, nothing may ever "
        "write to STDOUT except the protocol"
    )

    lines = [line for line in completed.stdout.decode("utf-8").split("\n") if line]
    assert len(lines) == 2, f"expected one response per request and none for the notification: {lines}"
    responses = [json.loads(line) for line in lines]
    assert [response["id"] for response in responses] == [1, 2]
    assert responses[0]["result"]["serverInfo"]["name"] == SERVER_NAME
    assert [tool["name"] for tool in responses[1]["result"]["tools"]] == [
        "list_projects", "get_survey", "get_hypotheses", "list_interviews", "get_facts",
        "get_standing", "get_interview_summary",
    ]


def test_the_config_names_no_absolute_or_machine_specific_path(config):
    """The `open-console.bat` failure, not repeated. `cd /d "C:\\projects\\
    Interview Agent 2.0"` in that file makes it work on exactly one directory
    on exactly one machine -- it is already broken inside this project's own
    git worktrees."""
    offenders = []
    for path, text in _strings(config):
        if re.match(r"\A[A-Za-z]:[\\/]", text) or text.startswith("/") or text.startswith("\\\\"):
            offenders.append((path, text))
        if "\\" in text:
            offenders.append((path, f"{text} (backslash separator)"))
        if "Interview Agent" in text and "/" in text:
            offenders.append((path, f"{text} (a hardcoded checkout directory)"))
    assert offenders == [], (
        f"the committed MCP config carries machine-specific paths: {offenders}. Keep every "
        "path workspace-relative -- the server is launched with the checkout as its cwd"
    )


# --- no secrets ------------------------------------------------------------------------


# Deliberately not just `sk-`. A test that only knew one vendor's prefix would
# pass on a config carrying a bare 40-character hex token, which is the shape
# most secrets actually have.
_KEY_SHAPED_VALUE = re.compile(
    r"""(
        sk-[A-Za-z0-9_-]{8,}          # OpenAI / Anthropic style
      | sk-ant-[A-Za-z0-9_-]{8,}
      | ghp_[A-Za-z0-9]{8,}           # GitHub
      | xox[baprs]-[A-Za-z0-9-]{8,}   # Slack
      | AIza[A-Za-z0-9_-]{10,}        # Google
      | \b[A-Fa-f0-9]{32,}\b          # a bare hex token
      | \b[A-Za-z0-9+/]{40,}={0,2}\b  # a bare base64 blob
    )""",
    re.VERBOSE,
)

_SECRET_SHAPED_NAME = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|bearer|auth)", re.IGNORECASE
)


def test_the_committed_config_contains_no_key_shaped_value(config):
    """THE failure: a live API key committed to git, where deleting it later
    does not remove it. `requirements.txt` pins `anthropic`, so this project
    HAS a key on the operator's machine, and an MCP config with an `env` block
    is precisely where a future contributor will be tempted to put it."""
    offenders = [
        (path, text) for path, text in _strings(config) if _KEY_SHAPED_VALUE.search(text)
    ]
    assert offenders == [], (
        f"the committed MCP config carries a key-shaped value at {[p for p, _ in offenders]}. "
        "This file is in git history forever -- the server needs no credential of any kind, "
        "and any it ever did need belongs in the operator's environment"
    )


def test_no_config_key_is_even_NAMED_like_a_credential(config):
    """The version of the test above that fires BEFORE a real key lands.

    A value-shape scan only fires once a genuine secret has been typed in --
    by which time it is already committed. This one fires on the empty
    `"ANTHROPIC_API_KEY": ""` placeholder that a contributor adds first,
    which is the moment the mistake is still free to undo."""
    offenders = [
        path for path, text in _strings(config)
        if path.endswith("(key)") and _SECRET_SHAPED_NAME.search(text)
    ]
    assert offenders == [], (
        f"the committed MCP config names a credential-shaped setting at {offenders}. "
        "Even an empty placeholder is an invitation to fill it in and commit it"
    )


def test_the_secret_scan_is_not_vacuous():
    """The mutation both tests above cannot survive without: a scan whose
    patterns never match anything passes on any file at all, forever."""
    poisoned = {
        "mcpServers": {
            SERVER_NAME: {
                "command": ".venv/Scripts/python.exe",
                "args": ["-m", "app.mcp"],
                "env": {"ANTHROPIC_API_KEY": "sk-ant-api03-NOTAREALKEY0123456789abcdef"},
            }
        }
    }
    values = [text for path, text in _strings(poisoned) if not path.endswith("(key)")]
    assert any(_KEY_SHAPED_VALUE.search(text) for text in values), (
        "the value-shape scan does not recognize a real key shape"
    )
    names = [text for path, text in _strings(poisoned) if path.endswith("(key)")]
    assert any(_SECRET_SHAPED_NAME.search(text) for text in names), (
        "the name scan does not recognize a credential-shaped setting name"
    )
    # And the absolute-path scan, poisoned the same way.
    hardcoded = {"mcpServers": {SERVER_NAME: {"command": "C:/projects/Interview Agent 2.0/x"}}}
    assert any(
        re.match(r"\A[A-Za-z]:[\\/]", text)
        for _path, text in _strings(hardcoded)
    ), "the absolute-path scan does not recognize a drive-letter path"


# --- it is actually committable ---------------------------------------------------------


def test_the_config_is_not_swallowed_by_gitignore():
    """A registration file that git ignores is a registration file that
    reaches nobody else's checkout -- the feature would work only on the
    machine it was written on, and silently. Checked with real
    `git check-ignore` rather than by reading the patterns, for the reason
    gate 29 already states: pattern anchoring is exactly the part humans get
    wrong (`context/*` is root-anchored and does not follow into
    `projects/`; `sessions/` is unanchored and does)."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v", ".mcp.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, (
        f".mcp.json is git-ignored by: {result.stdout.strip()!r} -- it would never reach "
        "another checkout, and the server would appear unregistered to everyone but its author"
    )


def test_the_readme_documents_the_claude_desktop_registration():
    """Claude Code reads `.mcp.json` from the repository root; Claude Desktop
    does not -- it reads one global config, so it needs an ABSOLUTE path and
    a documented cwd. That asymmetry is the whole reason the README section
    has to exist, and this pins that it does."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Reaching the corpus from another AI agent (MCP)")
    assert len(section) == 2, "the README lost its MCP registration section"
    body = section[1].split("\n## ")[0]
    assert "claude_desktop_config.json" in body
    assert ".mcp.json" in body

    # PARSED, not grepped. The first version of this test asserted `"cwd" in
    # body` and a mutation that deleted the `cwd` line from the JSON example
    # SURVIVED it -- because the surrounding prose says the word "cwd" too, so
    # the guard was satisfied by the explanation of the thing it was meant to
    # check. A registration copied from a README missing that key fails at
    # startup with an import error the operator sees only in Claude Desktop's
    # own log. So the example itself is parsed and inspected.
    blocks = []
    for chunk in body.split("```json")[1:]:
        try:
            blocks.append(json.loads(chunk.split("```")[0]))
        except json.JSONDecodeError as exc:  # a copy-pasteable example must parse
            pytest.fail(f"a JSON example in the MCP README section is not valid JSON: {exc}")
    assert blocks, "the MCP section shows no JSON registration example at all"

    entries = [block["mcpServers"][SERVER_NAME] for block in blocks]
    assert all(entry["args"] == ["-m", "app.mcp"] for entry in entries), entries
    desktop = [entry for entry in entries if "cwd" in entry]
    assert desktop, (
        "no README example carries a `cwd`. Claude Desktop launches the server from an "
        "arbitrary directory and `python -m app.mcp` needs the repository root on sys.path "
        "-- without that key the registration fails at startup with an import error"
    )
    assert all(entry["cwd"] in entry["command"] for entry in desktop), (
        "the Desktop example's interpreter must live inside the cwd it names, or the two "
        "halves of the example are describing different checkouts"
    )
