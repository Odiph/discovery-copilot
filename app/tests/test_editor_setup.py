"""The checked-in editor setup has to work on a machine that is not this one.

Everything under `.vscode/` and every launcher batch file is *committed*, which
means it is a promise made to someone who has never seen this repository. Each
guard below names the concrete failure it exists to prevent -- most of them
observed, not imagined:

* `open-console.bat` shipped `cd /d "C:\\projects\\Interview Agent 2.0"` and so
  worked on exactly one laptop.
* The Test Explorer's "Run All Tests" button would fire `live_reasoning` --
  a real, billed call to the reasoning vendor -- unless the marker filter
  deselects it.
* `sessions/` and `projects/` hold verbatim customer speech. They are
  gitignored, but gitignore does not stop VS Code's search index or the
  Copilot-class extensions that read it.

The guards are written to catch the *sneaky* version of each break, not the
obvious one: a glob narrowed from `projects/**` to `projects/*`, an exclude
flipped from `true` to `false` with the key left in place, a marker expression
that mentions `live_reasoning` while *selecting* it, a drive letter smuggled in
as `${env:HOMEDRIVE}`, a path escaping the workspace via `${workspaceFolder}/..`.

VS Code reads JSON-with-comments (jsonc), and these files use comments heavily
-- the comments are where the reasons live. So this module carries its own
string-aware comment stripper rather than pretending the files are plain JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VSCODE_DIR = REPO_ROOT / ".vscode"

# Batch files this agent owns and has made workspace-relative. Anything else
# matching root/*.bat is a NEW file that has not been through these guards --
# `test_every_batch_file_is_accounted_for` fails on it rather than letting it
# slip past unscanned. That is the difference between a guard and a allowlist.
IN_SCOPE_BATCH_FILES = (
    "open-console.bat",
    "start-overlay.bat",
    # Moved out of the known-unscanned list once its hardcoded
    # `cd /d "C:\\projects\\Interview Agent 2.0"` was replaced with `%~dp0`.
    # An exemption that outlives the reason for it is worse than no exemption:
    # the guard goes on reporting full coverage while quietly not covering a
    # file nobody remembers excluding.
    "record-test-session.bat",
)

# Deliberately EMPTY, and kept rather than deleted. An empty tuple states that
# nothing is currently exempt; removing the concept would let the next
# genuinely-unscannable file be added to IN_SCOPE without anyone having to
# write down why it cannot be scanned.
KNOWN_UNSCANNED_BATCH_FILES: "tuple[str, ...]" = ()

OFFLINE_MARKERS = ("live", "hardware", "live_reasoning")

# Paths that must never be reachable from the editor's search index: each one
# is (or sits under) verbatim customer speech.
SENSITIVE_PATHS = (
    "sessions/2026-08-12T19-00-00/transcript.md",
    "projects/acme/context/facts.json",
    "projects/acme/sub-node/sessions/s1/session.json",
)


# --------------------------------------------------------------------------
# jsonc
# --------------------------------------------------------------------------
def strip_jsonc_comments(text: str) -> str:
    """Removes `//` and `/* */` comments, leaving string literals alone.

    A naive `re.sub(r"//.*", "", text)` would corrupt any value containing a
    URL or a `//` inside a string, and would silently change what this module
    believes the file says. This walks the text instead.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\":
                    if i + 1 < n:
                        out.append(text[i + 1])
                        i += 2
                        continue
                elif text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_jsonc(path: Path) -> object:
    return json.loads(strip_jsonc_comments(path.read_text(encoding="utf-8")))


def walk_strings(node: object) -> list[str]:
    """Every string anywhere in a parsed document, keys included.

    Keys included on purpose: an absolute path can hide in a `search.exclude`
    key just as easily as in a task argument, and a scan of values only would
    miss it.
    """
    found: list[str] = []
    if isinstance(node, str):
        found.append(node)
    elif isinstance(node, dict):
        for key, value in node.items():
            found.append(key)
            found.extend(walk_strings(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(walk_strings(value))
    return found


# --------------------------------------------------------------------------
# The files under guard
# --------------------------------------------------------------------------
def committed_json_files() -> list[Path]:
    assert VSCODE_DIR.is_dir(), ".vscode/ is missing -- the editor setup is not committed"
    return sorted(p for p in VSCODE_DIR.rglob("*") if p.is_file())


def committed_batch_files() -> list[Path]:
    return [REPO_ROOT / name for name in IN_SCOPE_BATCH_FILES]


def guarded_files() -> list[Path]:
    return committed_json_files() + committed_batch_files()


def scan_units(path: Path) -> list[tuple[str, str]]:
    """(label, text) pairs to run the text guards over, for one file.

    For a JSON file this is BOTH the raw source -- comments and all, because a
    comment naming a dead absolute path misleads a newcomer exactly as much as
    a value does -- and every parsed string, because JSON escaping (`C:\\\\x`)
    changes what the raw text looks like.
    """
    raw = path.read_text(encoding="utf-8")
    units = [(f"{path.name} (raw source)", raw)]
    if path.suffix == ".json":
        for value in walk_strings(json.loads(strip_jsonc_comments(raw))):
            units.append((f"{path.name} (parsed value {value!r})", value))
    return units


# --------------------------------------------------------------------------
# Guard: no absolute or machine-specific path
# Prevents: the open-console.bat bug -- a checkout that works on exactly one
# machine, failing on every other with a bewildering error.
# --------------------------------------------------------------------------
# `(?<![A-Za-z])` keeps URL schemes out (`http://` would otherwise match on
# its `p:/`) while still catching a bare drive letter in either slash flavour.
DRIVE_LETTER = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
# The MSYS/Git-Bash spelling of the same thing, anchored so an ordinary
# `app/b/c` path segment cannot trip it.
MSYS_ROOT = re.compile(r"""(?:^|[\s"'=(])/[A-Za-z]/""")
# A UNC share is absolute too, and points off the machine entirely.
UNC_SHARE = re.compile(r"(?:^|[\s\"'=(])\\\\[A-Za-z0-9]")
# Substitutions that resolve to a machine-specific absolute path without ever
# writing a drive letter. This is how a hardcoded path comes back.
MACHINE_SPECIFIC = (
    "${env:",
    "${userHome}",
    "%USERPROFILE%",
    "%HOMEDRIVE%",
    "%HOMEPATH%",
    "%APPDATA%",
    "%LOCALAPPDATA%",
)


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: p.name)
def test_no_absolute_or_machine_specific_path(path: Path) -> None:
    for label, text in scan_units(path):
        hit = DRIVE_LETTER.search(text)
        assert hit is None, f"{label}: drive-letter path {text[max(0, hit.start() - 30):hit.end() + 40]!r}"
        hit = MSYS_ROOT.search(text)
        assert hit is None, f"{label}: absolute MSYS path {text[max(0, hit.start() - 30):hit.end() + 40]!r}"
        hit = UNC_SHARE.search(text)
        assert hit is None, f"{label}: UNC path {text[max(0, hit.start() - 30):hit.end() + 40]!r}"
        for token in MACHINE_SPECIFIC:
            assert token not in text, f"{label}: machine-specific substitution {token!r}"


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: p.name)
def test_nothing_escapes_the_workspace_folder(path: Path) -> None:
    """`${workspaceFolder}/../.venv` is workspace-relative in form and absolute
    in effect -- it names a directory outside the checkout. Prevents a path
    that passes the drive-letter guard and still only works on one machine."""
    for label, text in scan_units(path):
        assert "${workspaceFolder}/.." not in text, f"{label}: path escapes the workspace"
        assert "${workspaceFolder}\\.." not in text, f"{label}: path escapes the workspace"


def test_every_batch_file_is_accounted_for() -> None:
    """Prevents the hole this guard could otherwise have: a NEW launcher .bat
    added later, carrying a fresh hardcoded path, that nothing scans."""
    on_disk = {p.name for p in REPO_ROOT.glob("*.bat")}
    accounted = set(IN_SCOPE_BATCH_FILES) | set(KNOWN_UNSCANNED_BATCH_FILES)
    assert on_disk <= accounted, (
        f"unscanned batch file(s) {sorted(on_disk - accounted)} -- add them to "
        "IN_SCOPE_BATCH_FILES so the path and secret guards cover them"
    )
    for name in IN_SCOPE_BATCH_FILES:
        assert (REPO_ROOT / name).is_file(), f"{name} is missing"


def test_batch_files_locate_themselves_relatively() -> None:
    """Prevents a launcher that runs from the wrong directory: %~dp0 is the
    only spelling of "where this file lives" that survives a different
    checkout path."""
    for path in committed_batch_files():
        text = path.read_text(encoding="utf-8")
        assert "%~dp0" in text, f"{path.name}: no %~dp0 -- it cannot find its own checkout"
        for line in text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("cd ") or stripped.startswith("pushd "):
                assert "%~dp0" in line, f"{path.name}: {line.strip()!r} does not use %~dp0"


def test_overlay_launcher_keeps_a_real_console() -> None:
    """Prevents the STOP button silently not stopping anything.

    The overlay stops an interview with GenerateConsoleCtrlEvent, which
    requires the calling process to OWN A CONSOLE. `pythonw.exe` has none, and
    a hidden window (`start /b`, a VBS `WindowStyle 0` shim) is the tempting
    way to get the console out of the way. Minimized is the only correct
    answer: the console still exists, it is just not in the way.
    """
    raw = (REPO_ROOT / "start-overlay.bat").read_text(encoding="utf-8")
    # REM lines are stripped before the executable checks: the file EXPLAINS
    # why pythonw.exe is wrong, and a guard that cannot tell an explanation
    # from an instruction would forbid saying so.
    text = "\n".join(
        line for line in raw.splitlines() if not line.strip().lower().startswith("rem ")
    )
    assert "pythonw" not in text.lower(), "pythonw.exe has no console -- STOP would break"
    assert re.search(r"(?i)\bstart\b[^\n]*\B/min\b", text), "the launcher does not start minimized"
    assert not re.search(r"(?i)\bstart\b[^\n]*\B/b\b", text), "`start /b` hides the console entirely"
    assert not re.search(r"(?i)wscript|cscript|windowstyle", text), (
        "a script-host shim would hide the console instead of minimizing it"
    )


# --------------------------------------------------------------------------
# Guard: the pytest marker filter really is offline
# Prevents: one click of "Run All Tests" in the Test Explorer making a paid
# call to the reasoning vendor, or failing on a machine with no microphone.
# --------------------------------------------------------------------------
class _MarkerNamespace:
    """Marker names resolve to whether that marker is on this imaginary test."""

    def __init__(self, active: tuple[str, ...]) -> None:
        self._active = set(active)

    def __getitem__(self, name: str) -> bool:
        if name.startswith("__"):
            raise KeyError(name)
        return name in self._active


def evaluate_marker_expression(expression: str, active: tuple[str, ...]) -> bool:
    """Would pytest select a test carrying exactly `active`?

    Evaluating the expression is the whole point. A substring check would pass
    `-m "live_reasoning"` -- which SELECTS the paid tests and nothing else --
    because the string is right there in it.
    """
    return bool(eval(compile(expression, "<marker expression>", "eval"), {"__builtins__": {}}, _MarkerNamespace(active)))  # noqa: S307


def marker_expressions() -> list[tuple[str, str]]:
    """Every offline marker filter committed anywhere in `.vscode/`."""
    found: list[tuple[str, str]] = []

    settings = load_jsonc(VSCODE_DIR / "settings.json")
    args = settings.get("python.testing.pytestArgs")
    assert isinstance(args, list) and args, "settings.json declares no python.testing.pytestArgs"
    assert "-m" in args, "settings.json's pytestArgs carry no marker filter at all"
    found.append(("settings.json pytestArgs", args[args.index("-m") + 1]))

    tasks = load_jsonc(VSCODE_DIR / "tasks.json")
    for task in tasks.get("tasks", []):
        task_args = task.get("args") or []
        if "pytest" not in task_args:
            continue
        # Only the `-m` AFTER `pytest`: the earlier one is `python -m pytest`.
        tail = task_args[task_args.index("pytest") + 1 :]
        assert "-m" in tail, f"task {task['label']!r} runs pytest with no marker filter"
        found.append((f"tasks.json {task['label']!r}", tail[tail.index("-m") + 1]))

    assert len(found) >= 2, "the marker filter is missing from settings.json or from the test task"
    return found


@pytest.mark.parametrize("label,expression", marker_expressions(), ids=lambda v: None)
def test_marker_filter_deselects_every_costly_marker(label: str, expression: str) -> None:
    for marker in OFFLINE_MARKERS:
        assert not evaluate_marker_expression(expression, (marker,)), (
            f"{label}: {expression!r} would RUN a `{marker}` test -- that costs money, "
            "needs a funded key, or needs real audio hardware"
        )
    assert not evaluate_marker_expression(expression, OFFLINE_MARKERS), (
        f"{label}: {expression!r} would run a test carrying all three markers"
    )


@pytest.mark.parametrize("label,expression", marker_expressions(), ids=lambda v: None)
def test_marker_filter_still_selects_the_offline_suite(label: str, expression: str) -> None:
    """Prevents the lazy fix: an expression that deselects EVERYTHING (a
    nonexistent marker name, `not live or live`) is offline-safe and useless,
    and would leave a newcomer with a green Test Explorer running nothing."""
    assert evaluate_marker_expression(expression, ()), (
        f"{label}: {expression!r} selects no ordinary test at all"
    )


def test_pytest_is_actually_enabled() -> None:
    """A marker filter nobody applies is not a guard. If pytest is off, the
    Test Explorer is empty and whoever turns it on gets no filter with it."""
    settings = load_jsonc(VSCODE_DIR / "settings.json")
    assert settings.get("python.testing.pytestEnabled") is True


# --------------------------------------------------------------------------
# Guard: customer speech is out of the editor's index
# Prevents: verbatim interview speech reaching VS Code's search index -- and
# through it any Copilot-class extension that grounds itself in "workspace
# context" -- despite being gitignored.
# --------------------------------------------------------------------------
def glob_to_regex(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    if "/" not in pattern and "*" not in pattern and "?" not in pattern:
        # VS Code treats a bare name as "this folder, at any depth".
        return re.compile(rf"^(?:.*/)?{re.escape(pattern)}(?:/.*)?$")
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "{":
            end = pattern.index("}", i)
            alts = pattern[i + 1 : end].split(",")
            out.append("(?:" + "|".join(re.escape(a) for a in alts) + ")")
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


@pytest.mark.parametrize("setting", ["search.exclude", "files.watcherExclude"])
@pytest.mark.parametrize("sensitive", SENSITIVE_PATHS)
def test_customer_speech_is_excluded(setting: str, sensitive: str) -> None:
    """Checks COVERAGE of a real sensitive path, not the presence of a key.

    The two sneaky breaks this is built for: flipping `"projects/**": true` to
    `false` (the key is still there, so a key-presence check passes and the
    directory is indexed anyway), and narrowing `projects/**` to `projects/*`
    (which matches the folder's direct children and none of the nested
    context/ and sessions/ where the speech actually is).
    """
    excludes = load_jsonc(VSCODE_DIR / "settings.json").get(setting)
    assert isinstance(excludes, dict) and excludes, f"{setting} is missing"
    enabled = [pattern for pattern, on in excludes.items() if on is True]
    assert any(glob_to_regex(pattern).match(sensitive) for pattern in enabled), (
        f"{setting} does not cover {sensitive!r} -- verbatim customer speech would be "
        f"indexed by the editor. Enabled patterns: {enabled}"
    )


# --------------------------------------------------------------------------
# Guard: the interpreter is the workspace venv
# Prevents: a newcomer's editor silently bound to a system Python that has
# none of this project's pinned packages.
# --------------------------------------------------------------------------
def test_interpreter_path_is_workspace_relative() -> None:
    interpreter = load_jsonc(VSCODE_DIR / "settings.json").get("python.defaultInterpreterPath")
    assert isinstance(interpreter, str) and interpreter
    assert interpreter.startswith("${workspaceFolder}"), (
        f"{interpreter!r} is not anchored to the workspace"
    )
    assert ".." not in interpreter, f"{interpreter!r} climbs out of the workspace"
    normalised = interpreter.replace("\\", "/").lower()
    assert normalised.endswith("/.venv/scripts/python.exe"), (
        f"{interpreter!r} is not the workspace venv's interpreter"
    )


def test_every_task_runs_the_workspace_venv_or_the_launcher() -> None:
    """Prevents a task quietly running a bare `python` -- the system 3.12,
    with no pytest and none of the pinned packages, silently doing the wrong
    thing. `py` is allowed for exactly one task: the one that CREATES the venv,
    which by definition cannot run inside it."""
    tasks = load_jsonc(VSCODE_DIR / "tasks.json")["tasks"]
    for task in tasks:
        command = task.get("command")
        if command is None:
            assert task.get("dependsOn"), f"task {task['label']!r} has neither command nor dependsOn"
            continue
        if command == "py":
            assert "venv" in task.get("args", []), (
                f"task {task['label']!r} runs the system launcher for something other than "
                "creating the venv"
            )
            continue
        assert command.startswith("${workspaceFolder}"), (
            f"task {task['label']!r} runs {command!r}, not the workspace venv"
        )
        assert command.replace("\\", "/").lower().endswith("/.venv/scripts/python.exe")


def test_a_task_warns_that_ctrl_c_is_the_only_safe_stop() -> None:
    """Prevents the failure app/interrupt.py and app/overlay/session.py exist
    to fix, arriving through the editor instead: VS Code's terminate button is
    TerminateProcess, so the interview dies without closed_at and with a
    truncated transcript. The running tasks have to say so where it is read."""
    tasks = load_jsonc(VSCODE_DIR / "tasks.json")["tasks"]
    by_label = {task["label"]: task for task in tasks}
    running = [
        label
        for label in by_label
        if any(verb in label.lower() for verb in ("overlay", "start an interview"))
    ]
    assert running, "no overlay/start task to warn on"
    for label in running:
        detail = by_label[label].get("detail", "").lower()
        assert "ctrl+c" in detail, f"task {label!r} does not name Ctrl+C as the stop"
        assert any(word in detail for word in ("terminate", "trash")), (
            f"task {label!r} does not warn against VS Code's terminate/trash-can button"
        )


def test_setup_task_creates_the_venv_and_installs() -> None:
    """The highest-value entry, and the one no in-app `doctor` command could
    ever replace: a repair command would need the very venv that is missing."""
    tasks = load_jsonc(VSCODE_DIR / "tasks.json")["tasks"]
    by_label = {task["label"]: task for task in tasks}
    setup = next((t for label, t in by_label.items() if label.lower().startswith("setup:") and t.get("dependsOn")), None)
    assert setup is not None, "no composite Setup task"
    assert setup.get("dependsOrder") == "sequence", "install could run before the venv exists"
    steps = [by_label[label] for label in setup["dependsOn"]]
    assert any(step.get("command") == "py" and "venv" in step.get("args", []) for step in steps)
    assert any(
        step.get("args", [])[:2] == ["-m", "pip"] and "requirements.txt" in step.get("args", [])
        for step in steps
    ), "no step installs requirements.txt"


def test_comment_stripper_leaves_string_literals_alone() -> None:
    """Guards the guard. Every check above reads these files through
    `strip_jsonc_comments`; a naive `//.*` regex would eat the rest of any line
    holding a `//` inside a string and hand the guards a document that does not
    say what the file says -- silently, and in the direction of passing."""
    source = """
    {
      // a real comment, with a "quote" in it
      "url": "http://example.com//x",       // trailing comment
      "escaped": "he said \\"//\\" and meant it",
      "slash": "C://drive" /* block */
    }
    """
    parsed = json.loads(strip_jsonc_comments(source))
    assert parsed == {
        "url": "http://example.com//x",
        "escaped": 'he said "//" and meant it',
        "slash": "C://drive",
    }


def test_recommended_extensions_are_the_two_that_matter() -> None:
    recommendations = load_jsonc(VSCODE_DIR / "extensions.json")["recommendations"]
    assert recommendations == ["ms-python.python", "ms-python.debugpy"]


# --------------------------------------------------------------------------
# Guard: no credential is committed
# Prevents: a key pasted into a task env block or a `set` line and pushed --
# the .env file is gitignored, but nothing under .vscode/ is.
# --------------------------------------------------------------------------
VENDOR_PREFIXES = re.compile(r"(?<![A-Za-z])(sk-|sk_live|xoxb-|ghp_|AKIA[A-Z0-9]|AIza)")
# A key set anywhere in these files, however short and however unremarkable.
# The entropy rule below would miss `set SONIOX_API_KEY=hunter2`; this catches
# it by the shape of the assignment instead.
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?)\b\s*[:=]\s*[\"']?([^\s\"',}\]]+)"
)
# A long opaque token with no vendor prefix at all -- the shape a rotated or
# self-hosted credential has.
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_\-]{24,}")
PLACEHOLDER = re.compile(r"^(\.\.\.|<.*>|\$\{.*\}|%.*%|\"\")$")


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: p.name)
def test_no_credential_shaped_value(path: Path) -> None:
    for label, text in scan_units(path):
        hit = VENDOR_PREFIXES.search(text)
        assert hit is None, f"{label}: vendor key prefix {hit.group(0)!r}"

        for name, value in CREDENTIAL_ASSIGNMENT.findall(text):
            assert PLACEHOLDER.match(value), f"{label}: {name} assigned a literal value {value!r}"

        for token in OPAQUE_TOKEN.findall(text):
            has_digit = any(c.isdigit() for c in token)
            has_letter = any(c.isalpha() for c in token)
            assert not (has_digit and has_letter), (
                f"{label}: opaque credential-shaped token {token!r}"
            )
