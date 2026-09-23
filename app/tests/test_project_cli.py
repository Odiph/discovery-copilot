"""Phase A: `--project` on the CLI.

`_apply_project` is the entire read-side of selecting a project, and these
tests are mostly about what it must NOT change. The default path -- no
`--project` -- has to stay byte-for-byte the behaviour every existing
invocation, `record-test-session.bat`, and the gates that read
`REPO_ROOT / "context"` already depend on.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.main import _apply_project, build_arg_parser, main
from app.projects import ProjectPathError, resolve

PROJECT_AWARE_COMMANDS = ("start", "render", "author", "extract", "summary", "standing")

# `render`, `extract` and `summary` take a positional session id; `standing`
# takes an OPTIONAL one, and is listed here with it omitted deliberately -- the
# flag has to survive on the shape an operator actually types when they want a
# whole project's standing rather than one interview's delta.
_MINIMAL_ARGV = {
    "start": ["start"],
    "render": ["render", "20260810T120000"],
    "author": ["author"],
    "extract": ["extract", "20260810T120000"],
    "summary": ["summary", "20260810T120000"],
    "standing": ["standing"],
}


def _parse(*argv: str):
    return build_arg_parser().parse_args(list(argv))


# --- the flag exists exactly where it should ------------------------------------------


@pytest.mark.parametrize("command", PROJECT_AWARE_COMMANDS)
def test_every_command_that_touches_a_root_accepts_a_project(command):
    args = _parse(*_MINIMAL_ARGV[command], "--project", "acme/onboarding")
    assert args.project == "acme/onboarding"


@pytest.mark.parametrize("command", PROJECT_AWARE_COMMANDS)
def test_the_flag_defaults_to_none_which_is_what_means_unchanged(command):
    assert _parse(*_MINIMAL_ARGV[command]).project is None


def test_precheck_refuses_the_flag_rather_than_silently_ignoring_it():
    # precheck touches neither root. Accepting `--project` there would be
    # worse than not offering it: an operator who passed it would reasonably
    # believe the check ran against that project.
    with pytest.raises(SystemExit):
        _parse("precheck", "--project", "acme")


# --- what _apply_project does, and does not, change ------------------------------------


def _config(tmp_path):
    from app import config as config_mod

    return config_mod.load_config()


def test_no_project_returns_the_very_same_config_object(tmp_path):
    # Identity, not equality. A `--project`-less invocation must not go
    # through a copy that could differ in some field nobody thought to
    # compare.
    cfg = _config(tmp_path)
    assert _apply_project(cfg, _parse("start")) is cfg


def test_a_project_points_both_roots_at_that_node_and_nothing_else(tmp_path):
    cfg = _config(tmp_path)
    applied = _apply_project(cfg, _parse("start", "--project", "acme/onboarding"))
    expected = resolve("acme/onboarding")

    assert applied.context_dir == expected.context_dir
    assert applied.sessions_root == expected.sessions_root

    # Every OTHER field must be untouched. Written as a whole-object
    # comparison rather than spot-checks so a future field cannot be quietly
    # captured by project selection without this failing.
    changed = {
        name
        for name in (field.name for field in dataclasses.fields(cfg))
        if getattr(cfg, name) != getattr(applied, name)
    }
    assert changed == {"context_dir", "sessions_root"}


def test_the_original_config_is_not_mutated(tmp_path):
    cfg = _config(tmp_path)
    before = cfg.context_dir
    _apply_project(cfg, _parse("start", "--project", "acme"))
    assert cfg.context_dir == before


def test_a_command_without_the_flag_at_all_is_tolerated(tmp_path):
    # `precheck` has no `project` attribute. `_apply_project` is not called
    # on that path today, but it must not become a landmine if it ever is.
    cfg = _config(tmp_path)
    assert _apply_project(cfg, _parse("precheck")) is cfg


def test_a_hostile_project_path_is_refused_before_any_root_is_built(tmp_path):
    with pytest.raises(ProjectPathError):
        _apply_project(_config(tmp_path), _parse("start", "--project", "../../etc"))


# --- the operator-facing failure -------------------------------------------------------


@pytest.mark.parametrize("command", PROJECT_AWARE_COMMANDS)
def test_a_rejected_project_path_exits_one_with_a_message_not_a_traceback(command, capsys):
    exit_code = main([*_MINIMAL_ARGV[command], "--project", "../../etc"])
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert stderr.startswith("ERROR:")
    assert "'.'" in stderr or ".." in stderr, "the message must name the rule that was broken"


def test_the_rejected_path_itself_appears_so_the_operator_can_see_the_typo(capsys):
    main(["render", "20260810T120000", "--project", "acme onboarding"])
    assert "acme onboarding" in capsys.readouterr().err


def test_start_rejects_the_path_before_it_opens_a_single_audio_device(monkeypatch, capsys):
    # Ordering that matters in the room, not just in the code: a bad
    # `--project` must fail before the process has taken the microphone and
    # the loopback device. Failing after would leave the operator's audio
    # grabbed and released for a typo, seconds before an interview.
    import app.main as main_mod

    def _must_not_run():
        raise AssertionError("an audio device was opened despite an invalid --project")

    monkeypatch.setattr(main_mod, "construct_pyaudio", _must_not_run)
    assert main(["start", "--project", "../escape"]) == 1
    assert capsys.readouterr().err.startswith("ERROR:")
