"""Tests for the project export (`app/session/export.py`).

Two properties carry the weight. An export must not land where git can see it,
because it holds verbatim customer speech and this project has already produced
one untracked file that was a `git add -A` from publication. And it must be
verifiable, because a corpus restored from a copy nobody can check cannot
support the product's central claim that a decision rule pre-dated its evidence.

What these cannot prove is stated where it matters: verifying an archive proves
its files hash to what the manifest recorded at export time, not that the
manifest was honest. Anyone who can edit a file can edit the manifest beside it.
This detects corruption and partial writes, which is what happens to backups; it
is not a tamper-proofing scheme and is not described as one.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.session.export import (
    ExportRefused, collect, export_project, sha256_of, verify_archive,
)


def _project(tmp_path, *, with_audio=False):
    root = tmp_path / "projects" / "acme"
    (root / "context").mkdir(parents=True)
    (root / "context" / "hypotheses.json").write_text(
        json.dumps([{"id": "h1", "statement": "s", "falsifier": "f",
                     "criterion": "c", "criterion_registered_after": ["S1"]}]),
        encoding="utf-8",
    )
    session = root / "sessions" / "S1"
    session.mkdir(parents=True)
    (session / "transcript.jsonl").write_text(
        json.dumps({"kind": "segment", "seq": 0, "speaker": "interviewee",
                    "text": "we went back to paper"}) + "\n",
        encoding="utf-8",
    )
    (session / "session.json").write_text(json.dumps({"closed_at": "x"}), encoding="utf-8")
    if with_audio:
        (session / "interviewee.wav").write_bytes(b"RIFF" + b"\0" * 200)
    return root


# --- where it may not go ---------------------------------------------------------


def test_an_export_into_a_git_visible_path_is_refused(tmp_path, monkeypatch):
    """An export holds verbatim customer speech. A copy sitting untracked in the
    working tree is one `git add -A` from being published -- the exact shape of
    mistake this repo already made with a credential file."""
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: True)
    with pytest.raises(ExportRefused, match="NOT git-ignored"):
        export_project(project, tmp_path / "leak.zip", repo_root=tmp_path)


def test_nothing_is_written_when_the_destination_is_refused(tmp_path, monkeypatch):
    project = _project(tmp_path)
    target = tmp_path / "leak.zip"
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: True)
    with pytest.raises(ExportRefused):
        export_project(project, target, repo_root=tmp_path)
    assert not target.exists(), "a refused export must leave nothing behind"


def test_an_ignored_destination_inside_the_repo_is_allowed(tmp_path, monkeypatch):
    """Refusing every in-repo path would push exports somewhere the operator has
    to remember, and `projects/` itself is ignored. The question is what git can
    SEE, not where the file happens to sit."""
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "ok.zip", repo_root=tmp_path)
    assert result.archive.exists()


def test_an_existing_archive_is_not_silently_replaced(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    target = tmp_path / "backup.zip"
    export_project(project, target, repo_root=tmp_path)
    with pytest.raises(ExportRefused, match="already exists"):
        export_project(project, target, repo_root=tmp_path)
    export_project(project, target, repo_root=tmp_path, overwrite=True)


def test_a_missing_project_is_refused(tmp_path):
    with pytest.raises(ExportRefused, match="not a project directory"):
        export_project(tmp_path / "nope", tmp_path / "x.zip", repo_root=tmp_path)


# --- what it carries -------------------------------------------------------------


def test_the_irreplaceable_files_are_all_present(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)
    with zipfile.ZipFile(result.archive) as bundle:
        names = set(bundle.namelist())
    assert "context/hypotheses.json" in names
    assert "sessions/S1/transcript.jsonl" in names
    assert "sessions/S1/session.json" in names


def test_an_unknown_new_artifact_is_carried_without_being_named(tmp_path, monkeypatch):
    """`declined_proposals.json` was added the same week this was written. A
    backup that named known filenames would have silently omitted it from every
    export until somebody noticed, which is the worst way to find out."""
    project = _project(tmp_path)
    (project / "context" / "something_invented_later.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)
    with zipfile.ZipFile(result.archive) as bundle:
        assert "context/something_invented_later.json" in bundle.namelist()


def test_audio_is_excluded_and_the_exclusion_is_recorded(tmp_path, monkeypatch):
    """Stated, never implied. A restorer must be told what is deliberately
    missing rather than infer it from an absence."""
    project = _project(tmp_path, with_audio=True)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)
    assert result.audio_skipped == 1
    with zipfile.ZipFile(result.archive) as bundle:
        assert not any(name.endswith(".wav") for name in bundle.namelist())
        manifest = json.loads(bundle.read("MANIFEST.json"))
    assert manifest["audio_excluded"] == 1


def test_collect_skips_audio_by_extension_not_by_size(tmp_path):
    project = _project(tmp_path, with_audio=True)
    files, skipped = collect(project)
    assert skipped == 1
    assert all(path.suffix != ".wav" for path in files)


# --- verification ----------------------------------------------------------------


def test_a_fresh_export_verifies_clean(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)
    assert verify_archive(result.archive) == []


def test_a_corrupted_file_is_detected(tmp_path, monkeypatch):
    """The failure that actually happens to backups: a file that changed between
    export and restore. Without checksums a restored corpus would carry a moved
    pre-registration boundary with nothing able to notice."""
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)

    # Rebuild the archive with one file's bytes altered and the manifest left
    # untouched -- exactly what silent corruption looks like.
    with zipfile.ZipFile(result.archive) as bundle:
        contents = {name: bundle.read(name) for name in bundle.namelist()}
    contents["context/hypotheses.json"] = b'[{"id": "h1", "criterion_registered_after": []}]'
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as bundle:
        for name, blob in contents.items():
            bundle.writestr(name, blob)

    assert verify_archive(tampered) == ["context/hypotheses.json"]


def test_a_missing_file_is_detected_not_skipped(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("app.session.export._is_git_visible", lambda target, repo_root: False)
    result = export_project(project, tmp_path / "b.zip", repo_root=tmp_path)
    with zipfile.ZipFile(result.archive) as bundle:
        contents = {n: bundle.read(n) for n in bundle.namelist()
                    if n != "sessions/S1/transcript.jsonl"}
    partial = tmp_path / "partial.zip"
    with zipfile.ZipFile(partial, "w") as bundle:
        for name, blob in contents.items():
            bundle.writestr(name, blob)
    assert verify_archive(partial) == ["sessions/S1/transcript.jsonl"]


# --- the guard itself, unmocked --------------------------------------------------
# Every test above stubs `_is_git_visible`, which makes them tests of what the
# caller does with the answer and NOT of the answer. A mocked guard is an
# untested guard, so these two ask the real function about the real repository.


def test_the_real_guard_sees_an_unignored_path_in_the_repo():
    from app.session.export import _is_git_visible

    repo = Path(__file__).resolve().parents[2]
    # `README.md` is tracked, so git can certainly see it -- and a path next to
    # it that nothing ignores is the shape an export would take by accident.
    assert _is_git_visible(repo / "customer-backup.zip", repo) is True


def test_the_real_guard_clears_an_ignored_path_and_paths_outside_the_repo(tmp_path):
    from app.session.export import _is_git_visible

    repo = Path(__file__).resolve().parents[2]
    # `.gitignore:50` excludes `projects/` wholesale.
    assert _is_git_visible(repo / "projects" / "backup.zip", repo) is False
    # Outside the repository entirely, the question is not asked at all.
    assert _is_git_visible(tmp_path / "backup.zip", repo) is False


def test_the_checksum_is_over_content_not_over_the_name(tmp_path):
    one = tmp_path / "a.txt"
    two = tmp_path / "b.txt"
    one.write_text("same", encoding="utf-8")
    two.write_text("same", encoding="utf-8")
    assert sha256_of(one) == sha256_of(two)
    two.write_text("different", encoding="utf-8")
    assert sha256_of(one) != sha256_of(two)
