"""A portable copy of everything a project cannot regenerate.

**The gap.** Sessions, facts, hypotheses, criteria and the
`criterion_registered_after` boundaries live only under `projects/`, which
`.gitignore:50` excludes wholesale, with standing gates asserting none of it
ever becomes committable. That discipline is correct and it has an unstated
consequence: "never escapes" and "has no backup" are the same property. There
was no export, snapshot or copy verb anywhere in the product. At interview 30 a
dead disk does not degrade this tool, it ends it -- including the evidentiary
value of every pre-registration boundary, which is the one thing here that
cannot be reconstructed from memory or from the customers.

**Checksums, because a restored corpus has to be believable.** The product's
central claim is that a decision rule was written before the evidence that
judges it. A corpus restored from a copy nobody can verify cannot support that
claim: the boundary might have moved in transit, and there would be no way to
tell. So the manifest records a SHA-256 per file. That does not make an export
tamper-PROOF -- anyone who can edit a file can edit the manifest beside it --
but it makes accidental corruption and partial writes detectable, which is what
actually happens to backups.

**Audio is deliberately excluded.** `--retain-audio` is off by default so it is
usually absent anyway, but where it exists it is large and it is the rawest form
of customer speech in the system. An export is a copy that leaves the guarded
tree; it should carry the least that still makes the corpus whole. The manifest
records that audio was skipped and how many files, so the absence is a stated
fact rather than something a restorer discovers.

**This module never decides WHERE the copy goes.** It refuses to write anywhere
git can see, and otherwise takes the caller's path. That guard exists because
this project has already produced a credential file that landed one `git add -A`
from the history; an export is the same shape of mistake with customer speech
in it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "MANIFEST.json"

# Extensions that are audio, excluded by policy rather than by size.
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


class ExportRefused(Exception):
    """Raised before anything is written. Never carries file contents."""


@dataclass(frozen=True)
class ExportResult:
    archive: Path
    files: int
    bytes_written: int
    audio_skipped: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_git_visible(target: Path, repo_root: Path) -> bool:
    """True when git would show `target` as an untracked or tracked path.

    Asks GIT rather than reasoning about `.gitignore` patterns, because the
    patterns are the thing most likely to be wrong -- `.gitignore` named `.env`
    exactly for months while every neighbour was committable. A path outside the
    repository entirely is never git-visible and the question is not asked."""
    try:
        target.relative_to(repo_root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(target)],
        cwd=str(repo_root), capture_output=True,
    )
    # 0 = ignored, 1 = NOT ignored (so git can see it), 128 = not a repo.
    return result.returncode == 1


def collect(project_dir: Path) -> "tuple[list[Path], int]":
    """Every file worth carrying, and how many audio files were skipped.

    Walks the whole project rather than naming known filenames. A future
    artifact -- `declined_proposals.json` was one, added the same week -- is
    included the day it exists, instead of being silently left out of every
    backup until somebody notices. The cost of that choice is that an unknown
    file is copied rather than refused, which is the right way round for a
    backup."""
    files: "list[Path]" = []
    audio_skipped = 0
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in AUDIO_SUFFIXES:
            audio_skipped += 1
            continue
        files.append(path)
    return files, audio_skipped


def export_project(
    project_dir: Path,
    archive: Path,
    *,
    repo_root: Path,
    overwrite: bool = False,
) -> ExportResult:
    """Writes a zip of `project_dir` to `archive`, with a checksum manifest.

    Refuses, before writing anything, if the project does not exist, if the
    archive already exists without `overwrite`, or if the destination is
    somewhere git can see -- an export carries verbatim customer speech, and a
    copy of it sitting untracked in the working tree is one `git add -A` from
    being published."""
    if not project_dir.is_dir():
        raise ExportRefused(f"{project_dir} is not a project directory")
    archive = archive.resolve()
    if archive.exists() and not overwrite:
        raise ExportRefused(f"{archive} already exists -- pass --overwrite to replace it")
    if _is_git_visible(archive, repo_root):
        raise ExportRefused(
            f"{archive} is inside the repository and NOT git-ignored. An export holds "
            "verbatim customer speech; write it outside the repo, or to an ignored path."
        )
    archive.parent.mkdir(parents=True, exist_ok=True)

    files, audio_skipped = collect(project_dir)
    entries = []
    for path in files:
        entries.append({
            "path": path.relative_to(project_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        })

    manifest = {
        "project": project_dir.name,
        "files": entries,
        "file_count": len(entries),
        # Stated, never implied. A restorer must be told what is deliberately
        # missing rather than left to infer it from an absence.
        "audio_excluded": audio_skipped,
        "audio_note": "Audio is excluded by policy. Everything else is verbatim.",
    }

    written = 0
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(project_dir).as_posix())
            written += path.stat().st_size
        bundle.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))

    return ExportResult(
        archive=archive, files=len(files), bytes_written=written, audio_skipped=audio_skipped,
    )


def verify_archive(archive: Path) -> "list[str]":
    """Re-checks every checksum in an export. Returns the paths that FAILED.

    A backup nobody has ever read is a hope, not a backup. This is the cheap
    version of restoring it: an empty list means every file in the archive
    hashes to what the manifest says it did at export time."""
    failures: "list[str]" = []
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(MANIFEST_NAME))
        for entry in manifest["files"]:
            try:
                blob = bundle.read(entry["path"])
            except KeyError:
                failures.append(entry["path"])
                continue
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                failures.append(entry["path"])
    return failures
