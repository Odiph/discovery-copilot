"""The reporting CLIs must survive a console that is not UTF-8.

**The failure this pins.** `benchmark/eval/uptake_report.py` and
`render_review.py` print transcript text straight to stdout. On Windows a real
console is UTF-8, but a REDIRECTED stdout (`... > report.txt`, or any pipe)
falls back to `locale.getpreferredencoding()`, which is cp1252 on a Western-locale Windows machine.
Printing a Hebrew segment through that raises `UnicodeEncodeError` and the
report dies part-written -- after the header, so the run looks like it started
fine. P1's number was unreadable for any interview with Hebrew in it.

**Why a subprocess and not a unit test of the helper.** Asserting that
`use_utf8_console()` calls `reconfigure` proves the helper was WRITTEN, not that
the script is protected: the script has to actually call it, before the first
write, on both streams. Only running the real entry point with a real cp1252
stdout exercises that. The env var is how a non-UTF-8 stdout is reproduced
deterministically on a machine whose console happens to be UTF-8 -- it sets the
same fallback that redirection sets.

`render_review`'s crash is worse than its `--out` path suggests: writing to a
file has always been safe (explicit `encoding="utf-8"`), so the DEFAULT
invocation is the broken one. It also emits a literal that is not cp1252-encodable
regardless of transcript language, so English sessions are not automatically safe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Reused verbatim from `app/tests/rtl_probe.py`'s D-04 verification set rather
# than invented here, so the alphabet under test is the one the product was
# actually verified against. The embedded quote and digits are part of the
# point: this is real code-switched speech, not a lone Hebrew word.
HEBREW_QUESTION = "אז מה דעתך על ה-Slack שלנו?"
HEBREW_ANSWER = "זה עלה לנו 1,200 ש\"ח בחודש שעבר, לא יותר מ-15%."

# The three entry points whose crash was reproduced against a REAL session
# before any of this was written. `trigger_floor` is the reason this is a
# parametrized list and not two tests: its crash needed both a real transcript
# AND `--show` at a floor that actually drops a turn, so the print that dies is
# one a casual run never reaches. The other fourteen benchmark CLIs get the same
# one-line guard, but only these three have a demonstrated failure to pin.
#
# The third element is the string that script is expected to have printed.
# It differs by script and that is the point: `uptake_report` and
# `render_review` print the SUGGESTION, while `trigger_floor` prints the
# INTERVIEWEE turn it dropped. Asserting one shared string would have meant
# either a weaker assertion or a test that passes on the wrong output.
SCRIPTS = [
    ("uptake_report.py", ("--misses",), HEBREW_QUESTION),
    ("render_review.py", (), HEBREW_QUESTION),
    # The floor must exceed the fixture's turn length or nothing is dropped and
    # the Hebrew-printing branch never executes -- which is exactly how this
    # test first passed while covering nothing. 50 is far enough above the
    # ~10-word turn that the drop cannot stop happening if the fixture is
    # reworded.
    ("trigger_floor.py", ("--show", "50"), HEBREW_ANSWER),
]


def write_session(tmp_path: Path) -> Path:
    """A session whose suggestion is TAKEN, so both text-printing sites run.

    The unmatched path prints only the suggestion; the matched path also prints
    every matched segment. A fixture with no match would leave the second site
    -- a separate `print` on a separate line -- unexecuted, and the test would
    pass while half the bug survived."""
    session_dir = tmp_path / "20260101T000000"
    session_dir.mkdir()
    records = [
        {"kind": "segment", "seq": 1, "speaker": "interviewee", "text": HEBREW_ANSWER,
         "audio_start_s": 0.0, "audio_end_s": 10.0},
        {"kind": "suggestion", "seq": 2, "shown": True, "trigger_seq": 1,
         "text": HEBREW_QUESTION, "move": "deepen", "band": "clean",
         "rationale": HEBREW_ANSWER},
        # Same words as the suggestion, so similarity is 1.0 and the match is
        # not a near-threshold coin flip that could silently stop covering the
        # matched-segment print.
        {"kind": "segment", "seq": 3, "speaker": "interviewer", "text": HEBREW_QUESTION,
         "audio_start_s": 18.0, "audio_end_s": 20.0},
    ]
    (session_dir / "transcript.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8",
    )
    # `session.json`, NOT `metadata.json` -- render_review reads the former, and
    # an earlier draft of this fixture wrote the latter, so the `synthetic`
    # branch (the one emitting the non-cp1252-encodable warning marker) never
    # ran and the test was quietly weaker than it read.
    (session_dir / "session.json").write_text(
        json.dumps({"engine": "fake", "synthetic": True}), encoding="utf-8",
    )
    return session_dir


def run_cli(script: str, session_dir: Path, *args: str) -> "subprocess.CompletedProcess[bytes]":
    env = dict(os.environ)
    # The whole point of the test. Without this the child inherits the parent's
    # UTF-8 stdout and cannot fail, whatever the code does.
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        [sys.executable, str(REPO / "benchmark" / "eval" / script), str(session_dir), *args],
        capture_output=True, cwd=str(REPO), env=env, timeout=120,
    )


@pytest.mark.parametrize("script,args,expected", SCRIPTS)
def test_report_survives_a_cp1252_stdout(
    tmp_path: Path, script: str, args, expected: str,
) -> None:
    del expected  # this test is about not dying; the next one is about the words
    result = run_cli(script, write_session(tmp_path), *args)
    stderr = result.stderr.decode("utf-8", "replace")
    assert "UnicodeEncodeError" not in stderr, f"{script} still dies on a cp1252 stdout:\n{stderr}"
    assert result.returncode == 0, f"{script} exited {result.returncode}:\n{stderr}"


@pytest.mark.parametrize("script,args,expected", SCRIPTS)
def test_the_hebrew_survives_rather_than_becoming_question_marks(
    tmp_path: Path, script: str, args, expected: str,
) -> None:
    """Not crashing is not enough.

    `errors="replace"` over cp1252 would also stop the crash, and would turn
    every Hebrew character into `?` -- a report that runs to completion and
    says nothing. The fix has to widen the stream to UTF-8, not silence the
    error, and only asserting on the actual words distinguishes the two."""
    result = run_cli(script, write_session(tmp_path), *args)
    stdout = result.stdout.decode("utf-8", "replace")
    assert expected in stdout, (
        f"{script} ran but lost the Hebrew -- encoding errors were swallowed, not fixed.\n"
        f"{stdout[:400]}"
    )
