"""Make a reporting CLI's stdout able to carry the alphabet it is reporting on.

**The bug this exists to remove.** Every script here prints transcript text, and
transcripts are not ASCII -- interviews are code-switched Hebrew/English, and
some reports emit non-ASCII of their own (`render_review`'s warning marker is
not cp1252-encodable at all, so an all-English session is not automatically
safe). On Windows an attached console is UTF-8, but the moment stdout is
REDIRECTED -- `... > report.txt`, `| more`, a CI log -- Python falls back to
`locale.getpreferredencoding()`, which is cp1252 on a Western-locale Windows machine, and the first
Hebrew character raises `UnicodeEncodeError`.

That failure is nastier than a crash at startup: the header, the rate and the
bar all print first, so the run looks healthy right up to the point it dies
part-written. Saving the report to a file, the natural thing to do with a
report, is exactly what triggers it.

**Why widening the stream and not `errors="replace"` alone.** Replacing on
cp1252 also stops the crash, and turns every Hebrew word into `?` -- a report
that completes and says nothing, which is worse than one that fails loudly.
UTF-8 is what the transcript is already stored in (`read`/`read_records` both
open with an explicit `encoding="utf-8"`), so this only makes the OUTPUT side
agree with the input side. `errors="replace"` is kept as a second line of
defence for whatever the widening cannot cover; it should never fire.

Call it from the `if __name__ == "__main__":` guard, not at import and not from
`main()`. At import it would be a side effect on any process that merely imports
the module -- pytest included. In `main()` it would fire for the tests that call
`main()` directly (`test_trigger_sim`, `test_judge_critic`, `test_rehearse` all
do), reconfiguring the streams pytest is capturing on. The guard is the actual
CLI boundary: it runs exactly when a human is looking at a console, which is the
only case this is for.

**Applied to every entry point under `benchmark/`, including the ones with no
non-ASCII output today.** Deciding per script which one "can print Hebrew"
is a judgement that silently expires the next time someone adds a `print`, and
the failure it expires into is a part-written report. A uniform rule costs one
line per file and cannot rot.
"""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """Widen stdout and stderr to UTF-8, best effort, for the life of the process.

    Both streams, because a traceback naming a segment goes to stderr and would
    raise a second, more confusing `UnicodeEncodeError` inside the handler for
    the first.

    Failures are swallowed DELIBERATELY, and this is the one place that is the
    right call: the function's entire purpose is to stop an encoding detail from
    killing a report, so raising here would reintroduce the failure it exists to
    prevent, on a machine where the report might otherwise have printed fine.
    A stream that cannot be reconfigured (already-detached, or replaced by a
    test double with no `reconfigure`) is left exactly as it was.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Could not widen it. Settle for never crashing on it: the same
            # console now drops unencodable characters instead of raising.
            try:
                reconfigure(errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
