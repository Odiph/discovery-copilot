"""A child-process entry point for the SESS-02 kill test
(app/tests/test_durability.py). Deliberately its own module rather than a
test file, so it can be run as `python -m app.tests.crash_writer <dir>
<count>` -- a subprocess entry point, not something pytest should collect.

Opens a real `SessionStore`, appends records in a loop, and prints the
sequence number ONLY after `append_segment` has returned -- meaning the
record has already been flushed and fsync'd to disk (SESS-02, U1). The
parent process reads these lines as they arrive and kills this process
without warning at a point of its choosing; whatever was printed before that
kill is guaranteed durable by the store's own contract, and the test asserts
exactly that guarantee against what actually survives on disk.

Never closes the store. Closing is not required for durability -- only
`append_segment` returning is (that is the entire point of this module) --
and a real crash never gets the chance to close anything either.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app.session.store import SessionStore


def main(argv: "list[str] | None" = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    session_dir = Path(argv[0])
    count = int(argv[1])

    store = SessionStore(session_dir)
    store.write_metadata({"engine": "crash-writer-test"})

    for i in range(count):
        store.append_segment(
            speaker="interviewee",
            text=f"crash-writer segment {i}",
            audio_start_s=float(i),
            audio_end_s=float(i) + 1.0,
            ready_wall_s=float(i) + 1.1,
            detected_language="en",
            engine="fake",
        )
        # Printed and flushed only after append_segment has RETURNED -- the
        # parent's kill can never land before this line's record is durable.
        print(i, flush=True)

    # Keep the process alive after the loop so the parent has a wide window
    # in which to kill it "without warning" -- the interesting case is a
    # kill landing mid-loop, and a process that exits the instant it is done
    # writing would make that race harder to hit deterministically.
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
