"""Every wire word the phone compares against must BE the server's word.

**The defect, found by the 2026-08-12 audit.** Renaming `REASON_NO_AUDIO`
server-side from `"no-audio"` to `"silent"` left the whole Python suite green at
1092 passed. The overlay survived because `app/overlay/model.py` IMPORTS the
constant -- its own comment says why: *"Imported, never re-spelled. A local copy
of the literal `degraded` would go on passing after the server renamed its
tier."* `app/static/app.js` compared a bare string literal, so it fell into the
wrong branch forever, and the phone would have said "transcription stopped on
the laptop" for a call whose audio had merely gone quiet -- the exact false
alarm the degraded tier was built to avoid.

This is `message["candidates"]` against a wire pinned to `primary`, one surface
later, and the third time this project has shipped a seam whose two sides were
each tested and never compared.

**Why the fix is imports and not a scanner.** A gate that greps `app.js` for
`message.reason === "..."` and checks the harvested set would work today and
would be defeated by a `switch`, a lookup table, or a ternary -- the same
one-language-feature-across blindness that let `dict(fact)` past gate 20. Moving
the literals into `live-state.mjs` and importing them makes the JS side
structurally unable to hold a stale copy: there is only one place the word
exists on that side of the wire.

**What this file then has to prove**, because the imports alone move the
problem rather than solving it: that the ONE JS copy equals the ONE Python copy.
It reads every constant back out of `live-state.mjs` through a real node
process -- the mechanism `test_pong_message_type_literal_comes_from_one_place_
not_two` already established for `MESSAGE_TYPE_PONG` -- and compares the whole
mapping by equality, so a word added on either side has to be added on both.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.main import ALL_SPEAKERS
from app.web import server as web_server
from app.web import status as web_status

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_STATE = REPO_ROOT / "app" / "static" / "live-state.mjs"
APP_JS = REPO_ROOT / "app" / "static" / "app.js"

# The authoritative mapping: JS export name -> the Python value it must equal.
#
# Written out by hand ON PURPOSE, and it is the one hand-written list in this
# file. Deriving it (say, by matching names) would make the test agree with
# itself: the whole failure being guarded against is two sides that look right
# in isolation. A human deciding "this JS name means that Python constant" is
# the content of the check.
EXPECTED: "dict[str, str]" = {
    "MESSAGE_TYPE_PING": web_server.MESSAGE_TYPE_PING,
    "MESSAGE_TYPE_PONG": web_server.MESSAGE_TYPE_PONG,
    "MESSAGE_TYPE_HELLO": web_server.MESSAGE_TYPE_HELLO,
    "MESSAGE_TYPE_SEGMENT": web_server.MESSAGE_TYPE_SEGMENT,
    "MESSAGE_TYPE_SERVER_STATUS": web_server.MESSAGE_TYPE_SERVER_STATUS,
    "MESSAGE_TYPE_SUGGESTION": web_server.MESSAGE_TYPE_SUGGESTION,
    "MESSAGE_TYPE_SURVEY": web_server.MESSAGE_TYPE_SURVEY,
    "STATUS_HEALTHY": web_status.STATUS_HEALTHY,
    "STATUS_DEGRADED": web_status.STATUS_DEGRADED,
    "STATUS_FAILED": web_status.STATUS_FAILED,
    "FAILED_SIDE_COACH": web_status.FAILED_SIDE_COACH,
    "REASON_NO_AUDIO": web_status.REASON_NO_AUDIO,
    "REASON_LAGGING": web_status.REASON_LAGGING,
    "REASON_NO_TRANSCRIPT": web_status.REASON_NO_TRANSCRIPT,
    "SPEAKER_INTERVIEWEE": "interviewee",
    "SPEAKER_INTERVIEWER": "interviewer",
}

_needs_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is not on PATH -- the JS half of the wire vocabulary could not be read",
)


def _read_js_constants() -> "dict[str, str]":
    """Every string export of `live-state.mjs`, read through a real node.

    Not a regex over the source: a regex would happily read a constant that
    the module never actually exports (a commented-out line, a name inside a
    template literal), and the property under test is what the BROWSER would
    import."""
    node = shutil.which("node")
    script = (
        f"import('file://{LIVE_STATE.as_posix()}').then(m => {{"
        "  const out = {};"
        "  for (const [k, v] of Object.entries(m)) {"
        "    if (typeof v === 'string') { out[k] = v; }"
        "  }"
        "  process.stdout.write(JSON.stringify(out));"
        "});"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"node could not import live-state.mjs:\n{result.stderr}"
    return json.loads(result.stdout)


@_needs_node
def test_every_shared_wire_word_is_the_same_string_on_both_sides():
    js = _read_js_constants()
    assert js, "live-state.mjs exported no string constants at all -- an empty read proves nothing"

    missing = sorted(set(EXPECTED) - set(js))
    assert not missing, (
        f"live-state.mjs no longer exports {missing}. app.js imports these; the browser would "
        "get `undefined` and every comparison against them would be false"
    )
    mismatched = {
        name: (js[name], EXPECTED[name]) for name in EXPECTED if js[name] != EXPECTED[name]
    }
    assert not mismatched, (
        f"the two sides of the wire disagree: {mismatched}. Each pair is (JS, Python) -- the "
        "phone is comparing against a word the server no longer sends"
    )


@_needs_node
def test_the_js_side_holds_no_wire_word_this_test_does_not_check():
    """Equality over the SET, not just over the pairs above.

    A constant exported from `live-state.mjs` and absent from `EXPECTED` is a
    word the phone may branch on with nothing holding it to the server. That is
    the state this whole file exists to end, so a new one has to be added here
    -- which is the moment someone decides which Python constant it mirrors."""
    js = _read_js_constants()
    # `DEFAULT_JITTER_FRACTION` and friends are numbers, already filtered out.
    # Anything left that is a string is wire vocabulary by definition.
    unchecked = sorted(set(js) - set(EXPECTED))
    assert not unchecked, (
        f"live-state.mjs exports string constants nothing pins to Python: {unchecked}"
    )


def test_the_speaker_labels_come_from_mains_own_tuple():
    """`SPEAKER_INTERVIEWEE` is the one entry above whose Python value is
    typed as a literal, because `app/main.py`'s constants are the home. This
    ties that literal back to the tuple so the mapping cannot go stale."""
    assert set(ALL_SPEAKERS) == {"interviewee", "interviewer"}
    assert set(web_status.all_failed_sides()) == {"interviewee", "interviewer", "coach"}


def test_app_js_compares_no_bare_wire_literal_any_more():
    """The structural half. Imports fix the problem only while nobody types a
    literal back in, and the natural way to add a sixth branch is to copy the
    line above it -- which, before this change, meant copying a bare string.

    Scoped to the four wire FIELDS rather than to all strings: `app.js` is full
    of legitimate literals (class names, sentences), and a blanket ban would be
    an allowlist nobody would maintain."""
    source = APP_JS.read_text(encoding="utf-8")
    offenders = re.findall(
        r'message\.(?:type|status|reason|failed_side)\s*===\s*"([^"]*)"', source
    )
    assert offenders == [], (
        f"app.js compares a wire field against a hand-typed literal: {offenders}. Import the "
        "constant from live-state.mjs instead -- a local copy goes on passing after the server "
        "renames the word, and the branch is then wrong for the rest of the run"
    )


def test_the_scan_above_finds_a_bare_literal_when_there_is_one():
    """The scan passing when it finds nothing is also what a broken scan
    returns -- gate 17/20/30's shape, applied to this walk."""
    sample = 'if (message.reason === "no-audio") { x(); }'
    assert re.findall(
        r'message\.(?:type|status|reason|failed_side)\s*===\s*"([^"]*)"', sample
    ) == ["no-audio"]


def test_the_server_message_types_the_phone_knows_are_the_ones_the_server_sends():
    """The set-level check the per-name mapping above cannot make: a message
    type the server broadcasts and `app.js` has no branch for is silently
    dropped by its final `else`.

    `ping` is excluded because it travels the other way -- it is the one
    client-originated type, and `ACCEPTED_CLIENT_MESSAGE_TYPES` pins it."""
    handled = {
        EXPECTED[name] for name in EXPECTED
        if name.startswith("MESSAGE_TYPE_") and name != "MESSAGE_TYPE_PING"
    }
    assert handled == set(web_server.SERVER_MESSAGE_TYPES), (
        f"the phone and the server disagree about what can arrive: "
        f"server-only {sorted(set(web_server.SERVER_MESSAGE_TYPES) - handled)}, "
        f"phone-only {sorted(handled - set(web_server.SERVER_MESSAGE_TYPES))}"
    )


def test_the_status_vocabularies_are_pinned_by_equality():
    """`app/web/status.py` had no collections at all until this commit, which
    is why nothing could be asserted about it. Equality, following gate 8."""
    assert set(web_status.ALL_STATUSES) == {"healthy", "degraded", "failed"}
    assert set(web_status.ALL_REASONS) == {
        "stream-failure", "coach-failure", "no-audio", "lagging", "no-transcript",
    }
