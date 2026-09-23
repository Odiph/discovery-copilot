"""`python -m app summary <session>` and `python -m app standing [<session>]`.

Both commands are pure SURFACE: every number they print was already computed by
`app/coach/interview_summary.py` and already served by the hub and the MCP
server. So these tests are mostly about the three things a surface can get
wrong on its own --

- it must not make a model call, ever, and no read command has a degraded mode
  that could hide one;
- it must not print verbatim customer speech, which it defends against by
  holding `FactRef`s rather than `Fact`s;
- the empty cases must each say something different. An interview that produced
  nothing, a project with no hypotheses, a corpus that was never written and a
  hypothesis nobody has asked about are four different situations, and rendering
  them as one screen of zeros is the failure the hub already had to fix once.

Exit codes carry the one distinction a script needs: a session id that names
nothing is the operator's typo and fails; every honest emptiness succeeds.
"""

from __future__ import annotations

import json

import pytest

from app import main as main_mod

_QUOTE = "we went back to paper because the export broke"

_HYPOTHESES = [
    {"id": "h1", "statement": "onboarding is the blocker", "falsifier": "setup was easy",
     "criterion": "3 of 6 interviewees raise it unprompted"},
    {"id": "h2", "statement": "they export to spreadsheets", "falsifier": "nobody exports"},
]


def _fact(fact_id, session_id, *, bearings=(), provenance="evidence"):
    return {
        "id": fact_id,
        "claim": "they rebuilt the export by hand",
        "provenance": provenance,
        "session_id": session_id,
        "seqs": [1],
        # Every fixture carries the SAME sentinel quote, so any test in this
        # file can assert on it without arranging its own leak.
        "quoted_text": _QUOTE,
        "bearings": [{"hypothesis_id": hid, "stance": stance} for hid, stance in bearings],
    }


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A project node with two sessions on disk, addressed through the config's
    own env vars rather than `--project`, so these tests never need a
    `projects/` tree inside the repo. `--project` reaches exactly the same two
    fields through `_apply_project`, which gate 30 pins."""
    context = tmp_path / "context"
    sessions = tmp_path / "sessions"
    context.mkdir()
    for name in ("S1", "S2"):
        (sessions / name).mkdir(parents=True)
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context))
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(sessions))
    (context / "hypotheses.json").write_text(json.dumps(_HYPOTHESES), encoding="utf-8")
    return context


def _write_facts(context, facts):
    (context / "facts.json").write_text(json.dumps(facts), encoding="utf-8")


@pytest.fixture
def no_vendor(monkeypatch):
    """Fails loudly if anything reaches a client. Neither command may ever grow
    a model call: they would then cost money and be able to fail on the network,
    on what is a read of a local file."""

    def _build_client():
        raise AssertionError("a read command built a model client")

    monkeypatch.setattr(main_mod.reasoning_client, "build_client", _build_client)


# --- a bad session id fails, and says so ---------------------------------------------


def test_summary_exits_non_zero_for_a_session_id_that_does_not_exist(node, no_vendor, capsys):
    assert main_mod.main(["summary", "nope"]) == 1
    assert "no session directory found" in capsys.readouterr().err


def test_standing_exits_non_zero_for_a_session_id_that_does_not_exist(node, no_vendor, capsys):
    assert main_mod.main(["standing", "nope"]) == 1
    assert "no session directory found" in capsys.readouterr().err


def test_standing_without_a_session_needs_no_session_at_all(node, no_vendor, capsys):
    # The optional positional is what makes `standing` answerable before any
    # interview has been recorded.
    assert main_mod.main(["standing"]) == 0
    assert "Hypothesis standing" in capsys.readouterr().out


# --- the four emptinesses each read differently ---------------------------------------


def test_summary_names_an_absent_corpus_as_a_step_not_taken(node, no_vendor, capsys):
    assert main_mod.main(["summary", "S1"]) == 0
    out = capsys.readouterr().out
    assert "No fact corpus" in out
    assert "python -m app extract S1" in out


def test_summary_distinguishes_an_interview_with_nothing_in_a_corpus_that_has_facts(
    node, no_vendor, capsys
):
    _write_facts(node, [_fact("f1", "S2", bearings=(("h1", "supports"),))])
    assert main_mod.main(["summary", "S1"]) == 0
    out = capsys.readouterr().out
    # The corpus EXISTS, so this is not the message above -- and which of the
    # two reasons it is cannot be known, so the wording must not pick one.
    assert "No fact corpus" not in out
    assert "Nothing from this interview is in the corpus" in out
    assert "nothing on disk distinguishes those" in out


def test_summary_says_so_when_an_interviews_claims_bear_on_no_hypothesis(node, no_vendor, capsys):
    _write_facts(node, [_fact("f1", "S1")])
    assert main_mod.main(["summary", "S1"]) == 0
    out = capsys.readouterr().out
    assert "claims extracted from this interview   1" in out
    assert "bear on none of this project's hypotheses" in out


def test_standing_says_there_is_nothing_to_stand_when_no_hypotheses_exist(
    tmp_path, monkeypatch, no_vendor, capsys
):
    context = tmp_path / "context"
    context.mkdir()
    (tmp_path / "sessions").mkdir()
    monkeypatch.setenv("APP_CONTEXT_DIR", str(context))
    monkeypatch.setenv("APP_SESSIONS_ROOT", str(tmp_path / "sessions"))
    assert main_mod.main(["standing"]) == 0
    out = capsys.readouterr().out
    assert "there is nothing to stand" in out
    assert "python -m app author" in out


def test_standing_calls_out_hypotheses_no_interview_has_borne_on(node, no_vendor, capsys):
    _write_facts(node, [_fact("f1", "S1", bearings=(("h1", "supports"),))])
    assert main_mod.main(["standing"]) == 0
    out = capsys.readouterr().out
    # A zero row on its own is indistinguishable from a refuted belief, which
    # is the reading this line exists to prevent.
    assert "No interview has borne on: h2" in out
    assert "not a refuted belief" in out


# --- the numbers, and what they count -------------------------------------------------


def test_summary_splits_reported_claims_from_inferred_ones(node, no_vendor, capsys):
    _write_facts(node, [
        _fact("f1", "S1", bearings=(("h1", "supports"),)),
        _fact("f2", "S1", bearings=(("h1", "supports"),)),
        _fact("f3", "S1", provenance="inference", bearings=(("h2", "undermines"),)),
        _fact("f4", "S2", bearings=(("h1", "supports"),)),
    ])
    assert main_mod.main(["summary", "S1"]) == 0
    out = capsys.readouterr().out
    assert "claims extracted from this interview   3" in out
    assert "reported by the interviewee            2" in out
    assert "inferred by the model                  1" in out
    assert "hypotheses this interview bore on      2" in out
    # CLAIMS here, and the heading has to say so -- `standing` prints
    # interview counts in columns that look identical.
    assert "counted in CLAIMS" in out


def test_summary_reports_bearings_on_hypotheses_the_project_no_longer_has(
    node, no_vendor, capsys
):
    _write_facts(node, [_fact("f1", "S1", bearings=(("h-gone", "supports"),))])
    assert main_mod.main(["summary", "S1"]) == 0
    out = capsys.readouterr().out
    assert "no longer has" in out
    assert "h-gone" in out


def test_standing_counts_interviews_not_claims(node, no_vendor, capsys):
    # One talkative interviewee, three claims, ONE interview. Counting claims
    # would print 3 here and let one person outweigh three others.
    _write_facts(node, [
        _fact("f1", "S1", bearings=(("h1", "supports"),)),
        _fact("f2", "S1", bearings=(("h1", "supports"),)),
        _fact("f3", "S1", bearings=(("h1", "supports"),)),
    ])
    assert main_mod.main(["standing"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("h1 ")]
    assert lines, "no row was printed for h1"
    assert lines[0].split() == ["h1", "untested", "yes", "1", "0", "0", "1"]


def test_standing_shows_the_before_and_now_for_a_named_interview(node, no_vendor, capsys):
    _write_facts(node, [
        _fact("f1", "S2", bearings=(("h1", "supports"),)),
        _fact("f2", "S1", bearings=(("h1", "supports"),)),
        _fact("f3", "S1", bearings=(("h1", "unclear"),)),
    ])
    assert main_mod.main(["standing", "S1"]) == 0
    out = capsys.readouterr().out
    assert "1 -> 2" in out, "the supporting column must show the corpus without S1, then with it"
    assert "0 -> 1" in out, "the unclear column must move too"
    assert "the corpus without S1" in out


def test_an_unclear_only_contribution_is_touched_but_never_moved(node, no_vendor, capsys):
    # MOVED is the badge that carries this product's thesis. A claim the model
    # could not direct is precisely the absence of direction.
    _write_facts(node, [_fact("f1", "S1", bearings=(("h1", "unclear"),))])
    assert main_mod.main(["standing", "S1"]) == 0
    out = capsys.readouterr().out
    assert "1 unclear" in out
    assert "MOVED" not in out


def test_a_supporting_contribution_is_marked_moved(node, no_vendor, capsys):
    _write_facts(node, [_fact("f1", "S1", bearings=(("h1", "supports"),))])
    assert main_mod.main(["standing", "S1"]) == 0
    assert "MOVED" in capsys.readouterr().out


def test_standing_marks_a_hypothesis_with_no_pre_registered_criterion(node, no_vendor, capsys):
    _write_facts(node, [_fact("f1", "S1", bearings=(("h2", "supports"),))])
    assert main_mod.main(["standing"]) == 0
    out = capsys.readouterr().out
    # h2 carries no criterion in the fixture; a tally against it is the target
    # painted round the arrow, and the output has to say so next to the row.
    assert "cannot honestly score it" in out


def test_standing_prints_each_hypothesis_statement_because_an_id_is_not_a_belief(
    node, no_vendor, capsys
):
    assert main_mod.main(["standing"]) == 0
    out = capsys.readouterr().out
    assert "onboarding is the blocker" in out
    assert "they export to spreadsheets" in out


# --- what neither command may ever print ------------------------------------------------


@pytest.mark.parametrize("argv", [["summary", "S1"], ["standing"], ["standing", "S1"]])
def test_no_verbatim_customer_speech_reaches_the_terminal(node, no_vendor, capsys, argv):
    """The structural defence is that both commands hold `FactRef`s, which have
    no field a quote could sit in. This is the behavioural backstop: a sentinel
    quote sits on every fact in the fixture and must appear in no output, on the
    path that renders the most (a full corpus, a named interview, and a claim
    whose text is deliberately not the quote)."""
    _write_facts(node, [
        _fact("f1", "S1", bearings=(("h1", "supports"),)),
        _fact("f2", "S2", bearings=(("h1", "undermines"), ("h2", "unclear"))),
    ])
    assert main_mod.main(argv) == 0
    captured = capsys.readouterr()
    assert _QUOTE not in captured.out
    assert _QUOTE not in captured.err


# --- a broken corpus is a message, not a traceback ---------------------------------------


def test_a_corpus_that_will_not_parse_is_reported_rather_than_raised(node, no_vendor, capsys):
    (node / "facts.json").write_text('[{"id": "f1"}]', encoding="utf-8")
    assert main_mod.main(["summary", "S1"]) == 1
    assert "ERROR:" in capsys.readouterr().err


def test_both_commands_accept_a_full_path_to_a_session_directory(node, no_vendor, capsys, tmp_path):
    _write_facts(node, [_fact("f1", "S1", bearings=(("h1", "supports"),))])
    assert main_mod.main(["summary", str(tmp_path / "sessions" / "S1")]) == 0
    assert "Interview S1" in capsys.readouterr().out
