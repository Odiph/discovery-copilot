"""Phase A: cross-hierarchy reads.

The user's rule, and the whole of what this file protects: **reads may merge,
writes never do**, and merging is allowed only within one hierarchy and only
on explicit request. The brief and the survey always come from the filing node
alone -- they are what this interview is about, and quietly merging another
node's survey would change the questions being asked in the room.
"""

from __future__ import annotations

import json

import pytest

from app.coach.context import FACT_ID_PREFIX, load_brief, load_grounded_brief
from app.main import _resolve_extra_fact_dirs, build_arg_parser
from app.projects import ProjectPathError


def _parse(*argv: str):
    return build_arg_parser().parse_args(list(argv))


def _node(tmp_path, name, *, facts=(), claim_word="onboarding"):
    """A project node with a minimal brief, a survey that makes `claim_word`
    a relevant query token, and the given facts."""
    context_dir = tmp_path / name / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "brief.json").write_text(
        json.dumps(
            [{"id": f"b-{name}", "text": f"brief for {name}", "source": "pm", "provenance": "evidence"}]
        ),
        encoding="utf-8",
    )
    (context_dir / "survey.json").write_text(
        # No trailing punctuation on the keyword, deliberately: `_query_tokens`
        # casefolds and splits on WHITESPACE only, so "onboarding." and
        # "onboarding" are different tokens and score zero overlap. A fixture
        # that got this wrong would make every retrieval assertion below pass
        # or fail for a reason unrelated to what it claims to test.
        json.dumps([{"id": f"q-{name}", "order": 1, "text": f"Tell me about {claim_word} please"}]),
        encoding="utf-8",
    )
    if facts:
        (context_dir / "facts.json").write_text(json.dumps(list(facts)), encoding="utf-8")
    return context_dir


def _fact(fact_id, session_id, claim, *, seqs=(1,)):
    return {
        "id": fact_id,
        "claim": claim,
        "provenance": "evidence",
        "session_id": session_id,
        "seqs": list(seqs),
        "quoted_text": claim,
        "bearings": [],
    }


# --- the default path is unchanged ------------------------------------------------------


def test_no_extra_sources_is_byte_identical_to_load_brief(tmp_path):
    # `load_brief` IS `load_grounded_brief` with no sources -- D-03's single
    # positional signature is preserved rather than widened, because the
    # CTX-03 seam is proven by a caller that only ever calls
    # `load_brief(context_dir)`.
    context_dir = _node(tmp_path, "acme", facts=[_fact("f1", "S1", "onboarding was slow")])
    assert load_grounded_brief(context_dir, extra_fact_dirs=()) == load_brief(context_dir)


def test_a_filing_node_fact_keeps_its_unqualified_id(tmp_path):
    # Byte stability of every single-project run depends on this: the filing
    # node's mapped facts must look exactly as they did before Phase A.
    context_dir = _node(tmp_path, "acme", facts=[_fact("f1", "S1", "onboarding was slow")])
    (item,) = [i for i in load_brief(context_dir) if i.id.startswith(FACT_ID_PREFIX)]
    assert item.id == "fact:f1"
    assert item.citation == "session:S1#seq:1"
    assert "project" not in item.source


# --- what merges, and what must not -----------------------------------------------------


def test_an_extra_nodes_facts_are_merged_in(tmp_path):
    filing = _node(tmp_path, "acme", facts=[_fact("f1", "S1", "onboarding was slow")])
    other = _node(tmp_path, "acme-billing", facts=[_fact("f2", "S2", "onboarding needed a call")])
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert {i.text for i in items if i.id.startswith(FACT_ID_PREFIX)} == {
        "onboarding was slow",
        "onboarding needed a call",
    }


def test_the_brief_comes_from_the_filing_node_alone(tmp_path):
    filing = _node(tmp_path, "acme")
    other = _node(tmp_path, "acme-billing", facts=[_fact("f2", "S2", "onboarding needed a call")])
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    hand_written = [i for i in items if not i.id.startswith(FACT_ID_PREFIX)]
    assert [i.id for i in hand_written] == ["b-acme"]


def test_the_survey_is_never_merged(tmp_path):
    # Asserted through the retrieval it drives: the extra node's survey names
    # a word the filing node's does not, and a fact relevant ONLY to that
    # word must not be selected. If surveys merged, this fact would appear.
    filing = _node(tmp_path, "acme", claim_word="onboarding")
    other = _node(tmp_path, "acme-billing", claim_word="dunning",
                  facts=[_fact("f2", "S2", "dunning emails bounced")])
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert [i for i in items if i.id.startswith(FACT_ID_PREFIX)] == []


def test_reading_a_node_leaves_its_files_byte_identical(tmp_path):
    # "Reads merge, writes never do", asserted rather than trusted.
    filing = _node(tmp_path, "acme")
    other = _node(tmp_path, "acme-billing", facts=[_fact("f2", "S2", "onboarding needed a call")])
    before = {p.name: p.read_bytes() for p in sorted(other.iterdir())}
    load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert {p.name: p.read_bytes() for p in sorted(other.iterdir())} == before


# --- fact ids must be qualified ---------------------------------------------------------


def test_a_merged_fact_is_qualified_with_the_node_it_came_from(tmp_path):
    filing = _node(tmp_path, "acme")
    other = _node(tmp_path, "acme-billing", facts=[_fact("f2", "S2", "onboarding needed a call")])
    (item,) = [
        i for i in load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
        if i.id.startswith(FACT_ID_PREFIX)
    ]
    assert item.id == "fact:acme/billing:f2"
    assert item.citation == "project:acme/billing#session:S2#seq:1"
    assert "acme/billing" in item.source


def test_two_nodes_with_the_same_session_id_stay_distinguishable(tmp_path):
    # The reason qualification is not cosmetic. Session ids are
    # `%Y%m%dT%H%M%S` and `create_session_dir` only checks collisions WITHIN
    # one root, so two projects genuinely can mint the same one. Unqualified,
    # these two facts would present as one provenance chain across two
    # customers.
    filing = _node(tmp_path, "acme", facts=[_fact("f1", "20260810T120000", "onboarding was slow")])
    other = _node(
        tmp_path, "acme-billing",
        facts=[_fact("f2", "20260810T120000", "onboarding needed a call")],
    )
    items = [
        i for i in load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
        if i.id.startswith(FACT_ID_PREFIX)
    ]
    assert len({i.citation for i in items}) == 2
    assert len({i.id for i in items}) == 2


# --- pooling, not per-directory selection ----------------------------------------------


def test_the_fact_cap_applies_once_across_all_sources(tmp_path):
    # Per-directory selection was the obvious alternative and would apply the
    # cap once per node -- three sources would put four times the facts into
    # the brief, spending cache tokens and diluting it, which is the exact
    # effect the cap exists to prevent.
    from app import config as config_mod

    cap = config_mod.DEFAULT_COACH_MAX_RETRIEVED_FACTS
    filing = _node(
        tmp_path, "acme",
        facts=[_fact(f"a{n}", "S1", f"onboarding note {n}", seqs=(n,)) for n in range(cap)],
    )
    other = _node(
        tmp_path, "acme-billing",
        facts=[_fact(f"b{n}", "S2", f"onboarding note {n}", seqs=(n,)) for n in range(cap)],
    )
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert len([i for i in items if i.id.startswith(FACT_ID_PREFIX)]) == cap


def test_an_extra_node_with_no_corpus_yet_does_not_suppress_the_filing_nodes_facts(tmp_path):
    # The pre-Phase-A early return on an absent facts.json was safe only
    # because there was exactly one source. Checked per source now.
    filing = _node(tmp_path, "acme", facts=[_fact("f1", "S1", "onboarding was slow")])
    other = _node(tmp_path, "acme-billing")
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert [i.text for i in items if i.id.startswith(FACT_ID_PREFIX)] == ["onboarding was slow"]


def test_a_filing_node_with_no_corpus_still_reads_the_extra_nodes(tmp_path):
    filing = _node(tmp_path, "acme")
    other = _node(tmp_path, "acme-billing", facts=[_fact("f2", "S2", "onboarding needed a call")])
    items = load_grounded_brief(filing, extra_fact_dirs=[("acme/billing", other)])
    assert [i.text for i in items if i.id.startswith(FACT_ID_PREFIX)] == ["onboarding needed a call"]


# --- the CLI rule: same hierarchy, explicit, no duplicates ------------------------------


def test_a_source_under_a_different_root_is_refused():
    with pytest.raises(ProjectPathError, match="different root"):
        _resolve_extra_fact_dirs(
            _parse("start", "--project", "acme/onboarding", "--read-from", "globex/billing")
        )


def test_a_source_in_the_same_hierarchy_is_accepted():
    (node, context_dir), = _resolve_extra_fact_dirs(
        _parse("start", "--project", "acme/onboarding", "--read-from", "acme/billing")
    )
    assert node == "acme/billing"
    assert context_dir.name == "context"


def test_read_from_without_a_project_is_refused():
    # The root context/ directory is not part of the project tree, so there
    # is no hierarchy the request could be checked against.
    with pytest.raises(ProjectPathError, match="needs --project"):
        _resolve_extra_fact_dirs(_parse("start", "--read-from", "acme/billing"))


def test_naming_the_filing_node_as_a_source_is_refused():
    with pytest.raises(ProjectPathError, match="filing node itself"):
        _resolve_extra_fact_dirs(
            _parse("start", "--project", "acme/onboarding", "--read-from", "acme/onboarding")
        )


def test_a_repeated_source_is_refused_rather_than_silently_deduplicated():
    with pytest.raises(ProjectPathError, match="twice"):
        _resolve_extra_fact_dirs(
            _parse(
                "start", "--project", "acme/onboarding",
                "--read-from", "acme/billing", "--read-from", "acme/billing",
            )
        )


def test_a_hostile_source_path_is_refused():
    with pytest.raises(ProjectPathError):
        _resolve_extra_fact_dirs(
            _parse("start", "--project", "acme", "--read-from", "../../etc")
        )


def test_no_read_from_resolves_to_an_empty_source_set():
    assert _resolve_extra_fact_dirs(_parse("start", "--project", "acme")) == ()
    assert _resolve_extra_fact_dirs(_parse("start")) == ()


def test_several_sources_are_kept_in_the_order_the_operator_gave_them():
    resolved = _resolve_extra_fact_dirs(
        _parse(
            "start", "--project", "acme/onboarding",
            "--read-from", "acme/billing", "--read-from", "acme/support",
        )
    )
    assert [node for node, _ in resolved] == ["acme/billing", "acme/support"]


def test_only_start_offers_read_from():
    # render/author/extract do not ground the coach, so the flag would be
    # accepted and ignored there.
    for command, argv in (
        ("render", ["render", "S1"]), ("author", ["author"]), ("extract", ["extract", "S1"]),
    ):
        with pytest.raises(SystemExit):
            _parse(*argv, "--read-from", "acme/billing")


# --- the operator can always see which knowledge is in play -----------------------------


def test_hello_carries_the_project_and_the_full_source_set():
    from app.web.hub import hello_payload

    payload = hello_payload(
        session_id="s", server_started_at="t",
        heartbeat_interval_ms=1, heartbeat_timeout_ms=2,
        reconnect_backoff_base_ms=3, reconnect_backoff_cap_ms=4,
        project="acme/onboarding",
        grounding_sources=["acme/billing", "acme/support"],
        coach_enabled=True,
    )
    assert payload["project"] == "acme/onboarding"
    assert payload["grounding_sources"] == ["acme/billing", "acme/support"]


def test_an_unprojected_run_says_so_explicitly_rather_than_omitting_the_keys():
    # The client branches on `project` being falsy. A MISSING key and a null
    # one behave the same in JS, but only one of them is inspectable in a
    # captured wire log when someone is working out why the strip did not
    # appear.
    from app.web.hub import hello_payload

    payload = hello_payload(
        session_id="s", server_started_at="t",
        heartbeat_interval_ms=1, heartbeat_timeout_ms=2,
        reconnect_backoff_base_ms=3, reconnect_backoff_cap_ms=4,
        project=None, grounding_sources=(), coach_enabled=False,
    )
    assert payload["project"] is None and payload["grounding_sources"] == []


def test_the_source_list_carries_node_paths_and_nothing_else():
    # Gate 12's no-text rule restated where it is easy to break: a future
    # edit that helpfully attached each node's fact COUNT, label or brief
    # would be putting derived corpus content onto a screen that may be
    # sitting in a cafe.
    from app.web.hub import hello_payload

    payload = hello_payload(
        session_id="s", server_started_at="t",
        heartbeat_interval_ms=1, heartbeat_timeout_ms=2,
        reconnect_backoff_base_ms=3, reconnect_backoff_cap_ms=4,
        project="acme", grounding_sources=["acme/billing"], coach_enabled=True,
    )
    assert payload["grounding_sources"] == ["acme/billing"]
    assert all(isinstance(entry, str) for entry in payload["grounding_sources"])


def test_the_page_ships_a_grounding_region_that_starts_hidden():
    from app.tests.test_web_serving import STATIC_DIR

    markup = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="grounding-region"' in markup
    assert "hidden" in markup.split('id="grounding-region"')[1].split(">")[0], (
        "an unprojected run must look exactly as it did before this strip existed"
    )


def test_the_client_writes_the_project_with_textcontent_never_innerhtml():
    # A project path is operator-typed input that reaches the DOM. This
    # page's whole security posture is that nothing it renders can be script.
    from app.tests.test_web_serving import STATIC_DIR

    source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    body = source.split("function renderGrounding")[1].split("function clearChildren")[0]
    assert "textContent" in body
    assert "innerHTML" not in body
