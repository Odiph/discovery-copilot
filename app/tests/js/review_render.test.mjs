// app/tests/js/review_render.test.mjs -- node's built-in test runner
// (`node --test`), no npm, no package.json, no dependency of any kind, the
// same rule `live_state.test.mjs` follows. `app/tests/test_web_review_render.py`
// shells out to node so this file runs inside the ordinary pytest run.
//
// What it defends: the hub renders the CHANGE this interview made. A payload
// assertion cannot catch a page that computes a delta correctly and then draws
// a running total, and the whole defect this file exists for was exactly that
// -- `standing_for` already tagged every entry `from_this_interview` and the
// client rendered one bold font weight. So this drives the REAL `review.js`
// against a minimal DOM and reads what a human would read.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REVIEW_JS = path.join(HERE, "..", "..", "static", "review", "review.js");

// --- the smallest DOM that can host this page ------------------------------------------------

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.className = "";
    this.id = "";
    this.hidden = false;
    this.disabled = false;
    this.children = [];
    this._text = null;
    this.listeners = {};
    this.scrolled = false;
    // The talk-share bars set `style.width`, which is the only inline style
    // this page writes. A plain object is enough to record it, and recording
    // it is what lets a test assert the BAR agrees with the number printed
    // beside it -- a bar drawn at a width that contradicts its own label is
    // exactly the defect a textContent-only harness cannot see.
    this.style = {};
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = value; this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  click() { (this.listeners.click || []).forEach((fn) => fn({ preventDefault() {} })); }
  scrollIntoView() { this.scrolled = true; }
}

function walk(node, out) {
  out.push(node);
  node.children.forEach((child) => walk(child, out));
  return out;
}

/** Everything a human would read out of this subtree, in order. */
function readable(node) {
  return walk(node, [])
    .map((n) => (n._text === null || n._text === undefined ? "" : String(n._text)))
    .filter((t) => t !== "")
    .join(" | ");
}

function findAll(node, predicate) {
  return walk(node, []).filter(predicate);
}

const PAGE_IDS = [
  "heading", "subheading", "extraction", "standings", "no-hypotheses",
  "no-proposals", "proposals", "summary", "transcript", "no-transcript",
  "page-error", "coverage", "coverage-note", "picker-row", "session-picker",
  "picker-note", "new-hypothesis", "save-banner", "participant",
  "talk-share", "talk-share-note",
];

function makeDocument() {
  const byId = {};
  PAGE_IDS.forEach((id) => {
    const node = new FakeNode("div");
    node.id = id;
    byId[id] = node;
  });
  return {
    byId,
    createElement(tag) { return new FakeNode(tag); },
    getElementById(id) {
      if (byId[id]) { return byId[id]; }
      for (const root of PAGE_IDS.map((key) => byId[key])) {
        const hit = findAll(root, (n) => n.id === id);
        if (hit.length) { return hit[0]; }
      }
      return null;
    },
  };
}

const source = fs.readFileSync(REVIEW_JS, "utf8");

const DEFAULT_SESSIONS = {
  project: "acme",
  sessions: [{ id: "S1", rendered: true }],
  selected: "S1",
};

async function render(review, proposals, sessions, options) {
  const failPost = (options || {}).failPost;
  // What a successful POST answers. `{ saved: true }` is the authoring save's
  // own body and stays the default; the decision route answers something else
  // entirely (`applied`/`recorded`), and a harness that could only speak one
  // of the two would make the other's branches untestable.
  const postBody = (options || {}).postBody;
  const document = makeDocument();
  // Routed by PREFIX, never by equality: the page appends `?session=` to both
  // the review and the proposals request, and an equality match would quietly
  // send those to the wrong branch and answer with the wrong body.
  const requested = [];
  const posted = [];
  const fetch = (url, options) => {
    requested.push(url);
    if (options && options.method === "POST") {
      posted.push({ url, body: JSON.parse(options.body) });
      // What the real save path answers: a body with no `error` on success,
      // and `{"error": ...}` with a 4xx on a refusal.
      return Promise.resolve({
        ok: !failPost,
        json: () => Promise.resolve(
          failPost ? { error: failPost } : (postBody || { saved: true })
        ),
      });
    }
    let body = review;
    if (url.startsWith("/api/proposals")) { body = { proposals: proposals || [] }; }
    else if (url.startsWith("/api/sessions")) {
      body = sessions === undefined ? DEFAULT_SESSIONS : sessions;
    } else if (url.startsWith("/api/hypotheses")) {
      // The stored array the page read-modify-writes, projected from the same
      // standings the page is rendering so the two cannot disagree.
      body = (review.standings || []).map((s) => ({
        id: s.id, statement: s.statement, falsifier: s.falsifier,
        criterion: s.criterion, state: s.state,
      }));
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });
  };
  new Function("document", "fetch", source)(document, fetch);
  // Three chained fetches now (sessions -> review -> proposals), each a few
  // `.then` hops deep; drain the microtask queue generously.
  for (let i = 0; i < 60; i += 1) { await Promise.resolve(); }
  document.requested = requested;
  document.posted = posted;
  document.settle = async () => {
    for (let i = 0; i < 60; i += 1) { await Promise.resolve(); }
  };
  return document;
}

/** The first descendant of `root` whose tag matches, in document order. */
function firstTag(root, tag) {
  return findAll(root, (n) => n.tagName === tag)[0];
}

// --- fixtures ---------------------------------------------------------------------------------

function factEntry(overrides) {
  return Object.assign({
    fact_id: "f1",
    claim: "they went back to paper",
    quoted_text: "we went back to paper",
    provenance: "evidence",
    session_id: "S1",
    seqs: [12],
    from_this_interview: true,
  }, overrides || {});
}

function standing(overrides) {
  const base = {
    id: "h1",
    statement: "Teams abandon the tool after onboarding",
    falsifier: "a team still using it at ninety days",
    state: "open",
    criterion: "",
    is_scorable: false,
    supporting: [],
    contrary: [],
    unclear: [],
    supporting_interviews: 0,
    contrary_interviews: 0,
    unclear_interviews: 0,
    total_interviews: 0,
    before: { supporting_interviews: 0, contrary_interviews: 0, unclear_interviews: 0 },
    this_interview: { supporting_facts: 0, contrary_facts: 0, unclear_facts: 0 },
    touched_this_interview: false,
    moved_this_interview: false,
  };
  return Object.assign(base, overrides || {});
}

function review(overrides) {
  return Object.assign({
    project: "acme",
    session_id: "S1",
    extraction: null,
    standings: [],
    transcript: [],
    summary: null,
    // Present and null by default, because that is what the server sends for
    // every interview recorded before the participant field existed -- which
    // is all of them. A fixture that defaulted to a name would make the empty
    // state, the common one, the case nothing exercised.
    participant: null,
    // Matches the default `transcript: []` above: no words, so nothing to
    // measure. Defaulting to a measurable meter would put a balanced-looking
    // pair of bars under every fixture that never mentioned talk share, and
    // the absence -- which is what an empty transcript actually renders -- would
    // be the case nothing exercised.
    talk_share: {
      measurable: false, total_words: 0, speakers: {},
      unattributed_words: 0, interviewer_key: "you",
    },
    // The ABSENT states, because they are the common ones. The question
    // critique is a paid model call the operator has to run deliberately, and
    // the rejection log only exists for projects extracted since it was
    // written -- so a fixture that defaulted either to data would make the
    // "never run" rendering, which is what most readers will actually see, the
    // case nothing exercised. That is the exact mistake `extraction: null` made
    // until the 2026-08-12 audit.
    question_critique: { status: "not-run", error: null, summary: null },
    rejections: {
      status: "not-kept", error: null, total: 0, reasons: {},
      sessions_recorded: 0, this_interview: null,
    },
  }, overrides || {});
}

/** A measurable meter. `you`/`them` are the labels the server projects onto
    interviewer/interviewee, and the client is told which is which rather than
    guessing -- so the fixture says it too. */
function meter(speakers, extra) {
  const total = Object.keys(speakers).reduce((n, k) => n + speakers[k], 0)
    + ((extra || {}).unattributed_words || 0);
  const out = {};
  Object.keys(speakers).forEach((name) => {
    out[name] = { words: speakers[name], share: speakers[name] / total };
  });
  return Object.assign({
    measurable: true, total_words: total, speakers: out,
    unattributed_words: 0, interviewer_key: "you",
  }, extra || {});
}

const cardText = (document) => readable(document.byId.standings);

// --- touched is not moved ---------------------------------------------------------------------

test("a fact this interview could not direct does not read as movement", async () => {
  // The badge carries the whole thesis of the product. A call that produced
  // one undirected claim must not light it.
  const document = await render(review({
    standings: [standing({
      unclear: [factEntry({})],
      unclear_interviews: 1,
      this_interview: { supporting_facts: 0, contrary_facts: 0, unclear_facts: 1 },
      touched_this_interview: true,
      moved_this_interview: false,
    })],
  }));
  const text = cardText(document);
  assert.ok(!text.includes("moved this interview"), `read as moved: ${text}`);
  assert.ok(text.includes("asked, unresolved"), `did not report the touch: ${text}`);
});

test("a directed fact from this interview does read as movement", async () => {
  const document = await render(review({
    standings: [standing({
      contrary: [factEntry({})],
      contrary_interviews: 1,
      this_interview: { supporting_facts: 0, contrary_facts: 1, unclear_facts: 0 },
      touched_this_interview: true,
      moved_this_interview: true,
    })],
  }));
  assert.ok(cardText(document).includes("moved this interview"));
});

// --- the delta ---------------------------------------------------------------------------------

test("the page renders a before and an after, not one running total", async () => {
  const document = await render(review({
    standings: [standing({
      supporting: [factEntry({ from_this_interview: false }), factEntry({ fact_id: "f2" })],
      contrary: [factEntry({ fact_id: "f3" })],
      supporting_interviews: 2,
      contrary_interviews: 1,
      before: { supporting_interviews: 2, contrary_interviews: 0, unclear_interviews: 0 },
      this_interview: { supporting_facts: 1, contrary_facts: 1, unclear_facts: 0 },
      touched_this_interview: true,
      moved_this_interview: true,
    })],
  }));
  const text = cardText(document);
  assert.ok(text.includes("2 for / 0 against"), `no before: ${text}`);
  assert.ok(text.includes("2 for / 1 against"), `no after: ${text}`);
  assert.ok(/was .*→.* now/.test(text) || text.includes("was") && text.includes("now"),
    `the before and after are not shown as a change: ${text}`);
});

test("an interview that contributed nothing says so instead of showing the corpus", async () => {
  // The flattering failure: a rich standing built by five earlier interviews
  // read as the size of this afternoon's work.
  const document = await render(review({
    standings: [standing({
      supporting: [factEntry({ from_this_interview: false, session_id: "S0" })],
      supporting_interviews: 5,
      before: { supporting_interviews: 5, contrary_interviews: 0, unclear_interviews: 0 },
    })],
  }));
  const text = cardText(document);
  assert.ok(/this interview (added )?nothing|unchanged by this interview/i.test(text),
    `silent about contributing nothing: ${text}`);
  assert.ok(!text.includes("moved this interview"));
});

// --- the falsifier ------------------------------------------------------------------------------

test("the falsifier reaches the screen, not merely the payload", async () => {
  // It is what makes the anti-confirmation critic checkable, and this is the
  // one page where a human decides whether the belief survived.
  const document = await render(review({ standings: [standing({})] }));
  assert.ok(cardText(document).includes("a team still using it at ninety days"));
});

// --- symmetry ------------------------------------------------------------------------------------

test("the unclear column is drawn even when it is empty", async () => {
  // Same symmetry the other two columns already keep: a claim the model could
  // not direct must not vanish without trace, and neither must its absence.
  const document = await render(review({ standings: [standing({})] }));
  const text = cardText(document);
  assert.ok(/unresolved/i.test(text), `no unclear column: ${text}`);
});

// --- citations are followable ---------------------------------------------------------------------

test("a citation into this interview is an anchor onto the transcript turn", async () => {
  const document = await render(review({
    session_id: "S1",
    standings: [standing({
      supporting: [factEntry({ seqs: [12, 13, 14] })],
      supporting_interviews: 1,
      before: { supporting_interviews: 0, contrary_interviews: 0, unclear_interviews: 0 },
      this_interview: { supporting_facts: 1, contrary_facts: 0, unclear_facts: 0 },
      touched_this_interview: true,
      moved_this_interview: true,
    })],
    transcript: [
      { seq: 12, speaker: "them", text: "we went back to paper" },
      { seq: 13, speaker: "you", text: "why?" },
      { seq: 14, speaker: "them", text: "it was faster" },
    ],
  }));
  const anchors = findAll(document.byId.standings, (n) => n.tagName === "a");
  assert.equal(anchors.length, 1, "the citation is not a link");
  assert.equal(anchors[0].href, "#turn-12");

  const turns = findAll(document.byId.transcript, (n) => String(n.id).startsWith("turn-"));
  assert.deepEqual(turns.map((t) => t.id), ["turn-12", "turn-13", "turn-14"]);

  anchors[0].click();
  const lit = turns.filter((t) => /\bhit\b/.test(t.className)).map((t) => t.id);
  assert.deepEqual(lit, ["turn-12", "turn-13", "turn-14"], "the cited span was not highlighted");
  assert.ok(turns[0].scrolled, "the transcript was not brought to the citation");
});

test("a citation into an earlier interview is not a link into this transcript", async () => {
  // The dangling anchor is the sneaky failure: it looks followable, and it
  // scrolls the reader to a turn from a different conversation, or nowhere.
  const document = await render(review({
    session_id: "S1",
    standings: [standing({
      supporting: [factEntry({ session_id: "S0", from_this_interview: false, seqs: [12] })],
      supporting_interviews: 1,
      before: { supporting_interviews: 1, contrary_interviews: 0, unclear_interviews: 0 },
    })],
    transcript: [{ seq: 12, speaker: "them", text: "a different conversation" }],
  }));
  assert.equal(findAll(document.byId.standings, (n) => n.tagName === "a").length, 0);
  assert.ok(cardText(document).includes("#12"), "the citation itself disappeared");
});

test("a highlight is cleared before the next citation is followed", async () => {
  const document = await render(review({
    session_id: "S1",
    standings: [standing({
      supporting: [factEntry({ fact_id: "f1", seqs: [12] })],
      contrary: [factEntry({ fact_id: "f2", seqs: [14] })],
      supporting_interviews: 1,
      contrary_interviews: 1,
      before: { supporting_interviews: 0, contrary_interviews: 0, unclear_interviews: 0 },
      this_interview: { supporting_facts: 1, contrary_facts: 1, unclear_facts: 0 },
      touched_this_interview: true,
      moved_this_interview: true,
    })],
    transcript: [
      { seq: 12, speaker: "them", text: "first" },
      { seq: 14, speaker: "them", text: "second" },
    ],
  }));
  const anchors = findAll(document.byId.standings, (n) => n.tagName === "a");
  anchors[0].click();
  anchors[1].click();
  const turns = findAll(document.byId.transcript, (n) => String(n.id).startsWith("turn-"));
  const lit = turns.filter((t) => /\bhit\b/.test(t.className)).map((t) => t.id);
  assert.deepEqual(lit, ["turn-14"], "the page accumulates highlights");
});

test("a turn with no sequence number is still rendered", async () => {
  // Anchoring must not become a reason for a line of the transcript to vanish.
  const document = await render(review({
    session_id: "S1",
    standings: [],
    transcript: [{ seq: null, speaker: "them", text: "a line with no seq" }],
  }));
  assert.ok(readable(document.byId.transcript).includes("a line with no seq"));
});

// --- the extraction banner: rendered here for the first time ---------------------------------
//
// Added after the 2026-08-12 audit, which found that `extraction` was in
// PAGE_IDS and that NO test in this file ever rendered it -- every fixture
// above passes `extraction: null`, which makes `renderExtraction` return on its
// first line. So `review.py` emitted status literals, `review.js` keyed a
// lookup table on the same literals by hand, and a rename on either side gave
// `headline === undefined`, an empty banner, and a fully green suite.
//
// `app/tests/test_web_review_extraction_status.py` holds the other half: it
// pins the JS table's keys against `review.ALL_EXTRACTION_STATUSES`, so the two
// hand-typed lists cannot agree with each other while disagreeing with Python.

function extraction(overrides) {
  return Object.assign({
    status: "extracted",
    facts_total: 3,
    facts_this_interview: 2,
    facts_without_bearings: 1,
    hypotheses_loaded: 2,
  }, overrides || {});
}

const bannerText = (document) => readable(document.byId.extraction);

test("an un-extracted interview says so instead of showing an empty banner", async () => {
  const document = await render(review({
    extraction: extraction({ status: "not-extracted", facts_this_interview: 0 }),
  }));
  assert.equal(document.byId.extraction.hidden, false);
  assert.ok(
    bannerText(document).includes("has not been extracted"),
    `the banner said nothing: ${bannerText(document)}`,
  );
  assert.ok(
    bannerText(document).includes("python -m app extract"),
    "the remedy is the only place in the product that names the command",
  );
});

test("a corpus with nothing from this interview is not called un-extracted", async () => {
  // THE SPLIT. Before it, this rendered "This interview has not been
  // extracted" above a remedy telling the operator to run the command they may
  // well have just run.
  const document = await render(review({
    extraction: extraction({
      status: "no-facts-from-this-interview",
      facts_total: 20,
      facts_this_interview: 0,
    }),
  }));
  const text = bannerText(document);
  // The DEFINITIVE claim, not the phrase. The honest wording legitimately
  // contains "has not been extracted" inside "Either it has not been
  // extracted, or ..." -- what must be gone is the page asserting it.
  assert.ok(
    !text.includes("This interview has not been extracted"),
    `still states un-extracted as a fact: ${text}`,
  );
  assert.ok(text.includes("no claims from this interview"), text);
  assert.ok(
    text.includes("nothing on disk distinguishes"),
    "the page must not pick one of the two causes it cannot tell apart",
  );
});

test("an extracted interview reports its own counts, not the corpus total", async () => {
  // The trap this wording was written against: a corpus made rich by EARLIER
  // interviews must not make a session that contributed nothing look examined.
  const document = await render(review({
    extraction: extraction({ facts_total: 40, facts_this_interview: 2, facts_without_bearings: 2 }),
  }));
  const text = bannerText(document);
  assert.ok(text.includes("2 facts extracted from this interview"), text);
  assert.ok(text.includes("2 bearing on no hypothesis"), text);
  assert.ok(text.includes("40 facts in the corpus"), text);
});

test("a singular count is not written as '1 facts'", async () => {
  const document = await render(review({
    extraction: extraction({ facts_this_interview: 1, facts_without_bearings: 0 }),
  }));
  assert.ok(bannerText(document).includes("1 fact extracted"), bannerText(document));
});

test("no hypotheses is worded as a fact about the project, not about the call", async () => {
  const document = await render(review({
    extraction: extraction({ status: "no-hypotheses", hypotheses_loaded: 0 }),
  }));
  assert.ok(bannerText(document).includes("No hypotheses are authored"), bannerText(document));
});

test("an unreadable corpus is stated rather than rendered as zero", async () => {
  const document = await render(review({
    extraction: extraction({ status: "unreadable" }),
  }));
  assert.ok(bannerText(document).includes("could not be read"), bannerText(document));
});

test("every status the page knows about carries a real sentence", async () => {
  // Exhaustive over the vocabulary rather than over the statuses someone
  // remembered to test. A status added to review.py without a sentence here
  // fails once the Python-side pin adds it to this list.
  const statuses = [
    "extracted", "not-extracted", "no-facts-from-this-interview",
    "no-hypotheses", "unreadable",
  ];
  for (const status of statuses) {
    const document = await render(review({ extraction: extraction({ status }) }));
    const text = bannerText(document);
    assert.ok(text.length > 20, `status ${status} rendered an empty banner: "${text}"`);
    assert.ok(
      !text.includes("does not recognise"),
      `status ${status} fell through to the unknown-status branch`,
    );
    assert.equal(
      document.byId.extraction.className,
      "extraction " + status,
      `status ${status} did not reach the CSS class, so it renders in the wrong colour`,
    );
  }
});

test("a status this page has never heard of says so instead of going blank", async () => {
  // The exact failure the audit predicted for a rename: `said[status]` is
  // `undefined`, and `undefined` used to be appended as an empty headline.
  const document = await render(review({
    extraction: extraction({ status: "renamed-by-someone" }),
  }));
  const text = bannerText(document);
  assert.ok(text.includes("does not recognise"), `went quiet on an unknown status: "${text}"`);
  assert.ok(text.includes("renamed-by-someone"), "name the status so it can be found in the code");
});

// --- survey coverage ---------------------------------------------------------------------------

function coverage(overrides) {
  return Object.assign({
    session: {
      reportable: true,
      survey_source: "session",
      asked: [{ id: "s1", order: 0, text: "how do you record" }],
      unasked: [{ id: "s2", order: 1, text: "what happened last time" }],
      orphaned_question_ids: [],
      total: 2,
    },
    project: { sessions_read: 3, never_asked: [], times_asked: { s1: 1 } },
    error: null,
  }, overrides || {});
}

const coverageText = (document) =>
  readable(document.byId["coverage-note"]) + " | " + readable(document.byId.coverage);

test("an interview with no survey does not read as a zero score", async () => {
  // The defect this test exists for: "0 of 0 covered" over a call that never
  // had a survey. Every real recording this product has made is this case.
  const document = await render(review({
    coverage: coverage({
      session: {
        reportable: false, survey_source: "none", asked: [], unasked: [],
        orphaned_question_ids: [], total: 0,
      },
      project: null,
    }),
  }));
  const text = coverageText(document);
  assert.ok(text.includes("ran without a survey"), text);
  assert.ok(!text.includes("0 of 0"), `a missing survey was scored: "${text}"`);
});

test("a survey nobody answered is still scored, because that is a real zero", async () => {
  const document = await render(review({
    coverage: coverage({
      session: {
        reportable: true, survey_source: "session", asked: [],
        unasked: [{ id: "s1", order: 0, text: "how do you record" }],
        orphaned_question_ids: [], total: 1,
      },
    }),
  }));
  assert.ok(coverageText(document).includes("0 of 1"), coverageText(document));
});

test("question text taken from today's survey is flagged as today's", async () => {
  // The session stored no survey of its own, so the wording on screen may
  // have been edited since the call. Silent, this is a lie rather than a
  // rounding error.
  const document = await render(review({
    coverage: coverage({ session: Object.assign(coverage().session, { survey_source: "project" }) }),
  }));
  assert.ok(coverageText(document).includes("today's"), coverageText(document));
});

test("question text from the session's own survey is not flagged", async () => {
  const document = await render(review({ coverage: coverage() }));
  assert.ok(!coverageText(document).includes("may have been edited"), coverageText(document));
});

test("a question asked but since deleted is named on the page", async () => {
  const document = await render(review({
    coverage: coverage({
      session: Object.assign(coverage().session, { orphaned_question_ids: ["s9"] }),
    }),
  }));
  assert.ok(coverageText(document).includes("s9"), coverageText(document));
});

test("what the project has never asked is reported, not just this call", async () => {
  // The number that decides the NEXT interview, and the one a per-session
  // view can never answer.
  const document = await render(review({
    coverage: coverage({
      project: {
        sessions_read: 6,
        never_asked: [{ id: "s4", order: 3, text: "what did you do next" }],
        times_asked: {},
      },
    }),
  }));
  const text = coverageText(document);
  assert.ok(text.includes("never been asked"), text);
  assert.ok(text.includes("6 interviews"), text);
  assert.ok(text.includes("what did you do next"), text);
  // Singular, like the extraction banner's own '1 fact' rule: one question is
  // the most likely count on a real project and it must not read as a typo.
  assert.ok(text.includes("1 question has"), text);
  assert.ok(!text.includes("1 questions"), text);
});

test("a coverage read that failed says so instead of rendering as empty", async () => {
  const document = await render(review({
    coverage: coverage({ error: "ContextValidationError: bad survey", session: null }),
  }));
  assert.ok(coverageText(document).includes("could not be read"), coverageText(document));
});

// --- writing a decision rule -------------------------------------------------------------------

const THREE_SESSIONS = {
  project: "acme",
  sessions: [{ id: "S3", rendered: true }, { id: "S2", rendered: true }, { id: "S1", rendered: true }],
  selected: "S3",
};

function cardFor(document) {
  return document.byId.standings.children[0];
}

function editorOf(document) {
  return findAll(cardFor(document), (n) => n.className === "criterion-editor")[0];
}

test("a hypothesis with no rule is offered one on the card that says it is missing", async () => {
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
  );
  const editor = editorOf(document);
  assert.ok(editor, "no way to write the rule on a card that just said there is none");
  assert.equal(firstTag(editor, "button").textContent, "Write the rule");
});

test("replacing an existing rule names the interviews it will stop pre-dating", async () => {
  // The price, beside the button that charges it. A silent edit here is
  // exactly the target-painting the boundary exists to catch.
  const document = await render(
    review({
      standings: [standing({
        criterion: "3 of 6 say so", is_scorable: true, criterion_registered_after: ["S1"],
      })],
    }), [], THREE_SESSIONS,
  );
  const text = readable(editorOf(document));
  assert.ok(text.includes("pre-dates 2 interviews"), text);
  assert.ok(text.includes("S3") && text.includes("S2"), text);
  assert.ok(!text.includes("S1"), `S1 is inside the boundary and is not a cost: "${text}"`);
});

test("a rule that pre-dates nothing says replacing it is free", async () => {
  const document = await render(
    review({
      standings: [standing({
        criterion: "3 of 6 say so", is_scorable: true,
        criterion_registered_after: ["S1", "S2", "S3"],
      })],
    }), [], THREE_SESSIONS,
  );
  assert.ok(readable(editorOf(document)).includes("gives up nothing"), readable(editorOf(document)));
});

test("saving a rule sends the WHOLE array, not just the one hypothesis", async () => {
  // Sending one entry would look tidier and would silently delete every other
  // hypothesis, because the save path replaces the file.
  const document = await render(
    review({
      standings: [
        standing({ id: "h1", criterion: "", is_scorable: false }),
        standing({ id: "h2", criterion: "keep me", is_scorable: true }),
      ],
    }), [], THREE_SESSIONS,
  );
  const editor = editorOf(document);
  firstTag(editor, "input").value = "3 of 6 describe this unprompted";
  firstTag(editor, "button").click();
  await document.settle();

  const post = document.posted.find((p) => p.url === "/api/hypotheses");
  assert.ok(post, "nothing was posted");
  assert.equal(post.body.length, 2, "the other hypothesis was dropped from the save");
  assert.equal(post.body[0].criterion, "3 of 6 describe this unprompted");
  assert.equal(post.body[1].criterion, "keep me");
});

test("an empty rule is refused in the page rather than saved as blank", async () => {
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
  );
  const editor = editorOf(document);
  firstTag(editor, "input").value = "   ";
  firstTag(editor, "button").click();
  await document.settle();
  assert.equal(document.posted.length, 0, "blank text was posted");
  assert.ok(readable(editor).includes("Write the rule first"), readable(editor));
});

test("a refused save is reported on the card, not swallowed", async () => {
  // A failed save that reports nothing is worse than one that refuses loudly:
  // the operator walks away believing the rule is on disk. The save path
  // really does refuse -- an unreadable file is a 409 -- so this is the
  // ordinary path, not a hypothetical.
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
    { failPost: "hypotheses.json is not readable" },
  );
  const editor = editorOf(document);
  firstTag(editor, "input").value = "a real rule";
  firstTag(editor, "button").click();
  await document.settle();
  const text = readable(editor);
  assert.ok(text.includes("Not saved"), `a refusal was swallowed: "${text}"`);
  assert.ok(text.includes("not readable"), `the reason was dropped: "${text}"`);
  assert.ok(!text.includes("Saved."), `a refusal reported success: "${text}"`);
});

test("a successful save is confirmed somewhere the re-read does not destroy", async () => {
  // Found by DOING it in a browser, not by reading the code: the confirmation
  // used to be written onto the card, and the re-read that follows a
  // successful save rebuilt every card a moment later. The operator saw
  // nothing, and the test that "proved" otherwise passed because it was
  // holding a node already detached from the page.
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
  );
  const before = document.requested.filter((url) => url.startsWith("/api/review")).length;
  const editor = editorOf(document);
  firstTag(editor, "input").value = "a real rule";
  firstTag(editor, "button").click();
  await document.settle();

  // Re-read, because the boundary the page is displaying was just recomputed
  // on the server and the card would otherwise keep showing the old price.
  assert.ok(
    document.requested.filter((url) => url.startsWith("/api/review")).length > before,
    "the page did not re-read itself after a save",
  );
  const banner = document.byId["save-banner"];
  assert.equal(banner.hidden, false, "the save was silent");
  assert.ok(banner.textContent.includes("Saved"), banner.textContent);
  // And it is NOT inside anything the re-render empties.
  assert.ok(
    !findAll(document.byId.standings, (n) => n === banner).length,
    "the confirmation is inside a container the next render clears",
  );
});

test("moving to a different interview clears the save confirmation", async () => {
  // The other half of the banner's rule. It survives the re-read after a save
  // (below), but a confirmation left standing over a DIFFERENT call asserts
  // that call was saved. Cleared on the picker, never in `clearRendered`.
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
  );
  const editor = editorOf(document);
  firstTag(editor, "input").value = "a real rule";
  firstTag(editor, "button").click();
  await document.settle();
  assert.ok(document.byId["save-banner"].textContent.includes("Saved"), "no confirmation to clear");

  const picker = document.byId["session-picker"];
  picker.value = "S2";
  (picker.listeners.change || []).forEach((fn) => fn({}));
  await document.settle();
  assert.equal(document.byId["save-banner"].hidden, true,
    "a save confirmation followed the reader to another interview");
  assert.equal(document.byId["save-banner"].textContent, "");
});

test("the save's own re-read does not wipe the confirmation", async () => {
  // The property the banner exists for. A successful save calls `load()`,
  // which empties every rendered container -- so this asserts the clearing
  // really did happen (the cards were rebuilt) AND the confirmation outlived
  // it. Asserting only the banner would pass on a page that never re-read.
  const document = await render(
    review({ standings: [standing({ criterion: "", is_scorable: false })] }), [], THREE_SESSIONS,
  );
  const before = document.requested.filter((url) => url.startsWith("/api/review")).length;
  const editor = editorOf(document);
  firstTag(editor, "input").value = "a real rule";
  firstTag(editor, "button").click();
  await document.settle();

  assert.ok(
    document.requested.filter((url) => url.startsWith("/api/review")).length > before,
    "the page never re-read, so this proves nothing about surviving a re-render",
  );
  assert.equal(document.byId.standings.children.length, 1, "the cards were not rebuilt");
  assert.ok(
    document.byId["save-banner"].textContent.includes("Saved"),
    "the re-read wiped the confirmation",
  );
});

// --- adding a hypothesis -----------------------------------------------------------------------

test("a new hypothesis needs both a statement and a falsifier", async () => {
  const document = await render(review(), [], THREE_SESSIONS);
  const form = document.byId["new-hypothesis"];
  const inputs = findAll(form, (n) => n.tagName === "input");
  inputs[0].value = "players defer lookups";
  inputs[1].value = "";
  firstTag(form, "button").click();
  await document.settle();
  assert.equal(document.posted.length, 0, "a hypothesis with no falsifier was saved");
  assert.ok(readable(form).includes("what would falsify it"), readable(form));
});

test("a new hypothesis is appended with an id that does not collide", async () => {
  const document = await render(
    review({ standings: [standing({ id: "h1" }), standing({ id: "h2" })] }), [], THREE_SESSIONS,
  );
  const form = document.byId["new-hypothesis"];
  const inputs = findAll(form, (n) => n.tagName === "input");
  inputs[0].value = "players defer lookups";
  inputs[1].value = "they alt-tab mid-match and call it cheap";
  firstTag(form, "button").click();
  await document.settle();

  const post = document.posted.find((p) => p.url === "/api/hypotheses");
  assert.equal(post.body.length, 3);
  const ids = post.body.map((h) => h.id);
  assert.equal(new Set(ids).size, 3, `ids collided: ${ids.join(", ")}`);
  assert.equal(post.body[2].state, "untested");
});

test("the form says a rule written now pre-dates nothing", async () => {
  // Said BEFORE the form is filled. A criterion typed here is typed after
  // every interview already on disk, and the record will say so whether or
  // not the page warned first.
  const document = await render(review(), [], THREE_SESSIONS);
  const text = readable(document.byId["new-hypothesis"]);
  assert.ok(text.includes("pre-dates none of the 3 interviews"), text);
});

// --- the interview picker ----------------------------------------------------------------------

test("the picker lists every interview and marks the un-rendered ones", async () => {
  const document = await render(review(), [], {
    project: "acme",
    sessions: [{ id: "S2", rendered: false }, { id: "S1", rendered: true }],
    selected: "S1",
  });
  const options = document.byId["session-picker"].children;
  assert.equal(options.length, 2);
  assert.ok(options[0].textContent.includes("not rendered"), options[0].textContent);
  assert.equal(options[1].textContent, "S1");
  assert.equal(document.byId["picker-row"].hidden, false);
});

test("a project with no interviews says so rather than showing an empty control", async () => {
  const document = await render(review(), [], { project: "acme", sessions: [], selected: null });
  assert.equal(document.byId["session-picker"].hidden, true);
  assert.ok(
    document.byId["picker-note"].textContent.includes("No interviews recorded"),
    document.byId["picker-note"].textContent,
  );
});

test("with no session on the command line the newest interview opens", async () => {
  // `selected` is null when `python -m app hub --project X` was run without a
  // session id, which is now the normal way to open the page.
  const document = await render(review(), [], {
    project: "acme",
    sessions: [{ id: "S2", rendered: true }, { id: "S1", rendered: true }],
    selected: null,
  });
  assert.ok(
    document.requested.some((url) => url === "/api/review?session=S2"),
    document.requested.join(", "),
  );
});

test("switching interview asks for that interview's proposals too", async () => {
  // The proposals card is the one place on this page that writes. Left behind
  // on the command line's session, it would offer a decision about a
  // different interview than the one on screen.
  const document = await render(review({ session_id: "S2" }), [], {
    project: "acme",
    sessions: [{ id: "S2", rendered: true }],
    selected: "S2",
  });
  assert.ok(
    document.requested.some((url) => url === "/api/proposals?session=S2"),
    document.requested.join(", "),
  );
});

// --- a rejection the operator can still see next time -----------------------------------------
//
// A rejection is real information: someone read the evidence, considered a
// state change and judged it insufficient. The server now keeps that; these
// tests are about the other half, which is that keeping it is worthless unless
// the operator MEETS it again when the same proposal comes back.
//
// What they cannot prove: that the operator reads the notice, that the date
// shown is the right date (the server supplies it verbatim and the page never
// parses or reformats it -- deliberately, because a client that reformatted a
// timestamp would be a second opinion about when something happened), or that
// the two buttons look equally weighted, which is CSS and is guarded in
// `test_web_accept.py` instead.

function proposal(overrides) {
  return Object.assign({
    target_hypothesis_id: "h1",
    proposed_state: "supported",
    questions: ["did that actually happen?"],
    seqs: [4],
    // Present and null by default, matching what the server sends for a
    // proposal nobody has declined -- which is the common case, and so must be
    // the case the default fixture exercises.
    previously_declined: null,
  }, overrides || {});
}

function proposalButtons(document) {
  return findAll(document.byId.proposals, (n) => n.tagName === "button");
}

test("a proposal nobody has declined carries no decline notice at all", async () => {
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal()]);
  const text = readable(document.byId.proposals);
  assert.ok(!/declined/i.test(text), `an undeclined proposal was marked: "${text}"`);
});

test("a proposal declined before is marked as declined before", async () => {
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: {
      count: 2, last_declined_at: "2026-06-30T17:45:00Z", sessions: ["S2", "S3"],
    },
  })]);
  const text = readable(document.byId.proposals);
  assert.ok(/already declined this proposal 2 times/i.test(text),
    `the earlier judgement was not handed back: "${text}"`);
});

test("a previously declined proposal is distinguishable in the badge row", async () => {
  // Added because deleting the badge SURVIVED the first round of mutation
  // testing: the prose notice below it still said everything, so nothing
  // failed. The badge is not decoration -- the badge row is what a reader
  // scans when several proposal cards are stacked, and a card that only
  // reveals "you already decided this" in a paragraph is one a hurried
  // operator scrolls past. Asserted on the badge subtree specifically, not on
  // the card's text, or the notice below would keep answering for it.
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: { count: 1, last_declined_at: "2026-03-04T09:15:00Z", sessions: ["S1"] },
  })]);
  const badges = findAll(document.byId.proposals, (n) => /\bbadge\b/.test(n.className));
  const labels = badges.map((b) => b.textContent);
  assert.ok(labels.some((t) => /previously declined/i.test(t)),
    `nothing in the badge row marks this card: ${JSON.stringify(labels)}`);
});

test("an undeclined proposal grows no extra badge", async () => {
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal()]);
  // Asserted on the DECLINED badge alone, not on the whole badge row: this
  // card legitimately carries other badges (the state transition, and
  // "no criterion" for an unscorable hypothesis), and pinning the row's exact
  // contents here would make this test fail for reasons that have nothing to
  // do with declines -- which is what the first draft of it did.
  const badges = findAll(document.byId.proposals, (n) => /\bbadge\b/.test(n.className));
  const marks = badges.map((b) => b.textContent).filter((t) => /declined/i.test(t));
  assert.deepEqual(marks, [], "a proposal nobody declined was badged as declined");
});

test("the marker says WHEN it was declined, not merely that it was", async () => {
  // An un-dated "you declined this before" is nearly useless: how much weight
  // yesterday's decision deserves is entirely a question of how long ago it
  // was and what has arrived since.
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: {
      count: 2, last_declined_at: "2026-06-30T17:45:00Z", sessions: ["S2", "S3"],
    },
  })]);
  const text = readable(document.byId.proposals);
  assert.ok(text.includes("2026-06-30T17:45:00Z"), `no date reached the screen: "${text}"`);
});

test("the marker names the interviews the proposal was declined after", async () => {
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: {
      count: 2, last_declined_at: "2026-06-30T17:45:00Z", sessions: ["S2", "S3"],
    },
  })]);
  const text = readable(document.byId.proposals);
  assert.ok(/interviews: S2, S3/.test(text), `the interviews were dropped: "${text}"`);
});

test("a single earlier decline is not written as '1 times'", async () => {
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: { count: 1, last_declined_at: "2026-03-04T09:15:00Z", sessions: ["S1"] },
  })]);
  const text = readable(document.byId.proposals);
  assert.ok(/declined this proposal once/i.test(text), text);
  assert.ok(/interview: S1/.test(text), `singular interview wording: "${text}"`);
});

test("a decline with no recorded time says so rather than printing nothing", async () => {
  // A record written by something that did not stamp a time is still a record
  // that a human declined this. Dropping the whole notice over a missing field
  // would be the original defect in miniature.
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: { count: 1, last_declined_at: null, sessions: [] },
  })]);
  const text = readable(document.byId.proposals);
  assert.ok(/already declined this proposal once/i.test(text), text);
  assert.ok(/does not say when/i.test(text), `the missing time was silent: "${text}"`);
});

test("a previously declined proposal is still offered for decision", async () => {
  // Marked, never hidden. Hiding it would replace one silent loss with
  // another, and a proposal declined in March may be right in June.
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [proposal({
    previously_declined: { count: 1, last_declined_at: "2026-03-04T09:15:00Z", sessions: ["S1"] },
  })]);
  assert.equal(document.byId.proposals.children.length, 1, "the card was hidden");
  const labels = proposalButtons(document).map((b) => b.textContent);
  assert.deepEqual(labels, ["Accept", "Reject"], `both decisions must stay live: ${labels}`);
});

test("rejecting reports that the decision was recorded, not that nothing happened", async () => {
  const document = await render(
    review({ standings: [standing({ id: "h1" })] }), [proposal()], undefined,
    { postBody: { applied: false, decision: "reject", recorded: true, record_error: null } },
  );
  proposalButtons(document)[1].click();
  await document.settle();
  const text = readable(document.byId.proposals);
  assert.ok(/the decision was recorded/i.test(text), `the record was not confirmed: "${text}"`);
  assert.ok(!/nothing was changed/i.test(text), `the old, now-false wording survived: "${text}"`);
});

test("a rejection whose record was lost says so instead of confirming it", async () => {
  // The operator is the only one who can do anything about it. A page that
  // said "recorded" regardless would be the same silent loss wearing a
  // confirmation.
  const document = await render(
    review({ standings: [standing({ id: "h1" })] }), [proposal()], undefined,
    {
      postBody: {
        applied: false, decision: "reject", recorded: false,
        record_error: "declined_proposals.json is not valid JSON",
      },
    },
  );
  proposalButtons(document)[1].click();
  await document.settle();
  const text = readable(document.byId.proposals);
  assert.ok(/could NOT be recorded/.test(text), `the loss was swallowed: "${text}"`);
  assert.ok(text.includes("declined_proposals.json is not valid JSON"),
    `the reason was withheld: "${text}"`);
});

test("a decision is filed against the interview on screen", async () => {
  // The card follows the picker everywhere else on this page; the record it
  // writes has to follow it too, or a rejection made while reading S2 is filed
  // against S1 forever.
  const document = await render(
    review({ session_id: "S2", standings: [standing({ id: "h1" })] }), [proposal()],
    { project: "acme", sessions: [{ id: "S2", rendered: true }], selected: "S2" },
    { postBody: { applied: false, decision: "reject", recorded: true, record_error: null } },
  );
  proposalButtons(document)[1].click();
  await document.settle();
  const post = document.posted.find((p) => p.url === "/api/decision");
  assert.ok(post, `nothing was posted: ${document.posted.map((p) => p.url).join(", ")}`);
  assert.equal(post.body.session, "S2");
  assert.equal(post.body.decision, "reject");
});

test("accepting still reports the state it moved to", async () => {
  // The other branch of the same status line, so a change to the rejection
  // wording cannot quietly take the acceptance message with it.
  const document = await render(
    review({ standings: [standing({ id: "h1" })] }), [proposal({ proposed_state: "contested" })],
    undefined, { postBody: { applied: true, decision: "accept", saved: 1 } },
  );
  proposalButtons(document)[0].click();
  await document.settle();
  const text = readable(document.byId.proposals);
  assert.ok(/Accepted — now contested/.test(text), text);
});

test("switching interview leaves nothing of the previous one on the page", async () => {
  // A standing card left over from another call, sitting under the new
  // interview's heading, is read as this call's evidence.
  const document = await render(review({ standings: [standing({ id: "h1" })] }), [], {
    project: "acme",
    sessions: [{ id: "S2", rendered: true }, { id: "S1", rendered: true }],
    selected: "S1",
  });
  const before = document.byId.standings.children.length;
  const picker = document.byId["session-picker"];
  picker.value = "S2";
  (picker.listeners.change || []).forEach((fn) => fn({}));
  for (let i = 0; i < 60; i += 1) { await Promise.resolve(); }
  assert.equal(
    document.byId.standings.children.length, before,
    "the standings accumulated across a switch instead of being replaced",
  );
});

// --- who was interviewed ---------------------------------------------------------------------

// The EMPTY state first, and it gets more tests than the filled one, because
// it is what this box renders almost every time the page is opened: every
// interview recorded before the field existed carries no label. A feature
// whose common path draws a blank where a name belongs reads as a page that
// failed to load something.

test("an interview with no participant says so in words rather than leaving a blank", async () => {
  const document = await render(review({ participant: null }));
  const text = readable(document.byId.participant);
  assert.ok(/no participant recorded/i.test(text), `the absence was silent: "${text}"`);
});

test("the empty participant state explains what the absence costs the counting", async () => {
  // Not a nag and not an error — the sentence exists because the standings
  // below count interviews, and that is a count of PEOPLE only while nobody
  // was interviewed twice. Nothing on the page can tell whether that held.
  const document = await render(review({ participant: null }));
  const text = readable(document.byId.participant);
  assert.ok(/normal state/i.test(text), `the ordinary case reads as a defect: "${text}"`);
  assert.ok(/interviewed twice|count of people/i.test(text),
    `the absence is reported without saying what it costs: "${text}"`);
});

test("a recorded participant reaches the screen and pre-fills the box", async () => {
  const document = await render(review({ participant: "Dana (ops lead)" }));
  const host = document.byId.participant;
  assert.ok(readable(host).includes("Dana (ops lead)"), readable(host));
  assert.equal(firstTag(host, "input").value, "Dana (ops lead)");
  assert.ok(!/no participant recorded/i.test(readable(host)), readable(host));
});

test("no interview selected offers no participant box at all", async () => {
  // There is nothing to attach a name to, and a disabled input would offer an
  // action that cannot work.
  const document = await render(review({ session_id: null, participant: null }), [], {
    project: "acme", sessions: [], selected: null,
  });
  assert.equal(document.byId.participant.children.length, 0);
});

test("recording a participant posts it against the interview on screen", async () => {
  // Not against whichever session the command line named. A name on the wrong
  // call is a wrong fact written to disk, not merely a misread number.
  const document = await render(review({ session_id: "S3", participant: null }), [], THREE_SESSIONS);
  const host = document.byId.participant;
  firstTag(host, "input").value = "  Dana (ops lead) ";
  firstTag(host, "button").click();
  await document.settle();

  const post = document.posted.find((p) => p.url === "/api/participant");
  assert.ok(post, `nothing was posted: ${document.posted.map((p) => p.url).join(", ")}`);
  assert.equal(post.body.session, "S3");
  assert.equal(post.body.participant, "Dana (ops lead)");
});

test("an empty box with nothing stored is refused in the page, not sent", async () => {
  const document = await render(review({ participant: null }), [], THREE_SESSIONS);
  const host = document.byId.participant;
  firstTag(host, "button").click();
  await document.settle();
  assert.equal(document.posted.length, 0, "saved nothing over nothing");
  assert.ok(/type who this was/i.test(readable(host)), readable(host));
});

test("emptying the box on a labelled interview is a clear, and it is sent", async () => {
  // The operator who deletes what is there is saying they were wrong about who
  // this was. Refusing would leave a wrong name on an interview forever.
  const document = await render(review({ participant: "Dana" }), [], THREE_SESSIONS);
  const host = document.byId.participant;
  firstTag(host, "input").value = "";
  firstTag(host, "button").click();
  await document.settle();
  const post = document.posted.find((p) => p.url === "/api/participant");
  assert.ok(post, "a deliberate clear was swallowed");
  assert.equal(post.body.participant, "");
});

test("a saved participant is confirmed in the banner and the page re-reads", async () => {
  const document = await render(review({ participant: null }), [], THREE_SESSIONS);
  const before = document.requested.filter((url) => url.startsWith("/api/review")).length;
  const host = document.byId.participant;
  firstTag(host, "input").value = "Dana";
  firstTag(host, "button").click();
  await document.settle();

  assert.ok(
    document.requested.filter((url) => url.startsWith("/api/review")).length > before,
    "the page did not re-read itself after a save",
  );
  const banner = document.byId["save-banner"];
  assert.equal(banner.hidden, false, "the save was silent");
  assert.ok(banner.textContent.includes("Dana"), banner.textContent);
  // And it must NOT claim to have written the hypotheses file, which is what a
  // shared hardcoded confirmation would have said.
  assert.ok(!banner.textContent.includes("hypotheses.json"), banner.textContent);
});

test("a refused participant save is reported beside the box, not swallowed", async () => {
  const document = await render(review({ participant: null }), [], THREE_SESSIONS, {
    failPost: "that session is not one of this project's interviews",
  });
  const host = document.byId.participant;
  firstTag(host, "input").value = "Dana";
  firstTag(host, "button").click();
  await document.settle();
  const text = readable(host);
  assert.ok(text.includes("Not saved"), `a refusal was swallowed: "${text}"`);
  assert.ok(text.includes("not one of this project's interviews"), `the reason was dropped: "${text}"`);
});

test("switching interview leaves no participant of the previous one on screen", async () => {
  // One call's name standing over another's is the same failure as a leftover
  // standing card, and worse: it names a person.
  const document = await render(review({ session_id: "S3", participant: "Dana" }), [], THREE_SESSIONS);
  assert.ok(readable(document.byId.participant).includes("Dana"), "nothing to leave behind");
  const before = document.byId.participant.children.length;

  const picker = document.byId["session-picker"];
  picker.value = "S2";
  (picker.listeners.change || []).forEach((fn) => fn({}));
  await document.settle();
  assert.equal(
    document.byId.participant.children.length, before,
    "the participant box accumulated across a switch instead of being replaced",
  );
});

test("the picker names each interview by its participant when one is recorded", async () => {
  // What turns the control from a column of timestamps into a list a human can
  // choose from. The id is APPENDED to, never replaced: it is what every other
  // surface in the product calls an interview.
  const document = await render(review(), [], {
    project: "acme",
    sessions: [
      { id: "S2", rendered: false, participant: "Dana (ops lead)" },
      { id: "S1", rendered: true, participant: null },
    ],
    selected: "S1",
  });
  const options = document.byId["session-picker"].children;
  assert.ok(options[0].textContent.includes("S2"), options[0].textContent);
  assert.ok(options[0].textContent.includes("Dana (ops lead)"), options[0].textContent);
  assert.ok(options[0].textContent.includes("not rendered"), options[0].textContent);
  // The unlabelled interview reads exactly as it did before, with no stray
  // separator hanging off the end of it.
  assert.equal(options[1].textContent, "S1");
});

// --- who did the talking -----------------------------------------------------------------------
// The only measure on this page that costs no model call. What these defend is
// not the arithmetic (app/tests/test_talk_share.py owns that) but the three
// ways a bar chart can lie: by drawing an absence as a balance, by drawing a
// width that disagrees with its own label, and by surviving into the next
// interview.

test("the interviewer's own share is the headline, as a plain percentage", async () => {
  const document = await render(review({
    talk_share: meter({ you: 65, them: 35 }),
  }));
  const note = document.byId["talk-share-note"].textContent;
  assert.ok(note.includes("65%"), `did not report the interviewer's share: ${note}`);
  assert.ok(note.includes("100"), `did not say how many words that was of: ${note}`);
});

test("no verdict is attached to the interviewer's share", async () => {
  // Nothing in this product has measured what "too much" is. A page that calls
  // 65% too high is inventing a threshold and dressing it as a finding.
  const document = await render(review({
    talk_share: meter({ you: 80, them: 20 }),
  }));
  const text = readable(document.byId["talk-share"])
    + " | " + document.byId["talk-share-note"].textContent;
  ["too much", "too high", "should", "poor", "warning", "dominat"].forEach((word) => {
    assert.ok(!text.toLowerCase().includes(word), `passed judgement (${word}): ${text}`);
  });
});

test("an unmeasurable interview draws no bars at all", async () => {
  // NOT a 50/50 bar and not "0%". There is no transcript to count, which is a
  // fact about the recording -- drawing a balanced pair would assert a fact
  // about the conversation instead.
  const document = await render(review({ transcript: [] }));
  const host = document.byId["talk-share"];
  assert.equal(findAll(host, (n) => n.className === "share-row").length, 0);
  const note = document.byId["talk-share-note"].textContent;
  assert.ok(note.includes("nothing to measure"), note);
  assert.ok(!note.includes("%"), `printed a percentage for an absence: ${note}`);
});

test("each bar's drawn width agrees with the number printed beside it", async () => {
  // The defect a textContent-only assertion cannot see: a correct label above
  // a bar drawn from the wrong number. The bar is what the eye reads.
  //
  // The word counts deliberately do NOT sum to 100. They did at first, which
  // made every word count numerically equal to its own percentage -- so a bar
  // drawn from `words` instead of `share` produced identical output and the
  // test passed against a mutation that had genuinely broken it. Any total but
  // 100 separates the two quantities; 20 is used so the shares stay round.
  const document = await render(review({
    talk_share: meter({ you: 13, them: 7 }),
  }));
  const fills = findAll(document.byId["talk-share"],
    (n) => n.className && n.className.indexOf("share-fill") === 0);
  assert.equal(fills.length, 2);
  assert.equal(fills[0].style.width, "65.0%");
  assert.equal(fills[1].style.width, "35.0%");
  // And the label beside it reads the same number the bar was drawn at.
  const values = findAll(document.byId["talk-share"], (n) => n.className === "share-value");
  assert.ok(values[0].textContent.startsWith("65%"), values[0].textContent);
  assert.ok(values[0].textContent.includes("13 words"), values[0].textContent);
});

test("the interviewer's row is emphasised without being coloured as a fault", async () => {
  const document = await render(review({
    talk_share: meter({ you: 65, them: 35 }),
  }));
  const fills = findAll(document.byId["talk-share"],
    (n) => n.className && n.className.indexOf("share-fill") === 0);
  assert.ok(fills[0].className.includes("mine"), fills[0].className);
  assert.ok(!fills[1].className.includes("mine"), fills[1].className);
  // Emphasis only. No error/flag class on the bar itself.
  assert.ok(!fills[0].className.includes("flag"), fills[0].className);
});

test("the client does not hardcode which speaker is the interviewer", async () => {
  // `SPEAKER_LABELS` belongs to the server. A client that had guessed "you"
  // would silently stop emphasising anything the day that label changed.
  const document = await render(review({
    talk_share: meter({ interviewer: 60, interviewee: 40 }, { interviewer_key: "interviewer" }),
  }));
  const fills = findAll(document.byId["talk-share"],
    (n) => n.className && n.className.indexOf("share-fill") === 0);
  assert.ok(fills[0].className.includes("mine"), fills[0].className);
  assert.ok(document.byId["talk-share-note"].textContent.includes("60%"),
    document.byId["talk-share-note"].textContent);
});

test("unattributed speech is said out loud, not folded into the bars", async () => {
  // Speech nobody is credited with means the shares do not sum to 100%. A
  // reader who notices that without being told will distrust the rest.
  const document = await render(review({
    talk_share: meter({ you: 50, them: 30 }, { unattributed_words: 20 }),
  }));
  const text = readable(document.byId["talk-share"]);
  assert.ok(text.includes("20 words could not be attributed"), text);
  assert.ok(text.includes("do not add up to 100%"), text);
});

test("a fully attributed interview does not warn about attribution", async () => {
  const document = await render(review({
    talk_share: meter({ you: 60, them: 40 }),
  }));
  assert.ok(!readable(document.byId["talk-share"]).includes("could not be attributed"),
    readable(document.byId["talk-share"]));
});

test("the page says the count is in words rather than in seconds", async () => {
  // Load-bearing, not a footnote: the stored per-segment start times are
  // derived, so a reader who assumes this is a timing split would trust a
  // number that was never measured.
  const document = await render(review({
    talk_share: meter({ you: 60, them: 40 }),
  }));
  const text = readable(document.byId["talk-share"]);
  assert.ok(text.includes("words, not in seconds"), text);
});

test("a speaker who is not the interviewer still gets a row", async () => {
  // Nothing here assumes an interview has exactly two people in it.
  const document = await render(review({
    talk_share: meter({ you: 50, them: 30, observer: 20 }),
  }));
  assert.equal(findAll(document.byId["talk-share"],
    (n) => n.className === "share-row").length, 3);
});

test("an interviewer absent from the transcript is reported, not silently skipped", async () => {
  const document = await render(review({
    talk_share: meter({ them: 40 }),
  }));
  const note = document.byId["talk-share-note"].textContent;
  assert.ok(note.includes("do not appear"), note);
});

test("switching interview does not leave the previous one's meter on screen", async () => {
  // The most quietly wrong thing this page could do: a plausible-looking ratio
  // sitting under another interview's heading.
  const document = await render(review({
    session_id: "S1", talk_share: meter({ you: 65, them: 35 }),
  }), [], THREE_SESSIONS);
  assert.ok(readable(document.byId["talk-share"]).includes("65%"),
    readable(document.byId["talk-share"]));

  document.byId["session-picker"].value = "S2";
  (document.byId["session-picker"].listeners.change || []).forEach((fn) => fn({}));
  await document.settle();

  // The stub answers the SAME payload for every session, so the assertion is
  // about there being exactly one meter rather than two stacked -- a leftover
  // card would double every row.
  assert.equal(findAll(document.byId["talk-share"],
    (n) => n.className === "share-row").length, 2);
});

// --- the summary block ------------------------------------------------------------------------
//
// The server had been computing an `InterviewSummary` on every request since
// the module existed, keeping three integers of it for one hypothesis's
// contribution line, and publishing `summary: null` -- under which the page
// printed "Not built yet". These pin the numbers onto the screen, which is the
// half a payload assertion cannot reach.

function summary(overrides) {
  return Object.assign({
    interviews: 1,
    fact_count: 3,
    evidence_facts: 2,
    inference_facts: 1,
    hypotheses_touched: 2,
    projects: ["acme"],
  }, overrides || {});
}

const summaryText = (document) => readable(document.byId.summary);

test("the interview's counts are on the screen, not merely in the payload", async () => {
  const document = await render(review({ summary: summary({}) }));
  const text = summaryText(document);
  assert.ok(text.includes("3 claims"), `no fact count: ${text}`);
  assert.ok(text.includes("2 hypotheses"), `no hypothesis count: ${text}`);
  assert.ok(text.includes("2 evidence"), `no evidence count: ${text}`);
  assert.ok(text.includes("1 inference"), `no inference count: ${text}`);
  assert.ok(text.includes("acme"), `the node the claims were filed at is missing: ${text}`);
  assert.ok(!/not built yet/i.test(text), `still says the summary is unbuilt: ${text}`);
});

test("the summary is counts, not a generated paragraph", async () => {
  // The line this page will not cross. Every claim is already rendered above
  // with its own quote and citation; prose here would be a second account of
  // the same evidence, and it would cost a model call on a read.
  const document = await render(review({ summary: summary({}) }));
  // Read off the summary's OWN nodes rather than off the whole container. The
  // container now also hosts the three debrief panels, whose caveats are long
  // on purpose and are hand-written constants in this repo -- the property
  // being defended here is that the CLAIM COUNTS are not a generated
  // paragraph, and measuring the container would have quietly turned this into
  // a length budget for prose nobody generated.
  const text = findAll(document.byId.summary,
    (n) => /^summary-/.test(n.className || "")).map((n) => n.textContent).join(" | ");
  assert.ok(text.length > 0, "the summary's own nodes disappeared");
  assert.ok(text.length < 220, `the summary block is prose-length: ${text}`);
  assert.ok(/\d/.test(text), "a summary with no numbers in it is not this summary");
});

test("the evidence and inference split is shown, not just the total", async () => {
  // An interview whose claims are mostly the model's reading is a weaker
  // result than the same count of things the interviewee reported, and the
  // fact count alone cannot say which this was.
  const document = await render(review({
    summary: summary({ fact_count: 10, evidence_facts: 1, inference_facts: 9 }),
  }));
  const text = summaryText(document);
  assert.ok(text.includes("1 evidence") && text.includes("9 inference"), text);
});

test("a provenance the page has never heard of is reported, not swallowed", async () => {
  // Without this the two known numbers simply stop adding up to the headline
  // and nothing on the page says why.
  const document = await render(review({
    summary: summary({ fact_count: 5, evidence_facts: 2, inference_facts: 1 }),
  }));
  assert.ok(summaryText(document).includes("2 of neither kind"), summaryText(document));
});

test("singular counts are not written as '1 claims'", async () => {
  const document = await render(review({
    summary: summary({ fact_count: 1, evidence_facts: 1, inference_facts: 0,
                       hypotheses_touched: 1 }),
  }));
  const text = summaryText(document);
  assert.ok(!text.includes("1 claims") && !text.includes("1 hypothesiss"), text);
  assert.ok(text.includes("1 claim from") && text.includes("1 hypothesis."), text);
});

test("touching no hypothesis is worded so a reader stops on it", async () => {
  // "0 hypotheses" is the number a reader skims past. A call that reached none
  // of the beliefs it was meant to test is the result worth stopping on.
  const document = await render(review({ summary: summary({ hypotheses_touched: 0 }) }));
  assert.ok(summaryText(document).includes("no hypothesis"), summaryText(document));
});

test("an interview with nothing extracted says so rather than reading as zeroes", async () => {
  const document = await render(review({
    summary: summary({ fact_count: 0, evidence_facts: 0, inference_facts: 0,
                       hypotheses_touched: 0, projects: [] }),
  }));
  assert.ok(/no claims from this interview/i.test(summaryText(document)), summaryText(document));
});

test("no summary with a session selected names the unreadable corpus", async () => {
  // The two situations that produce a null summary are distinguishable from
  // the rest of the payload, so the page must not print one sentence that is
  // wrong half the time.
  const document = await render(review({ session_id: "S1", summary: null }));
  assert.ok(/could not be read/i.test(summaryText(document)), summaryText(document));
});

test("no summary and no session selected says no interview is selected", async () => {
  const document = await render(review({ session_id: null, summary: null }));
  assert.ok(/no interview is selected/i.test(summaryText(document)), summaryText(document));
});

test("switching interview does not leave the previous one's counts on screen", async () => {
  // Same failure as the leftover meter, and quieter: a plausible fact count
  // sitting under another interview's heading, with nothing about it wrong
  // except which call it describes.
  const document = await render(review({
    session_id: "S1", summary: summary({}),
  }), [], THREE_SESSIONS);
  assert.ok(summaryText(document).includes("3 claims"), summaryText(document));

  document.byId["session-picker"].value = "S2";
  (document.byId["session-picker"].listeners.change || []).forEach((fn) => fn({}));
  await document.settle();

  // The stub answers the SAME payload for every session, so the assertion is
  // about there being exactly ONE headline rather than two stacked.
  assert.equal(
    findAll(document.byId.summary, (n) => n.className === "summary-headline").length, 1);
});

// --- the graphical standing (direction D) -----------------------------------------------------
//
// These exist because a bar is the one thing a textContent assertion cannot
// check. The whole reason the charts are divs with inline widths rather than
// SVG is so this harness can read the width and hold it against the number
// printed beside it.

const segments = (document) =>
  findAll(document.byId.standings, (n) => /\bstanding-seg\b/.test(n.className || ""));

test("the standing bar's widths agree with the counts printed beside them", async () => {
  const document = await render(review({
    standings: [standing({
      supporting_interviews: 3,
      contrary_interviews: 1,
      unclear_interviews: 0,
    })],
  }));
  const segs = segments(document);
  assert.equal(segs.length, 2, `expected two segments, got ${segs.length}`);
  // 3 of 4 and 1 of 4 -- a bar drawn at any other width is telling the reader
  // something the numbers do not say.
  assert.equal(segs[0].style.width, "75.0%");
  assert.equal(segs[1].style.width, "25.0%");
  assert.ok(/3 for/.test(cardText(document)), cardText(document));
  assert.ok(/1 against/.test(cardText(document)), cardText(document));
});

test("an unresolved claim gets its own segment rather than being folded away", async () => {
  const document = await render(review({
    standings: [standing({
      supporting_interviews: 1, contrary_interviews: 1, unclear_interviews: 2,
    })],
  }));
  const classes = segments(document).map((n) => n.className);
  assert.ok(classes.some((c) => /\bunclear\b/.test(c)),
    `the unresolved interviews vanished from the bar: ${classes.join(", ")}`);
  assert.ok(/2 unresolved/.test(cardText(document)), cardText(document));
});

test("a hypothesis nothing has borne on draws no bar at all", async () => {
  // NOT an empty track. A drawn-but-empty meter reads as "measured, and it
  // came to nothing"; this is "not measured".
  const document = await render(review({ standings: [standing({})] }));
  assert.equal(segments(document).length, 0, "an empty bar was drawn");
  assert.ok(/no interview has borne on this yet/i.test(cardText(document)),
    cardText(document));
});

test("the bar's identity never rests on colour alone", async () => {
  const document = await render(review({
    standings: [standing({ supporting_interviews: 2, contrary_interviews: 1 })],
  }));
  const text = cardText(document);
  // Every segment's meaning is also a word and a number in the key, so the
  // chart survives greyscale, colour blindness, and a screen reader.
  assert.ok(/2 for/.test(text) && /1 against/.test(text), text);
  assert.ok(/interviews, not claims/.test(text),
    `the bar does not say what it counts: ${text}`);
});

// --- the criterion meter ----------------------------------------------------------------------

const meterFill = (document) =>
  findAll(document.byId.standings, (n) => /\bmeter-fill\b/.test(n.className || ""))[0];
const meterMark = (document) =>
  findAll(document.byId.standings, (n) => /\bmeter-threshold\b/.test(n.className || ""))[0];

test("a countable rule is drawn with its threshold on the coordinate it marks", async () => {
  const document = await render(review({
    standings: [standing({
      criterion: "At least 3 of 6 interviewees describe it unprompted.",
      is_scorable: true,
      supporting_interviews: 2,
    })],
  }));
  assert.equal(meterFill(document).style.width, "33.3%", "2 of 6");
  assert.equal(meterMark(document).style.left, "50.0%", "the rule needs 3 of 6");
  assert.ok(/2 of 6/.test(cardText(document)), cardText(document));
  assert.ok(/needs 3/.test(cardText(document)), cardText(document));
});

test("a met threshold says so rather than leaving the reader to compare two numbers", async () => {
  const document = await render(review({
    standings: [standing({
      criterion: "At least 3 of 6 interviewees describe it unprompted.",
      is_scorable: true,
      supporting_interviews: 3,
    })],
  }));
  assert.ok(/threshold met/i.test(cardText(document)), cardText(document));
});

test("a rule this page cannot read without guessing keeps its sentence and gets no bar", async () => {
  // The real h4 criterion. A ratio has no denominator this code can recover,
  // and a meter drawn from a guess would be the exact target-painting the
  // criterion field exists to prevent -- with a progress bar to make it look
  // measured.
  const document = await render(review({
    standings: [standing({
      criterion: "Read-only uses outnumber edit-or-run uses at least 2 to 1.",
      is_scorable: true,
      supporting_interviews: 2,
    })],
  }));
  assert.equal(meterFill(document), undefined, "a bar was invented from an unparseable rule");
  assert.ok(/read-only uses outnumber/i.test(cardText(document)),
    `the rule itself stopped being shown: ${cardText(document)}`);
});

test("a hypothesis with no rule at all gets no meter", async () => {
  const document = await render(review({
    standings: [standing({ criterion: "", is_scorable: false, supporting_interviews: 4 })],
  }));
  assert.equal(meterFill(document), undefined,
    "a hypothesis with no pre-registered rule was scored against one anyway");
  assert.ok(/no decision rule/i.test(cardText(document)), cardText(document));
});

test("the meter names which number it is putting on the bar", async () => {
  // The rule is prose and this meter reads it as being about SUPPORTING
  // interviews. A rule written about the other direction would be drawn
  // backwards, so the page says which number it used.
  const document = await render(review({
    standings: [standing({
      criterion: "At least 3 of 6 describe it.", is_scorable: true, supporting_interviews: 1,
    })],
  }));
  assert.ok(/counting supporting interviews/i.test(cardText(document)), cardText(document));
});

test("a rule already exceeded does not overflow its own track", async () => {
  const document = await render(review({
    standings: [standing({
      criterion: "At least 2 of 4 describe it.", is_scorable: true, supporting_interviews: 9,
    })],
  }));
  assert.equal(meterFill(document).style.width, "100.0%");
});

// --- the three debrief panels (direction D) ---------------------------------------------------
//
// What these defend is not the arithmetic -- the server's own tests own that --
// but the one property all three panels turn on: an ABSENT measurement and a
// measurement of zero must never render alike. Every artifact behind these
// panels has to be produced deliberately, so "never run" is the rendering most
// readers will actually see, and a page that draws it as a score is inventing
// a number. The bars are divs with inline widths for exactly the reason the
// standing bar is: it lets a test hold a drawn width against the printed one.

function critique(overrides) {
  return Object.assign({ status: "read", error: null, summary: null }, overrides || {});
}

function critiqueSummary(overrides) {
  return Object.assign({
    turns: 20,
    judged: 17,
    unjudged: 3,
    clean: 11,
    clean_rate: 0.63,
    marked_questions: 10,
    marked_question_clean_rate: 0.71,
    unmarked_clean_rate: 0.52,
    flags: {
      leading: 9, hypothetical_or_future_tense: 2, pitch: 0,
      opinion_seeking: 3, confirmation_only: 5, generic: 0,
    },
    flags_not_judged: ["generic"],
  }, overrides || {});
}

function rejectionLog(overrides) {
  return Object.assign({
    status: "kept", error: null, total: 20,
    reasons: { "unlocatable-quote": 12, "span-too-long": 6, "mixed-speaker-span": 2 },
    sessions_recorded: 4,
    this_interview: { total: 3, reasons: { "unlocatable-quote": 3 } },
  }, overrides || {});
}

const panelFills = (document, extra) =>
  findAll(document.byId.summary, (n) =>
    /\bpanel-fill\b/.test(n.className || "")
    && (!extra || new RegExp("\\b" + extra + "\\b").test(n.className)));

const panelValues = (document) =>
  findAll(document.byId.summary, (n) => n.className === "panel-value").map((n) => n.textContent);

// --- how you interviewed ----------------------------------------------------------------------

test("a critique that was never run reads as absent, never as a score", async () => {
  // The common case, and the one that has to be got right: a page that drew
  // this as 0% clean would be reporting an interview nobody measured as the
  // worst one on disk.
  const document = await render(review({
    question_critique: { status: "not-run", error: null, summary: null },
  }));
  const text = summaryText(document);
  assert.ok(/has not been run/i.test(text), `the absence was silent: ${text}`);
  assert.ok(/absent measurement, not a low one/i.test(text), text);
  assert.equal(panelFills(document, "clean").length, 0,
    "a rate was drawn for an unmeasured interview");
  assert.ok(!/\d+% clean/.test(text), `printed a clean rate for an absent critique: ${text}`);
});

test("an un-run critique names the command that would run it", async () => {
  // The only place in the product that tells a reader this step exists, the
  // same job the extraction banner's remedy does.
  const document = await render(review({
    session_id: "S1",
    question_critique: { status: "not-run", error: null, summary: null },
  }));
  assert.ok(summaryText(document).includes("python -m app critique-questions S1"),
    summaryText(document));
});

test("the two split clean rates are drawn at the widths printed beside them", async () => {
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const fills = panelFills(document, "clean");
  assert.equal(fills.length, 2, `expected a bar per split rate, got ${fills.length}`);
  assert.equal(fills[0].style.width, "71.0%");
  assert.equal(fills[1].style.width, "52.0%");
  const values = panelValues(document);
  assert.ok(values[0].startsWith("71% clean"), values[0]);
  assert.ok(values[1].startsWith("52% clean"), values[1]);
  // And the denominators, so a percentage over three turns is not read as one
  // over thirty. 17 judged, 10 of them punctuated as questions.
  assert.ok(values[0].includes("10 judged turns"), values[0]);
  assert.ok(values[1].includes("7 judged turns"), values[1]);
});

test("a split with nothing to judge draws no bar and prints no percentage", async () => {
  // `None` is the artifact's own answer for "nothing of this kind could be
  // judged". 0% would file an unmeasured group with the worst possible result.
  const document = await render(review({
    question_critique: critique({
      summary: critiqueSummary({ marked_questions: 0, marked_question_clean_rate: null }),
    }),
  }));
  assert.equal(panelFills(document, "clean").length, 1, "an absent rate was drawn as a bar");
  const text = summaryText(document);
  assert.ok(/no turn punctuated as a question was judged/i.test(text), text);
  assert.ok(!/0% clean/.test(text), `an absent rate rendered as zero: ${text}`);
});

test("unmarked turns are never called statements", async () => {
  // The group holds leading STATEMENTS and the roughly half of real questions
  // the transcript gave no question mark. Labelled "statements", a low number
  // here becomes a verdict on something that was never measured.
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const text = summaryText(document);
  assert.ok(text.includes("Unmarked turns"), text);
  assert.ok(/not .statements./i.test(text), `the caveat is missing: ${text}`);
  assert.ok(/no question mark/i.test(text), text);
  // And no ROW is labelled as statements -- the caveat above legitimately says
  // the word, so this is asserted on the labels rather than on the prose.
  const labels = findAll(document.byId.summary, (n) => n.className === "panel-who")
    .map((n) => n.textContent);
  assert.ok(!labels.some((l) => /statement/i.test(l)),
    `a bar was labelled as statements: ${labels.join(", ")}`);
});

test("the combined rate is still shown, named as the number the split came from", async () => {
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  assert.ok(/Combined: 63% clean over 17 judged turns/.test(summaryText(document)),
    summaryText(document));
});

test("generic reads as not judged, never as a flag you passed", async () => {
  // Always 0 here BY CONSTRUCTION: no trigger utterance can be supplied for a
  // batched turn, so the model is told to answer "no" rather than guess. Drawn
  // like the other zeroes, it would be a pass the page invented.
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const text = summaryText(document);
  assert.ok(text.includes("Generic"), text);
  assert.ok(/not judged here/i.test(text), `generic rendered as a result: ${text}`);
  assert.ok(/no zero to read as a pass/i.test(text), text);
});

test("a flag that was judged and never fired still reads as a zero", async () => {
  // The other side of the same distinction: `pitch: 0` is a real zero and must
  // be legible as one, or the generic caveat above has swallowed both.
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const rows = findAll(document.byId.summary, (n) => n.className === "panel-row");
  const pitch = rows.filter((r) => readable(r).indexOf("Pitch") === 0)[0];
  assert.ok(pitch, "no row for a flag that never fired");
  assert.ok(/\b0\b/.test(readable(pitch)), readable(pitch));
  assert.ok(!/not judged/i.test(readable(pitch)),
    `a real zero was reported as unjudged: ${readable(pitch)}`);
  assert.equal(findAll(pitch, (n) => /\bpanel-fill\b/.test(n.className || "")).length, 0,
    "a zero count was drawn as a bar");
});

test("flag bars are drawn against the largest count and say so", async () => {
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const fills = panelFills(document, "flag");
  // leading 9 is the largest; confirmation-only 5 is 55.6% of it.
  assert.equal(fills[0].style.width, "100.0%");
  assert.equal(fills[3].style.width, "55.6%");
  assert.ok(/share no denominator/i.test(summaryText(document)),
    `nothing says what the bars are relative to: ${summaryText(document)}`);
});

test("turns the critic could not reach are said out loud", async () => {
  // They are absent from every rate above, and a reader who is not told reads
  // those rates as covering the whole interview.
  const document = await render(review({
    question_critique: critique({ summary: critiqueSummary({ unjudged: 3 }) }),
  }));
  assert.ok(/3 turns could not be judged/i.test(summaryText(document)), summaryText(document));
});

test("an interview the critic could not be reached for shows no rate at all", async () => {
  const document = await render(review({
    question_critique: critique({
      summary: critiqueSummary({
        judged: 0, unjudged: 20, clean: 0, clean_rate: null,
        marked_questions: 0, marked_question_clean_rate: null, unmarked_clean_rate: null,
      }),
    }),
  }));
  const text = summaryText(document);
  assert.ok(/none could be judged/i.test(text), text);
  assert.ok(/an unjudged interview is not a bad one/i.test(text), text);
  assert.equal(panelFills(document, "clean").length, 0, "a rate was drawn with nothing judged");
  assert.ok(!/% clean/.test(text), `printed a clean rate over nothing: ${text}`);
});

test("an unreadable critique says so rather than rendering as zeroes", async () => {
  const document = await render(review({
    question_critique: { status: "unreadable", error: "JSONDecodeError: line 3", summary: null },
  }));
  const text = summaryText(document);
  assert.ok(/could not be read/i.test(text), text);
  assert.ok(text.includes("JSONDecodeError: line 3"), `the reason was dropped: ${text}`);
});

test("a critique status this page has never heard of says so instead of going blank", async () => {
  const document = await render(review({
    question_critique: { status: "renamed-by-someone", error: null, summary: null },
  }));
  const text = summaryText(document);
  assert.ok(/does not recognise/.test(text), text);
  assert.ok(text.includes("renamed-by-someone"), text);
});

// --- what extraction refused ------------------------------------------------------------------

test("no rejection log at all is silence, not a count of zero", async () => {
  // Extraction printed its rejections to a terminal and kept nothing until the
  // log existed. A project extracted before then knows nothing, and drawing
  // that as "refused nothing" would manufacture the very finding the log was
  // written to make checkable.
  const document = await render(review({
    rejections: {
      status: "not-kept", error: null, total: 0, reasons: {},
      sessions_recorded: 0, this_interview: null,
    },
  }));
  const text = summaryText(document);
  assert.ok(/No rejection log exists/i.test(text), text);
  assert.ok(/silence, not a count of zero/i.test(text), text);
  assert.equal(panelFills(document, "reason").length, 0);
});

test("a log that exists and holds nothing is a real zero and says so", async () => {
  const document = await render(review({
    rejections: rejectionLog({
      total: 0, reasons: {}, sessions_recorded: 2, this_interview: null,
    }),
  }));
  const text = summaryText(document);
  assert.ok(/refused no claim/i.test(text), text);
  assert.ok(/real zero, not a missing file/i.test(text), text);
  assert.ok(!/No rejection log exists/i.test(text), `a real zero read as an absent log: ${text}`);
});

test("each reason's bar width agrees with the share printed beside it", async () => {
  const document = await render(review({ rejections: rejectionLog({}) }));
  const fills = panelFills(document, "reason");
  assert.equal(fills.length, 3);
  assert.equal(fills[0].style.width, "60.0%", "12 of 20");
  assert.equal(fills[1].style.width, "30.0%", "6 of 20");
  const values = panelValues(document);
  assert.ok(values.some((v) => v === "12 · 60%"), values.join(" | "));
  assert.ok(values.some((v) => v === "6 · 30%"), values.join(" | "));
});

test("the reason a rejection was recorded under reaches the screen verbatim", async () => {
  // The whole point of the log: "is it really ~30% unlocatable-quote?" is a
  // question about a named reason, and a prettified label would stop matching
  // the word the pipeline actually writes.
  const document = await render(review({ rejections: rejectionLog({}) }));
  assert.ok(summaryText(document).includes("unlocatable-quote"), summaryText(document));
});

test("this interview's own share of the refusals is reported", async () => {
  const document = await render(review({ rejections: rejectionLog({}) }));
  assert.ok(/3 of them came from this interview/i.test(summaryText(document)),
    summaryText(document));
});

test("an interview with no rows in the log is reported as ambiguous, not as clean", async () => {
  // An extraction that refused nothing writes no row naming this session, and
  // neither does one that never ran. Picking either is how a page ends up
  // telling an operator to re-run a command they just ran.
  const document = await render(review({
    rejections: rejectionLog({ this_interview: { total: 0, reasons: {} } }),
  }));
  const text = summaryText(document);
  assert.ok(/no rejection from this interview/i.test(text), text);
  assert.ok(/never run on it/i.test(text), `the page picked one of two causes: ${text}`);
  assert.ok(/keeps no marker/i.test(text), text);
});

test("an unreadable rejection log is stated rather than rendered as zero", async () => {
  const document = await render(review({
    rejections: {
      status: "unreadable", error: "the rejection log is not a list of rejections",
      total: 0, reasons: {}, sessions_recorded: 0, this_interview: null,
    },
  }));
  const text = summaryText(document);
  assert.ok(/could not be read/i.test(text), text);
  assert.ok(text.includes("not a list of rejections"), text);
  assert.ok(!/refused no claim/i.test(text), `a broken log read as a clean run: ${text}`);
});

// --- evidence against inference ---------------------------------------------------------------

test("the evidence and inference split is drawn at widths that match its counts", async () => {
  const document = await render(review({
    summary: summary({ fact_count: 10, evidence_facts: 7, inference_facts: 2 }),
  }));
  const segs = findAll(document.byId.summary, (n) => /\bpanel-seg\b/.test(n.className || ""));
  assert.equal(segs.length, 3, "the claims of neither kind were dropped from the bar");
  assert.equal(segs[0].style.width, "70.0%");
  assert.equal(segs[1].style.width, "20.0%");
  assert.equal(segs[2].style.width, "10.0%");
  assert.ok(/\bunknown\b/.test(segs[2].className), segs[2].className);
});

test("a fully accounted-for interview draws no third segment", async () => {
  const document = await render(review({
    summary: summary({ fact_count: 4, evidence_facts: 3, inference_facts: 1 }),
  }));
  const segs = findAll(document.byId.summary, (n) => /\bpanel-seg\b/.test(n.className || ""));
  assert.equal(segs.length, 2);
  assert.equal(segs[0].style.width, "75.0%");
});

test("the provenance bar carries the caveat that the evidence count is an upper bound", async () => {
  // Extraction checks that a quote is verbatim and never that the claim follows
  // from it, so a high evidence share is not the good news it looks like.
  const document = await render(review({ summary: summary({}) }));
  assert.ok(/upper bound, not a finding/i.test(summaryText(document)), summaryText(document));
});

test("an interview with no claims draws no provenance bar at all", async () => {
  const document = await render(review({
    summary: summary({ fact_count: 0, evidence_facts: 0, inference_facts: 0, projects: [] }),
  }));
  assert.equal(
    findAll(document.byId.summary, (n) => /\bpanel-seg\b/.test(n.className || "")).length, 0,
    "a proportion was drawn over nothing");
});

// --- the panels survive an unreadable corpus, and do not survive a switch ----------------------

test("an unreadable corpus does not delete the question critique", async () => {
  // `renderSummary` returns early on a null summary, and the two new panels are
  // called separately for exactly this reason: whether the corpus parses has no
  // bearing on whether the interviewer's own questions were judged.
  const document = await render(review({
    session_id: "S1",
    summary: null,
    question_critique: critique({ summary: critiqueSummary({}) }),
  }));
  const text = summaryText(document);
  assert.ok(/could not be read/i.test(text), "the corpus failure stopped being reported");
  assert.ok(/71% clean/.test(text), `the critique went down with the corpus: ${text}`);
});

test("switching interview does not stack two sets of panels", async () => {
  const document = await render(review({
    session_id: "S1",
    question_critique: critique({ summary: critiqueSummary({}) }),
    rejections: rejectionLog({}),
  }), [], THREE_SESSIONS);
  assert.equal(panelFills(document, "clean").length, 2, "nothing to leave behind");

  document.byId["session-picker"].value = "S2";
  (document.byId["session-picker"].listeners.change || []).forEach((fn) => fn({}));
  await document.settle();

  // The stub answers the same payload for every session, so a leftover panel
  // would double every row.
  assert.equal(panelFills(document, "clean").length, 2);
  assert.equal(panelFills(document, "reason").length, 3);
});
