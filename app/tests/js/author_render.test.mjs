// app/tests/js/author_render.test.mjs -- node's built-in test runner, no npm,
// no package.json, no dependency of any kind. The same rule
// `live_state.test.mjs` and `review_render.test.mjs` follow, and gate 6
// actively forbids a JS manifest anywhere in this repo.
//
// The first JS test of the authoring page. `app/tests/test_web_reconnect.py`'s
// node wrapper globs `*.test.mjs`, so this file runs inside the ordinary pytest
// run without a wrapper of its own.
//
// WHAT IT DEFENDS. `/api/state` was added by 3936e0a and had no consumer:
// `grep "api/state" app/static` returned nothing. So the read half of the
// never-clobber guard was missing, and the page kept its original behaviour --
// `_read_existing` returns [] for a MISSING file and for a DAMAGED one, so the
// operator sees an identical empty form either way, retypes one question into
// what looks like a fresh node, presses Save, and only then gets a 409.
//
// Nothing is lost when that happens: the save-side guard is real and refuses
// to write. What is lost is the retyping, and the refusal lands at the worst
// possible moment -- after the work rather than before it.

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const AUTHOR_JS = path.join(HERE, "..", "..", "static", "author", "author.js");
const AUTHOR_HTML = path.join(HERE, "..", "..", "static", "author", "index.html");
const AUTHOR_CSS = path.join(HERE, "..", "..", "static", "author", "author.css");

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.className = "";
    this.id = "";
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.children = [];
    this._text = null;
    this.listeners = {};
    this.attributes = {};
  }
  get textContent() { return this._text; }
  set textContent(value) { this._text = value; this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  // The page sets dir="auto" on every free-text box (Hebrew is a first-class
  // case here). Without this the form throws mid-render.
  setAttribute(name, value) { this.attributes[name] = value; }
  getAttribute(name) { return this.attributes[name]; }
  click() { (this.listeners.click || []).forEach((fn) => fn({ preventDefault() {} })); }
}

function walk(node, out) {
  out.push(node);
  node.children.forEach((child) => walk(child, out));
  return out;
}

function findAll(node, predicate) {
  return walk(node, []).filter(predicate);
}

function readable(node) {
  return walk(node, [])
    .map((n) => (n._text === null || n._text === undefined ? "" : String(n._text)))
    .filter((t) => t !== "")
    .join(" | ");
}

// Taken from the REAL index.html rather than hand-listed, so a slot renamed in
// the markup fails here instead of being quietly absent from the fixture --
// which is precisely how the review page's `extraction` node sat in a test
// harness for a whole commit without any test rendering it.
const html = fs.readFileSync(AUTHOR_HTML, "utf8");
const PAGE_IDS = Array.from(html.matchAll(/id="([^"]+)"/g)).map((m) => m[1]);

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
    // `labelledField` builds its label out of a text node, so without this the
    // whole hypothesis form throws inside a `.then` and renders EMPTY -- which
    // an assertion looking for a missing field cannot tell from the field
    // being missing. Worth the note: the first cut of this harness had exactly
    // that, and every criterion test failed for the wrong reason.
    createTextNode(text) {
      const node = new FakeNode("#text");
      node.textContent = text;
      return node;
    },
    getElementById(id) { return byId[id] || null; },
  };
}

const source = fs.readFileSync(AUTHOR_JS, "utf8");

async function render(state, { stateFails = false } = {}) {
  const document = makeDocument();
  const fetch = (url) => {
    if (url === "/api/state") {
      if (stateFails) { return Promise.reject(new Error("offline")); }
      return Promise.resolve({ ok: true, json: () => Promise.resolve(state) });
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
  };
  new Function("document", "fetch", source)(document, fetch);
  for (let i = 0; i < 20; i += 1) { await Promise.resolve(); }
  return document;
}

const HEALTHY = { brief: "present", survey: "present", hypotheses: "missing" };

// --- the page asks at all ---------------------------------------------------------------

test("the page actually requests /api/state", () => {
  // The whole defect was an endpoint with no consumer. Asserted against the
  // source so it fails even if a future refactor stops rendering the banner.
  assert.ok(source.includes("/api/state"), "author.js never asks what is on disk");
});

test("every artifact section has a slot for the banner", () => {
  for (const name of ["brief", "survey", "hypotheses"]) {
    assert.ok(
      PAGE_IDS.includes(name + "-damage"),
      `index.html has no ${name}-damage slot, so that section can never warn`,
    );
  }
});

// --- a damaged file is named, and only that file -----------------------------------------

test("a damaged file is called out in its own section", async () => {
  const document = await render({ ...HEALTHY, survey: "unreadable" });
  const survey = readable(document.byId["survey-damage"]);
  assert.equal(document.byId["survey-damage"].hidden, false);
  assert.ok(survey.includes("could not be read"), survey);
  assert.ok(survey.includes("context/survey.json"), "say WHICH file, so it can be fixed");
});

test("the other two sections stay quiet", async () => {
  // Per-section, not one page-level banner. A blanket warning over a page
  // whose other two forms are fine teaches the operator to dismiss it.
  const document = await render({ ...HEALTHY, survey: "unreadable" });
  assert.equal(document.byId["brief-damage"].hidden, true);
  assert.equal(document.byId["hypotheses-damage"].hidden, true);
});

test("a healthy page shows no banner anywhere", async () => {
  // The vacuous-pass guard for every test above: a renderer that showed the
  // banner unconditionally would satisfy all of them.
  const document = await render(HEALTHY);
  for (const name of ["brief", "survey", "hypotheses"]) {
    assert.equal(document.byId[name + "-damage"].hidden, true, `${name} warned with no cause`);
  }
});

test("a missing file is not treated as a damaged one", async () => {
  // THE discriminating case, and the reason /api/state exists at all. "Nobody
  // has authored this yet" is the ordinary state of a fresh node and must not
  // be dressed up as a fault.
  const document = await render({ brief: "missing", survey: "missing", hypotheses: "missing" });
  for (const name of ["brief", "survey", "hypotheses"]) {
    assert.equal(document.byId[name + "-damage"].hidden, true, `${name} called missing damaged`);
  }
});

test("all three damaged reports all three", async () => {
  const document = await render({
    brief: "unreadable", survey: "unreadable", hypotheses: "unreadable",
  });
  for (const name of ["brief", "survey", "hypotheses"]) {
    assert.equal(document.byId[name + "-damage"].hidden, false);
  }
});

// --- and when the page cannot ask --------------------------------------------------------

test("a failed check says it could not check, not that the file is broken", async () => {
  // Saying "this file could not be read" when the truth is "this page could
  // not ask" is the same class of error the banner exists to prevent.
  const document = await render(null, { stateFails: true });
  const text = readable(document.byId["brief-damage"]);
  assert.equal(document.byId["brief-damage"].hidden, false, "a failed check must not be silent");
  assert.ok(text.includes("could not check"), text);
  assert.ok(!text.includes("exists on disk"), `claimed to know the file is damaged: ${text}`);
});

test("an unrecognised status is not rendered as damage", async () => {
  // The far side of the same rule. A status this page does not know is not
  // evidence of a broken file, and inventing a red banner for it would make
  // the page cry wolf on the day a fourth state is added server-side.
  const document = await render({ brief: "somethingelse", survey: "present", hypotheses: "present" });
  assert.equal(document.byId["brief-damage"].hidden, true);
});

// --- the banner is visible -----------------------------------------------------------------

test("the damage class has a style, so the banner is not invisible", () => {
  // review.js's own convention: a status string that reaches the DOM as a
  // class name and has no CSS rule renders as ordinary text.
  const css = fs.readFileSync(AUTHOR_CSS, "utf8");
  assert.ok(css.includes(".damage"), "the banner has no styling and reads as body copy");
});

// --- the hypothesis form: criterion, and the fourth state --------------------------------
//
// Two gaps found together on 2026-08-12, and they are the same gap: the model
// grew and this form did not.
//
// `criterion` -- the pre-registered "we are right if..." rule -- has been on
// `Hypothesis` since 763787c, and `criterion_registered_after` (the sessions
// that already existed when it was written) landed the same day. The page
// rendered neither. So the ONLY way to author a criterion was to hand-edit
// context/hypotheses.json, and the next save from this page would have written
// the hypothesis back without it. The repo's own research found that no shipped
// tool has a pre-registered criterion; this one had the data model, the
// provenance stamp, and no input box.
//
// `contested` is a legal RESTING state -- evidence points both ways and the
// honest record says so -- and the state dropdown offered three options.

function hypothesisForm(document) {
  return readable(document.byId["hypotheses-items"]);
}

async function renderWith(hypotheses) {
  const document = makeDocument();
  const posted = [];
  const fetch = (url, init) => {
    if (init && init.method === "POST") {
      posted.push({ url, body: JSON.parse(init.body) });
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ saved: 1 }) });
    }
    if (url === "/api/state") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(HEALTHY) });
    }
    if (url === "/api/hypotheses") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(hypotheses) });
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve([]) });
  };
  new Function("document", "fetch", source)(document, fetch);
  for (let i = 0; i < 20; i += 1) { await Promise.resolve(); }
  document.posted = posted;
  return document;
}

/** Press Save and return the body that went to /api/hypotheses. */
async function save(document) {
  document.byId["save-hypotheses"].click();
  for (let i = 0; i < 20; i += 1) { await Promise.resolve(); }
  const post = document.posted.filter((p) => p.url === "/api/hypotheses").pop();
  assert.ok(post, "pressing Save sent no request to /api/hypotheses");
  return post.body;
}

test("the hypothesis form has a criterion field at all", async () => {
  const document = await renderWith([
    { id: "h1", statement: "onboarding blocks activation", falsifier: "three breeze through",
      criterion: "", state: "untested" },
  ]);
  const text = hypothesisForm(document);
  assert.ok(/we are right if/i.test(text), `no criterion field on the form: ${text}`);
});

test("the criterion label says WHEN to write it, not just what it is", async () => {
  // The field is worthless if it is filled in after the interviews that would
  // judge it -- that is the whole content of a PRE-registered rule, and the
  // label is the only place a human is told.
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", criterion: "", state: "untested" },
  ]);
  assert.ok(/before/i.test(hypothesisForm(document)), hypothesisForm(document));
});

test("an existing criterion is loaded back into the box", async () => {
  // The round-trip. A form that renders the field but drops the stored value
  // erases a criterion on the next save -- worse than having no field, because
  // the provenance stamp would move too.
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", state: "untested",
      criterion: "we are right if 3 of 10 raise it unprompted" },
  ]);
  const boxes = findAll(document.byId["hypotheses-items"], (n) => n.tagName === "textarea");
  const values = boxes.map((n) => n.value);
  assert.ok(
    values.includes("we are right if 3 of 10 raise it unprompted"),
    `the stored criterion was not loaded into any box: ${JSON.stringify(values)}`,
  );
});

test("the criterion box starts empty and carries no example text", async () => {
  // D-04/§K14, and here it is load-bearing rather than tidy: an example left
  // in place and saved is a decision rule nobody wrote, and
  // `criterion_registered_after` would stamp it with the interviews that
  // already existed -- giving a fabricated rule the strongest provenance this
  // product can issue.
  const document = await renderWith([
    { id: "h1", statement: "", falsifier: "", criterion: "", state: "untested" },
  ]);
  const boxes = findAll(document.byId["hypotheses-items"], (n) => n.tagName === "textarea");
  assert.ok(boxes.length >= 3, `expected statement, falsifier and criterion boxes, got ${boxes.length}`);
  for (const box of boxes) {
    assert.equal(box.value, "", "a hypothesis box started with text in it");
  }
});

test("the state dropdown offers every state the model allows", async () => {
  // Pinned against the list app/coach/corpus.py::ALL_HYPOTHESIS_STATES holds.
  // `contested` was added to the model and never to this list, so the state
  // was unreachable from the only surface that authors hypotheses -- and worse,
  // a hypothesis hand-edited to `contested` would be written back as whatever
  // the select happened to show.
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", state: "contested" },
  ]);
  const selects = findAll(document.byId["hypotheses-items"], (n) => n.tagName === "select");
  assert.equal(selects.length, 1, "expected exactly one state dropdown");
  const options = selects[0].children.map((o) => o.value).sort();
  assert.deepEqual(
    options,
    ["contested", "contradicted", "supported", "untested"],
    "the dropdown and app/coach/corpus.py::ALL_HYPOTHESIS_STATES disagree",
  );
});

test("a stored contested state survives a render", async () => {
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", state: "contested" },
  ]);
  const selects = findAll(document.byId["hypotheses-items"], (n) => n.tagName === "select");
  assert.equal(selects[0].value, "contested", "the form silently downgraded a contested hypothesis");
});

// --- the save path, which is where the field was actually being lost --------------------
//
// Added after a mutation showed the gap. Deleting the `criterion` line from the
// POST builder -- leaving the box rendered and the value loaded, which is
// EXACTLY the state this page shipped in for `criterion` -- left all sixteen
// tests above green. Every one of them reads the DOM, and none of them pressed
// Save. Rendering a field the form then drops is worse than not rendering it:
// the operator types a decision rule, watches it appear, saves, and it is gone.

test("pressing Save sends the criterion that is in the box", async () => {
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", state: "untested",
      criterion: "we are right if 3 of 10 raise it unprompted" },
  ]);
  const body = await save(document);
  assert.equal(body.length, 1);
  assert.equal(body[0].criterion, "we are right if 3 of 10 raise it unprompted");
});

test("pressing Save sends the state that is in the dropdown", async () => {
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", state: "contested" },
  ]);
  const body = await save(document);
  assert.equal(body[0].state, "contested");
});

test("a blank criterion is omitted, never sent as an empty string", async () => {
  // The same rule the falsifier follows. It matters here because `criterion`
  // decides `is_scorable`: the difference between "no rule was written" and
  // "a rule was written and it is empty" is the difference between a
  // hypothesis the hub refuses to tally and one it tallies against nothing.
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", criterion: "   ", state: "untested" },
  ]);
  const body = await save(document);
  assert.ok(!("criterion" in body[0]), `sent a blank criterion: ${JSON.stringify(body[0])}`);
});

test("loading a criterion and saving without touching it does not lose it", async () => {
  // The round trip, stated as the failure it prevents. Before the field
  // existed on the form, opening this page and pressing Save on an untouched
  // hypothesis ERASED a hand-authored criterion -- and moved its provenance
  // stamp with it, because the server recomputes the stamp whenever the text
  // changes.
  const stored = "we are right if 3 of 10 raise it unprompted";
  const document = await renderWith([
    { id: "h1", statement: "s", falsifier: "f", criterion: stored, state: "untested" },
  ]);
  const body = await save(document);
  assert.equal(body[0].criterion, stored, "an untouched criterion was dropped by Save");
});
