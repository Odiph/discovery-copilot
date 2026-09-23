// The hub's client. textContent only -- this page renders verbatim customer
// speech and hand-authored hypothesis text, none of which may ever be script.
(function () {
  "use strict";

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = text; }
    return node;
  }

  function badge(text, extra) {
    return el("span", extra ? "badge " + extra : "badge", text);
  }

  // The session the transcript below belongs to. A citation is only followable
  // into THIS transcript; a citation from an earlier interview is drawn as
  // text, because an anchor that scrolls to a turn from a different
  // conversation is worse than no anchor.
  var sessionId = null;
  var litTurns = [];

  function highlight(seqs) {
    litTurns.forEach(function (node) {
      node.className = node.className.replace(/\s*\bhit\b/, "");
    });
    litTurns = [];
    seqs.forEach(function (seq) {
      var node = document.getElementById("turn-" + seq);
      if (!node) { return; }
      node.className = node.className + " hit";
      litTurns.push(node);
    });
    if (litTurns.length) { litTurns[0].scrollIntoView({ block: "center" }); }
  }

  function citedSeqs(fact) {
    // The whole span, not its endpoints: a claim drawn from #12-14 is three
    // turns of evidence and highlighting two of them hides the middle one.
    var out = [];
    for (var seq = fact.seqs[0]; seq <= fact.seqs[fact.seqs.length - 1]; seq += 1) {
      out.push(seq);
    }
    return out;
  }

  function citation(fact) {
    var span = fact.seqs.length > 1
      ? fact.seqs[0] + "–" + fact.seqs[fact.seqs.length - 1]
      : String(fact.seqs[0]);
    var text = fact.provenance + " · " + fact.session_id + " #" + span;
    if (fact.session_id !== sessionId) { return el("span", "cite", text); }
    var link = el("a", "cite", text);
    link.href = "#turn-" + fact.seqs[0];
    link.addEventListener("click", function () { highlight(citedSeqs(fact)); });
    return link;
  }

  function factNode(fact) {
    // Every claim carries its own quote and citation, and the citation lands
    // ON the turn it came from. The loudest complaint users report about
    // screens like this is not hallucination -- it is evidence buried one
    // click too deep. Here the quote is zero clicks and its context is one.
    var wrap = el("div", fact.from_this_interview ? "fact new" : "fact");
    wrap.appendChild(el("span", null, fact.claim));
    if (fact.quoted_text) {
      wrap.appendChild(el("span", "quote", "“" + fact.quoted_text + "”"));
    }
    wrap.appendChild(citation(fact));
    return wrap;
  }

  function column(title, facts, className, emptyText) {
    var col = el("div", "column " + className);
    col.appendChild(el("h4", null, title));
    if (!facts.length) {
      col.appendChild(el("p", "empty", emptyText));
    } else {
      facts.forEach(function (fact) { col.appendChild(factNode(fact)); });
    }
    return col;
  }

  function tally(counts) {
    // INTERVIEWS, not facts. Counting facts would let one talkative
    // interviewee who repeated themselves four times outweigh four separate
    // people -- the vote-counting failure evidence synthesis abandoned
    // decades ago.
    return counts.supporting_interviews + " for / " + counts.contrary_interviews + " against";
  }

  // --- the graphical half of a standing (design direction D) --------------------------
  //
  // Drawn with divs and inline widths, never SVG, and that is a testability
  // decision rather than a stylistic one: `app/tests/js/review_render.test.mjs`
  // drives this file against a minimal fake DOM that records `style.width`, so
  // a bar built this way can be asserted to AGREE with the number printed
  // beside it. A bar whose width contradicts its own label is exactly the
  // defect a textContent-only harness cannot see, and an SVG one would be
  // invisible to it.
  //
  // The palette is this page's OWN `--supports`/`--contrary`/`--dim`, not a
  // new one. Those three were already tuned for both canvases with the
  // contrast working shown at the top of review.css; introducing a second
  // diverging pair here would mean re-doing that work to end up somewhere
  // equivalent, and would leave two pairs to keep in agreement.

  function barSegment(className, share, title) {
    var seg = el("div", className);
    seg.style.width = share.toFixed(1) + "%";
    // The accessible name carries the COUNT, so the meaning never depends on
    // reading a width or a colour.
    seg.title = title;
    return seg;
  }

  function standingBar(standing) {
    var forCount = standing.supporting_interviews;
    var againstCount = standing.contrary_interviews;
    var unclearCount = standing.unclear_interviews;
    var total = forCount + againstCount + unclearCount;

    var wrap = el("div", "standing-bar");
    if (!total) {
      // No bar at all rather than an empty track. A drawn-but-empty meter
      // reads as "measured, and it came to nothing"; this is "not measured".
      wrap.appendChild(el("p", "note", "No interview has borne on this yet."));
      return wrap;
    }

    var track = el("div", "standing-track");
    // Order is fixed -- supports, unclear, undermines -- so the eye compares
    // the same edge across cards. Sorting by size would make two cards with
    // identical numbers look different.
    if (forCount) {
      track.appendChild(barSegment("standing-seg for", pctOf(forCount, total),
        forCount + " supporting"));
    }
    if (unclearCount) {
      track.appendChild(barSegment("standing-seg unclear", pctOf(unclearCount, total),
        unclearCount + " unresolved"));
    }
    if (againstCount) {
      track.appendChild(barSegment("standing-seg against", pctOf(againstCount, total),
        againstCount + " undermining"));
    }
    wrap.appendChild(track);

    var key = el("div", "standing-key");
    key.appendChild(el("span", "k for", forCount + " for"));
    if (unclearCount) { key.appendChild(el("span", "k unclear", unclearCount + " unresolved")); }
    key.appendChild(el("span", "k against", againstCount + " against"));
    key.appendChild(el("span", "k dim", "interviews, not claims"));
    wrap.appendChild(key);
    return wrap;
  }

  function pctOf(n, total) {
    return total > 0 ? (n / total) * 100 : 0;
  }

  // The ONE criterion shape this page will draw. Deliberately narrow.
  //
  // A criterion is free prose ("at least 3 of 6 interviewees describe...",
  // "read-only uses outnumber edit-or-run at least 2 to 1"). Only the first of
  // those has an unambiguous target and denominator, so only the first gets a
  // meter. Everything else keeps its sentence and gets no bar -- a chart drawn
  // from a rule this code guessed at would be the exact target-painting the
  // criterion field exists to prevent, except now with a progress bar to make
  // it look measured.
  function parseCriterionTarget(text) {
    var match = /\bat least\s+(\d+)\s+of\s+(\d+)\b/i.exec(text || "");
    if (!match) { return null; }
    var need = parseInt(match[1], 10);
    var of = parseInt(match[2], 10);
    if (!need || !of || need > of) { return null; }
    return { need: need, of: of };
  }

  function criterionMeter(standing) {
    if (!standing.is_scorable) { return null; }
    var target = parseCriterionTarget(standing.criterion);
    if (!target) { return null; }

    var have = standing.supporting_interviews;
    var met = have >= target.need;

    var wrap = el("div", "criterion-meter" + (met ? " met" : ""));
    var track = el("div", "meter-track");
    var fill = el("div", "meter-fill");
    fill.style.width = Math.min(100, pctOf(have, target.of)).toFixed(1) + "%";
    track.appendChild(fill);

    // The threshold is the whole point of the drawing. Without it this is a
    // progress bar toward the sample size, which is a different and much less
    // interesting fact than progress toward the RULE.
    var mark = el("div", "meter-threshold");
    mark.style.left = pctOf(target.need, target.of).toFixed(1) + "%";
    mark.title = "the rule needs " + target.need;
    track.appendChild(mark);
    wrap.appendChild(track);

    wrap.appendChild(el("p", "meter-read",
      have + " of " + target.of + " · the rule needs " + target.need
      + (met ? " · threshold met" : "")));

    // Named out loud because the page is choosing an interpretation. The rule
    // is prose; this meter reads it as being about SUPPORTING interviews, and
    // a rule written about the other direction would be drawn backwards with
    // nothing to signal it. Saying which number is on the bar lets a reader
    // catch that in one glance.
    wrap.appendChild(el("p", "note",
      "Counting supporting interviews against the rule's target."));
    return wrap;
  }

  function contributionLine(standing) {
    // The sentence this page exists for. Every surveyed research tool ends at
    // "evidence captured" and shows a LEVEL; a level cannot tell an operator
    // whether the call they just finished changed anything.
    var mine = standing.this_interview;
    var parts = [];
    if (mine.supporting_facts) { parts.push(mine.supporting_facts + " supporting"); }
    if (mine.contrary_facts) { parts.push(mine.contrary_facts + " contrary"); }
    if (mine.unclear_facts) { parts.push(mine.unclear_facts + " undirected"); }
    if (!parts.length) {
      return "This interview added nothing to this hypothesis.";
    }
    // Named as CLAIMS beside numbers named interviews, so a reader never has
    // to remember which kind of count they are looking at.
    return "This interview added " + parts.join(", ") + " claim"
      + (mine.supporting_facts + mine.contrary_facts + mine.unclear_facts === 1 ? "" : "s")
      + ".";
  }

  // Every interview under this project, filled from `/api/sessions` at boot.
  // Needed to PRICE a criterion edit: everything here that is not already
  // inside a hypothesis's boundary is currently pre-dated by its rule, and an
  // edit moves all of it inside.
  var allSessions = [];

  function predatedBy(standing) {
    var boundary = standing.criterion_registered_after || [];
    return allSessions.filter(function (id) { return boundary.indexOf(id) === -1; });
  }

  function saveHypotheses(mutate) {
    // Read-modify-write the WHOLE array, because that is the shape the save
    // path takes and the stamping rule reads the previous file to decide
    // whether each criterion changed. Sending one entry would look tidier and
    // would silently delete every other hypothesis.
    return fetch("/api/hypotheses")
      .then(function (r) { return r.json(); })
      .then(function (existing) {
        var next = mutate(Array.isArray(existing) ? existing : []);
        return fetch("/api/hypotheses", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(next),
        });
      })
      .then(function (r) { return r.json(); });
  }

  function announceSaved(message) {
    // The page-level banner, deliberately NOT the card. A successful save is
    // followed by a re-read that rebuilds every card, so a confirmation
    // written onto one is destroyed before it can be read.
    var banner = document.getElementById("save-banner");
    banner.textContent = message;
    banner.hidden = false;
  }

  // The confirmation the hypothesis saves announce. Passed to `saveResult`
  // rather than hardcoded inside it, because the participant save writes a
  // different file for a different reason and telling the operator it wrote
  // `context/hypotheses.json` would be a plain lie about what just happened.
  var HYPOTHESES_SAVED =
    "Saved to context/hypotheses.json. " +
    "Any criterion written now records the interviews it did not pre-date.";

  function saveResult(node, promise, onSuccess, message) {
    return promise.then(function (body) {
      if (body && body.error) {
        // A REFUSAL stays on the card that asked for it, and that asymmetry is
        // deliberate: nothing re-renders on a failure, so the message survives
        // exactly where the operator is looking, next to the text it refused.
        node.textContent = "Not saved: " + body.error;
        node.className = "save-status failed";
        return;
      }
      if (onSuccess) { onSuccess(); }
      announceSaved(message || HYPOTHESES_SAVED);
    }).catch(function () {
      node.textContent = "Not saved: the request failed.";
      node.className = "save-status failed";
    });
  }

  function criterionEditor(standing) {
    var box = el("div", "criterion-editor");
    var input = el("input");
    input.type = "text";
    input.value = standing.criterion || "";
    input.placeholder = "e.g. 3 of 6 interviewees describe this unprompted";

    var button = el("button", null, standing.criterion ? "Replace the rule" : "Write the rule");
    var status = el("span", "save-status");

    // The COST, shown before the click rather than after it. Editing a
    // criterion is allowed here because the alternative is a rule that never
    // gets written at all -- but a casual edit on the post-interview surface
    // is exactly the target-painting the boundary exists to catch, so the
    // price is on screen next to the button that charges it.
    var cost = predatedBy(standing);
    if (standing.criterion && cost.length) {
      box.appendChild(el("p", "flag",
        "This rule currently pre-dates " + cost.length + " interview" +
        (cost.length === 1 ? "" : "s") + " — " + cost.join(", ") +
        ". Replacing it gives that up: the new wording was written after those calls."));
    } else if (standing.criterion) {
      box.appendChild(el("p", "note",
        "This rule pre-dates none of the interviews on disk, so replacing it gives up nothing."));
    }

    button.addEventListener("click", function () {
      var text = (input.value || "").trim();
      if (!text) {
        status.textContent = "Write the rule first.";
        status.className = "save-status failed";
        return;
      }
      button.disabled = true;
      saveResult(status, saveHypotheses(function (existing) {
        return existing.map(function (entry) {
          return entry && entry.id === standing.id
            ? Object.assign({}, entry, { criterion: text })
            : entry;
        });
      }), function () { load(currentSession); });
    });

    box.appendChild(input);
    box.appendChild(button);
    box.appendChild(status);
    return box;
  }

  function newHypothesisForm() {
    var box = el("div", "card new-hypothesis");
    var fields = {};
    [
      ["statement", "What you believe", "e.g. Players defer lookups to the end of a session"],
      ["falsifier", "This would be wrong if", "what you would have to hear to abandon it"],
      ["criterion", "Decide it by (optional)", "e.g. 3 of 6 describe this unprompted"],
    ].forEach(function (spec) {
      box.appendChild(el("label", null, spec[1]));
      var input = el("input");
      input.type = "text";
      input.placeholder = spec[2];
      fields[spec[0]] = input;
      box.appendChild(input);
    });

    // Said before the form is filled, not after it is submitted. A criterion
    // typed here is typed AFTER every interview already on disk, and the
    // record will say so.
    if (allSessions.length) {
      box.appendChild(el("p", "note",
        "A rule written now pre-dates none of the " + allSessions.length + " interview" +
        (allSessions.length === 1 ? "" : "s") + " already recorded under this project."));
    }

    var button = el("button", null, "Add this hypothesis");
    var status = el("span", "save-status");
    button.addEventListener("click", function () {
      var statement = (fields.statement.value || "").trim();
      var falsifier = (fields.falsifier.value || "").trim();
      if (!statement || !falsifier) {
        // Refused in the page rather than by the server, because the server's
        // message is about a schema and this one is about the thing that
        // makes a hypothesis a hypothesis.
        status.textContent = "A hypothesis needs both a statement and what would falsify it.";
        status.className = "save-status failed";
        return;
      }
      button.disabled = true;
      saveResult(status, saveHypotheses(function (existing) {
        var ids = {};
        existing.forEach(function (entry) { if (entry && entry.id) { ids[entry.id] = true; } });
        var n = existing.length + 1;
        while (ids["h" + n]) { n += 1; }
        return existing.concat([{
          id: "h" + n,
          statement: statement,
          falsifier: falsifier,
          criterion: (fields.criterion.value || "").trim(),
          state: "untested",
        }]);
      }), function () { load(currentSession); });
    });
    box.appendChild(button);
    box.appendChild(status);
    return box;
  }

  function standingCard(standing) {
    var card = el("div", standing.moved_this_interview
      ? "card touched moved"
      : (standing.touched_this_interview ? "card touched" : "card"));
    card.appendChild(el("h3", null, standing.statement));

    var badges = el("div", "badges");
    badges.appendChild(badge(standing.state));
    // MOVED and TOUCHED are different claims and the badge makes the stronger
    // one. A fact the model could not direct is precisely the absence of
    // direction; letting it light "moved" would put the whole thesis of this
    // product behind a call that settled nothing.
    if (standing.moved_this_interview) {
      badges.appendChild(badge("moved this interview", "moved"));
    } else if (standing.touched_this_interview) {
      badges.appendChild(badge("asked, unresolved", "touched"));
    }
    badges.appendChild(badge(tally(standing) + " (interviews)"));
    if (!standing.is_scorable) {
      badges.appendChild(badge("no criterion — cannot be scored", "unscorable"));
    }
    card.appendChild(badges);

    // The shape of the evidence, above the sentence describing it. Both stay:
    // the bar is read at a glance and the sentence carries the before/now
    // change, which no single bar can show without becoming two bars.
    card.appendChild(standingBar(standing));

    card.appendChild(el("p", "delta",
      "was " + tally(standing.before) + " → now " + tally(standing) + ". "
      + contributionLine(standing)));

    // The falsifier, on the page where a human decides whether the belief
    // survived. Every bearing above was DIRECTED against it, so a reader
    // checking a stance -- or the anti-confirmation critic's work -- needs it
    // in front of them rather than in the payload.
    card.appendChild(el("p", "falsifier", "This is wrong if: " + standing.falsifier));

    // The pre-registered rule, or the loud absence of one. No shipped tool in
    // the surveyed set has this field, and without it a tally is a target
    // painted around an arrow already fired.
    if (standing.criterion) {
      card.appendChild(el("p", "criterion", "We are right if: " + standing.criterion));
      // Only for a rule whose target this page can read without guessing --
      // see `parseCriterionTarget`. A criterion in any other shape keeps its
      // sentence and gets no bar, which is the honest outcome: the absence of
      // a meter here means "this rule is not in a shape that can be charted",
      // never "this rule is not being met".
      var meter = criterionMeter(standing);
      if (meter) { card.appendChild(meter); }
    } else {
      card.appendChild(el("p", "criterion missing",
        "No decision rule was written before interviewing. Any count below is "
        + "evidence about this hypothesis, not a verdict on it."));
    }

    // Writing the rule is offered HERE, on the card that just showed it
    // missing, because the moment an operator wants a decision rule is the
    // moment they see a tally they cannot honestly read.
    card.appendChild(criterionEditor(standing));

    var ledger = el("div", "ledger");
    ledger.appendChild(column("Supports", standing.supporting, "for",
      "Nothing in the corpus supports this yet."));
    // The empty state is a QUESTION, not a blank. Symmetry has to be
    // structural, or the absence of contrary evidence reads as its absence in
    // the world.
    ledger.appendChild(column("Undermines", standing.contrary, "against",
      "No disconfirming evidence found — was any sought?"));
    card.appendChild(ledger);

    // Always drawn, exactly like the two columns above. A claim the model
    // could not direct vanishing without trace is the same failure as a
    // missing contrary column wearing a politer face.
    card.appendChild(column("Bears on it, unresolved", standing.unclear, "unclear",
      "Nothing was said that bears on this without pointing one way."));
    return card;
  }

  function decide(proposal, verb, statusNode, buttons, session) {
    buttons.forEach(function (b) { b.disabled = true; });
    statusNode.textContent = "Saving…";
    fetch("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hypothesis_id: proposal.target_hypothesis_id,
        proposed_state: proposal.proposed_state,
        decision: verb,
        // The interview on screen, sent so a rejection is filed against the
        // call it was proposed in rather than against whatever this process
        // was started with. The card follows the picker everywhere else on
        // this page; the record it writes has to follow it too.
        session: session || null
      })
    })
      .then(function (r) { return r.json().then(function (b) { return [r.ok, b]; }); })
      .then(function (pair) {
        if (!pair[0]) {
          statusNode.textContent = "Not applied — " + (pair[1].error || "refused");
          buttons.forEach(function (b) { b.disabled = false; });
          return;
        }
        if (pair[1].applied) {
          statusNode.textContent =
            "Accepted — now " + proposal.proposed_state + ". Reload to see the standing.";
          return;
        }
        // A rejection changes no hypothesis AND is written down, and the page
        // says both. "Nothing was changed" was the old wording and it is no
        // longer true of the second half -- and the half it was hiding is the
        // one the operator needs to be able to rely on later, when this same
        // proposal comes back in another interview.
        //
        // The failure case is spelled out rather than folded into the success
        // sentence: if the record could not be kept, the operator is the only
        // one who can do anything about it, and a page that said "recorded"
        // regardless would be the same silent loss wearing a confirmation.
        statusNode.textContent = pair[1].recorded === false
          ? "Rejected — the hypothesis is unchanged, but this decision could NOT be "
            + "recorded: " + (pair[1].record_error || "reason unknown")
          : "Rejected — the hypothesis is unchanged, and the decision was recorded.";
      })
      .catch(function () {
        statusNode.textContent = "Not applied — the request failed.";
        buttons.forEach(function (b) { b.disabled = false; });
      });
  }

  // What this page says about a proposal the operator has already declined
  // once. Returns null when there is nothing to say, so the common case adds
  // no node at all.
  //
  // The card is MARKED, never hidden or reordered. A proposal declined in
  // March may be perfectly acceptable in June on evidence that did not exist
  // then, so removing it would replace one silent loss (the forgotten
  // rejection) with another (the withheld proposal). What the operator is
  // owed is their own earlier judgement, and WHEN they made it -- an
  // un-dated "you declined this before" is nearly useless, because how much
  // weight yesterday's decision deserves is entirely a question of how long
  // ago it was and what has arrived since.
  function declineNotice(declined) {
    if (!declined) { return null; }
    var box = el("div", "declined-before");
    var times = declined.count === 1 ? " once" : " " + declined.count + " times";
    // The count is of DECLINES, not of proposals -- the log records decisions
    // and has never seen how often the coach raised this. Worded so it cannot
    // be read as the other number.
    box.appendChild(el("p", "declined-headline",
      "You have already declined this proposal" + times + "."));
    box.appendChild(el("p", "declined-when",
      declined.last_declined_at
        ? "Most recently " + declined.last_declined_at + "."
        : "The log does not say when — the record carries no time."));
    if (declined.sessions && declined.sessions.length) {
      box.appendChild(el("p", "declined-where",
        (declined.sessions.length === 1 ? "Declined after interview: " : "Declined after interviews: ")
        + declined.sessions.join(", ") + "."));
    }
    // Said out loud, because the buttons below are live and look exactly as
    // they did the first time.
    box.appendChild(el("p", "declined-note",
      "It is offered again anyway — evidence that did not exist then may have "
      + "arrived since. Deciding again is a new judgement, not a correction."));
    return box;
  }

  function proposalCard(proposal, standing, session) {
    var card = el("div", "card proposal");
    card.appendChild(el("h3", null,
      standing ? standing.statement : proposal.target_hypothesis_id));

    var badges = el("div", "badges");
    badges.appendChild(badge(
      (standing ? standing.state : "unknown") + " → " + proposal.proposed_state, "moved"));
    if (proposal.previously_declined) {
      badges.appendChild(badge("previously declined", "declined"));
    }
    if (!standing) {
      badges.appendChild(badge(
        "no such hypothesis in the corpus — " + proposal.target_hypothesis_id, "unscorable"));
    } else if (!standing.is_scorable) {
      badges.appendChild(badge("no criterion — cannot be scored", "unscorable"));
    }
    card.appendChild(badges);

    // Above the evidence and the buttons, not below them: the operator's own
    // earlier verdict is context for reading everything under it.
    var notice = declineNotice(proposal.previously_declined);
    if (notice) { card.appendChild(notice); }

    // The pre-registered rule sits beside the button, not a scroll away: the
    // whole point of writing it before the interviewing is that it is what
    // this decision gets judged against.
    if (standing && standing.criterion) {
      card.appendChild(el("p", "criterion", "We are right if: " + standing.criterion));
    } else if (standing) {
      card.appendChild(el("p", "criterion missing",
        "No decision rule was written before interviewing. Accepting this is a "
        + "judgement call, not a scored result."));
    }

    var asked = el("div", "asked");
    asked.appendChild(el("h4", null, "Asked in this interview"));
    proposal.questions.forEach(function (q, i) {
      asked.appendChild(el("p", "quote", "#" + proposal.seqs[i] + " " + q));
    });
    card.appendChild(asked);

    // Both columns, same as the standing card. Accepting a state on a
    // supporting-only view is exactly the confirmation machine `Bearing`
    // exists to prevent.
    if (standing) {
      var ledger = el("div", "ledger");
      ledger.appendChild(column("Supports", standing.supporting, "for",
        "Nothing in the corpus supports this yet."));
      ledger.appendChild(column("Undermines", standing.contrary, "against",
        "No disconfirming evidence found — was any sought?"));
      card.appendChild(ledger);
    }

    var actions = el("div", "actions");
    var accept = el("button", "accept", "Accept");
    var reject = el("button", "reject", "Reject");
    var status = el("p", "decision-status", "");
    var buttons = [accept, reject];
    accept.addEventListener("click", function () {
      decide(proposal, "accept", status, buttons, session);
    });
    reject.addEventListener("click", function () {
      decide(proposal, "reject", status, buttons, session);
    });
    actions.appendChild(accept);
    actions.appendChild(reject);
    card.appendChild(actions);
    card.appendChild(status);
    return card;
  }

  // Above the standings on purpose. Every zero below is conditional on
  // extraction having run, and until that is stated the page's default
  // rendering is a confident false negative: "Nothing supports this yet"
  // reads as a finding whether or not anything was ever looked at.
  function renderExtraction(extraction, project, sessionId) {
    var box = document.getElementById("extraction");
    if (!extraction) { box.hidden = true; return; }
    box.hidden = false;
    box.className = "extraction " + extraction.status;

    var said = {
      "not-extracted": "This interview has not been extracted. Nothing below reflects it.",
      // Split out from "not-extracted" after the 2026-08-12 audit. The corpus
      // exists, so extraction HAS run in this project -- and nothing on disk
      // says whether this particular interview was extracted and yielded
      // nothing, or was never extracted at all. The sentence says exactly
      // that rather than picking one, because picking one is how the previous
      // wording ended up telling the operator to re-run a command that had
      // already run.
      "no-facts-from-this-interview":
        "The corpus has no claims from this interview. Either it has not been extracted, " +
        "or extraction found nothing in it -- nothing on disk distinguishes the two.",
      "no-hypotheses": "No hypotheses are authored for this project, so no fact can bear on " +
        "anything. The empty columns below say nothing about this interview.",
      "unreadable": "The fact corpus could not be read, so nothing below reflects this interview.",
      "extracted": null
    };
    // An UNKNOWN status must not render a blank banner. `said[status]` is
    // `undefined` for a word this page has never heard of, and `undefined`
    // silently produced an empty headline -- the exact failure the audit
    // predicted for a rename, with every test green because no test rendered
    // this function at all.
    var headline = said[extraction.status];
    if (headline === undefined) {
      headline = "The page does not recognise the extraction status '" +
        extraction.status + "'. Treat every count below as unexplained.";
    } else if (headline === null) {
      headline = extraction.facts_this_interview + " fact" +
        (extraction.facts_this_interview === 1 ? "" : "s") + " extracted from this interview" +
        ", " + extraction.facts_without_bearings + " bearing on no hypothesis.";
    }
    box.appendChild(el("div", "extraction-headline", headline));

    if (extraction.status === "not-extracted" ||
        extraction.status === "no-facts-from-this-interview") {
      // The remedy, not just the diagnosis. Nothing in `run_hub` triggers or
      // mentions extraction, so a reader has no way to know this step exists.
      //
      // Shown for both, because in both cases running it is the action that
      // resolves the question -- but the headline above has already said
      // whether it is known to be needed, so the operator is not being told
      // to repeat something they just did without knowing that is what they
      // are being told.
      box.appendChild(el("div", "extraction-remedy",
        "Run: python -m app extract " + (sessionId || "<session>") +
        " --project " + (project || "<project>")));
    }
    box.appendChild(el("div", "extraction-counts",
      extraction.facts_total + " facts in the corpus · " +
      extraction.hypotheses_loaded + " hypotheses loaded"));
  }

  // WHO was interviewed, and the box to say so in.
  //
  // The EMPTY state is the design here, not the filled one. Every interview
  // recorded before this field existed carries no label, so "nobody is named"
  // is what this box renders almost every time anyone opens the page — and a
  // blank space where a name belongs reads as a page that failed to load
  // something, which is exactly the wrong reaction to the ordinary case. So
  // the absence is written out in words, and it says what the absence costs
  // rather than merely reporting it: the count of interviews further down this
  // page is the product's central rule, and it is a count of PEOPLE only while
  // nobody is re-interviewed and nobody records two calls in one sitting.
  // Nothing on this page can tell whether that held. A human with the labels
  // in front of them can.
  //
  // What this deliberately does NOT do is nag. No red, no warning icon, no
  // "missing" — an unlabelled interview is not a defect, it is an interview
  // nobody has got round to naming, and most of them never will be.
  function renderParticipant(data) {
    var host = document.getElementById("participant");
    if (!data.session_id) {
      // No interview on screen means there is nothing to attach a name to.
      // Drawing a disabled input here would offer an action that cannot work.
      return;
    }
    var label = data.participant || null;

    host.appendChild(el("p", label ? "participant-name" : "participant-name empty",
      label ? "Interviewed: " + label : "No participant recorded for this interview."));
    if (!label) {
      host.appendChild(el("p", "note",
        "That is the normal state — nothing recorded who was in the room until now. "
        + "The standings below count interviews, which is a count of people only if "
        + "nobody was interviewed twice."));
    }

    var input = el("input");
    input.type = "text";
    input.value = label || "";
    input.placeholder = "a name, a role or a pseudonym";

    var button = el("button", null, label ? "Change who this was" : "Record who this was");
    var status = el("span", "save-status");

    button.addEventListener("click", function () {
      var text = (input.value || "").trim();
      if (!text && !label) {
        // Refused in the page rather than by the server. Saving nothing over
        // nothing is not a failure worth a round trip, and the server's
        // wording would be about a body while this one is about the box in
        // front of the operator.
        status.textContent = "Type who this was first.";
        status.className = "save-status failed";
        return;
      }
      button.disabled = true;
      // An empty box with a label already stored is a CLEAR, and it is
      // allowed: the operator who deletes what is there is saying they were
      // wrong about who this was, and refusing that would leave a wrong name
      // on an interview with no way to take it off.
      saveResult(
        status,
        fetch("/api/participant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // The session travels with the request, exactly as it does for the
          // proposals card. Left behind, this would attach a name to whichever
          // interview the command line named rather than the one on screen —
          // and unlike a misread standing, a name on the wrong call is a wrong
          // fact written to disk.
          body: JSON.stringify({ session: currentSession, participant: text }),
        }).then(function (r) { return r.json(); }),
        function () { load(currentSession); },
        text
          ? "Saved. This interview is recorded as: " + text
          : "Saved. This interview is recorded with no participant."
      );
    });

    host.appendChild(input);
    host.appendChild(button);
    host.appendChild(status);
  }

  function questionList(items, className) {
    var list = el("ul", "questions " + className);
    items.forEach(function (item) { list.appendChild(el("li", null, item.text)); });
    return list;
  }

  function renderTalkShare(meter) {
    var host = document.getElementById("talk-share");
    var note = document.getElementById("talk-share-note");
    if (!meter || !meter.measurable) {
      // An ABSENCE, never "0%" and never a 50/50 bar. There is no transcript
      // to count, which is a fact about the recording; it is not a fact about
      // how the conversation was balanced, and a drawn bar would assert one.
      note.textContent = "No transcript to count, so there is nothing to measure here.";
      return;
    }

    var speakers = meter.speakers || {};
    var names = Object.keys(speakers);
    var mine = speakers[meter.interviewer_key];

    // The headline is the interviewer's own share, because that is the number
    // this was built for and the only one the reader can act on. Stated as a
    // plain percentage with no verdict attached: nothing here says 65% is too
    // much, because nothing in this product has measured what "too much" is,
    // and a threshold invented for a progress bar would be a guess wearing the
    // costume of a finding.
    note.textContent = mine
      ? "You spoke " + Math.round(mine.share * 100) + "% of the "
        + meter.total_words + " words in this interview."
      : "You do not appear in this transcript. " + meter.total_words + " words were counted.";

    names.forEach(function (name) {
      var entry = speakers[name];
      var row = el("div", "share-row");
      row.appendChild(el("span", "share-who", name));

      // A meter element rather than a coloured div: it carries the value to a
      // screen reader without needing a label that could drift from the bar.
      var track = el("div", "share-track");
      var fill = el("div", "share-fill");
      fill.style.width = (entry.share * 100).toFixed(1) + "%";
      // The interviewer's row is the one being reported on. Emphasis only --
      // never a warning colour, because no threshold has been established and
      // colouring the common case like a problem teaches the operator to
      // ignore the one colour this page reserves for real ones.
      if (name === meter.interviewer_key) { fill.className = "share-fill mine"; }
      track.appendChild(fill);
      row.appendChild(track);

      row.appendChild(el("span", "share-value",
        Math.round(entry.share * 100) + "% · " + entry.words + " words"));
      host.appendChild(row);
    });

    if (meter.unattributed_words) {
      // Said out loud rather than folded into the bars. Speech nobody is
      // credited with means the shares below do not sum to 100%, and a reader
      // who spots that without being told will reasonably distrust the rest.
      host.appendChild(el("p", "flag",
        meter.unattributed_words + " word" + (meter.unattributed_words === 1 ? "" : "s")
        + " could not be attributed to a speaker, so the shares above do not add up to 100%."));
    }

    host.appendChild(el("p", "note",
      "Counted in words, not in seconds — the stored per-segment start times are "
      + "derived, not measured, so a timing-based split would count each speaker's "
      + "silence as their own talking."));
  }

  function renderCoverage(coverage) {
    var host = document.getElementById("coverage");
    var note = document.getElementById("coverage-note");
    if (!coverage) { note.textContent = ""; return; }

    if (coverage.error) {
      note.textContent = "Coverage could not be read: " + coverage.error;
      return;
    }

    var session = coverage.session;
    if (!session || !session.reportable) {
      // NOT "0 of 0 covered". A call that ran without a survey did not miss
      // anything; every real recording this product has made so far is this
      // case, so it is the common path and it must not read as a failing score.
      note.textContent = session
        ? "This interview ran without a survey, so there is nothing to have covered."
        : "No interview selected.";
    } else {
      note.textContent = session.asked.length + " of " + session.total +
        " survey questions were asked in this interview.";
      if (session.survey_source === "project") {
        // The question TEXT below came from today's survey.json, not from
        // what was on screen that night. Said out loud because today's
        // wording attached to last month's answer is a silent lie, not a
        // rounding error.
        host.appendChild(el("p", "flag",
          "This interview stored no survey of its own, so the wording below is " +
          "today's — it may have been edited since the call."));
      }
      if (session.asked.length) {
        host.appendChild(el("h3", null, "Asked"));
        host.appendChild(questionList(session.asked, "asked"));
      }
      if (session.unasked.length) {
        host.appendChild(el("h3", null, "Not asked"));
        host.appendChild(questionList(session.unasked, "unasked"));
      }
      if (session.orphaned_question_ids.length) {
        // An asked question whose text is gone is a fact about the record.
        host.appendChild(el("p", "flag",
          "Asked, but no longer in the survey: " +
          session.orphaned_question_ids.join(", ")));
      }
    }

    var project = coverage.project;
    if (!project) { return; }
    host.appendChild(el("h3", null, "Still unasked across the project"));
    if (project.never_asked.length) {
      // The number that decides what to do NEXT time, which a per-session
      // view can never answer.
      host.appendChild(el("p", "note",
        project.never_asked.length +
        (project.never_asked.length === 1 ? " question has" : " questions have") +
        " never been asked in any of the " + project.sessions_read +
        " interview" + (project.sessions_read === 1 ? "" : "s") + " under this project."));
      host.appendChild(questionList(project.never_asked, "unasked"));
    } else {
      host.appendChild(el("p", "note",
        "Every survey question has been asked at least once across " +
        project.sessions_read + " interview" + (project.sessions_read === 1 ? "" : "s") + "."));
    }
  }

  function plural(n, word) {
    return n + " " + word + (n === 1 ? "" : "s");
  }

  function hypothesesTouched(n) {
    // "0 hypotheses" is the number a reader skims past; "no hypothesis" is the
    // one they stop on, and stopping is the correct reaction to a call that
    // reached none of the beliefs it was meant to test.
    if (!n) { return "no hypothesis"; }
    return n === 1 ? "1 hypothesis" : n + " hypotheses";
  }

  // The shape of the call as a whole, in numbers the server already counted.
  // Deliberately no prose: every claim is drawn above with its own quote and
  // its citation, and a generated paragraph here would be a second account of
  // the same evidence that a reader could not check against anything.
  function renderSummary(summary, session) {
    var host = document.getElementById("summary");
    if (!summary) {
      // The two situations that produce no summary are distinguishable from
      // the rest of the payload, so the page says which rather than printing
      // one sentence that is wrong half the time.
      host.appendChild(el("p", "note", session
        ? "The fact corpus could not be read, so there is nothing to count."
        : "No interview is selected, so there is nothing to count."));
      return;
    }
    if (!summary.fact_count) {
      host.appendChild(el("p", "note",
        "No claims from this interview are in the corpus, so there is nothing to count."));
      return;
    }

    host.appendChild(el("div", "summary-headline",
      plural(summary.fact_count, "claim") + " from this interview, bearing on "
      + hypothesesTouched(summary.hypotheses_touched) + "."));

    // EVIDENCE against INFERENCE, because the fact count alone hides it: an
    // interview whose claims are mostly the model's reading is a weaker result
    // than the same number of things the interviewee actually reported.
    var split = [
      summary.evidence_facts + " evidence — reported by them",
      summary.inference_facts + " inference — the model's reading"
    ];
    // A provenance this page has never heard of must not be swallowed by
    // arithmetic that assumes two kinds. Silently, the two numbers would
    // simply stop adding up to the headline and nothing would say why.
    var rest = summary.fact_count - summary.evidence_facts - summary.inference_facts;
    if (rest) { split.push(rest + " of neither kind"); }
    host.appendChild(el("div", "summary-split", split.join(" · ")));
    host.appendChild(provenanceBar(summary, rest));

    if (summary.projects.length) {
      // Where the claims were FILED, which is more than one node when a
      // subtree was rolled up -- and is still one interview.
      host.appendChild(el("div", "summary-projects",
        "Filed under " + summary.projects.join(", ")));
    }
  }

  // --- the three debrief panels (design direction D) ----------------------------------
  //
  // All three live inside the "This interview in numbers" container, alongside
  // the claim counts, because that is what they are: arithmetic over artifacts
  // already on disk, costing no model call on a page load. They are drawn with
  // divs and inline widths for the same testability reason the standing bar is
  // -- see the note above `barSegment` -- and they reuse this page's own
  // --supports / --contrary / --dim rather than introducing a palette.
  //
  // THE RULE ALL THREE OBEY: "not run yet" and "ran and found nothing" are
  // different sentences and are never allowed to render alike. Every one of
  // these panels is drawn from an artifact the operator has to produce
  // deliberately, so the absent case is the COMMON case -- and an absent
  // measurement drawn as a zero is a score the page invented.

  /** One labelled bar. `share === null` draws NO fill node at all, which is
      how this page says "not measured" -- a zero-width fill inside a drawn
      track reads as "measured, and it came to nothing". */
  function panelRow(host, label, share, valueText, fillClass) {
    var row = el("div", "panel-row");
    row.appendChild(el("span", "panel-who", label));
    var track = el("div", "panel-track");
    if (share !== null && share !== undefined) {
      var fill = el("div", fillClass ? "panel-fill " + fillClass : "panel-fill");
      fill.style.width = share.toFixed(1) + "%";
      track.appendChild(fill);
    }
    row.appendChild(track);
    row.appendChild(el("span", "panel-value", valueText));
    host.appendChild(row);
    return row;
  }

  function pct(n) { return Math.round(n * 100) + "%"; }

  // --- evidence against inference -----------------------------------------------------
  //
  // The proportion the claim count alone hides. Drawn NEUTRALLY -- both
  // segments are --dim, separated by weight rather than by hue -- for the
  // reason the talk-share bars are neutral: nothing in this product has
  // established what a healthy ratio is, and a green "evidence" segment would
  // read as a grade. The design note is blunter still, and it is carried onto
  // the page below the bar: the pipeline verifies that a quote is verbatim and
  // never that the claim follows from it, so the evidence count is an upper
  // bound rather than a finding.
  //
  // The only coloured segment is the one for a provenance this page has never
  // heard of, which really is a fault and is already named in words above.
  function provenanceBar(summary, rest) {
    var wrap = el("div", "provenance-bar");
    var track = el("div", "panel-track wide");
    var total = summary.fact_count;
    track.appendChild(barSegment("panel-seg evidence", pctOf(summary.evidence_facts, total),
      summary.evidence_facts + " evidence"));
    track.appendChild(barSegment("panel-seg inference", pctOf(summary.inference_facts, total),
      summary.inference_facts + " inference"));
    if (rest) {
      track.appendChild(barSegment("panel-seg unknown", pctOf(rest, total),
        rest + " of neither kind"));
    }
    wrap.appendChild(track);
    wrap.appendChild(el("p", "note",
      "Read the evidence share as an upper bound, not a finding: extraction "
      + "checks that a quote is verbatim, never that the claim follows from it."));
    return wrap;
  }

  // --- how you interviewed -------------------------------------------------------------

  // The rubric's flag names, as a reader would say them. Keyed on the names the
  // SERVER sends, and an unknown key is prettified rather than dropped -- the
  // 2026-08-12 audit's finding about the extraction banner applies exactly
  // here: a hand-keyed lookup on the producer's literals renders a nameless row
  // the day one is renamed, with every test green.
  // `review.QUESTION_FLAGS` is pinned against the writing module's own keys on
  // the Python side, so this table cannot drift alone.
  var FLAG_LABELS = {
    leading: "Leading",
    hypothetical_or_future_tense: "Hypothetical / future",
    pitch: "Pitch",
    opinion_seeking: "Opinion-seeking",
    confirmation_only: "Confirmation-only",
    generic: "Generic"
  };

  function flagLabel(name) {
    return FLAG_LABELS[name] || name.replace(/_/g, " ");
  }

  function cleanRateRow(host, label, rate, judged, absentText) {
    // A `null` rate is the artifact's own way of saying nothing of this kind
    // could be judged, and it is passed through as an ABSENCE: no fill, no
    // percentage, and a sentence in the place the number would have been.
    // Rendering it as 0% would report an unmeasured group as the worst
    // possible result.
    if (rate === null || rate === undefined) {
      panelRow(host, label, null, absentText);
      return;
    }
    panelRow(host, label, rate * 100,
      pct(rate) + " clean of " + plural(judged, "judged turn"), "clean");
  }

  function questionFlagBars(host, summary) {
    var flags = summary.flags || {};
    var notJudged = summary.flags_not_judged || [];
    var names = Object.keys(flags);
    // Scaled to the LARGEST flag rather than to a total, because one turn can
    // break several rules at once and the counts therefore do not share a
    // denominator. Said out loud below, so nobody reads a full-width bar as
    // "all of them".
    var largest = 0;
    names.forEach(function (name) {
      if (notJudged.indexOf(name) !== -1) { return; }
      if (typeof flags[name] === "number" && flags[name] > largest) { largest = flags[name]; }
    });

    names.forEach(function (name) {
      if (notJudged.indexOf(name) !== -1) {
        // NOT a zero, and never drawn as one. The flag was not asked, so there
        // is nothing here to have passed -- see `QUESTION_FLAGS_NOT_JUDGED`.
        panelRow(host, flagLabel(name), null,
          "not judged here — no zero to read as a pass");
        return;
      }
      var count = flags[name];
      if (typeof count !== "number") {
        panelRow(host, flagLabel(name), null, "unreadable in the critique file");
        return;
      }
      // A count of 0 for a flag that WAS judged is a real zero and gets no bar
      // rather than a hairline one -- but its value reads "0", not "not
      // judged", which is the whole distinction this panel turns on.
      panelRow(host, flagLabel(name), count ? pctOf(count, largest) : null,
        String(count), "flag");
    });

    host.appendChild(el("p", "note",
      "One turn can break several rules, so these counts share no denominator — "
      + "each bar is drawn against the largest count, not against the interview."));
  }

  function renderQuestionCritique(critique, project, session) {
    var host = document.getElementById("summary");
    host.appendChild(el("h3", "panel-heading", "How you interviewed"));

    if (!critique) {
      host.appendChild(el("p", "note",
        "This page received no question critique field at all, so nothing here "
        + "describes the questions you asked."));
      return;
    }

    var said = {
      "no-session": "No interview is selected, so there are no questions to judge.",
      "not-run": "The asked-question critique has not been run for this interview. "
        + "Nothing below scores your questions — this is an absent measurement, "
        + "not a low one.",
      "unreadable": "The critique for this interview could not be read, so nothing "
        + "below reflects it.",
      "read": null
    };
    var headline = said[critique.status];
    if (headline === undefined) {
      // The same guard the extraction banner carries, for the same reason: an
      // unknown status used to append `undefined` as an empty paragraph, which
      // is a silent nothing where a caveat belongs.
      headline = "The page does not recognise the critique status '"
        + critique.status + "'. Treat everything below as unexplained.";
    }
    if (headline !== null) {
      host.appendChild(el("p", "note", headline));
      if (critique.error) { host.appendChild(el("p", "flag", critique.error)); }
      if (critique.status === "not-run") {
        // The remedy, exactly as the extraction banner gives one. Nothing else
        // in the product tells a reader this command exists.
        host.appendChild(el("div", "extraction-remedy",
          "Run: python -m app critique-questions " + (session || "<session>")
          + " --project " + (project || "<project>")));
      }
      return;
    }

    var summary = critique.summary || {};
    if (typeof summary.judged !== "number") {
      host.appendChild(el("p", "note",
        "The critique's counts could not be read, so no rate is shown."));
      return;
    }

    if (!summary.judged) {
      // Turns were collected and NONE could be banded. A rate of 0% here would
      // report an unreachable critic as a failed interview.
      host.appendChild(el("p", "note",
        "Of " + plural(summary.turns || 0, "collected turn")
        + ", none could be judged — the critic could not be reached. No rate is "
        + "shown, because an unjudged interview is not a bad one."));
    } else {
      cleanRateRow(host, "Marked questions", summary.marked_question_clean_rate,
        summary.marked_questions,
        "no turn punctuated as a question was judged");
      cleanRateRow(host, "Unmarked turns", summary.unmarked_clean_rate,
        summary.judged - (summary.marked_questions || 0),
        "no unmarked turn was judged");

      if (summary.clean_rate !== null && summary.clean_rate !== undefined) {
        host.appendChild(el("p", "note",
          "Combined: " + pct(summary.clean_rate) + " clean over "
          + plural(summary.judged, "judged turn")
          + " — the single number these two were split out of."));
      }

      // ⚠️ The caveat that makes the second bar readable at all. Without it a
      // reader takes "unmarked" for "statements" and reads a low number as a
      // verdict on statements, when half the group is genuine questions the
      // transcript simply gave no question mark.
      host.appendChild(el("p", "note",
        "“Unmarked” is not “statements”. It holds leading statements and the "
        + "roughly half of real questions the transcript gave no question mark, "
        + "so nothing is filtered away here — only separated."));
    }

    if (summary.unjudged) {
      // Reachability said out loud, exactly as the unattributed words are on
      // the talk-share meter: turns missing from the denominator mean the rates
      // above do not describe the whole interview.
      host.appendChild(el("p", "flag",
        plural(summary.unjudged, "turn") + " could not be judged at all, so the "
        + "rates above cover only part of this interview."));
    }

    questionFlagBars(host, summary);
  }

  // --- what extraction refused ---------------------------------------------------------

  function renderRejections(rejections) {
    var host = document.getElementById("summary");
    host.appendChild(el("h3", "panel-heading", "What extraction refused"));

    if (!rejections) {
      host.appendChild(el("p", "note",
        "This page received no rejection field at all, so nothing here says what "
        + "extraction threw away."));
      return;
    }
    if (rejections.status === "not-kept") {
      // SILENCE, not zero. Extraction printed its rejections to a terminal and
      // kept nothing until this log existed, which is the whole reason the
      // "~30% unlocatable-quote" figure became unreproducible.
      host.appendChild(el("p", "note",
        "No rejection log exists for this project, so nothing is known about what "
        + "extraction threw away. That is silence, not a count of zero."));
      return;
    }
    if (rejections.status === "unreadable") {
      host.appendChild(el("p", "note",
        "The rejection log could not be read, so nothing below reflects it."));
      if (rejections.error) { host.appendChild(el("p", "flag", rejections.error)); }
      return;
    }
    if (rejections.status !== "kept") {
      host.appendChild(el("p", "note",
        "The page does not recognise the rejection status '" + rejections.status
        + "'. Treat everything below as unexplained."));
      return;
    }

    if (!rejections.total) {
      // A REAL zero, and it is worth saying so plainly. `rejection_store`
      // writes an empty slice on purpose so that this case leaves a mark, which
      // is what lets the page distinguish it from the silence above.
      host.appendChild(el("p", "note",
        "The log exists and holds nothing: extraction has run under this project "
        + "and refused no claim. A real zero, not a missing file."));
    } else {
      var reasons = rejections.reasons || {};
      Object.keys(reasons).forEach(function (reason) {
        // Share of ALL rejections, so the width and the printed number are the
        // same quantity and cannot contradict each other.
        panelRow(host, reason, pctOf(reasons[reason], rejections.total),
          reasons[reason] + " · " + pct(reasons[reason] / rejections.total), "reason");
      });
      host.appendChild(el("p", "note",
        plural(rejections.total, "refused claim") + " across "
        + plural(rejections.sessions_recorded, "interview") + " under this project."));
    }

    var mine = rejections.this_interview;
    if (!mine) { return; }
    if (mine.total) {
      host.appendChild(el("p", "note",
        mine.total + " of them came from this interview: "
        + Object.keys(mine.reasons).map(function (reason) {
          return mine.reasons[reason] + " " + reason;
        }).join(", ") + "."));
    } else {
      // AMBIGUOUS, and said as such. An extraction that refused nothing writes
      // no row naming this session, and neither does an extraction that never
      // ran -- nothing in the log separates the two, and picking one is how a
      // page ends up telling an operator to re-run a command they just ran.
      host.appendChild(el("p", "note",
        "The log holds no rejection from this interview. Either extraction "
        + "refused nothing here, or it was never run on it — the log keeps no "
        + "marker that separates the two."));
    }
  }

  function renderProposals(standings, session) {
    var byId = {};
    standings.forEach(function (s) { byId[s.id] = s; });
    // The session travels with the request. The proposals card is the one
    // place on this page that writes, so it following the picker is not a
    // cosmetic detail: left behind, it would offer a decision about a
    // different interview than the one on screen.
    fetch("/api/proposals" + query(session))
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var list = data.proposals || [];
        document.getElementById("no-proposals").hidden = list.length > 0;
        var host = document.getElementById("proposals");
        list.forEach(function (p) {
          host.appendChild(proposalCard(p, byId[p.target_hypothesis_id], session));
        });
      });
  }

  function query(session) {
    return session ? "?session=" + encodeURIComponent(session) : "";
  }

  function clearRendered() {
    // Every container the render fills, emptied together. Switching interview
    // has to leave NOTHING of the previous one behind: a standing card left
    // over from another call, sitting under the new interview's heading, is
    // read as this call's evidence.
    // `participant` is in this list even though it lives in the header, beside
    // the save banner which deliberately is NOT. The distinction is what each
    // one is ABOUT: the banner is about an action the operator just took and
    // has to outlive the re-read that follows it, while the participant box is
    // about the interview on screen and must be rebuilt the instant that
    // changes. Left behind, it would offer one call's name over another's.
    // `talk-share` is here for the same reason `transcript` is: it describes
    // the interview on screen and nothing else. Left behind across a switch it
    // would report one call's balance under another call's heading, which is
    // the most quietly wrong thing this page could do -- the number would look
    // entirely plausible for the interview it was now sitting above.
    // `summary` is here for exactly that reason too: it is counts ABOUT the
    // interview on screen and nothing else.
    ["proposals", "standings", "transcript", "coverage", "new-hypothesis",
     "participant", "talk-share", "summary"].forEach(function (id) {
      document.getElementById(id).textContent = "";
    });
    document.getElementById("extraction").textContent = "";
    document.getElementById("extraction").hidden = true;
    document.getElementById("page-error").hidden = true;
    litTurns = [];
  }

  function render(data) {
    if (data.error) {
      var err = document.getElementById("page-error");
      err.textContent = data.error;
      err.hidden = false;
      return;
    }
    // Set BEFORE anything renders: a citation decides whether it is
    // followable by comparing against this.
    sessionId = data.session_id;
    document.getElementById("heading").textContent = data.project;
    document.getElementById("subheading").textContent =
      data.session_id ? "Interview " + data.session_id : "No session selected";

    renderParticipant(data);
    renderExtraction(data.extraction, data.project, data.session_id);
    renderCoverage(data.coverage);
    renderTalkShare(data.talk_share);

    var standings = document.getElementById("standings");
    document.getElementById("no-hypotheses").hidden = data.standings.length > 0;
    data.standings.forEach(function (s) { standings.appendChild(standingCard(s)); });
    document.getElementById("new-hypothesis").appendChild(newHypothesisForm());

    // Proposals are rendered from the standings already fetched, so the
    // statement, criterion and evidence a reader judges by have exactly one
    // source rather than two that can disagree.
    renderProposals(data.standings, data.session_id);

    renderSummary(data.summary, data.session_id);
    // Called SEPARATELY rather than from inside `renderSummary`, which returns
    // early on an unreadable corpus and on an interview with no claims. Neither
    // of those has any bearing on whether the interviewer's own questions were
    // judged or on what extraction refused, and nesting them would let one
    // absent artifact silently delete two unrelated panels.
    renderQuestionCritique(data.question_critique, data.project, data.session_id);
    renderRejections(data.rejections);

    var transcript = document.getElementById("transcript");
    document.getElementById("no-transcript").hidden = data.transcript.length > 0;
    data.transcript.forEach(function (turn) {
      var line = el("div", "turn");
      // The anchor a citation lands on. `seq` was already in the payload and
      // was being dropped here, which is what made every citation a dead
      // string. A turn with no seq is still rendered -- anchoring must never
      // become a reason for a line of the transcript to disappear.
      if (turn.seq !== null && turn.seq !== undefined) { line.id = "turn-" + turn.seq; }
      line.appendChild(el("span", "who", turn.speaker));
      line.appendChild(el("span", null, turn.text));
      transcript.appendChild(line);
    });
  }

  // The interview currently on screen. Held so a save can re-read the page it
  // was made from rather than snapping back to whatever the command line chose.
  var currentSession = null;

  function load(session) {
    currentSession = session;
    clearRendered();
    fetch("/api/review" + query(session))
      .then(function (r) { return r.json(); })
      .then(render);
  }

  // The picker is fetched SEPARATELY from the review and never blocks on it.
  // The operator whose corpus will not load is exactly the one who needs to be
  // able to switch to another interview, so a failing review must still leave
  // a working picker.
  fetch("/api/sessions")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var picker = document.getElementById("session-picker");
      var sessions = data.sessions || [];
      allSessions = sessions.map(function (session) { return session.id; });
      sessions.forEach(function (session) {
        // The participant, when there is one, is what turns this control from
        // a column of timestamps into a list a human can choose from. Appended
        // rather than substituted: the id is what every other surface in the
        // product names an interview by, and replacing it would mean the
        // picker and the command line disagreed about what to call the same
        // call. An interview with no label reads exactly as it did before,
        // which is the common case and must not grow a stray separator.
        var parts = [session.id];
        if (session.participant) { parts.push(session.participant); }
        if (!session.rendered) { parts.push("not rendered"); }
        var option = el("option", null, parts.join(" · "));
        option.value = session.id;
        picker.appendChild(option);
      });
      if (sessions.length) {
        document.getElementById("picker-row").hidden = false;
        picker.value = data.selected || sessions[0].id;
        document.getElementById("picker-note").textContent =
          sessions.length === 1 ? "1 interview" : sessions.length + " interviews";
        picker.addEventListener("change", function () {
          // The save confirmation is cleared HERE and not in `clearRendered`,
          // and the distinction is the whole point of it. It has to survive
          // the re-read that follows a successful save -- that is why it lives
          // outside every container the render empties -- but it must not
          // survive a deliberate move to a different interview, where it would
          // sit over another call asserting that call had been saved.
          var banner = document.getElementById("save-banner");
          banner.textContent = "";
          banner.hidden = true;
          load(picker.value);
        });
      } else {
        // Said out loud rather than left as an empty control. A project with
        // no interviews yet is a normal state and reads as breakage if the
        // page simply shows nothing.
        document.getElementById("picker-row").hidden = false;
        picker.hidden = true;
        document.getElementById("picker-note").textContent =
          "No interviews recorded under this project yet.";
      }
      load(data.selected || (sessions.length ? picker.value : null));
    })
    .catch(function () { load(null); });
})();
