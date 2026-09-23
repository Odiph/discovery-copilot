// app/static/author/author.js -- the authoring page's wiring
// (04-03-PLAN.md §E4). No build step, no framework, no third-party origin
// (gate 6) -- the same discipline app/static/app.js follows for the phone
// surface. This page's entire job is accepting typed text, so ordinary
// <input>/<textarea> elements are exactly right here: 03-01-PLAN.md §P5's
// "never textarea, never contenteditable" rule is about the PHONE surface's
// TRANSCRIPT rendering, where those element types change bidi behaviour --
// it does not apply to this authoring surface, on a different app, on a
// different binding (04-03-PLAN.md §E3's own clarification).
(function () {
  "use strict";

  var briefContainer = document.getElementById("brief-items");
  var surveyContainer = document.getElementById("survey-items");
  var hypothesesContainer = document.getElementById("hypotheses-items");
  var briefStatus = document.getElementById("brief-status");
  var surveyStatus = document.getElementById("survey-status");
  var hypothesesStatus = document.getElementById("hypotheses-status");

  var briefItems = [];
  var surveyItems = [];
  var hypothesesItems = [];
  var freshIdCounter = 0;

  function freshId(prefix) {
    freshIdCounter += 1;
    return prefix + "-" + Date.now().toString(36) + "-" + freshIdCounter;
  }

  function labelledField(labelText, inputEl) {
    var label = document.createElement("label");
    label.className = "field";
    label.appendChild(document.createTextNode(labelText));
    label.appendChild(inputEl);
    return label;
  }

  // --- Brief ---------------------------------------------------------------

  function renderBrief() {
    briefContainer.textContent = "";
    briefItems.forEach(function (item, index) {
      var row = document.createElement("fieldset");
      row.className = "item-row";

      var textInput = document.createElement("textarea");
      textInput.setAttribute("dir", "auto");
      textInput.value = item.text || "";
      textInput.addEventListener("input", function () {
        item.text = textInput.value;
      });
      row.appendChild(labelledField("Text", textInput));

      var sourceInput = document.createElement("input");
      sourceInput.type = "text";
      sourceInput.setAttribute("dir", "auto");
      sourceInput.value = item.source || "";
      sourceInput.addEventListener("input", function () {
        item.source = sourceInput.value;
      });
      row.appendChild(labelledField("Source", sourceInput));

      var citationInput = document.createElement("input");
      citationInput.type = "text";
      citationInput.setAttribute("dir", "auto");
      citationInput.value = item.citation || "";
      citationInput.addEventListener("input", function () {
        item.citation = citationInput.value;
      });
      row.appendChild(labelledField("Citation", citationInput));

      var provenanceGroup = document.createElement("div");
      provenanceGroup.className = "provenance-group";
      var groupName = "provenance-" + item.id;
      [
        ["evidence", "Evidence -- I was told this, or read it somewhere"],
        ["inference", "Inference -- I'm guessing"],
      ].forEach(function (pair) {
        var value = pair[0];
        var text = pair[1];
        var optionLabel = document.createElement("label");
        optionLabel.className = "provenance-option";
        var radio = document.createElement("input");
        radio.type = "radio";
        radio.name = groupName;
        radio.value = value;
        // CTX-02: no default selection, ever. `item.provenance` starts
        // `undefined` for a freshly-added item and stays that way until the
        // user makes an explicit choice -- never pre-checked.
        radio.checked = item.provenance === value;
        radio.addEventListener("change", function () {
          item.provenance = value;
        });
        optionLabel.appendChild(radio);
        optionLabel.appendChild(document.createTextNode(" " + text));
        provenanceGroup.appendChild(optionLabel);
      });
      row.appendChild(provenanceGroup);

      var removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.textContent = "Remove";
      removeButton.addEventListener("click", function () {
        briefItems.splice(index, 1);
        renderBrief();
      });
      row.appendChild(removeButton);

      briefContainer.appendChild(row);
    });
  }

  document.getElementById("add-brief-item").addEventListener("click", function () {
    briefItems.push({ id: freshId("brief"), text: "", source: "", citation: "", provenance: undefined });
    renderBrief();
  });

  document.getElementById("save-brief").addEventListener("click", function () {
    briefStatus.textContent = "Saving...";
    var payload = briefItems.map(function (item) {
      var out = { id: item.id, text: item.text, source: item.source || null, citation: item.citation || null };
      // Omitted entirely (never a guessed default) when the user has not
      // yet chosen -- the server rejects a missing provenance by naming
      // this item's id, rather than silently accepting one.
      if (item.provenance) {
        out.provenance = item.provenance;
      }
      return out;
    });
    postJson("/api/brief", payload, briefStatus);
  });

  // --- Survey ----------------------------------------------------------------

  function renderSurvey() {
    surveyContainer.textContent = "";
    surveyItems.forEach(function (item, index) {
      var row = document.createElement("fieldset");
      row.className = "item-row";

      var textInput = document.createElement("input");
      textInput.type = "text";
      textInput.setAttribute("dir", "auto");
      textInput.value = item.text || "";
      textInput.addEventListener("input", function () {
        item.text = textInput.value;
      });
      row.appendChild(labelledField("Question " + (index + 1), textInput));

      var upButton = document.createElement("button");
      upButton.type = "button";
      upButton.textContent = "Move up";
      upButton.disabled = index === 0;
      upButton.addEventListener("click", function () {
        var swap = surveyItems[index - 1];
        surveyItems[index - 1] = surveyItems[index];
        surveyItems[index] = swap;
        renderSurvey();
      });
      row.appendChild(upButton);

      var downButton = document.createElement("button");
      downButton.type = "button";
      downButton.textContent = "Move down";
      downButton.disabled = index === surveyItems.length - 1;
      downButton.addEventListener("click", function () {
        var swap = surveyItems[index + 1];
        surveyItems[index + 1] = surveyItems[index];
        surveyItems[index] = swap;
        renderSurvey();
      });
      row.appendChild(downButton);

      var removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.textContent = "Remove";
      removeButton.addEventListener("click", function () {
        surveyItems.splice(index, 1);
        renderSurvey();
      });
      row.appendChild(removeButton);

      surveyContainer.appendChild(row);
    });
  }

  document.getElementById("add-survey-item").addEventListener("click", function () {
    surveyItems.push({ id: freshId("survey"), text: "" });
    renderSurvey();
  });

  document.getElementById("save-survey").addEventListener("click", function () {
    surveyStatus.textContent = "Saving...";
    // The server always derives `order` from array position (§E4) -- this
    // client sends items in the order the user arranged them and never
    // computes or sends an order value itself.
    var payload = surveyItems.map(function (item) {
      return { id: item.id, text: item.text };
    });
    postJson("/api/survey", payload, surveyStatus);
  });

  // --- Hypotheses (05-03-PLAN.md §K13/§K14) -----------------------------------

  function renderHypotheses() {
    hypothesesContainer.textContent = "";
    hypothesesItems.forEach(function (item, index) {
      var row = document.createElement("fieldset");
      row.className = "item-row";

      var statementInput = document.createElement("textarea");
      statementInput.setAttribute("dir", "auto");
      statementInput.value = item.statement || "";
      statementInput.addEventListener("input", function () {
        item.statement = statementInput.value;
      });
      row.appendChild(labelledField("Hypothesis", statementInput));

      var falsifierInput = document.createElement("textarea");
      falsifierInput.setAttribute("dir", "auto");
      // D-04/§K14: no placeholder text and no pre-filled value, ever -- the
      // same rule the provenance radio group above already follows, for the
      // same reason. Placeholder text that reads like an example is exactly
      // what gets left in place and saved; an empty box is the only honest
      // starting state for a field the user has not yet answered.
      falsifierInput.value = item.falsifier || "";
      falsifierInput.addEventListener("input", function () {
        item.falsifier = falsifierInput.value;
      });
      row.appendChild(labelledField("What evidence would make this hypothesis FALSE?", falsifierInput));

      var criterionInput = document.createElement("textarea");
      criterionInput.setAttribute("dir", "auto");
      // No placeholder and no pre-filled value, exactly as the falsifier
      // above -- and here the rule is load-bearing rather than tidy. An
      // example criterion left in place and saved would be a PRE-REGISTERED
      // DECISION RULE nobody wrote, and `criterion_registered_after` would
      // stamp it with the interviews that already existed, giving a
      // fabricated rule the strongest provenance the product can issue.
      criterionInput.value = item.criterion || "";
      criterionInput.addEventListener("input", function () {
        item.criterion = criterionInput.value;
      });
      row.appendChild(labelledField(
        "We are right if... (optional; write it BEFORE the interviews that will judge it)",
        criterionInput,
      ));

      var stateSelect = document.createElement("select");
      [
        ["untested", "Untested"],
        ["supported", "Supported"],
        ["contradicted", "Contradicted"],
        // A legal RESTING state, not a transitional one: evidence points both
        // ways and the honest record says so. It was added to the model in
        // 763787c and this list was never widened, so the only way to reach
        // it was to hand-edit the file -- and the next save from this page
        // would have written the hypothesis back with whatever the select was
        // showing instead.
        ["contested", "Contested"],
      ].forEach(function (pair) {
        var option = document.createElement("option");
        option.value = pair[0];
        option.textContent = pair[1];
        stateSelect.appendChild(option);
      });
      // Deliberately DOES default to "untested" -- unlike the falsifier
      // above and the provenance radio group in the brief section. A
      // brand-new hypothesis IS untested by definition; that is not a guess
      // about the world, whereas a defaulted falsifier or provenance would
      // assert something the user never confirmed (§K14).
      stateSelect.value = item.state || "untested";
      stateSelect.addEventListener("change", function () {
        item.state = stateSelect.value;
      });
      row.appendChild(labelledField("State", stateSelect));

      var removeButton = document.createElement("button");
      removeButton.type = "button";
      removeButton.textContent = "Remove";
      removeButton.addEventListener("click", function () {
        hypothesesItems.splice(index, 1);
        renderHypotheses();
      });
      row.appendChild(removeButton);

      hypothesesContainer.appendChild(row);
    });
  }

  document.getElementById("add-hypothesis-item").addEventListener("click", function () {
    hypothesesItems.push({
      id: freshId("hyp"), statement: "", falsifier: "", criterion: "", state: "untested",
    });
    renderHypotheses();
  });

  document.getElementById("save-hypotheses").addEventListener("click", function () {
    hypothesesStatus.textContent = "Saving...";
    var payload = hypothesesItems.map(function (item) {
      var out = { id: item.id, statement: item.statement, state: item.state || "untested" };
      // Omitted entirely (never sent as an empty string) when blank -- the
      // server rejects a missing falsifier by naming this item's id, rather
      // than silently accepting one nobody actually wrote (D-04/§K14).
      if (item.falsifier && item.falsifier.trim()) {
        out.falsifier = item.falsifier;
      }
      // Same rule, and here it decides whether a hypothesis is SCORABLE at
      // all. Sending "" would be indistinguishable from omitting it on the
      // model (`criterion` defaults to empty), but it would also mean this
      // page could never round-trip a criterion someone had written: load,
      // touch nothing, save, and the field is gone. Omitting when blank keeps
      // the two situations apart at the only place a human can see them.
      if (item.criterion && item.criterion.trim()) {
        out.criterion = item.criterion;
      }
      return out;
    });
    postJson("/api/hypotheses", payload, hypothesesStatus);
  });

  // --- Shared save/load plumbing ----------------------------------------------

  function postJson(url, payload, statusEl) {
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        return response.json().then(function (body) {
          if (!response.ok) {
            throw new Error(body.error || "save failed");
          }
          statusEl.textContent = "Saved " + body.saved + " item(s).";
        });
      })
      .catch(function (err) {
        statusEl.textContent = "Error: " + err.message;
      });
  }

  // --- what is on disk, before anything is typed ------------------------------
  //
  // The read half of the never-clobber guard. `_read_existing` returns [] for a
  // MISSING file and for a DAMAGED one, so without this the page shows the same
  // empty form either way: the operator retypes one question into what looks
  // like a fresh node, presses Save, and only then learns the file was
  // unreadable. The save itself is already refused with a 409 -- nothing is
  // lost -- but the work of retyping is, and the refusal arrives at the worst
  // possible moment.
  //
  // Per section, not one page-level banner: one artifact can be damaged while
  // the other two are fine, and it is the specific empty form that is
  // misleading.
  var DAMAGE_SAID = {
    unreadable:
      "This file exists on disk and could not be read, so the form below is EMPTY for a " +
      "reason that is not 'nothing authored yet'. Saving is refused until it is repaired " +
      "or moved aside.",
    // A DIFFERENT sentence, because it is a different claim. Saying "this file
    // could not be read" when the truth is "this page could not ask" would be
    // the same class of error the banner exists to prevent: a confident
    // statement about something the surface cannot see.
    unknown:
      "This page could not check whether the files on disk are readable, so an empty form " +
      "below may mean nothing authored yet, or may mean a damaged file.",
  };
  var ARTIFACTS = ["brief", "survey", "hypotheses"];

  function renderDamage(state) {
    ARTIFACTS.forEach(function (name) {
      var box = document.getElementById(name + "-damage");
      if (!box) { return; }
      var said = DAMAGE_SAID[state && state[name]];
      if (!said) {
        box.hidden = true;
        return;
      }
      box.hidden = false;
      var line = document.createElement("div");
      line.className = "damage-headline";
      line.textContent = said;
      box.appendChild(line);
      var where = document.createElement("div");
      where.className = "damage-path";
      where.textContent = "context/" + name + ".json";
      box.appendChild(where);
    });
  }

  fetch("/api/state")
    .then(function (response) {
      return response.json();
    })
    .then(renderDamage)
    // The only one of the four loads with a catch, deliberately. If this
    // request fails the page cannot know whether a file is damaged, and
    // silently rendering no banner would be the original defect again -- so it
    // says that it does not know, in those words.
    .catch(function () {
      renderDamage({ brief: "unknown", survey: "unknown", hypotheses: "unknown" });
    });

  fetch("/api/brief")
    .then(function (response) {
      return response.json();
    })
    .then(function (items) {
      briefItems = items;
      renderBrief();
    });

  fetch("/api/survey")
    .then(function (response) {
      return response.json();
    })
    .then(function (items) {
      surveyItems = items;
      renderSurvey();
    });

  fetch("/api/hypotheses")
    .then(function (response) {
      return response.json();
    })
    .then(function (items) {
      hypothesesItems = items;
      renderHypotheses();
    });
})();
