// Phase B: the local console's client.
//
// Everything that reaches the DOM here goes through textContent, never
// innerHTML. Project labels, survey questions, hypothesis statements and
// session ids are all operator- or customer-derived strings, and this page's
// posture is the same as the overlay's: nothing it renders can be script.
(function () {
  "use strict";

  var tree = document.getElementById("project-tree");
  var treeEmpty = document.getElementById("tree-empty");
  var detail = document.getElementById("detail");
  var detailHeading = document.getElementById("detail-heading");
  var surveyList = document.getElementById("survey-list");
  var hypothesisList = document.getElementById("hypothesis-list");
  var sessionList = document.getElementById("session-list");
  var sourcePicker = document.getElementById("source-picker");
  var factCount = document.getElementById("fact-count");
  var startButton = document.getElementById("start-button");

  var allNodes = [];
  var selected = null;

  function clearChildren(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function showError(element, message) {
    if (!message) {
      element.hidden = true;
      element.textContent = "";
      return;
    }
    element.textContent = message;
    element.hidden = false;
  }

  function renderTree(nodes) {
    allNodes = nodes;
    clearChildren(tree);
    treeEmpty.hidden = nodes.length > 0;
    nodes.forEach(function (node) {
      var item = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = node.label;
      // Depth is the hierarchy made visible. A project is a nested label, so
      // a flat list would misrepresent what the operator is choosing.
      button.style.paddingLeft = 0.5 + (node.depth - 1) * 1.1 + "rem";
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", function () {
        selectNode(node.path);
      });
      item.appendChild(button);
      tree.appendChild(item);
    });
  }

  function markSelected(path) {
    var buttons = tree.querySelectorAll("button");
    for (var i = 0; i < buttons.length; i += 1) {
      buttons[i].setAttribute("aria-pressed", allNodes[i].path === path ? "true" : "false");
    }
  }

  function rootOf(path) {
    return path.split("/")[0];
  }

  function renderSourcePicker(path) {
    clearChildren(sourcePicker);
    // Same root only, and never the filing node itself. The server re-checks
    // both on every Start; this filter exists so the operator is not offered
    // a choice that would be refused.
    allNodes
      .filter(function (node) {
        return node.path !== path && rootOf(node.path) === rootOf(path);
      })
      .forEach(function (node) {
        var item = document.createElement("li");
        var label = document.createElement("label");
        var box = document.createElement("input");
        box.type = "checkbox";
        box.value = node.path;
        box.className = "source-checkbox";
        var text = document.createElement("span");
        text.textContent = " " + node.path;
        label.appendChild(box);
        label.appendChild(text);
        item.appendChild(label);
        sourcePicker.appendChild(item);
      });
  }

  function renderDetail(data) {
    detailHeading.textContent = data.project;
    detail.hidden = false;

    clearChildren(surveyList);
    data.survey.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item.text;
      surveyList.appendChild(li);
    });
    showError(document.getElementById("survey-error"), data.survey_error);

    clearChildren(hypothesisList);
    data.hypotheses.forEach(function (item) {
      var li = document.createElement("li");
      var statement = document.createElement("span");
      statement.textContent = item.statement;
      var state = document.createElement("span");
      state.className = "state";
      state.textContent = " [" + item.state + "]";
      li.appendChild(statement);
      li.appendChild(state);
      hypothesisList.appendChild(li);
    });
    showError(document.getElementById("hypothesis-error"), data.hypotheses_error);

    // A count. The console never displays a fact: one carries a previous
    // customer's verbatim speech.
    factCount.textContent =
      data.fact_count === null ? "unknown" : String(data.fact_count) + " fact(s)";
    showError(document.getElementById("facts-error"), data.facts_error);

    clearChildren(sessionList);
    data.sessions.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item.id + (item.rendered ? "" : "  (not rendered yet)");
      sessionList.appendChild(li);
    });
    // The cap is stated, never silent. A truncated list that looked complete
    // would misreport how much a node has accumulated.
    if (data.session_total > data.sessions.length) {
      var more = document.createElement("li");
      more.className = "state";
      more.textContent =
        "... and " + (data.session_total - data.sessions.length) + " older ("
        + data.session_total + " in total)";
      sessionList.appendChild(more);
    }

    renderSourcePicker(data.project);
    showError(document.getElementById("start-error"), null);
  }

  function selectNode(path) {
    selected = path;
    markSelected(path);
    fetch("/api/project/" + path)
      .then(function (response) {
        return response.json();
      })
      .then(function (data) {
        if (data.error) {
          showError(document.getElementById("start-error"), data.error);
          return;
        }
        renderDetail(data);
      });
  }

  function selectedSources() {
    var boxes = sourcePicker.querySelectorAll(".source-checkbox");
    var chosen = [];
    for (var i = 0; i < boxes.length; i += 1) {
      if (boxes[i].checked) {
        chosen.push(boxes[i].value);
      }
    }
    return chosen;
  }

  startButton.addEventListener("click", function () {
    if (!selected) {
      return;
    }
    startButton.disabled = true;
    fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project: selected, read_from: selectedSources() }),
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (!result.ok) {
          showError(document.getElementById("start-error"), result.data.error || "start refused");
          startButton.disabled = false;
          return;
        }
        // The server stops itself here, so this page is the last thing the
        // operator sees from the console. Say so plainly rather than leaving
        // a dead button -- the next output is in the terminal.
        var done = document.getElementById("start-done");
        done.textContent =
          "Starting " + result.data.project + " -- the interview is now running in the terminal. "
          + "You can close this tab.";
        done.hidden = false;
      })
      .catch(function () {
        // The server stopping mid-response is the EXPECTED shape of success
        // here, not a failure worth alarming the operator about.
        var done = document.getElementById("start-done");
        done.textContent = "Starting -- check the terminal. You can close this tab.";
        done.hidden = false;
      });
  });

  fetch("/api/projects")
    .then(function (response) {
      return response.json();
    })
    .then(renderTree);
})();
