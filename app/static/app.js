// app/static/app.js -- the phone/second-monitor client's WIRING layer only
// (03-02-PLAN.md Task 1 §Q8). Every DECISION -- when to ping, when to declare
// dead, how long to back off, whether a trigger should reconnect, where a
// gap marker measures from, where a block belongs -- lives in the pure
// `live-state.mjs` module this file imports. This file owns only: the real
// DOM, the real WebSocket, and real timers/listeners that call into that
// module and act on what it decides.
//
// Receive-only by construction: this file never references any browser
// media-capture API of any kind, which is what keeps a plain, non-TLS
// WebSocket connection (never "wss") legal on a bare LAN IP
// (app/tests/test_phase_gates.py Gate 2).
//
// Rendering rule, inherited verbatim from app/session/render.py: order BLOCKS
// by session time (audio_end_s); never reorder, reflow, escape or annotate
// the TEXT inside one block. No code path here ever clears the transcript
// container -- it is only ever appended to or inserted into (SURF-06).
import {
  MESSAGE_TYPE_PING,
  MESSAGE_TYPE_PONG,
  MESSAGE_TYPE_HELLO,
  MESSAGE_TYPE_SEGMENT,
  MESSAGE_TYPE_SERVER_STATUS,
  MESSAGE_TYPE_SUGGESTION,
  MESSAGE_TYPE_SURVEY,
  STATUS_HEALTHY,
  STATUS_DEGRADED,
  FAILED_SIDE_COACH,
  REASON_NO_AUDIO,
  REASON_NO_TRANSCRIPT,
  SPEAKER_INTERVIEWEE,
  BackoffCalculator,
  HeartbeatMachine,
  GapTracker,
  formatGapForDisplay,
  findInsertionIndex,
  shouldReconnectOnTrigger,
  SeenSequenceTracker,
} from "./live-state.mjs";

(function () {
  "use strict";

  var transcriptRegion = document.getElementById("transcript-region");
  var serverStatusRegion = document.getElementById("server-status-region");
  var suggestionRegion = document.getElementById("suggestion-region");
  var nextQuestionRegion = document.getElementById("next-question-region");
  var groundingRegion = document.getElementById("grounding-region");

  // The token already lives in this page's own URL -- reuse it for the
  // socket and sibling-asset paths instead of duplicating it anywhere.
  var basePath = window.location.pathname.replace(/\/$/, "");
  var wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  var wsUrl = wsProtocol + "//" + window.location.host + basePath + "/ws";

  // Bounded backwards search window for placing a new block by audio_end_s --
  // cheap, and sufficient because segments almost always arrive close to
  // session-time order; only two concurrently-finishing stream workers can
  // ever produce a brief out-of-order pair.
  var TRAILING_WINDOW = 8;

  // How often app.js checks the heartbeat machine's decision. Independent of
  // the SERVER-configured heartbeat interval/timeout (served in `hello`,
  // below) -- this is just the wiring's own polling cadence, fine-grained
  // enough that the machine's elapsed-time checks stay accurate regardless
  // of the actual configured interval/timeout values.
  var HEARTBEAT_TICK_CADENCE_MS = 1000;

  // Used only until the first `hello` arrives (a first-connect failure
  // before any server config was ever received) -- the server's own values,
  // once known, always win (never hardcoded twice).
  var FALLBACK_BACKOFF_BASE_MS = 500;
  var FALLBACK_BACKOFF_CAP_MS = 10000;

  var socket = null;
  var cfg = null; // set from the server's `hello` message
  var heartbeatMachine = null;
  var heartbeatTickHandle = null;
  var backoff = null;
  var gapTracker = new GapTracker();
  var seenSequences = new SeenSequenceTracker();
  var reconnectTimer = null;

  function isSocketOpen() {
    return socket !== null && socket.readyState === WebSocket.OPEN;
  }

  function formatTimestamp(audioEndS) {
    var totalSeconds = Math.max(0, Math.floor(audioEndS));
    var minutes = Math.floor(totalSeconds / 60);
    var seconds = totalSeconds % 60;
    var mm = (minutes < 10 ? "0" : "") + minutes;
    var ss = (seconds < 10 ? "0" : "") + seconds;
    return mm + ":" + ss;
  }

  function buildUtteranceBlock(message) {
    var block = document.createElement("div");
    block.className = "utterance speaker-" + message.speaker;
    // dir="auto" on this ordinary block element computes to
    // unicode-bidi: isolate (03-RESEARCH.md §3). Base direction resolves
    // from the block's own first strong character; a garbled/unexpected
    // utterance can never destabilize the direction of the page around it.
    block.setAttribute("dir", "auto");
    block.dataset.audioEndS = String(message.audio_end_s);

    // <bdi> is the semantic, no-CSS-required tool for "isolate this
    // fragment and let the browser pick its own direction" -- exactly what
    // a leading timestamp needs so it never gets reordered by the Hebrew
    // text beside it.
    var timestamp = document.createElement("bdi");
    timestamp.className = "ts";
    timestamp.textContent = formatTimestamp(message.audio_end_s);
    block.appendChild(timestamp);

    // Text inserted as a text node, never string concatenation into markup --
    // verbatim reproduction and XSS-safety are the same discipline here.
    block.appendChild(document.createTextNode(message.text));

    return block;
  }

  function insertUtteranceBlock(block, audioEndS) {
    var children = transcriptRegion.children;
    var endTimes = [];
    for (var i = 0; i < children.length; i++) {
      var raw = children[i].dataset.audioEndS;
      // A non-utterance sibling (the gap marker, the server-status banner
      // never lives here) has no audio_end_s -- treat it as "earliest
      // possible" so a real utterance is never placed before it.
      endTimes.push(raw !== undefined ? parseFloat(raw) : -Infinity);
    }
    var index = findInsertionIndex(endTimes, audioEndS, TRAILING_WINDOW);
    if (index >= children.length) {
      transcriptRegion.appendChild(block);
    } else {
      transcriptRegion.insertBefore(block, children[index]);
    }
  }

  function appendUtterance(message) {
    // Belt-and-braces dedupe by `seq` (HIGH finding 1): the primary fix for
    // the stale-listener/double-socket bug is the identity guard in
    // connect() below, but this is a second, independent layer -- even if
    // two live sockets ever both deliver the same finalized segment, it
    // must never render twice.
    if (!seenSequences.shouldRender(message.seq)) {
      return;
    }
    var block = buildUtteranceBlock(message);
    insertUtteranceBlock(block, message.audio_end_s);
    // Always auto-scroll to the newest block -- glanceable, not
    // pin-when-scrolled-up. This surface is for glancing, not reading back
    // (COACH-06); reading back is what the post-call transcript is for.
    transcriptRegion.scrollTop = transcriptRegion.scrollHeight;
  }

  // Phase A. Set from `hello`, which is re-sent on every reconnect, so this
  // survives a sleeping phone without needing its own message type. Written
  // with textContent, never innerHTML: a project path is operator-typed
  // input reaching the DOM, and this page's whole security posture is that
  // nothing it renders can be script.
  function renderGrounding(project, sources) {
    if (!groundingRegion) {
      return;
    }
    if (!project) {
      // An unprojected run looks exactly as it did before this strip
      // existed. Hiding rather than writing "no project" keeps the common
      // case uncluttered and means the strip's PRESENCE is itself the
      // signal that non-default knowledge is in play.
      groundingRegion.hidden = true;
      return;
    }
    var text = project;
    if (sources && sources.length > 0) {
      text = text + " + " + sources.join(", ");
    }
    var body = groundingRegion.querySelector(".grounding-text");
    if (body) {
      body.textContent = text;
    }
    groundingRegion.hidden = false;
  }

  function clearChildren(el) {
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function buildSuggestionTextBlock(text, className) {
    var block = document.createElement("div");
    block.className = className;
    // 04-03-PLAN.md §E3: same bidi rule as the transcript's own utterance
    // blocks -- dir="auto" per block, text inserted as a text node, never
    // string-concatenated into markup. A code-switched suggestion is exactly
    // the content that breaks careless bidi handling (D-06).
    block.setAttribute("dir", "auto");
    block.appendChild(document.createTextNode(text));
    return block;
  }

  function renderSurvey(message) {
    // Phase C: the one region whose content never came from a model, and so
    // the one region that is always populated. Replaces its content on each
    // message for the same reason renderSuggestion does -- what matters is
    // the CURRENT next question, and a growing list is the opposite of
    // glanceable (COACH-06).
    clearChildren(nextQuestionRegion);

    // Interface chrome in English, isolated from the (possibly Hebrew)
    // question beside it -- the same rule the utterance timestamp and the
    // gap marker already follow.
    var label = document.createElement("p");
    label.className = "next-question-label";
    label.setAttribute("dir", "ltr");
    var total = message.total_count;
    var asked = message.asked_count;
    label.textContent =
      typeof total === "number" && total > 0
        ? "Next question · " + asked + " of " + total + " asked"
        : "Next question";
    nextQuestionRegion.appendChild(label);

    var body = document.createElement("p");
    if (typeof message.next_question_text === "string" && message.next_question_text.length > 0) {
      // The question is authored content in whatever script it was written
      // in -- dir="auto" per block, inserted as a text node, exactly like an
      // utterance and a suggestion.
      body.className = "next-question-text";
      body.setAttribute("dir", "auto");
      body.appendChild(document.createTextNode(message.next_question_text));
    } else {
      // A finished survey and an empty one are both real, server-reported
      // states -- shown as such, never left displaying the last question the
      // interviewer already asked.
      body.className = "next-question-resting";
      body.setAttribute("dir", "ltr");
      body.textContent =
        typeof total === "number" && total > 0
          ? "— every survey question asked —"
          : "— no survey for this interview —";
    }
    nextQuestionRegion.appendChild(body);
  }

  function renderSuggestion(message) {
    // 04-03-PLAN.md §E3: the ONE region that REPLACES its content on each
    // new set rather than appending -- the current suggestion is what
    // matters, and a growing list of stale ones is the opposite of
    // glanceable (COACH-06). Build every new block first, then swap the
    // whole set in -- never mutate an already-rendered block's text.
    clearChildren(suggestionRegion);

    // Phase C: the heading lives here, not in index.html, because this
    // region is cleared on every set -- see index.html's own comment. It
    // also names what these are: ADDITIONAL to the next survey question
    // above, never a replacement for it.
    var heading = document.createElement("div");
    heading.className = "suggestion-label";
    heading.setAttribute("dir", "ltr");
    heading.textContent = "Follow-ups";
    suggestionRegion.appendChild(heading);

    var primaryBlock = buildSuggestionTextBlock(message.primary, "suggestion-primary");
    suggestionRegion.appendChild(primaryBlock);

    // D-05/04-CONTEXT.md: a primary standing alone (no alternatives) is a
    // normal, complete state -- no placeholder, no gap, no "no alternatives"
    // text is ever rendered here for that case.
    if (message.alternatives && message.alternatives.length > 0) {
      var label = document.createElement("div");
      label.className = "suggestion-alt-label";
      // Interface chrome in English, not transcript content -- isolated
      // exactly like the gap marker and the utterance timestamp fragments.
      label.setAttribute("dir", "ltr");
      label.textContent = "Also consider:";
      suggestionRegion.appendChild(label);

      for (var i = 0; i < message.alternatives.length; i++) {
        // Visually subordinate, no rank/number chrome (COACH-06: a rank
        // badge invites reading rather than glancing).
        var altBlock = buildSuggestionTextBlock(message.alternatives[i], "suggestion-alternative");
        suggestionRegion.appendChild(altBlock);
      }
    }
  }

  function renderGapMarker(durationMs) {
    // D-02: client-local, its own block, never merged into a neighbour,
    // never persisted anywhere. Appended at the end -- it represents "just
    // reconnected, right now", not a point in session time to sort into.
    var block = document.createElement("div");
    block.className = "gap-marker";
    // Interface chrome in English, not transcript content -- isolated from
    // the Hebrew around it exactly like the utterance timestamp fragments.
    block.setAttribute("dir", "ltr");
    block.textContent =
      "⟳ reconnected — about " + formatGapForDisplay(durationMs) + " missing · see the post-call transcript";
    transcriptRegion.appendChild(block);
    transcriptRegion.scrollTop = transcriptRegion.scrollHeight;
  }

  function renderServerStatus(message) {
    // A laptop-side failure (03-RESEARCH §6) -- visually and verbally
    // distinct from the gap marker, in its own region, so a user under
    // pressure can tell "check your phone" from "your laptop stopped
    // transcribing" apart. Never names transcript text or a vendor. 04-03
    // §E5: this function must never reach into the suggestion region --
    // a coach failure leaves the last shown suggestion in place, not blank.
    if (message.status === STATUS_HEALTHY) {
      serverStatusRegion.hidden = true;
      serverStatusRegion.textContent = "";
      return;
    }
    serverStatusRegion.hidden = false;
    serverStatusRegion.setAttribute("dir", "ltr");
    // 04-03-PLAN.md §E5/OPS-05: a dead reasoning layer is NOT a dead
    // recording -- the transcript is fine, only the suggestions stopped.
    // Deliberately a DIFFERENT, calmer sentence from the stream-failure one
    // below (asserted as a string inequality in app/tests/test_web_suggestion.py,
    // never a fixed-text match, so tuning the wording later cannot silently
    // regress the distinction).
    // Degraded is checked BEFORE the two failure sentences, and it must be:
    // without a branch of its own it falls through to "Transcription stopped
    // on the laptop", which is FALSE here -- transcription is running, it is
    // the audio reaching it that is wrong -- and would be exactly the false
    // alarm the failure vocabulary was built to avoid.
    if (message.status === STATUS_DEGRADED) {
      // Written as direct assignments, never a ternary: gate 16 finds these
      // sentences by scanning for a plain assignment of a string literal to
      // this region, and a ternary hides both of its branches from that scan.
      // Dodging the scan is the drift the gate exists to catch, so the shape
      // here is deliberate. (This comment deliberately does not spell the
      // pattern out -- writing it verbatim makes the scanner match the
      // comment itself, which is how this note came to exist.)
      if (message.reason === REASON_NO_AUDIO && message.failed_side === SPEAKER_INTERVIEWEE) {
        // Actionable, and pointed at the likeliest cause. The commonest
        // version of this is the interviewee muting themselves or the call
        // app losing the device, neither of which the operator can see from
        // a transcript that has simply stopped moving.
        serverStatusRegion.textContent = "No audio from the call — check they are unmuted and still connected.";
      } else if (message.reason === REASON_NO_AUDIO) {
        serverStatusRegion.textContent = "No audio from your microphone — check it is unmuted.";
      } else if (message.reason === REASON_NO_TRANSCRIPT && message.failed_side === SPEAKER_INTERVIEWEE) {
        // Sound IS arriving on that device; none of it is coming back as
        // words while your own side transcribes fine. The overwhelmingly
        // likeliest cause is that the call is playing through a different
        // output than the one being captured -- which is what a Bluetooth
        // headset does the moment a call app opens its microphone.
        serverStatusRegion.textContent = "Nothing is being transcribed from the call — check you are capturing the output the call is playing through.";
      } else if (message.reason === REASON_NO_TRANSCRIPT) {
        serverStatusRegion.textContent = "Nothing is being transcribed from your microphone — check the right one is selected.";
      } else {
        serverStatusRegion.textContent = "Audio is arriving late — the transcript is behind.";
      }
      return;
    }
    if (message.failed_side === FAILED_SIDE_COACH) {
      serverStatusRegion.textContent = "Suggestions unavailable — transcript continues.";
    } else {
      serverStatusRegion.textContent = "Transcription stopped on the laptop — your recording is still running.";
    }
  }

  function stopHeartbeatTicking() {
    if (heartbeatTickHandle !== null) {
      clearInterval(heartbeatTickHandle);
      heartbeatTickHandle = null;
    }
  }

  function startHeartbeatTicking() {
    stopHeartbeatTicking();
    heartbeatMachine = new HeartbeatMachine({
      intervalMs: cfg.heartbeatIntervalMs,
      timeoutMs: cfg.heartbeatTimeoutMs,
      now: Date.now(),
    });
    heartbeatTickHandle = setInterval(function () {
      var action = heartbeatMachine.tick(Date.now(), {
        open: isSocketOpen(),
        visible: document.visibilityState === "visible",
      });
      if (action === null) {
        return;
      }
      if (action.type === "ping") {
        if (socket) {
          socket.send(JSON.stringify({ type: MESSAGE_TYPE_PING }));
        }
      } else if (action.type === "dead") {
        // The half-open case: declared dead purely from the missed pong,
        // regardless of what readyState still claims (SURF-07).
        handleDisconnectOrDeath();
        try {
          if (socket) {
            socket.close();
          }
        } catch (err) {
          // Nothing to do -- we already treat the connection as dead.
        }
      }
    }, HEARTBEAT_TICK_CADENCE_MS);
  }

  function handleDisconnectOrDeath() {
    stopHeartbeatTicking();
    gapTracker.recordDeath();
    scheduleReconnect();
  }

  function scheduleReconnect() {
    if (reconnectTimer !== null) {
      return; // already scheduled -- never stack multiple pending reconnects
    }
    if (backoff === null) {
      backoff = new BackoffCalculator({ baseMs: FALLBACK_BACKOFF_BASE_MS, capMs: FALLBACK_BACKOFF_CAP_MS });
    }
    var delayMs = backoff.nextDelayMs();
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, delayMs);
  }

  function maybeReconnectNow() {
    if (!shouldReconnectOnTrigger(isSocketOpen())) {
      return; // already open -- reconnecting it would be its own failure mode
    }
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket !== null) {
      try {
        socket.close();
      } catch (err) {
        // Already gone -- fine, connect() below replaces it.
      }
    }
    connect();
  }

  function handleOpen() {
    var gapMs = gapTracker.onOpen(Date.now());
    if (gapMs !== null) {
      renderGapMarker(gapMs);
    }
    if (backoff !== null) {
      backoff.reset();
    }
  }

  function handleMessage(event) {
    var message;
    try {
      message = JSON.parse(event.data);
    } catch (err) {
      return;
    }
    // D-02 fix (HIGH finding 2): ANY message proves the connection was
    // alive at this instant -- segments and pongs both flow constantly.
    // This is the honest anchor a later gap is measured from, never the
    // moment death happened to be detected (which can lag the true start
    // of an outage by nearly a full sleep duration).
    gapTracker.recordMessage(Date.now());
    if (message.type === MESSAGE_TYPE_SEGMENT) {
      appendUtterance(message);
    } else if (message.type === MESSAGE_TYPE_HELLO) {
      cfg = {
        heartbeatIntervalMs: message.heartbeat_interval_ms,
        heartbeatTimeoutMs: message.heartbeat_timeout_ms,
        backoffBaseMs: message.reconnect_backoff_base_ms,
        backoffCapMs: message.reconnect_backoff_cap_ms,
      };
      if (backoff === null) {
        backoff = new BackoffCalculator({ baseMs: cfg.backoffBaseMs, capMs: cfg.backoffCapMs });
      }
      renderGrounding(message.project, message.grounding_sources);
      startHeartbeatTicking();
    } else if (message.type === MESSAGE_TYPE_PONG) {
      if (heartbeatMachine) {
        heartbeatMachine.onPong();
      }
    } else if (message.type === MESSAGE_TYPE_SERVER_STATUS) {
      renderServerStatus(message);
    } else if (message.type === MESSAGE_TYPE_SUGGESTION) {
      renderSuggestion(message);
    } else if (message.type === MESSAGE_TYPE_SURVEY) {
      renderSurvey(message);
    }
  }

  function handleCloseOrError() {
    handleDisconnectOrDeath();
  }

  // HIGH finding 1 fix: without this, a stale socket's late `close`/`error`
  // event (e.g. the closing-handshake timeout after the peer vanished
  // mid-sleep) could fire `handleDisconnectOrDeath()` -> `connect()` AFTER a
  // newer socket had already replaced it, producing two live sockets with
  // handlers both attached -- every segment then renders twice for the rest
  // of the interview, and the phantom "dead" call also silently kills the
  // heartbeat on the live socket (SURF-07 disabled).
  //
  // Two guards close both routes described by the finding:
  //   1. `connect()` is a no-op whenever the current socket is already OPEN
  //      or CONNECTING -- covers every call site (the initial connect(), the
  //      reconnect timer callback, and a trigger racing during CONNECTING)
  //      with one check, rather than duplicating it at each call site.
  //   2. Each socket's own handlers close over that SPECIFIC socket
  //      (`thisSocket`) and no-op if it is no longer the CURRENT one
  //      (`thisSocket !== socket`) -- so a stale event from a replaced
  //      socket can never trigger a second connect() or kill the new
  //      socket's heartbeat.
  function connect() {
    if (socket !== null && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return; // already live -- never create a second socket over it
    }
    var thisSocket = new WebSocket(wsUrl);
    socket = thisSocket;
    thisSocket.addEventListener("open", function () {
      if (thisSocket !== socket) {
        return; // stale -- this socket has already been replaced
      }
      handleOpen();
    });
    thisSocket.addEventListener("message", function (event) {
      if (thisSocket !== socket) {
        return;
      }
      handleMessage(event);
    });
    thisSocket.addEventListener("close", function () {
      if (thisSocket !== socket) {
        return;
      }
      handleCloseOrError();
    });
    thisSocket.addEventListener("error", function () {
      if (thisSocket !== socket) {
        return;
      }
      handleCloseOrError();
    });
  }

  // The three reconnect triggers (03-RESEARCH §2/§Q3) -- none sufficient
  // alone, each a no-op when the socket is already open.
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "visible") {
      maybeReconnectNow();
    }
    // Going hidden pauses the heartbeat implicitly (HeartbeatMachine.tick
    // checks `visible` itself every tick) -- no separate stop needed here.
  });
  window.addEventListener("pageshow", function (event) {
    if (event.persisted) {
      maybeReconnectNow(); // restored from bfcache -- reconnect unconditionally
    }
  });
  window.addEventListener("online", function () {
    maybeReconnectNow();
  });
  window.addEventListener("pagehide", function () {
    // Let bfcache work cleanly (03-RESEARCH §2) -- an explicit close here is
    // the documented pattern, not a leak: `handleCloseOrError` schedules a
    // reconnect, but the page is about to be frozen or unloaded either way.
    if (socket) {
      socket.close();
    }
  });

  connect();
})();
