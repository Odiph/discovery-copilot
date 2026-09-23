// app/static/live-state.mjs -- every PURE decision the phone/second-monitor
// client makes (03-02-PLAN.md Task 1 §Q2-Q5, §Q8). Imports nothing, touches
// no DOM, creates no timer, holds no socket reference. It takes an event and
// a time value in, and returns an action out -- `app/static/app.js` is the
// only file allowed to own a real WebSocket, a real timer, or `document`.
//
// This split exists so the two hardest-to-verify-by-eye behaviours in this
// phase -- declaring a half-open socket dead on a missed pong even though
// `readyState` still claims OPEN (SURF-07), and measuring exactly one gap per
// outage from the LAST MESSAGE ACTUALLY SEEN before the outage began, never
// from whenever detection happened to occur (D-02) -- are asserted by node's
// built-in test runner against an injected clock
// (`app/tests/js/live_state.test.mjs`), not eyeballed on a phone.
//
// Anchoring the gap on detection time (rather than last-message time) is
// the exact defect an independent review caught in the canonical 90-second-
// sleep scenario: the heartbeat is paused while the document is hidden, so
// on wake the half-open socket still reports OPEN until the heartbeat
// resumes and eventually times out -- detection can lag the true start of
// the outage by nearly the full sleep duration, which rendered "about 90s
// missing" as "about 3s missing". A dishonest marker is worse than none.
//
// Every constructor below takes its timing knobs as explicit config -- never
// a hardcoded interval -- because the server serves the real values inside
// the `hello` message (03-02-PLAN.md Task 1 action) so both sides read one
// source of truth (`app/config.py`).

"use strict";

// The one client-originated message type (D-01: this is the ONLY lever this
// protocol ever grows on the client->server direction -- see
// `ACCEPTED_CLIENT_MESSAGE_TYPES` in app/web/server.py, asserted equal to
// this literal by a real test rather than typed twice and hoped about).
export const MESSAGE_TYPE_PING = "ping";
export const MESSAGE_TYPE_PONG = "pong";

// --- the rest of the wire vocabulary, imported by app.js, never re-typed there ----------
//
// WHY THIS BLOCK EXISTS. On 2026-08-12 an audit renamed `REASON_NO_AUDIO`
// server-side from "no-audio" to "silent" and the entire Python suite stayed
// green at 1092. The overlay survived because `app/overlay/model.py` IMPORTS
// the constant. `app.js` compared a bare string literal, so it fell into the
// wrong branch forever -- the phone would have told the operator
// "transcription stopped on the laptop" for a call whose audio had simply gone
// quiet, which is the exact false alarm the degraded tier was built to avoid.
//
// This is `message["candidates"]` against a wire pinned to `primary`, one
// surface later. The lesson from that one is written in
// `app/overlay/model.py`'s own comment: "Imported, never re-spelled. A local
// copy of the literal `degraded` would go on passing after the server renamed
// its tier." The overlay took that lesson; the phone did not.
//
// Declared HERE rather than in app.js because app.js is loaded as a module in
// a browser and cannot be imported by the node test runner (it opens a real
// WebSocket on load). This file already crosses that boundary --
// `test_wire_vocabulary.py` reads every one of these back through node and
// compares it to the Python constant it mirrors, by equality over the whole
// set, so a value added on either side has to be added on both.

export const MESSAGE_TYPE_HELLO = "hello";
export const MESSAGE_TYPE_SEGMENT = "segment";
export const MESSAGE_TYPE_SERVER_STATUS = "server_status";
export const MESSAGE_TYPE_SUGGESTION = "suggestion";
export const MESSAGE_TYPE_SURVEY = "survey";

export const STATUS_HEALTHY = "healthy";
export const STATUS_DEGRADED = "degraded";
export const STATUS_FAILED = "failed";

export const FAILED_SIDE_COACH = "coach";
export const REASON_NO_AUDIO = "no-audio";
export const REASON_LAGGING = "lagging";
export const REASON_NO_TRANSCRIPT = "no-transcript";

export const SPEAKER_INTERVIEWEE = "interviewee";
export const SPEAKER_INTERVIEWER = "interviewer";

export const DEFAULT_JITTER_FRACTION = 0.2;

// --- Backoff: min(cap, base * 2^attempt) with +-jitterFraction jitter, never negative --------

export class BackoffCalculator {
  constructor({ baseMs, capMs, jitterFraction = DEFAULT_JITTER_FRACTION, randomFn = Math.random }) {
    this._baseMs = baseMs;
    this._capMs = capMs;
    this._jitterFraction = jitterFraction;
    this._randomFn = randomFn;
    this._attempt = 0;
  }

  /** The undisturbed (pre-jitter) delay for the CURRENT attempt count, without
   * consuming an attempt -- exposed so a test can assert doubling/capping
   * independently of the random jitter applied by `nextDelayMs()`. */
  undisturbedDelayMs() {
    return Math.min(this._capMs, this._baseMs * Math.pow(2, this._attempt));
  }

  /** Returns the next delay in ms and advances the internal attempt counter. */
  nextDelayMs() {
    const undisturbed = this.undisturbedDelayMs();
    this._attempt += 1;
    const jitter = undisturbed * this._jitterFraction * (this._randomFn() * 2 - 1);
    return Math.max(0, undisturbed + jitter);
  }

  /** A successful open resets backoff to its base (03-RESEARCH §2). */
  reset() {
    this._attempt = 0;
  }
}

// --- Reconnect triggers: reconnecting an already-open socket is its own failure mode --------

/**
 * Shared decision behind all three reconnect triggers (visibility returning,
 * a bfcache restore via `pageshow`, and `online`) -- §Q3: each produces a
 * reconnect action only when the socket is not already open; reconnecting an
 * open socket is itself a failure mode this function exists to prevent.
 */
export function shouldReconnectOnTrigger(isSocketOpen) {
  return !isSocketOpen;
}

// --- Heartbeat / liveness: the half-open-socket detector (§Q2) -------------------------------

/**
 * Decides whether to send a heartbeat ping this tick, or declare the
 * connection dead on a missed pong -- the half-open case, which does NOT
 * depend on any reported ready state. No timer of its own: `tick(now, ...)`
 * is driven by app.js's real `setInterval`, and `now` is whatever clock value
 * the caller supplies (a real one in production, an injected one in tests).
 */
export class HeartbeatMachine {
  constructor({ intervalMs, timeoutMs, now }) {
    this._intervalMs = intervalMs;
    this._timeoutMs = timeoutMs;
    this._lastPingAt = now;
    this._pingSentAt = null;
    this._awaitingPong = false;
  }

  /** Clears the outstanding-pong wait -- called when a `{"type":"pong"}`
   * message arrives. */
  onPong() {
    this._awaitingPong = false;
    this._pingSentAt = null;
  }

  /**
   * `open`/`visible` are booleans the caller already knows (socket readyState
   * and `document.visibilityState === "visible"`). Returns one of:
   * `{type: "ping"}`, `{type: "dead"}`, or `null` (nothing to do this tick).
   *
   * The heartbeat is paused entirely while the document is hidden -- a
   * throttled background timer firing a timeout is the OS starving the tab,
   * not a dead socket, and the visibility trigger is what covers the gap
   * once the tab is foregrounded again (§Q2).
   */
  tick(now, { open, visible }) {
    if (!open || !visible) {
      return null;
    }
    if (this._awaitingPong) {
      if (now - this._pingSentAt >= this._timeoutMs) {
        // The half-open case: declared dead purely from the missed pong.
        // The caller's reported `open` (readyState) is deliberately never
        // consulted here -- that is the entire point of this branch.
        this._awaitingPong = false;
        this._pingSentAt = null;
        this._lastPingAt = now;
        return { type: "dead" };
      }
      return null;
    }
    if (this._lastPingAt === null || now - this._lastPingAt >= this._intervalMs) {
      this._lastPingAt = now;
      this._pingSentAt = now;
      this._awaitingPong = true;
      return { type: "ping" };
    }
    return null;
  }
}

// --- Gap tracking: client-local, one marker per outage, measured from the LAST MESSAGE actually
// seen on the wire before the outage began -- never from whenever death happened to be detected
// (D-02, and the fix for an independent review's HIGH finding 2) -----------------------------

/**
 * Tracks the wall-clock time of the last message actually received on the
 * socket (segments and pongs both flow constantly while a connection is
 * genuinely alive), and computes the gap on the next successful reopen as
 * `now - lastMessageAt`. That upper-bounds the missed window honestly.
 *
 * Anchoring on DETECTION time instead (the earlier, defective design) is
 * dishonest by construction: the heartbeat is paused while the document is
 * hidden, so nothing runs during a phone sleep, and on wake the half-open
 * socket still reports OPEN until the heartbeat resumes and eventually
 * times out -- detection can lag the true start of the outage by nearly the
 * full sleep duration. The canonical ~90s-sleep scenario rendered as "about
 * 3s missing" under the old anchor; anchoring on the last real message
 * instead renders it honestly as "about 90s missing".
 *
 * `Date.now()` (wall-clock) is the correct clock family for both
 * `recordMessage`/`onOpen` inputs -- `performance.now()` can freeze during
 * device sleep and would undercount further. Entirely in-memory: no message
 * is ever sent for this, no field is ever added to the protocol (D-02) --
 * see `app/web/status.py`'s docstring for why the SEPARATE `server_status`
 * mechanism is what survives a reconnect instead.
 */
export class GapTracker {
  constructor() {
    this._deathDetected = false;
    this._lastMessageAt = null;
  }

  /**
   * Call whenever ANY message arrives on the socket -- this is the honest
   * anchor: the last instant the connection is known, for certain, to have
   * still been alive.
   */
  recordMessage(now) {
    this._lastMessageAt = now;
  }

  /**
   * Records that the connection was declared dead within the CURRENT
   * outage. Idempotent by design: a second, third, ... failed reconnect
   * attempt inside the SAME outage must not change anything -- the eventual
   * gap is measured from `_lastMessageAt`, which recording death again does
   * not touch.
   */
  recordDeath() {
    this._deathDetected = true;
  }

  /**
   * Call on a successful reopen (`onopen`). Returns the elapsed milliseconds
   * since the last message actually seen on the wire, or `null` if no death
   * was ever recorded within this outage (a fresh page load is not a gap,
   * D-02) or if no message was ever received to anchor from. Clears the
   * death flag immediately so a LATER outage measures itself off its own
   * last message, not the sum of every outage so far.
   */
  onOpen(now) {
    const hadDeath = this._deathDetected;
    this._deathDetected = false;
    if (!hadDeath || this._lastMessageAt === null) {
      return null;
    }
    return now - this._lastMessageAt;
  }
}

// --- Render dedupe: belt-and-braces guard against the same segment rendering twice -----------

/**
 * Tracks which `seq` values have already been rendered. Independent,
 * second-layer defence alongside app.js's connect()/handler identity guard
 * (the primary fix for the stale-listener double-socket bug, §HIGH finding
 * 1): even if two live sockets somehow both deliver the same finalized
 * segment, the transcript must never show it twice. `seq` is stable and
 * gap-free per `app/session/store.py`, so simple Set membership is exact --
 * no windowing or eviction needed for a single interview's segment count.
 */
export class SeenSequenceTracker {
  constructor() {
    this._seen = new Set();
  }

  /**
   * Returns `true` (and records `seq` as seen) the first time this `seq` is
   * passed in; returns `false` on every subsequent call with the same
   * `seq`, without side effects.
   */
  shouldRender(seq) {
    if (this._seen.has(seq)) {
      return false;
    }
    this._seen.add(seq);
    return true;
  }
}

/**
 * Rounds a gap duration for a glance, not a read (COACH-06): whole seconds
 * below a minute, whole minutes at or above -- never a raw millisecond count.
 */
export function formatGapForDisplay(durationMs) {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const minutes = Math.round(totalSeconds / 60);
  return `${minutes}m`;
}

// --- Block placement: order by end-of-utterance time over a bounded trailing window ---------

/**
 * Given the end-of-utterance times of the already-rendered blocks (in their
 * current DOM/array order) and a newly arriving one, returns the index at
 * which the new block belongs: `existingEndTimes.length` means "append at
 * the end" (the ordinary in-order case); any smaller index means "insert
 * before the block currently at that index". Mirrors `render.py`'s rule
 * exactly: order BLOCKS by `audio_end_s`, never reorder an already-placed
 * one -- only a bounded trailing window is ever searched, matching the
 * original app.js algorithm this module extracts verbatim (03-01-PLAN.md
 * §P5, the two-speaker reversed-arrival case this project has already fixed
 * once via commit 74cbd6f).
 */
export function findInsertionIndex(existingEndTimes, newEndTime, windowSize) {
  const scanLimit = Math.max(0, existingEndTimes.length - windowSize);
  for (let i = existingEndTimes.length - 1; i >= scanLimit; i--) {
    if (existingEndTimes[i] <= newEndTime) {
      return i + 1;
    }
  }
  return scanLimit;
}
