// app/tests/js/live_state.test.mjs -- node's built-in test runner (`node --test`),
// no npm, no package.json, no dependency of any kind (03-02-PLAN.md §Q8). Every
// test drives `live-state.mjs` with an injected clock (a plain number this file
// controls) -- never a real timer, never a real socket, never `document`.

import test from "node:test";
import assert from "node:assert/strict";

import {
  MESSAGE_TYPE_PING,
  MESSAGE_TYPE_PONG,
  BackoffCalculator,
  shouldReconnectOnTrigger,
  HeartbeatMachine,
  GapTracker,
  formatGapForDisplay,
  findInsertionIndex,
  SeenSequenceTracker,
} from "../../static/live-state.mjs";

// --- message type literals -------------------------------------------------------------------

test("the message type literals are the exact strings the server pins", () => {
  assert.equal(MESSAGE_TYPE_PING, "ping");
  assert.equal(MESSAGE_TYPE_PONG, "pong");
});

// --- HeartbeatMachine: the half-open-socket detector (SURF-07) -------------------------------

test("advancing the clock by one heartbeat interval yields exactly one send-ping action", () => {
  const machine = new HeartbeatMachine({ intervalMs: 15000, timeoutMs: 5000, now: 0 });
  const action = machine.tick(15000, { open: true, visible: true });
  assert.deepEqual(action, { type: "ping" });
});

test("a pong arriving before the timeout yields another ping on the next interval, no death", () => {
  const machine = new HeartbeatMachine({ intervalMs: 15000, timeoutMs: 5000, now: 0 });
  assert.deepEqual(machine.tick(15000, { open: true, visible: true }), { type: "ping" });
  machine.onPong();
  const action = machine.tick(30000, { open: true, visible: true });
  assert.deepEqual(action, { type: "ping" });
});

test("a missed pong declares the connection dead even though open/visible still say so -- the half-open case", () => {
  const machine = new HeartbeatMachine({ intervalMs: 15000, timeoutMs: 5000, now: 0 });
  assert.deepEqual(machine.tick(15000, { open: true, visible: true }), { type: "ping" });
  // No onPong() call. The reported ready state ("open: true") is passed in
  // unchanged -- this is the single most important assertion in this plan:
  // death must be declared from the missed pong alone, never from readyState.
  const action = machine.tick(15000 + 5000, { open: true, visible: true });
  assert.deepEqual(action, { type: "dead" });
});

test("while the document is hidden, advancing the clock yields no ping and no death", () => {
  const machine = new HeartbeatMachine({ intervalMs: 15000, timeoutMs: 5000, now: 0 });
  const action = machine.tick(1000000, { open: true, visible: false });
  assert.equal(action, null);
});

test("heartbeat stays quiet while the socket is not open", () => {
  const machine = new HeartbeatMachine({ intervalMs: 15000, timeoutMs: 5000, now: 0 });
  const action = machine.tick(15000, { open: false, visible: true });
  assert.equal(action, null);
});

// --- BackoffCalculator: doubling, capped, jittered, reset on open ----------------------------

test("backoff doubles per attempt and is capped", () => {
  const backoff = new BackoffCalculator({ baseMs: 500, capMs: 10000, randomFn: () => 0.5 });
  assert.equal(backoff.undisturbedDelayMs(), 500);
  backoff.nextDelayMs();
  assert.equal(backoff.undisturbedDelayMs(), 1000);
  backoff.nextDelayMs();
  assert.equal(backoff.undisturbedDelayMs(), 2000);
  for (let i = 0; i < 10; i++) backoff.nextDelayMs();
  assert.equal(backoff.undisturbedDelayMs(), 10000, "must never exceed the cap");
});

test("backoff delay is never negative and lands within +-20% of the undisturbed value", () => {
  const backoff = new BackoffCalculator({ baseMs: 500, capMs: 10000 });
  for (let i = 0; i < 500; i++) {
    const undisturbed = backoff.undisturbedDelayMs();
    const delay = backoff.nextDelayMs();
    assert.ok(delay >= 0, `delay must never be negative, got ${delay}`);
    assert.ok(
      delay >= undisturbed * 0.8 - 1e-9 && delay <= undisturbed * 1.2 + 1e-9,
      `delay ${delay} out of +-20% bounds of undisturbed ${undisturbed}`
    );
  }
});

test("a successful open resets backoff to its base", () => {
  const backoff = new BackoffCalculator({ baseMs: 500, capMs: 10000, randomFn: () => 0.5 });
  backoff.nextDelayMs();
  backoff.nextDelayMs();
  assert.notEqual(backoff.undisturbedDelayMs(), 500);
  backoff.reset();
  assert.equal(backoff.undisturbedDelayMs(), 500);
});

// --- Reconnect triggers: never reconnect an already-open socket ------------------------------

test("each reconnect trigger fires only when the socket is not already open", () => {
  assert.equal(shouldReconnectOnTrigger(false), true, "visibility-returning trigger, socket closed");
  assert.equal(shouldReconnectOnTrigger(true), false, "visibility-returning trigger, socket already open");
  // The same function backs all three triggers (visibility, pageshow-persisted,
  // online) -- they share identical open/closed semantics by construction.
  assert.equal(shouldReconnectOnTrigger(false), true, "bfcache-restore trigger, socket closed");
  assert.equal(shouldReconnectOnTrigger(true), false, "bfcache-restore trigger, socket already open");
  assert.equal(shouldReconnectOnTrigger(false), true, "network-online trigger, socket closed");
  assert.equal(shouldReconnectOnTrigger(true), false, "network-online trigger, socket already open");
});

// --- GapTracker: one marker per outage, measured from the LAST message actually seen on the
// wire before the outage began -- never from when detection happened (D-02 fix; WR/HIGH finding
// 2). Detection lags the true start of an outage by up to a full sleep duration: the heartbeat is
// paused while the document is hidden, so nothing runs during sleep, and on wake the half-open
// socket still reports OPEN until the heartbeat resumes and eventually times out. Anchoring on
// detection time (the old behaviour) therefore rendered the canonical 90-second sleep as "about
// 3s missing" -- exactly the dishonest-marker defect this section exists to prevent.

test("the canonical 90-second sleep: the gap is anchored on the last message before sleep, not on wake-time detection", () => {
  const tracker = new GapTracker();
  // Last segment/pong actually received before the phone slept.
  tracker.recordMessage(0);
  // Nothing arrives during ~90s of sleep (heartbeat is paused while hidden).
  // On wake, detection does not happen until t=91000 (one ping+timeout cycle
  // after the heartbeat resumes) -- but the gap must NOT be measured from
  // that late detection time.
  tracker.recordDeath();
  const gapMs = tracker.onOpen(92000);
  assert.equal(gapMs, 92000, "must be measured from the last real message (t=0), not from detection (t=91000)");
});

test("reopening with no recorded death yields no marker -- a fresh page load is not a gap", () => {
  const tracker = new GapTracker();
  const gapMs = tracker.onOpen(5000);
  assert.equal(gapMs, null);
});

test("reopening with a death recorded but no message ever received yields no marker -- nothing to anchor from", () => {
  const tracker = new GapTracker();
  tracker.recordDeath();
  const gapMs = tracker.onOpen(5000);
  assert.equal(gapMs, null);
});

test("two consecutive failed reconnect attempts inside one outage still anchor on the last message before the outage, not on either attempt", () => {
  const tracker = new GapTracker();
  tracker.recordMessage(1000); // last message actually seen before the outage
  tracker.recordDeath(); // first detection -- must not move the anchor
  tracker.recordDeath(); // a failed retry re-detects death -- must not move the anchor either
  tracker.recordDeath(); // a second failed retry
  const gapMs = tracker.onOpen(10000);
  assert.equal(gapMs, 10000 - 1000, "must be measured from the last real message, never a detection timestamp");
});

test("a gap is cleared after being read, so a later outage measures itself off its OWN last message, not the sum", () => {
  const tracker = new GapTracker();
  tracker.recordMessage(1000);
  tracker.recordDeath();
  assert.equal(tracker.onOpen(2000), 1000);
  // Messages flow normally again after the reconnect, moving the anchor
  // forward before the next outage begins.
  tracker.recordMessage(5000);
  tracker.recordDeath();
  assert.equal(tracker.onOpen(5500), 500, "must anchor on the SECOND outage's own last message, not sum with the first");
});

test("gap duration is rounded for a glance: seconds under a minute, whole minutes above", () => {
  // Below the one-minute threshold: whole seconds, never a raw ms count.
  assert.equal(formatGapForDisplay(1500), "2s");
  assert.equal(formatGapForDisplay(45000), "45s");
  assert.equal(formatGapForDisplay(59000), "59s");
  // At/above one minute: whole minutes, never a seconds+minutes combo and
  // never a raw ms count (03-02-PLAN.md Task 2 §Q5: "rounded for glanceability").
  assert.equal(formatGapForDisplay(60000), "1m");
  assert.equal(formatGapForDisplay(90000), "2m");
  assert.equal(formatGapForDisplay(150000), "3m");
});

// --- Block placement: end-of-utterance ordering over a bounded trailing window ---------------

test("an ordinary in-order arrival appends at the end", () => {
  const index = findInsertionIndex([1.0, 2.0, 3.0], 4.0, 8);
  assert.equal(index, 3);
});

test("an empty transcript places the first block at index 0", () => {
  const index = findInsertionIndex([], 1.0, 8);
  assert.equal(index, 0);
});

test("the two-speaker reversed-arrival case places the block in end-time order", () => {
  // Two concurrently-finishing stream workers: the second speaker's segment
  // (ending at 3.0) arrives AFTER the first speaker's later-ending segment
  // (5.0) is already rendered -- this is the ordering bug fixed once already
  // (commit 74cbd6f) via audio_end_s, never audio_start_s.
  const existing = [5.0]; // one block already rendered, ending at t=5.0
  const index = findInsertionIndex(existing, 3.0, 8);
  assert.equal(index, 0, "the earlier-ending block must be inserted BEFORE the later-ending one");
});

test("block placement only searches the bounded trailing window", () => {
  // 10 existing blocks, strictly increasing; window of 3 means only the last
  // 3 are ever inspected. A new arrival earlier than everything in the
  // window still lands at the front of the window, never earlier.
  const existing = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const index = findInsertionIndex(existing, 0.5, 3);
  assert.equal(index, 7, "must land at the front of the trailing window, not before it");
});

// --- SeenSequenceTracker: belt-and-braces render dedupe by `seq` (HIGH finding 1) ------------
//
// The primary fix for the stale-listener/double-socket bug lives in app.js's connect()/handler
// identity guard (not unit-testable here -- it is real-DOM/real-WebSocket wiring). This tracker
// is the second, independent layer: even if two live sockets somehow both deliver the same
// finalized segment, the render path must never show it twice.

test("the same seq arriving twice is rendered only once", () => {
  const tracker = new SeenSequenceTracker();
  assert.equal(tracker.shouldRender(7), true, "first sighting of seq 7 must render");
  assert.equal(tracker.shouldRender(7), false, "a second sighting of the SAME seq 7 must not render again");
});

test("different seqs each render exactly once, independent of arrival order", () => {
  const tracker = new SeenSequenceTracker();
  assert.equal(tracker.shouldRender(3), true);
  assert.equal(tracker.shouldRender(1), true);
  assert.equal(tracker.shouldRender(2), true);
  assert.equal(tracker.shouldRender(1), false, "seq 1 was already rendered");
  assert.equal(tracker.shouldRender(3), false, "seq 3 was already rendered");
});
