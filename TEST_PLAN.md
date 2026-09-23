# Test plan

Four layers, in increasing order of what they cost and decreasing order of how
often they run. Each one exists because the layer below it cannot see the
failure the layer above catches.

| Layer | What it proves | Cost | Run it |
|---|---|---|---|
| 1. Automated suites | The code does what its authors intended | seconds | every change |
| 2. Coach eval | The coach earns its money on a real conversation | ~26 min + API spend | before shipping a coach change |
| 3. Manual hardware | Audio devices, the mic, the STOP button | ~15 min of a person | before a real interview |
| 4. Real interview | All of it, with a human who does not know the script | an hour and a participant | rarely, deliberately |

**The standing lesson this file is organised around:** a test that exercises
one side of a seam proves INTENT, not BEHAVIOUR. Layer 1 is almost entirely
intent. That is not a criticism of it — it is why layers 2 to 4 exist.

---

## Layer 1 — automated suites

```powershell
.\.venv\Scripts\python.exe -m pytest app/tests -m "not live and not hardware and not live_reasoning" -q
```

1321 tests, about 2m30s. Includes the phase gates, which are structural
assertions rather than behaviour tests (gate 19 pins which modules may write
`context/hypotheses.json`; gate 20 pins which may read a verbatim quote).

The node DOM suite for the hub client runs inside that pytest run, shelled out
from `test_web_review_render.py`:

```powershell
node --test app\tests\js\review_render.test.mjs
```

41 tests. It drives the real `review.js` against a minimal fake DOM. **Editing
`review.js` without updating it breaks the pytest run, not just node.**

The benchmark suite is separate and costs vendor calls:

```powershell
.\.venv\Scripts\python.exe -m pytest benchmark -q
```

**Excluded by marker, on purpose:** `live` (real vendor calls), `hardware`
(real audio devices), `live_reasoning` (real model calls). Everything those
markers cover is layer 2 or 3.

---

## Layer 2 — the coach eval

**What it is.** A synthetic 25-minute Claudictive discovery interview,
synthesised to audio and fed into the real pipeline as two capture devices.
The audio-to-coach spine is shipping code: `CaptureStream`, the Soniox session,
the coach pipeline, the coverage detector, the session writer.

**It is a REPORT FOR A HUMAN, not a gate.** `analyse_eval.py` has no thresholds
and always exits 0. Every number is one sample of a nondeterministic system,
and a threshold over one sample either never fires or fires on noise.

**Why it exists.** The coach is the only part of this product whose output is a
judgement rather than a value, and no unit test can tell a good suggestion from
a plausible one. Before this existed the coach had run against exactly one test
session in its whole life.

### Running it

```powershell
# 1. Build the audio (~2 min, SAPI, no network). Regenerates ~145 MB of WAVs,
#    all gitignored.
.\.venv\Scripts\python.exe benchmark\eval\build_audio.py
.\.venv\Scripts\python.exe benchmark\eval\split_channels.py

# 2. Run it. ~26 minutes of REAL TIME by design -- see below.
$env:APP_KEEPALIVE_ENABLED="0"
.\.venv\Scripts\python.exe benchmark\eval\run_eval_interview.py 1560

# 3. Score it against what was played in.
.\.venv\Scripts\python.exe benchmark\eval\analyse_eval.py
```

**It costs real money:** ~25 minutes of Soniox streaming on two channels, plus
one or more Anthropic calls per interviewer segment for the coach, the critic
and the coverage detector. Do not run it in a loop.

**It runs in real time deliberately.** `FileStream.get_read_available()` reports
only what the wall clock says should have arrived. Feeding the file as fast as
it reads would hand the engine a 25-minute conversation in seconds, and every
latency figure, coach cycle and queue-depth guard would be measuring a
situation that cannot occur.

### What the script is built to provoke

The interview is **entirely invented** — `benchmark/data/real/` is deliberately
not consulted, and no sentence in it was said by a real person. It is written
to argue with itself rather than to confirm the demo's hypotheses:

- **A deliberately skipped survey question.** `s3` is never asked; the
  interviewer says "let me skip ahead" to save time, which is how coverage is
  really lost — triaged under a clock, not forgotten. The subject then
  volunteers the answer anyway, so the TOPIC is covered while the QUESTION is
  not, and only one of those is recorded. `s7` is never asked either.
- **Follow-up bait.** Generalisations with no instance behind them, a strong
  word dropped and walked past, a number given and then retracted, and a
  contradiction against something said twenty minutes earlier.
- **Five deliberately bad interviewer questions in a row** — leading,
  hypothetical, pitch, opinion-seeking, confirmation-only — placed in the
  middle of an otherwise competent interview, which is how they actually occur.
- **Evidence in both directions.** The subject contradicts h1 outright in one
  place and is genuinely ambivalent about h4. An interview where everything
  lines up would be a poor test of a tool whose entire thesis is
  disconfirmation.

### What it measures

`analyse_eval.py` reports three things that fail independently:

1. **Did it hear it** — per-speaker word similarity against the script, plus a
   tail check that the last scripted turn actually played. The planted material
   sits near the END, so a run cut short by a mis-set `--seconds` would lose
   exactly what the eval exists for while every other number still looked fine.
2. **Did coverage fire** — which questions matched, and whether `s3` and `s7`
   were left unmarked. If one of them DID match, the report prints the
   utterance that triggered it rather than calling it a bug: the interviewer
   probes the substance of `s3` shortly after skipping it, the matcher sees one
   segment with no context and is told to match paraphrases, so a match there
   may be the detector complying with its spec. That is a judgement for a human.
3. **Did the coach earn it** — cycle counts, suggestions generated vs shown,
   and what it actually proposed at the four planted moments (the skip, the
   unfollowed generalisation, the retraction, the bad-question run).

### What it does NOT prove

**THE CRITIC IS NOT UNDER TEST.** This is the most important limitation on the
page, and it was originally claimed as a feature. The critic bands the coach's
OWN generated candidates; it never sees the interviewer's questions. So the
script's five deliberately bad interviewer questions provoke nothing that is
measured anywhere. Worse, the flag counts the report prints are flags on the
coach's own suggestions, where **all-clean is simultaneously the expected
result, the good result, and what a critic stuck returning `clean` would
produce.** There is no positive control, so this eval cannot tell a working
critic from a dead one. Testing the critic needs a separate harness that forces
known-bad candidates through `rank_candidates`.

**Speaker attribution is not measured, and cannot be.** Labels come from which
file the frames arrived on, and the two files have no bleed, so attribution is
correct by construction.

**Extraction is not run.** The harness runs `start` only. Facts, bearings and
the extraction rubric are untouched by this eval.

**The resampler runs at identity ratio only.** The audio is synthesised at
16 kHz mono precisely so nothing needs resampling. The real path — 48 kHz
stereo loopback down to 16 kHz mono — is not exercised.

**The web layer and the real STOP path are skipped.** The run uses `--no-serve`
and stops on the `--seconds` timer, so the hub, the status poll, the
suggestion and survey broadcasts, and the signal-driven ordered shutdown are
all layer 3, not layer 2.

**PortAudio and loopback capture are not exercised.** The two device reads are
the seam this harness replaces, and that is the honest cost of getting two
clean channels: the pipeline separates speakers by DEVICE, so audio played out
of one render device would land entirely on the interviewee side and the
coverage detector — which watches only the interviewer stream — would never
fire. Device behaviour is layer 3.

**The keepalive is disabled** (`APP_KEEPALIVE_ENABLED=0`). It exists to stop a
real loopback device idling, and there is no real device here.

**There is no bleed.** Two real microphones in a room hear each other. These
two files do not, which isolates what is being measured from a problem the
project already knows it has.

---

## Layer 3 — manual hardware checklist

Nothing here can be automated; all of it needs a person, a microphone and
speakers. Run it before any interview that matters.

- [ ] **Mic ON/OFF mid-interview.** Toggle off, keep talking, toggle back on.
      The overlay panel must report what HAPPENED, not what was asked for.
      *Verified 2026-08-13.*
      **Known unfixed bug:** after any toggle every interviewer `latency_s` is
      inflated by the off-duration — the vendor audio clock freezes while the
      wall clock advances. Expect it; it is not a new failure.
- [ ] **Full record → stop → render.** Stop with the overlay's STOP button, not
      Ctrl+C. **The pass condition is `closed_at` in `session.json`** — without
      it a session is indistinguishable from one killed mid-call, and
      `latency_summary` is never written. *Never yet done end to end.*
- [ ] **Pre-flight.** Unset `SONIOX_API_KEY` and press record. It must refuse
      BEFORE the microphone opens, leaving no orphaned `sessions/<id>/`.
- [ ] **Escape while recording** refuses; **Ctrl+Escape** ends it.
- [ ] **Two interviews at once must never be started.** Two PyAudio instances
      on one machine is an uncatchable Windows access violation that kills the
      process and destroys the first session. There is no guard; this is a
      procedure, not a test.
- [ ] **The hub over a real project** (`python -m app hub --project _unfiled`).
      The empty states are the ones only synthetic data has exercised: no
      survey, no hypotheses, `--no-coach` recordings.

### Launching the overlay

Use `start-overlay.bat`, or `python -m app overlay` with a **real console**.
Two ways to get this wrong, both found the hard way:

- **Never `pythonw.exe`, and never with redirected stdio.** Both leave the
  process with no console, and `GenerateConsoleCtrlEvent` requires one — so the
  STOP button silently stops stopping the interview.
- **Never launch it minimised.** `-WindowStyle Minimized` minimises the overlay
  window itself, not the console, and parks it off-screen.

---

## Layer 4 — a real interview

A participant who has not seen the script, on real hardware, with the coach on.
The only layer that can find out whether the suggestions are any good in front
of someone whose time is being spent.

Afterwards, run the close-out chain: stop → render → extract → hub.

---

## Things deliberately not tested

- **Soniox accuracy against a reference.** Two confounded WER numbers exist in
  this project's history and must never be quoted. Fixing that needs a clean
  benchmark run, not a test.
- **The demo project's content.** `projects/demo-claudictive` is a fixture for
  looking at the hub, not a thing under test. Rebuild it with
  `build_demo_claudictive.py` if a run dirties it.
