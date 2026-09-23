# Discovery Copilot

Live bilingual (Hebrew/English) interview transcription: captures the
interviewee's audio via WASAPI loopback and the interviewer's own microphone
as two independent streams, transcribes both live, and writes a durable,
speaker-labelled transcript to disk while the call is running.

On top of the transcript sits a **live coach**: during the call it suggests
your next question, held to the Mom Test (ask about past behaviour, never
pitch, never fish for compliments), and grounded in the survey and hypotheses
you wrote beforehand. Afterwards it extracts claims into a local evidence
corpus and shows where each hypothesis stands across interviews. Windows only.

## Quick start

1. **Read "Privacy, consent and data flows" below first.** You are recording
   other people.
2. Clone to a short path and install (Python 3.12+):
   ```
   git clone https://github.com/Odiph/discovery-copilot C:\src\discovery-copilot
   cd C:\src\discovery-copilot
   py -3 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   copy .env.example .env
   ```
   Put your `SONIOX_API_KEY` (speech-to-text) and `ANTHROPIC_API_KEY` (the
   coach) in `.env`.
3. Write what the interview is for: `.venv\Scripts\python.exe -m app author`
   opens a local page for the brief, survey and hypotheses. The coach judges
   every question against them.
4. Check both audio sides carry sound: `.venv\Scripts\python.exe -m app precheck`.
5. Record: double-click `start-overlay.bat`, or run
   `.venv\Scripts\python.exe -m app start`. Stop with Ctrl+C.
6. Afterwards: `render` for a readable transcript, `extract` to pull claims
   into the corpus, and `hub` to review where each hypothesis stands. Every
   subcommand has `--help`; the sections below cover each in depth.

The rest of this README is the full operating manual, written from real use,
so it goes into far more detail than a first run needs.

## Privacy, consent and data flows

This tool records conversations with other people. Read this before the
first call.

**Consent is yours to get.** This is not legal advice, and the rules differ
by place:

- **Some places require every party's consent**, for example California
  (Penal Code 632). Software cannot tell which rule applies to a given call,
  so ask everyone, every time.
- **In the EU (GDPR)** a voice recording is personal data: you need a lawful
  basis, and before recording you must tell the interviewee who you are, why
  you record, who receives the data (the vendors below) and how long you keep
  it.
- **In Israel** a participant may record a conversation they take part in, and
  the Protection of Privacy Law (including Amendment 13, in force since August
  2025) governs what you then do with the recording and transcript.

Say at the start of every call that you are recording, transcribing and using
an AI assistant, and get agreement before you press Start. Nothing in this
tool asks for or records consent on your behalf.

**What leaves your machine, and to whom:**

| Data | Sent to | When |
|---|---|---|
| Live audio of both sides | Soniox (speech-to-text) | Throughout every recorded call |
| Transcript text, your brief, survey, hypotheses and prior claims | Anthropic, or the OpenAI-compatible endpoint you configure (`COACH_BASE_URL`) | Every coaching cycle during a call, and when you run `extract` or `critique-questions` |
| Retained audio (`--retain-audio`, off by default) | Nothing, unless you run the benchmark, which uploads it to the STT vendors it compares | Only when you run `benchmark/` yourself |

Each vendor's own policy governs retention and training use, and they change,
so read them before recording anyone: [Soniox security and
privacy](https://soniox.com/docs/security-and-privacy), [Anthropic API data
retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention).
If you process other people's personal data, a data processing agreement with
each vendor is your responsibility. To avoid a vendor, set `COACH_BASE_URL` to
a model you run yourself; without `ANTHROPIC_API_KEY` and `COACH_BASE_URL` the
coach is simply off and only Soniox receives anything. Nothing is sent
anywhere else: no telemetry, no analytics.

**What stays on your machine:** transcripts, session metadata, extracted
claims and hypotheses, under `sessions/` and `projects/`. Both are
git-ignored in full. They hold verbatim speech, so treat the folders as
confidential and delete sessions you no longer need. If someone asks you to
erase their data, delete their session directory and remove that session's
entries from the project's `context/facts.json`; there is no delete command
yet.

**A note on the code comments.** Many comments cite design notes such as
`04-01-PLAN.md`, `03-RESEARCH.md` or `STATE.md`, and phase or gate numbers.
Those are the maintainer's private planning records and are not published;
the comments around each citation carry the reasoning that matters. The
benchmark's `edge-tts` dependency is LGPL-3.0, used as an unmodified library.

**Who can reach the running app:** during a call, a small page is served on
your local network (port 8765) so a phone on the same Wi-Fi can follow along.
Every route on it requires a random token generated fresh for that run, and it
is plain HTTP, so use it only on a network you trust. The console, the
authoring page and the hub listen on `127.0.0.1` only. The MCP server is
read-only and never returns quotes, transcripts or audio. See `SECURITY.md`.

## Setup (first run on a new machine)

Windows only, and not by accident: `PyAudioWPatch` ships `win_amd64` wheels
only, and WASAPI loopback captures the **host** machine's default render
endpoint. There is deliberately no devcontainer — a container cannot record
this. Python 3.12+ with the `py` launcher (a standard python.org install).

```
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

That is where the `.venv\Scripts\python.exe` named throughout this file comes
from. Every command below assumes it exists; a bare `python` is the system
interpreter, has none of these pinned packages, and will silently do something
else.

**Keep the checkout path short** (e.g. `C:\src\discovery-copilot`), or enable
Windows long paths. The `anthropic` package ships module names long enough that
a venv nested ~120 characters deep crosses Windows' 260-character limit, and
every import of the coach then fails with a `ModuleNotFoundError` that names a
file which is plainly there.

Then copy `.env.example` to `.env` in the repository root and fill it in —
never committed, never a CLI argument:

```
SONIOX_API_KEY=...
ANTHROPIC_API_KEY=...
```

`SONIOX_API_KEY` is the speech-to-text vendor: without it nothing transcribes.
`ANTHROPIC_API_KEY` is the coach's reasoning vendor: without it the transcript
still records perfectly and the live suggestions simply never appear.

**From VS Code**, none of this needs to be typed. Open the folder and run
**Ctrl+Shift+P → "Tasks: Run Task"**:

- **`Setup: create venv and install`** — the two commands above. Run this
  first; nothing else works until it has succeeded.
- **`Overlay (floating recorder)`**, **`Console`**, **`Start an interview`**,
  **`Precheck`**, **`Author`**, **`Hub`**, **`Render`**, **`Extract`** — one
  task per verb.
- **`Tests (offline)`** — the suite with every paid and hardware-bound test
  deselected.

**Stop a running interview with Ctrl+C in the terminal — never VS Code's
terminate / trash-can button.** That button is `TerminateProcess`: the process
dies where it stands, `session.json` never gets its `closed_at`, and the
transcript keeps whatever partial tail was mid-write. Outside VS Code,
double-click `start-overlay.bat` or `open-console.bat`; both are
workspace-relative and run from wherever the repository is checked out.

## Before a call (9:58 for a 10:00 call)

```
.venv\Scripts\python.exe -m app precheck
```

Play the call app's test audio (or wait for the interviewee to speak) and say
something into the microphone while it runs. It prints one PASS/FAIL row per
check — both devices resolve, both streams carry REAL audio above a level
threshold (not merely "the device opened"), and the STT engine answers within
its network timeout. **Any FAIL exits non-zero and prints a specific remedy.**
Fix every FAIL before starting — a precheck that passes on a muted device is
worse than no precheck at all.

Skip the engine-reachability row (e.g. to check audio only, with no network
call) with `--skip-stt`. Shorten or lengthen the listening window with
`--seconds N` (default 3).

**Before a REAL interview, add `--phone`:**

```
.venv\Scripts\python.exe -m app precheck --phone
```

This adds two more rows on top of the audio/engine checks above — the pinned
web port is free, and this machine's LAN address resolves — plus a third,
bounded row: it starts the server, shows the join URL, and waits (up
to 30s by default) for a real phone to actually connect. PASS means a phone
connected in time; FAIL means none did, and the row's remedy is the one-line
triage rule from the hostile-Wi-Fi runbook below. It never hangs past its
window. This is the one command that answers "will the phone actually work
tonight" — run it, get everything green, *then* start the call.

## Starting a call

```
.venv\Scripts\python.exe -m app start
```

Captures and transcribes both sides at once until you press Ctrl+C, or for a
fixed duration with `--seconds N`. To run only one side (debugging one half of
a broken setup), add `--only interviewee` or `--only interviewer`.

`start` also serves the live transcript to your phone from inside this same
process — one command, one process, no deployed service. It prints a join URL;
open that on the phone for the live view. (The QR is off by default — see "The join QR is off"
below.) The web port is pinned (`8765` by
default — change it with `APP_PORT` if it's ever taken) so the printed URL
always matches what's actually listening. The first time you run `start` on this
machine, Windows Firewall may prompt to allow inbound connections for this
interpreter — **accept it once, now, during a rehearsal, never live during a
real call.**

> **One process per interview, with one exception.** The overlay is a second
> process: it spawns `python -m app start` as a child
> (`app/overlay/session.py`). The rule that holds everywhere is narrower —
> **a web page never spawns a subprocess.** The
> console proves it: pressing Start stops the page's own server and the *same*
> process runs the interview. The overlay is the one deliberate exception, and
> it earns it by being a real supervisor rather than a launcher — it holds the
> child's stdin control pipe, drains its stdout so it cannot deadlock, delivers
> CTRL_BREAK on stop, and checks the child's exit code instead of assuming a
> clean close. Nothing outlives what launched it.

Need to run `start` with no browser and no server at all (e.g. the
single-side trap tests, or debugging capture in isolation)? Add `--no-serve`.
Serving is not a separate mode you opt into — it's what `start` does by
default; `--no-serve` is the one flag that opts back out.

**Pre-flight rules — not fixable in code, read them every time:**

- Fix your call output device (Zoom/Meet/Teams) *before* pressing start.
  **Do not change it mid-call** — the loopback stream stays bound to whatever
  was default when it opened; plugging in headphones mid-call is detected and
  triggers one automatic re-open, but is still slower and riskier than not
  changing it at all.
- Disable sleep / set the power plan to "never sleep" for the call's duration.
- Loopback captures **everything the machine plays**, not just the call app —
  close other audio sources (music, notifications) before starting.
- Confirm the printed LAN address matches your Wi-Fi adapter — not a VPN or
  virtual adapter — before trusting the join URL (compare leading octets with
  your phone's own Wi-Fi IP).
- If a second monitor is in use, set your screen-share selection to that
  single window or display *now* — never "entire desktop" or "all displays".

## Pre-call checklist (sixty seconds before a customer interview)

Run this down in order, every time, before a real call — not during one:

0. Author the brief, survey and any hypotheses (`python -m app author`) well
   before the sixty-second countdown starts, never during it — see
   "Grounding the coach" below. Confirm `context/brief.json`,
   `context/survey.json` and `context/hypotheses.json` are the ones for
   *this* interview, not left over from a previous customer. **Interview #1
   always has an empty `context/facts.json`** (nothing has been extracted
   yet) — with no facts extracted, the coach simply has no past interviews to
   draw on, which is not a fault.
1. Run `precheck --phone` and get **every row green**, including the phone
   actually connecting during the reachability window.
2. Confirm the printed LAN address matches your phone's own Wi-Fi address
   (not a VPN or virtual adapter — see the LAN-address row's own reminder).
3. Open the printed join URL on the phone and confirm the live page loads
   *now*, before the call, not during it.
4. If using a second monitor, set your screen-share selection to that single
   window or display now.
5. Disable notifications (browser and OS) for the surface's origin — a
   notification banner can leak the surface even when nothing else is shared.
6. Confirm the coach is live *before* the call, not during it: `start`'s own
   startup line reads `Coach enabled: brief has N item(s), survey has M
   item(s).` when `ANTHROPIC_API_KEY` is set and both files loaded. If it
   instead prints `Coach disabled: ...`, that is a decision to make now —
   capture and transcription still work either way, but a mid-call surprise
   about missing suggestions is what this step exists to prevent.
7. Start the session (`start`), then start the call.

## When the phone won't connect: the hostile-Wi-Fi runbook

**The one-line rule — the part to remember under pressure:** slow or no
response at all → assume isolation, hotspot first; fast refusal → firewall;
loads but the IP looks wrong → adapter or VPN; loads fine but frozen → not a
network problem at all, look at the laptop.

| Symptom | Likely cause | Fast diagnostic | Fix |
|---|---|---|---|
| Join URL opens, phone shows nothing, times out entirely, no error page | AP/client isolation — hotel, conference, co-working Wi-Fi blocking device-to-device traffic | This is the DEFAULT assumption on unfamiliar Wi-Fi. Do not debug further | Phone personal hotspot; the laptop joins the phone's hotspot |
| Join URL gives "connection refused", `ERR_CONNECTION_ABORTED`, or any red error FAST, not a long timeout | Windows Firewall blocking the port | `Get-NetConnectionProfile` — **if `NetworkCategory` is `Public`, that is the answer.** Do NOT go looking for a firewall prompt; on a Public network Windows blocks inbound silently and never prompts | Make the network Private and add the allow rule — see below |
| Page never loads and the printed LAN IP looks wrong — e.g. a different subnet from the phone's own Wi-Fi address | Wrong adapter: a VPN or virtual adapter owns the default route | Compare the printed IP's leading octets with the phone's own Wi-Fi IP | Disconnect the VPN and re-run `precheck` |
| Phone connects and shows the page, but the transcript never updates | NOT a networking problem — capture or STT failure | Look at the laptop's terminal and the on-screen status banner, not the network | Follow the server-status banner, not this table |

**The Public-network trap (the most common failure):**

Windows classifies unfamiliar Wi-Fi as **Public** and then blocks inbound
connections *silently* — no prompt, ever. The laptop looks completely healthy
from the laptop: the server answers 200 on both `127.0.0.1` and its own LAN
address. Only the phone sees the failure, and it looks like a fast red error.

Diagnose and fix, from an **Administrator** PowerShell:

```powershell
Get-NetConnectionProfile                  # NetworkCategory: Public  <- the tell

Set-NetConnectionProfile -InterfaceAlias 'Wi-Fi' -NetworkCategory Private
New-NetFirewallRule -DisplayName 'Discovery Copilot phone surface (8765)' `
  -Direction Inbound -Protocol TCP -LocalPort 8765 `
  -Profile Private -Action Allow
```

**The rule is scoped to `Private` on purpose.** On a hotel, conference or
co-working network — exactly the networks Windows marks Public — the port
should stay shut and the hotspot procedure below is the intended answer. Do
not widen this to `Public` to make a venue work; use the hotspot.

**The hotspot procedure (the default fix for the default problem):**

1. Turn on the phone's personal hotspot.
2. Join the **laptop** to the phone's hotspot (not the other way around).
3. Re-run `start`.
4. Use the **new** join URL — the address changes because the network
   changed; that is exactly why the join surface is regenerated every run.

A tunnel is a manual last resort only, never part of the normal path — reach
for the hotspot first, always.

## Screen-sharing during the call

- Share **only the video-call application window**. If full-screen sharing is
  unavoidable, pick the single monitor that does **not** have the surface
  open — never "entire desktop" or "all displays".
- Prefer the phone. Nothing on a phone screen can leak into a laptop
  screen-share by construction — that's why the phone is the design here, not
  a fallback.
- If you're using the second monitor instead, picking the correct single
  display share target is a checklist item, not something the app solves for
  you.
- Disable browser and OS notifications for the surface's origin — a
  notification banner is a leak vector independent of what you actually
  share. The browser window title is already generic; don't rename the tab.

## Secrets

Two secrets to manage, both the same discipline: the environment or a local
`.env` file, never committed, never a CLI argument.

- **`SONIOX_API_KEY`** (and `ASSEMBLYAI_API_KEY` if used) — the STT vendor.
- **`ANTHROPIC_API_KEY`** — the coach's reasoning vendor. If it's
  absent, `start` prints `Coach disabled: ...` and runs anyway — capture and
  transcription are completely unaffected; you just get no suggestions. See
  "Grounding the coach" below.

### Pointing the coach at a different model

The coach takes its client by injection and speaks two dialects, so any
OpenAI-compatible endpoint works — a local runtime (Ollama, vLLM, llama.cpp) or
a hosted provider (OpenRouter, Groq, Together, DeepInfra, Fireworks). Three
variables, all read from the environment like every other secret:

```
COACH_BASE_URL=https://openrouter.ai/api/v1
COACH_API_KEY=...
COACH_MODEL=google/gemini-2.5-flash
```

**`COACH_BASE_URL` wins outright.** When it is set, `ANTHROPIC_API_KEY` is not
consulted at all — which is deliberate, and is also why every test that
exercises the no-credential degrade path has to blank *both* routes (see
`app/tests/conftest.py`, which does it for the whole suite after this bit them
three separate times).

**`COACH_MODEL` is not optional here.** The default is a dated Anthropic model
id and means nothing to another provider, so pointing the coach elsewhere
without naming a model sends that endpoint an id it has never heard of.

**`EXTRACT_MODEL` buys quality with latency nobody is waiting on.** The live
coach answers inside a ~8s cycle while an interview is happening, so its model
is chosen under a latency bound. Extraction runs *after* the interview ends —
so a slow, strong model there costs money and nothing else. It is the one
place in this system where quality is free.

```
EXTRACT_MODEL=anthropic/claude-opus-4-5
```

Unset, it falls back to `COACH_MODEL` and changes nothing. It selects a
*model*, not a *provider*: there is one client construction site, so whatever
you name has to be served by the same `COACH_BASE_URL` endpoint.

Worth setting, because the extraction model is measurably the weak link: it
labels merely-implied claims `evidence`, at 3/18 on a hand-adjudicated real
interview. `app/coach/extract.py`'s own docstring carries the measurement and
the two failure modes.

**Choosing a model is a measurement, not a preference.** `benchmark/eval/judge_critic.py`
runs a labelled question set through the real critique path and reports whether
the model can hold a forced tool call against the real rubric at all. Run it
before trusting a model with an interview — it costs a fraction of a cent:

```
.venv/Scripts/python.exe benchmark/eval/judge_critic.py
```

Measured over 3 runs of 17 labelled cases each. `judged` is the gate
that matters: a model that cannot emit the tool call is not a cheap coach, it
is not a coach.

| model | judged | rules caught | band flips | s/call | $/interview |
|---|---|---|---|---|---|
| `google/gemini-2.5-flash` | 17/17 | 5/5 | 0 | 2.85 | 1.87 |
| `anthropic/claude-haiku-4.5` | 15.7/17 | 4.7/5 | 4 | 4.95 | 5.82 |
| `mistralai/mistral-nemo` | 7/17 | 1.3/5 | 13 | 4.29 | 0.10 |

Five further models — `openai/gpt-5-nano`, `openai/gpt-oss-120b`,
`qwen/qwen3.5-flash`, `z-ai/glm-4.7-flash` and both local `qwen2.5` sizes —
failed the tool-fidelity gate outright and are not candidates at any price.

Two caveats that keep those numbers honest. Both `$/interview` figures assume
**no prompt cache**, because the OpenAI-compatible seam drops `cache_control`
(see `app/coach/openai_seam.py`) and prefill is roughly 8× without one; Haiku
**direct** through the Anthropic SDK, where caching does apply, measured
**$1.05**. And this compares Haiku *through OpenRouter*, not Haiku direct — the
parse errors and band flips in that row may belong to the translation rather
than to the model.

The join token embedded in the URL is **not** a secret to manage — it's
generated fresh in memory every run, never written anywhere, and dies the
moment that run ends. There's nothing to store or protect about it beyond not
leaving old join URLs lying around.

## Projects: running several lines of enquiry at once

A project is a **nested label** — a product, a feature within it, a hypothesis
within that. Nesting is directory nesting, so a project path looks like
`acme/onboarding/activation-delay` and lives under `projects/`:

```
projects/acme/onboarding/
  context/     brief.json, survey.json, hypotheses.json, facts.json
  sessions/    interviews filed at THIS node
```

Pass `--project` to `start`, `render`, `author` or `extract`:

```
python -m app author  --project acme/onboarding
python -m app start   --project acme/onboarding
python -m app extract --project acme/onboarding 20260810T143000
```

`precheck` deliberately does **not** accept it — it touches neither directory,
so a project there would be an argument it silently ignored.

**Omit `--project` and nothing changes**: the root `context/` and `sessions/`
are used, exactly as before projects existed.

**All of `projects/` is git-ignored, blanket.** That rule has to stay blanket:
`projects/*` with a `!projects/*/context/*.example.json` negation looks like it
would let format samples live in the tree, but git never descends into an
excluded directory, so the negation is never consulted. The samples stay in the
root `context/`, which is what lets this rule be total. A test asserts it.

### Drawing on another project's knowledge

By default the coach is grounded in the filing node alone. `--read-from` adds
another node's **facts and hypotheses**, read-only, and is repeatable:

```
python -m app start --project acme/onboarding --read-from acme/billing
```

The rules, all enforced rather than documented:

- **Same hierarchy only.** A node under a different root is refused. Knowledge
  is shared within one product, never across two customers.
- **Reads merge; writes never do.** Extraction always writes to the filing
  node. A source node's files are left byte-identical.
- **The brief and survey always come from `--project` alone.** They are what
  *this* interview is about; merging another node's survey would quietly change
  the questions you ask in the room.
- **Facts are pooled, then selected once**, so the retrieved-fact cap still
  means what it says — three sources do not put three times the facts into the
  brief.
- **Merged fact ids are qualified** with the node they came from
  (`fact:acme/billing:<id>`). Session ids are only unique *within* one root, so
  two projects can mint the same one; unqualified, two customers' facts would
  read as a single provenance chain.

Every rejection is an error, never a silent skip — a dropped source would mean
the coach ran on less grounding than you selected while reporting success, and
you would blame the model for the thinner suggestions.

While a projected session runs, the overlay shows a small strip naming the
project and its extra sources, so you can always see which knowledge is in
play. It is hidden entirely for an unprojected run. Note it renders the node
paths on a screen that may be sitting in a cafe: if you name a node after a
customer, that name is on the screen. The session's `session.json` records the
same two fields for later.

### The console: pick a project, then start

```
python -m app console
```

or double-click **`open-console.bat`**. A loopback-only page opens showing the
project tree; select a node and you see its survey, its hypotheses with their
states, how many facts it has accumulated, and its past interviews. Tick any
in-branch nodes to also draw on, then press **Start**.

**The console and the interview are one process.** Pressing Start stops the
page's server and the same process runs the interview — **a web page never
spawns a subprocess** (see "Starting a call" above for the overlay, the one
process that does). So leave the terminal open and read it once you have
pressed Start; that is where the interview reports from. Ctrl+C before choosing
cancels without starting anything.

The console **only reads**. No route it exposes changes anything on disk, and
it shows a fact COUNT rather than any fact — a fact carries a previous
customer's verbatim speech, and this is a browser page. Authoring stays where
it was, on `python -m app author`.

It carries the same two guards as the authoring page (loopback bind, Host
header, JSON `Content-Type`) because the loopback bind stops the LAN but not
your own browser. Here the consequence of getting that wrong is worse than the
authoring hole those guards were added for: a cross-origin POST would choose
which customer's knowledge grounds your next interview.

One limitation, stated rather than discovered: `DEFAULT_APP_PORT` (8765) and
`DEFAULT_AUTHOR_PORT` (8766) are fixed and never auto-chosen, so two projects
cannot run at once. That is acceptable given the one-PyAudio-instance rule,
which already forbids two concurrent interviews on one machine.

### The join QR is off

By default `create_join` does not draw the terminal QR or write
`join-qr.png`. The rest of the join surface is unaffected: the token is still
fresh per run, the LAN IP is still resolved, and the join URL is still printed,
so a phone can still join by typing it.

Re-enable with `APP_EMIT_QR=1`. The renderers were left in place rather than
deleted, so that is the only change needed.

### Keeping test sessions apart

Keep rehearsals and test recordings in their own project, for example
`projects/_eval/`, and declare it `{"corpus": "demo"}` in its `project.json`
so its claims are never mistaken for customer evidence:

```
python -m app render --project _eval <session-id>
```

A bare `python -m app start` with nothing in the root `context/` tells you to
author a brief first; that is the default behaviour, not a fault.

## Grounding the coach: the brief, the survey, and hypotheses

The authoring page (`python -m app author`) now carries **three** things, all
under `context/` and all git-ignored (the brief can name a real customer —
this is the most sensitive plaintext this repository ever produces, and
hypotheses are right behind it: each one names a bet you're making about that
same customer):

- **`context/brief.json`** — what you know or suspect about the interviewee,
  as a list of items. Each one is labelled `"provenance": "evidence"` (you
  were told this, or read it somewhere) or `"provenance": "inference"` (you're
  guessing). There's no default — a missing or misspelled provenance value
  fails to load rather than silently becoming "evidence".
- **`context/survey.json`** — the ordered list of questions you plan to ask.
  The coach tracks which ones the *interviewer* (you) actually asked, from
  your own microphone stream — never from a suggestion shown, whether or not
  you followed it.
- **`context/hypotheses.json`** — what you believe about this
  customer, stated so it can be proven wrong. Every hypothesis needs a
  **falsification condition** — what evidence would make it FALSE — and the
  page refuses to save one without it: a hypothesis nobody could ever
  disprove gives the coach nothing to target when it steers toward the
  question that would settle it. A new hypothesis always starts
  **`"state": "untested"`**; the state is yours to change as evidence comes
  in — nothing in this codebase changes it for you yet. Today, hypotheses are
  read by `python -m app extract` for tagging only (each extracted fact may
  reference the hypothesis it bears on); nothing else reads or writes them.

Copy the shipped format samples to get started:

```
copy context\brief.example.json context\brief.json
copy context\survey.example.json context\survey.json
copy context\hypotheses.example.json context\hypotheses.json
```

The brief and survey are read **once**, at `start`, and copied into that
session's own metadata — editing them mid-call never re-grounds a session
already running. `start` **fails loudly**, before anything else happens, if
either file is missing or malformed. `start` **warns** (but proceeds) if
either file predates your previous session — a stale brief from a different
customer is a realistic and expensive mistake, but only you can tell "stale"
from "deliberately reused". Hypotheses are read separately, by
`python -m app extract`, at extraction time — see "Extracting facts" below.

If `ANTHROPIC_API_KEY` is absent, none of this is even read — the coach is
disabled and the session records a transcript only.

## Where the transcript goes

Every run creates a new timestamped directory under `sessions/` (gitignored —
confirmed with `git check-ignore`, never by reading `.gitignore`) containing:

- `transcript.jsonl` — one JSON record per line, append-only, `fsync`'d after
  every write so a crash never loses what was already captured. Segment
  records carry `speaker` (`interviewee` or `interviewer`), `text`,
  timestamps, and a *measured* latency. Health/trap events (idle, recovered,
  lagging, dropped, device-changed, dead, stream-failure, ...) interleave in
  the same file, in the order they happened.
- `session.json` — provenance: both device names, both native sample rates as
  reported by the driver, the resolved engine, start/close times, whether
  audio was retained (`audio_retained`), a per-speaker latency summary, and —
  when there is one — `participant`, who was interviewed.

  `participant` is optional and its ABSENCE is the normal state: pass
  `--participant "Dana (ops lead)"` to `start`, or fill it in afterwards on the
  hub, which also clears it. The key is omitted entirely when nobody is named
  rather than written as `null`. It exists because per-interview counting
  counts *sessions*, which is the same as counting people only while nobody is
  re-interviewed — the label does not change the count, it makes the count
  checkable by a human.
- `transcript.md` — only after `python -m app render` is run (see below).
- `interviewee.wav` / `interviewer.wav` — only if `--retain-audio` was passed
  (see below). Already covered by the existing `*.wav` ignore rule.

## Reading the transcript after the call

```
.venv\Scripts\python.exe -m app render <session-id-or-path>
```

Writes `transcript.md` into the session directory: both speakers interleaved
in strict time order, merged into paragraphs per speaker turn, with health
events (idle, device-changed, ...) shown inline at the point they happened,
and a per-speaker latency summary (count, median, p95) at the end. Segment
text is reproduced exactly as the engine emitted it -- mixed Hebrew/English
lines are never reordered or "fixed". Works on a session that crashed or is
still running; it says so in the output rather than failing.

**This same document is also where every coach decision gets audited.** Inline, in the same session-time order as the transcript itself:

- **Every suggestion the coach ever produced** -- shown, ranked down, *and
  dropped* -- with its move type (`advance`/`deepen`/`fill_gap`/
  `test_hypothesis`), its band, its five named per-rule flags, and its
  rationale. Read the dropped ones: a high drop rate concentrated on one
  flag means the critic rubric is over-firing on real calls, and that is
  the single most useful tuning signal the post-call document gives you.
- **Which past-interview facts grounded a suggestion.**
  A `grounded in:` line under a suggestion prints the fact's claim and its
  **verbatim quote from that earlier interview** -- this document is the
  ONLY place in the whole tool a past interview's exact words are ever
  shown; they never reach the phone or the live call itself. If a
  suggestion cites an id that resolves to nothing, that's normal, not
  corruption: fact ids are re-minted every time you re-run `extract`, so an
  id from before your last re-extraction won't match anymore.
- **A proposed hypothesis-state change.** A line reading
  `PROPOSED (not applied): hypothesis ... -> '...'` means the coach thinks
  the evidence points that way -- **nothing has changed yet.** The
  hypothesis set changes only when *you* change it, on the authoring page
  (`python -m app author`) -- there is no other accept surface, and reading
  this line is the whole workflow: decide whether you agree, then go set it
  yourself.
- **The FULL coverage transition history** -- every survey question the
  coach marked as asked, and what interviewer sentence it matched against;
  every segment it considered and found no match for too. **This is where
  you check whether the coach quietly stopped tracking a question:** a
  false-positive match silently removes that survey item from attention for
  the rest of the interview, and reading "matched X against Y" side by side
  is the only place that mistake is ever catchable after the fact.
- A closing **`## Coaching summary`** -- cycles completed, suggestions
  shown, candidates dropped broken down by which rule fired, coverage
  counts, how many suggestions were grounded in a past-interview fact (and
  how many distinct facts were cited), how many hypothesis-state changes
  were proposed (with the same not-applied reminder), and the coaching
  latency figures (session-start grounding, first cycle, and remaining
  cycles, reported separately). Reports all-zero, without failing, for a
  session where the coach never ran -- including every session recorded
  before this phase existed.

### Extracting facts

```
.venv\Scripts\python.exe -m app extract <session-id-or-path>
```

The one command to run after an interview. Reads the session's transcript
and writes cited, provenance-tagged facts into `context/facts.json` -- the
same directory the brief and survey live in, under the identical
git-ignore rule. **`context/facts.json` and `context/hypotheses.json` are
never committed, on purpose:** they carry verbatim customer speech, exactly
like `context/brief.json` already does. Nothing here changes that.

**Safe to re-run.** Running `extract` again on the SAME session *replaces*
that session's facts with the new run's -- it never doubles the corpus. This
matters in practice: a prompt fix, a model swap, or just re-running because
something looked off should never leave two versions of one interview's
facts sitting side by side, reading as if two different customers said the
same thing. Every OTHER session's facts are left completely untouched by a
re-run.

**A non-zero exit means at least one returned claim was rejected** -- the
run still writes every fact that WAS accepted; only the rejected ones are
dropped. Look at the printed report, not just the exit code: each rejection
names the claim and one of two reasons, and the two point at different
problems --

- `unresolved-citation` -- the model cited a segment that doesn't exist in
  this session's transcript (a miscounted or hallucinated `seq`).
- `unlocatable-quote` -- the claim was labelled `evidence`, but its quoted
  text isn't really, verbatim, in the segment it cites. Either the model
  paraphrased instead of quoting, or the claim is really an inference that
  got mislabelled.

**If extraction refuses to run at all** (exits non-zero *before* printing
any per-claim report), the EXISTING corpus at `context/facts.json` failed
to parse -- the run stops before the vendor is ever called, and the message
names the file (and, where possible, the offending item). Nothing was
overwritten or lost; fix or restore that file, then re-run.

**If you see `WARNING: session ... has no closed_at marker`**, the session
you pointed at was never closed -- extraction still runs; this is a
warning, not a refusal. Only you can tell a deliberate mid-interview sanity
check from a forgotten crash, so the fact set is written either way, but
treat it as possibly incomplete if the interview is genuinely still going.

**Roughly what one extraction costs:** every run prints the accepted and
rejected counts plus the call's own token usage, when the response reports
it. This is a per-interview batch cost, not a per-utterance one -- check the
printed numbers against a few real interviews rather than trusting a
one-line estimate; that is exactly what these numbers are for.

### More commands after the call

All take `--project PATH` like the rest, and none calls a model unless noted.

- **`summary <session>`**: what one interview produced: how many claims, how
  many were stated versus inferred, and which hypotheses they bore on.
- **`standing [session]`**: where every hypothesis stands across all extracted
  interviews. Name a session to also see what that one interview moved.
- **`critique-questions <session>`**: bands YOUR OWN questions from the
  interview against the same Mom Test rubric the coach is held to, and writes
  `questions_critique.json` into the session. **This one calls the reasoning
  vendor.** `--min-words` sets the shortest turn worth judging.
- **`export --to backup.zip [--verify]`**: writes a checksummed copy of the
  project's irreplaceable data (transcripts, facts, hypotheses) to a zip. It
  refuses a destination git can see, so a backup can never be committed by
  accident; `--verify` re-reads every checksum after writing.

## Retaining the raw audio (rehearsal sessions only)

```
.venv\Scripts\python.exe -m app start --retain-audio
```

**Off by default.** Add `--retain-audio` to also write one 16 kHz mono WAV
per speaker (`interviewee.wav`, `interviewer.wav`) alongside the transcript --
this is exactly the two-stream input the STT benchmark under `benchmark/`
needs. Point the existing benchmark harness at the retained pair; no new benchmark tooling
is needed.

**Use `--retain-audio` for a REHEARSAL or mock interview ONLY -- never a real
customer interview.** The benchmark this audio feeds uploads to two
third-party STT vendors you may not have agreed to share customer audio
with. A retained WAV pair from a real call is one command away from shipping
customer audio to them. `session.json`'s `audio_retained` field always records
whether a given session actually retained audio, so this is never left to
guesswork after the fact.

## Changing the STT engine

One environment variable, read once at startup — nothing else in the capture
or session code ever names a vendor:

```
set STT_ENGINE=soniox      REM the funded, live default
set STT_ENGINE=fake        REM offline, deterministic, no credentials -- for local testing
```

A `--engine` flag on both `start` and `precheck` overrides the environment
variable for a single run. Put `SONIOX_API_KEY` (and, if used,
`ASSEMBLYAI_API_KEY`) in a local `.env` file — never in source control.

## Troubleshooting

**Extraction failures aren't in the table below** — they're printed
CLI-side, not written as session events, and are covered in full under
"Extracting facts" above: a non-zero exit, an `unresolved-citation`
vs. `unlocatable-quote` rejection, a refusal to run at all, and the unclosed-
session warning are each a *different* condition that looks like "extraction
failed" from the terminal — that section is where you tell them apart.

Every health/trap condition is a named `event` record in `transcript.jsonl`
(and inline in `transcript.md` once rendered) — look for these names:

| Event | Meaning | What to do |
|---|---|---|
| `idle` | That stream has produced only zero-filled (synthesized) frames for a few seconds | Normal during a pause. If it never clears: check the interviewee's call app isn't muted, or that you're speaking into the right microphone |
| `recovered` | Real audio resumed after an `idle` period | Informational — no action needed |
| `lagging` | The STT engine can't keep up; the capture queue is backing up | Usually a slow local engine or a network stall. Capture is never blocked — but segments may be delayed |
| `dropped` | The queue overflowed and the oldest frame was discarded (carries a running count) | A sustained `lagging` condition turned into real data loss — switch to a faster engine (`--engine soniox`) or investigate the network |
| `device-changed` | The Windows default output device changed mid-call (e.g. headphones plugged in) | One automatic re-open is attempted; check the next event for the outcome |
| `reopened` | The automatic re-open after a `device-changed` succeeded | Informational |
| `dead` | A stream's device changed and the one re-open attempt failed | That side has stopped capturing. Restart `python -m app start` |
| `stream-failure` | An unhandled error inside one stream's worker thread | That side stopped; the other side and the session file are unaffected. Check the event's `error_type`/`message` |
| `keepalive-warning` | The silent keep-alive render stream (second layer against the silent-loopback trap) failed to open | Not fatal — the pacer's own non-blocking read is still the primary mitigation |
| `audio-retention-error` | `--retain-audio` was set but writing a WAV failed (e.g. disk full) | Retention is disabled for the rest of that session; capture and transcription are unaffected |
| `session-resumed` | `python -m app start` was pointed at an existing session directory and continued it rather than starting fresh | Informational — confirms a crash recovery picked up where it left off |
| `coverage-asked` | The coach decided your own words asked a survey question — carries which one, never the words themselves | Informational — the full history (matches AND near-misses) is what `coverage-considered` below is for |
| `coverage-considered` | A finalized interviewer segment was checked against the survey and matched nothing | Informational — normal for most of what you say |
| `coverage-queue-drop` | The coach's coverage queue was full (a sustained vendor slowdown) and the oldest pending check was dropped | Not fatal — a fresh check happens on your very next sentence |
| `reasoning-timeout` | A coach vendor call (`stage`: `coverage`, `generate` or `critique`) timed out or errored | Suggestions may be briefly missing; the transcript is completely unaffected. A single one is normal degradation — a SUSTAINED run of them (three or more in a row) is what raises the phone's own "suggestions unavailable" banner |
| `coach-cycle-skipped` | The interviewee kept talking before the previous suggestion cycle finished, so the queued-but-not-yet-started cycle was replaced by the newer one | Informational — this is "one cycle at a time, replace never queue" working as designed, not data loss |

### What a quiet suggestion region means (three different things that look identical)

"The suggestion region on the phone is quiet" is not one
condition — it is three, and only one of them is a problem:

| What you see on the phone | What it means | What to do |
|---|---|---|
| A calm banner reading **"Suggestions unavailable — transcript continues"** | The coach's reasoning layer failed a SUSTAINED run of cycles | The transcript is unaffected. Check your network and `ANTHROPIC_API_KEY`; this is never the "Transcription stopped on the laptop" banner, which is a different, more urgent condition |
| The region is quiet, **no banner at all**, everything else looks healthy | The coach had nothing worth suggesting this cycle | Nothing to do — this is a normal state, not a fault |
| **One suggestion, no alternatives underneath it** | Normal, complete state — never a partial or half-loaded render | Nothing to do |
| **The coach never suggests a hypothesis-testing question** | Two ordinary causes, both normal: (1) no hypotheses were authored for this interview — check `context/hypotheses.json` on the authoring page; (2) hypotheses exist, but nothing extracted yet is relevant enough to test one — check the post-call document's dropped candidates for whether `test_hypothesis` candidates were proposed and dropped, or never proposed at all | Author at least one hypothesis before the call if you want this move available; after the call, read the coaching summary's drop breakdown before assuming something is broken |

If you cannot immediately tell which of these three you are looking at,
that is itself worth noting after the call — see "Reading the transcript
after the call" below for where the coverage log settles any doubt about
what the coach actually did.

## Reaching the corpus from another AI agent (MCP)

`python -m app.mcp` serves this machine's interview corpus to your other AI
agents over the Model Context Protocol — JSON-RPC 2.0 on stdio, hand-rolled
against the standard library, no new dependency.

**What it will and will not hand over.** It is READ-ONLY: seven tools, none of
which writes a byte. It exposes extracted CLAIMS, hypothesis standings and
interview metadata — never `quoted_text`, never a transcript segment, never
audio. This is the only surface in this project that hands the corpus to a
process you did not write, so the quote boundary is enforced structurally
(every projection is written out field by field) and pinned by a test.

**Never put a key in an MCP config.** These files get committed. This server
needs no credential of any kind — it reads local files — and
`app/tests/test_mcp_harness.py` fails the build if `.mcp.json` ever grows a
key-shaped value, or even a setting merely *named* like one.

### Claude Code

Nothing to do — `.mcp.json` at the repository root registers it, and Claude
Code offers to enable it the first time you open this folder:

```json
{
  "mcpServers": {
    "discovery-copilot": {
      "type": "stdio",
      "command": ".venv/Scripts/python.exe",
      "args": ["-m", "app.mcp"]
    }
  }
}
```

The interpreter path is **relative on purpose**, so it is the venv of
whichever checkout you opened — including a git worktree. Never replace it
with an absolute path: that works in exactly one checkout on one machine.
Claude Code launches the server with the project directory as its cwd, which
is also what puts the repository root on `sys.path` so `-m app.mcp` can import
`app`.

Check it with `/mcp` inside Claude Code; the server should list as
`discovery-copilot` with seven tools.

### Claude Desktop

Claude Desktop reads ONE global config instead of a per-repository file, and
it launches servers from an arbitrary working directory. So the entry needs an
**absolute** interpreter path and an explicit **cwd** — without the cwd,
`python -m app.mcp` cannot import `app` and the server dies at startup with an
import error you will only see in Claude Desktop's own log.

Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add:

```json
{
  "mcpServers": {
    "discovery-copilot": {
      "command": "C:\\path\\to\\discovery-copilot\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp"],
      "cwd": "C:\\path\\to\\discovery-copilot"
    }
  }
}
```

Replace both paths with your own checkout, restart Claude Desktop, and the
tools appear under the connector menu.

### Another repository: the `discovery` Claude Code plugin

`.mcp.json` reaches the server only from inside this checkout. A product
repository opened in Claude Code gets it from the plugin in
`integrations/claude-code/`, which teaches that repository three things: the
tools (the server registration), which project it answers to
(`.discovery-copilot.json`, written by `/link-discovery`), and how to reason with
the evidence (the `discovery-evidence` skill: `demo` and `undeclared` claims are
never findings, counts are per interview, gaps become proposals rather than
assumptions).

Once per machine:

```
setx DISCOVERY_COPILOT_HOME "C:\path\to\discovery-copilot"
claude plugin marketplace add "C:\path\to\discovery-copilot\integrations\claude-code"
claude plugin install discovery@discovery-copilot-local
```

Then restart Claude Code, and in each product repository run
`/discovery:link-discovery <project>`. After that, `/discovery:evidence
<question>` answers from the interviews with citations, and
`/discovery:interview-gap` drafts hypotheses and survey questions into the
product repository's `discovery-proposals.md`, which you enter here through
`author`. Nothing in the plugin writes to this checkout.

- **The checkout is named only through `DISCOVERY_COPILOT_HOME`.** Claude Code
  copies an installed plugin into its own cache, so no relative path can
  reach this checkout, and an absolute one would be the `open-console.bat`
  mistake. The same copy means **an edit to the plugin reaches nobody until
  `claude plugin marketplace update discovery-copilot-local`** and a restart.
- **`/link-discovery` also writes deny rules** into the product repository's
  `.claude/settings.local.json` for this checkout's `projects/`, `sessions/`
  and `context/`. Measured, not assumed: without them, a session in the product
  repository read a transcript here directly, around everything the MCP
  server withholds. With them, the same read was refused. Claude Code asks
  before writing its own settings file; approve it.
- `app/tests/test_claude_plugin.py` launches the registration from a foreign
  directory and fails if the skill or commands name a tool the server lacks.

### The seven tools

| Tool | Answers |
| --- | --- |
| `list_projects` | every project node on this machine — start here |
| `get_survey` | the ordered questions you set out to ask |
| `get_hypotheses` | statements, falsifiers, states, decision criteria |
| `list_interviews` | which interviews exist, and whether each closed cleanly |
| `get_facts` | the extracted claims and which way each one bears |
| `get_standing` | where each hypothesis stands, counted per INTERVIEW |
| `get_interview_summary` | what one interview did, deterministically |

### `scope`: one node, or a whole sub-tree

`get_hypotheses`, `list_interviews`, `get_facts`, `get_standing` and
`get_interview_summary` take a `scope`:

- `"node"` (the default) reads only the project you named.
- `"subtree"` also reads every project nested underneath it and merges the
  results.

`subtree` is opt-in and never implicit, because it is a cross-project read —
the same rule the console's "draw on another project's knowledge" follows:
**reads may merge, within one hierarchy, on explicit request; writes never
merge.** A sub-tree never climbs to a parent, never reaches a sibling, and
never reaches another hierarchy — asking for `acme` does not pull in
`acme-corp`.

Every item in a merged answer carries the `project` node it came from, and
every count in a merged answer is **per-interview, never per-fact**: one
interviewee who made the same point five times moves `supporting_interviews`
by one. `supporting_facts` is reported alongside it, so you can see both
numbers and never mistake one for the other. If one node in the sub-tree has a
malformed artifact, only that node degrades — its path is named in `errors`
and the rest of the answer still arrives.

## Tests

```
.venv\Scripts\python.exe -m pytest app/tests benchmark/tests -q -m "not live and not hardware and not live_reasoning"
```

Add `-m hardware` (needs a working default output device and microphone, with
OS microphone access granted), `-m live` (needs a funded STT API key), or
`-m live_reasoning` (needs a funded `ANTHROPIC_API_KEY` and makes a paid call
to the reasoning vendor — a few cents) to run the tests deselected above.
