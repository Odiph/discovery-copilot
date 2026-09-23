# Contributing

Thanks for looking. This is a solo-maintained project, so small, focused pull
requests are the easiest to review.

## Before you open a pull request

1. Set up as in the README's Quick start (Windows, Python 3.12+, a short
   checkout path).
2. Run the offline suite. It needs no network, no audio hardware and no paid
   API call:
   ```
   .venv\Scripts\python.exe -m pytest app/tests benchmark/tests -m "not live and not hardware and not live_reasoning"
   ```
   and the JS tests, one file at a time (the directory form reports a
   spurious failure on Windows):
   ```
   node --test app/tests/js/author_render.test.mjs
   node --test app/tests/js/live_state.test.mjs
   node --test app/tests/js/review_render.test.mjs
   ```
3. Add or update a test for the behaviour you change. Test names describe
   the behaviour, not the function.

## The one rule that is not negotiable

**Never commit real recorded speech.** No transcript lines, no quotes, no
audio, no claims extracted from a real interview, including your own. Tests
use synthetic text shaped like the real thing. `sessions/`, `projects/` and
`context/*.json` are git-ignored for this reason; keep it that way.

## Security issues

Do not open a public issue. See `SECURITY.md`.
