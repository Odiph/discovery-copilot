---
description: Find what the interviews have not yet tested, and draft hypotheses and questions for the next interview
argument-hint: [optional focus, e.g. a feature or decision]
---

Find the gaps in this product's interview evidence and draft what to learn
next. Focus, if given: $ARGUMENTS

Follow the `discovery-evidence` skill. Read `.discovery-copilot.json` first; if
it is missing, stop and suggest `/link-discovery`.

1. Read `get_standing`, `get_hypotheses`, `get_survey` and `list_interviews`
   for the linked project and scope.
2. List the gaps, most decision-relevant first:
   - hypotheses with no interviews bearing on them, or only unclear ones;
   - hypotheses with `is_scorable: false`, which need a decision criterion
     before any count means anything;
   - interviews not yet extracted (`unextracted_projects`), which are free
     evidence waiting on one command: `python -m app extract --project <path>
     <session_id>`, run from the Discovery Copilot checkout;
   - with a focus given, parts of it no hypothesis covers.
3. Draft proposals for the biggest gaps, following the skill's section 5:
   hypotheses with a falsifier and a decision criterion, and survey questions
   about past behaviour. Keep it short: three hypotheses and six questions at
   most.
4. Append them to `discovery-proposals.md` under today's date, and tell the
   operator they go in through `python -m app author --project <path>` in the
   Discovery Copilot checkout. Do not write to the Discovery Copilot yourself.
