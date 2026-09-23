---
description: What the customer interviews say about a feature, decision or question, with citations
argument-hint: <feature, decision or question>
---

Answer from this product's interview evidence: $ARGUMENTS

Follow the `discovery-evidence` skill. Read `.discovery-copilot.json` first; if
it is missing, stop and suggest `/link-discovery`.

1. `get_standing` for the linked project and scope. Pick the hypotheses that
   bear on the question. Say so if none do.
2. `get_facts` for each of those hypotheses, for the claims behind the counts.
3. Answer in this shape:
   - **Bottom line**, in one or two sentences, including how much evidence it
     rests on (X of N interviews).
   - **Per hypothesis**: its statement, its `criterion`, supporting, contrary
     and unclear interviews out of `total_interviews`, and the two or three
     claims that carry the most weight, each cited by `id` (first eight
     characters) and `session_id`.
   - **What the interviews do not cover**: parts of the question no
     hypothesis or claim touches.
   - **Recommendation**: marked as unevidenced wherever it goes beyond the
     claims.

Before concluding anything, apply the skill's checks: the `corpus` of every
node the claims came from, `unextracted_projects`, and `is_scorable`.
