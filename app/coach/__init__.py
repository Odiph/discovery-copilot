"""Phase 4 -- "The Coach" (04-CONTEXT.md D-01..D-07, 04-RESEARCH.md).

This plan (04-01) builds the part of the coach that has no prior art anywhere
else in this repository: `reasoning_client.py` (the one vendor-client
construction site), `stub.py` (its offline stand-in), `context.py` (the
`ContextItem[]` grounding seam, CTX-01/02/03) and `coverage.py` (the
detector that watches the interviewer's own transcript stream and decides
whether a survey question was asked, D-03).

No generate stage, no critic, no ranking, no suggestion record exists here --
those are 04-02 and 04-03 (see 04-01-PLAN.md §C10). Nothing in this package
may import `anthropic` except `reasoning_client.py`; a standing AST-based
gate in `app/tests/test_phase_gates.py` proves it.
"""
