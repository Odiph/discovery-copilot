# Hand-authored text fixtures

`cases.json` holds hand-authored (reference, hypothesis) token pairs used by
`benchmark/tests/test_score.py` and `benchmark/tests/test_report.py`. No audio,
no network, no model — these are pure text-in / numbers-out cases for the S5
metric family and the S6 gate/selection logic.

Tokens are deliberately short placeholder Hebrew/English "words" (single Hebrew
letters, short English words) rather than naturalistic sentences, so that every
case's expected switch points, window membership and error counts can be
hand-verified by a reader from the token list alone. Real-sentence naturalism is
exercised by the synthetic audio fixture (`benchmark/fixtures/synthetic_script.json`)
and, eventually, real interview audio (plan 01-02) — this directory's job is
arithmetic correctness, not realism.

Each entry in `cases.json` documents, in its own `note` field, the arithmetic it
is designed to prove and which phase_specification S5 rule it exercises.
