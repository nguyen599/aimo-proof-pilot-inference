# Round-3 internal verifier audit

This audit reconstructs the production verifier/meta aggregation for the
round-3 P4/P5 proofs and joins it to the two-call GPT-5.6 grades in
`intermediate-r3-grading-gpt56-c24-r2-newkeys/`.

## Result

The internal evaluator is useful for shortlisting, but its validated-low cap
creates broad ties that discard meaningful pre-cap evidence.

| Problem | Internal rank | Candidate | Final score | Pre-cap score | Validated lows | GPT-5.6 / 7 |
|---|---:|---:|---:|---:|---:|---:|
| P4 | 1 | 22 | 0.5000 | 0.6875 | 5 | 6.0 |
| P4 | 2 | 27 | 0.5000 | 0.6250 | 3 | 5.0 |
| P4 | 3 | 21 | 0.5000 | 0.5000 | 2 | 5.5 |
| P4 | 4 | 29 | 0.5000 | 0.5000 | 7 | 4.5 |
| P4 | 7 | 2 | 0.3750 | 0.3750 | 7 | 6.0 |
| P4 | 8 | 16 | 0.3750 | 0.3750 | 7 | 6.0 |
| P5 | 1 | 21 | 0.9375 | 0.9375 | 1 | 6.0 |
| P5 | 2 | 35 | 0.8750 | 0.8750 | 1 | 7.0 |
| P5 | 4 | 29 | 0.2500 | 0.2500 | 5 | 6.0 |
| P5 | 7 | 15 | 0.1250 | 0.1250 | 7 | 6.0 |

P5's best proof remains near the top internally, so the evaluator has useful
directional signal. P4 candidate 22 also has the highest uncapped verifier
score and an external score of 6. The hard cap, however, maps four materially
different P4 candidates to the same final score of 0.5. Sorting those ties by
proof length can remove candidate 22 from a bounded selector prompt.

## Pipeline change

The selector shortlist now sorts by:

1. safety-capped `final_score`;
2. uncapped `pre_cap_score`;
3. proof length.

Eligibility and rejection still use `final_score`; `pre_cap_score` is only a
tie-breaker after the safety decision. Historical retention also continues to
prefer the earlier proof on an exact score tie. This keeps the validated-low
guard while preserving useful verifier evidence inside a capped group.

The remaining end-to-end test is whether historical retention plus this
shortlist ordering selects the externally strongest proof after round 4.
