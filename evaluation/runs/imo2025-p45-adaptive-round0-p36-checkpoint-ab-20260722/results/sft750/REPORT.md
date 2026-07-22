# IMO 2025 round-0 proof-generation quality

Problems 4, 5 each have 36 independent round-0 proof calls using the `adaptive` planning portfolio. A candidate is graded only when `usage.thinking_budget_applied` is false and the production parser accepts its final visible XML response. This is a structural completeness check; GPT-5.6 performs the mathematical grading.

| Problem | Total | No budget | No-budget rate | Parseable complete | Graded | Avg / 7 | Best / 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 36 | 33 | 91.7% | 33 | 33 | 2.773 | 5.5 |
| 5 | 36 | 26 | 72.2% | 26 | 26 | 2.538 | 5.0 |

Overall graded-candidate average: 2.669/7.

## GPT-5.6 call-score distribution

Two independent grader calls were made for every eligible candidate.

| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6 | 10 | 18 | 5 | 14 | 12 | 1 | 0 |
| 5 | 1 | 17 | 9 | 9 | 11 | 4 | 1 | 0 |

The exact per-candidate two-call scores and half-point mean distribution are in `analysis/final_summary.json`.

## Planning-strategy outcomes

| Problem | Strategy | Total | Complete | Graded | Avg / 7 | Best / 7 | >=6 | 7 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | baseline | 6 | 4 | 4 | 2.625 | 4.5 | 0 | 0 |
| 4 | p4_orbit_normal_form | 12 | 12 | 12 | 3.208 | 5.5 | 0 | 0 |
| 4 | p4_backward_divisibility | 6 | 6 | 6 | 3.083 | 5.0 | 0 | 0 |
| 4 | p4_transition_classification | 6 | 6 | 6 | 2.583 | 4.0 | 0 | 0 |
| 4 | counterexample_audit | 3 | 2 | 2 | 2.500 | 3.0 | 0 | 0 |
| 4 | independent_reformulation | 3 | 3 | 3 | 1.167 | 2.5 | 0 | 0 |
| 5 | baseline | 6 | 4 | 4 | 1.500 | 3.0 | 0 | 0 |
| 5 | p5_threshold_pairing | 12 | 9 | 9 | 2.278 | 4.0 | 0 | 0 |
| 5 | p5_alice_cauchy_spike | 6 | 6 | 6 | 3.833 | 5.0 | 0 | 0 |
| 5 | p5_bazza_pairing | 6 | 4 | 4 | 2.375 | 4.0 | 0 | 0 |
| 5 | game_regime_completeness | 3 | 1 | 1 | 2.000 | 2.0 | 0 | 0 |
| 5 | independent_reformulation | 3 | 2 | 2 | 2.500 | 4.0 | 0 | 0 |
