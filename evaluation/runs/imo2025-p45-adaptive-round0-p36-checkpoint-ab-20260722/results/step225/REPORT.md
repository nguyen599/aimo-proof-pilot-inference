# IMO 2025 round-0 proof-generation quality

Problems 4, 5 each have 36 independent round-0 proof calls using the `adaptive` planning portfolio. A candidate is graded only when `usage.thinking_budget_applied` is false and the production parser accepts its final visible XML response. This is a structural completeness check; GPT-5.6 performs the mathematical grading.

| Problem | Total | No budget | No-budget rate | Parseable complete | Graded | Avg / 7 | Best / 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 36 | 28 | 77.8% | 28 | 28 | 2.357 | 5.0 |
| 5 | 36 | 19 | 52.8% | 19 | 19 | 2.395 | 7.0 |

Overall graded-candidate average: 2.372/7.

## GPT-5.6 call-score distribution

Two independent grader calls were made for every eligible candidate.

| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 14 | 14 | 8 | 9 | 6 | 0 | 0 |
| 5 | 0 | 13 | 11 | 8 | 3 | 0 | 1 | 2 |

The exact per-candidate two-call scores and half-point mean distribution are in `analysis/final_summary.json`.

## Planning-strategy outcomes

| Problem | Strategy | Total | Complete | Graded | Avg / 7 | Best / 7 | >=6 | 7 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | baseline | 6 | 5 | 5 | 1.100 | 2.5 | 0 | 0 |
| 4 | p4_orbit_normal_form | 12 | 10 | 10 | 2.450 | 5.0 | 0 | 0 |
| 4 | p4_backward_divisibility | 6 | 5 | 5 | 2.200 | 4.0 | 0 | 0 |
| 4 | p4_transition_classification | 6 | 4 | 4 | 3.625 | 4.0 | 0 | 0 |
| 4 | counterexample_audit | 3 | 2 | 2 | 3.250 | 3.5 | 0 | 0 |
| 4 | independent_reformulation | 3 | 2 | 2 | 2.000 | 3.0 | 0 | 0 |
| 5 | baseline | 6 | 2 | 2 | 1.500 | 2.0 | 0 | 0 |
| 5 | p5_threshold_pairing | 12 | 6 | 6 | 2.333 | 5.0 | 0 | 0 |
| 5 | p5_alice_cauchy_spike | 6 | 5 | 5 | 3.500 | 7.0 | 1 | 1 |
| 5 | p5_bazza_pairing | 6 | 4 | 4 | 1.750 | 3.0 | 0 | 0 |
| 5 | game_regime_completeness | 3 | 2 | 2 | 2.000 | 3.0 | 0 | 0 |
| 5 | independent_reformulation | 3 | 0 | 0 | n/a | n/a | 0 | 0 |
