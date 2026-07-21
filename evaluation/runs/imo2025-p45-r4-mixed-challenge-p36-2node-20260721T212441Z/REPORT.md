# IMO 2025 round-0 proof-generation quality

Problems 4 and 5 each have 36 independent round-0 proof calls. A candidate is graded only when `usage.thinking_budget_applied` is false and the production parser accepts its final visible XML response. This is a structural completeness check; GPT-5.6 performs the mathematical grading.

| Problem | Total | No budget | No-budget rate | Parseable complete | Graded | Avg / 7 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 36 | 27 | 75.0% | 26 | 26 | 2.327 |
| 5 | 36 | 21 | 58.3% | 20 | 20 | 1.550 |

Overall graded-candidate average: 1.989/7.

## GPT-5.6 call-score distribution

Two independent grader calls were made for every eligible candidate.

| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 12 | 17 | 5 | 5 | 8 | 0 | 0 |
| 5 | 2 | 20 | 13 | 4 | 1 | 0 | 0 | 0 |

The exact per-candidate two-call scores and half-point mean distribution are in `analysis/final_summary.json`.
