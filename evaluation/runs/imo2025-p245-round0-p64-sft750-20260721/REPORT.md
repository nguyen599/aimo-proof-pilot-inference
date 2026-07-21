# IMO 2025 round-0 proof-generation quality

Each of problems 2, 4, and 5 has 64 independent round-0 proof calls. A candidate is graded only when `usage.thinking_budget_applied` is false and the production parser accepts its final visible XML response. This is a structural completeness check; GPT-5.6 performs the mathematical grading.

| Problem | Total | No budget | No-budget rate | Parseable complete | Graded | Avg / 7 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 30 | 46.9% | 30 | 30 | 0.000 |
| 4 | 64 | 47 | 73.4% | 45 | 45 | 2.489 |
| 5 | 64 | 39 | 60.9% | 39 | 39 | 1.385 |

Overall graded-candidate average: 1.456/7.

All 228 expected grader calls completed successfully with no missing or duplicate
candidate-attempt pairs. Three P4 calls initially returned HTTP 402 from a
depleted account; they were rerouted to the two funded accounts and completed.
P5 used 16-way concurrency, with eight calls assigned to each funded account.

## GPT-5.6 call-score distribution

Two independent grader calls were made for every eligible candidate.

| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 12 | 18 | 17 | 18 | 8 | 16 | 1 | 0 |
| 5 | 10 | 41 | 16 | 9 | 2 | 0 | 0 | 0 |

The exact per-candidate two-call scores and half-point mean distribution are in `analysis/final_summary.json`.
