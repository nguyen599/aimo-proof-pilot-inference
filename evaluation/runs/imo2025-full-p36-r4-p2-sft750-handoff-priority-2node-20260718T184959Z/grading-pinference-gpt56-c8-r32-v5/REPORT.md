# IMO 2025 Grading Report

The selected proof for each problem was graded 32 times with
`openai/gpt-5.6-sol` at high reasoning effort. Each problem score is the
arithmetic mean of its 32 grades.

## Results

| Problem | Mean / 7 | Percent | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.2500 | 46.43% | 0 | 0 | 2 | 20 | 10 | 0 | 0 | 0 |
| 2 | 0.3750 | 5.36% | 26 | 0 | 6 | 0 | 0 | 0 | 0 | 0 |
| 3 | 1.0625 | 15.18% | 7 | 16 | 9 | 0 | 0 | 0 | 0 | 0 |
| 4 | 7.0000 | 100.00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 |
| 5 | 0.1875 | 2.68% | 26 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 1.0000 | 14.29% | 0 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |

The sum of the six problem means is **12.875 / 42 (30.65%)**. The mean
problem score is **2.1458 / 7**.

## Validation

- 192 final records: 32 attempts for each of six problems.
- Every problem contains each attempt index from 0 through 31 exactly once.
- No duplicate records and no terminal grading failures.
- 18 malformed intermediate responses were retained for debugging; retries
  recovered every affected grading call.
- 19 calls required retries, with 22 retries in total.

The run used four API keys with global concurrency 8 and a maximum of two
concurrent calls per key. Reported usage was 748,224 input tokens, 942,884
output tokens, and 865,274 reasoning tokens, for a reported cost of $30.495.
See `summary.json` for complete machine-readable aggregates and
`records.jsonl` for individual grades.
