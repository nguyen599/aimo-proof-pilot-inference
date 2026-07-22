# P4 same-pool tournament-selector replay

This replay tests the teammate-style stratified tournament selector on the
completed P4 candidate pool from the baseline run. It does not regenerate,
verify, or refine any proof, so the only changed variable is final selection.

## Result

The tournament selector made the result worse and should not replace the
current selector.

| Selector | Candidate | GPT-5.6 calls | Mean / 7 |
| --- | --- | --- | ---: |
| Baseline | attempt 11 | 7, 6 | 6.5 |
| 64-ballot tournament | attempt 34 | 3, 4 | 3.5 |

The replay selected `p4-c34-final`, changing away from the original candidate.
All 64 ballots parsed successfully, so this was not caused by malformed or
missing selector outputs. Candidate 34 won 10 of 26 appearances, while the
original candidate 11 won 6 of 26. The winning proof had internal final score
`0.5`; pairwise votes overrode a substantially better externally graded proof.

## Configuration

- 35 completed candidate proofs; 10-candidate forced-wide pool.
- Four proofs per ballot, 64 independently shuffled ballots.
- Balanced appearances: each pool candidate appeared 25 or 26 times.
- SFT-750 selector checkpoint with FP8 model/KV and TP2/DP2.
- Selector thinking cutoff at 56,000 tokens with a forced structured close.
- External grading used `openai/gpt-5.6-sol`, high reasoning, two independent
  calls, arithmetic-mean aggregation, and the official IMO 2025 rubric.
- External grading had zero retries and zero failed calls.

`diagnostics.json` contains the compact ballot counts. `records.jsonl` is the
selected proof, and `grading/` contains the exact external grader outputs.
The full 10 MB ballot trace remains in the NII run directory because it is not
needed to reproduce the score comparison.

## Conclusion

Final selection is not the next bottleneck to optimize. On this same pool, the
baseline already found a near-correct P4 proof, while a much more expensive
tournament demoted it by three points. Keep the baseline selector behavior and
measure checkpoint effects on the initial proof pool next. Multi-parent
refinement remains unproven and should not be ported until the initial-pool A/B
shows whether the teammate checkpoint is actually better.
