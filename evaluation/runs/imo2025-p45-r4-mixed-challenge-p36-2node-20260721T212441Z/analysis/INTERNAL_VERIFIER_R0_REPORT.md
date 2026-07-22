# Round-zero verifier alignment

This analysis compares the mean of eight internal verifier scores with the
two-call GPT-5.6 external grade for every structurally complete, non-cutoff
round-zero proof. It covers 26 P4 candidates and 20 P5 candidates.

| Problem | Matched | Pearson | Spearman | External best / 7 | Best external score among internal top 3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 26 | 0.664 | 0.600 | 5.0 | 5.0 |
| 5 | 20 | 0.578 | 0.441 | 3.0 | 3.0 |

For P4, three initial candidates scored `5/7` externally. Two of them were the
top two candidates by internal verifier mean, so verifier ranking did retain a
credible starting proof. For P5, no initial candidate scored above `3/7`;
there was therefore no credible proof for later ranking or refinement to
recover.

Calibration is not reliable enough for correctness gating. P5 candidate 25
received eight internal `1` scores but only `3/7` from each external grader.
The internal verifier is still directionally useful for ranking, but an
internal strict pass must remain adversarially challenged.

The raw per-candidate internal means and score counts are in
`internal-r0-verifier.json`. The comparison excludes proofs that were not
externally graded because they reached the thinking budget or failed the
visible response parser.
