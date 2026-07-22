# Intermediate round-3 proof quality

This snapshot grades every structurally valid, non-cutoff round-3 refinement
that had finished by the export time. It is not the final selector output.

## Grading setup

- Grader: `openai/gpt-5.6-sol`, high reasoning
- Aggregation: arithmetic mean of two independent calls per proof
- Routing: three API keys, 24 global slots, eight slots per key
- Completed: 88/88 calls; no failed or invalid responses
- Cost: $18.1655

## Proof-pool quality

| Problem | Proofs | Mean / 7 | Best / 7 | Score distribution |
|---|---:|---:|---:|---|
| P4 | 27 | 4.389 | 6.0 | 2:1, 2.5:1, 3:3, 3.5:2, 4:5, 4.5:1, 5:10, 5.5:1, 6:3 |
| P5 | 17 | 3.029 | 7.0 | 1:4, 1.5:1, 2:4, 3:2, 3.5:2, 6:3, 7:1 |

The strongest P5 proof is `p5-c35-r3`, graded `[7, 7]`. Three additional P5
proofs scored 6.0. The strongest P4 round-3 proofs are `p4-c02-r3`,
`p4-c16-r3`, and `p4-c22-r3`, each graded `[6, 6]`. P4's historical
`p4-c02-r2` proof was graded `[7, 7]`, so selecting only the latest refinement
would discard the best known P4 proof.

## Paired refinement effect

Only candidates graded at both rounds are included here.

| Problem | Paired proofs | Round 2 mean | Round 3 mean | Mean delta | Improved / same / regressed |
|---|---:|---:|---:|---:|---:|
| P4 | 13 | 4.308 | 4.423 | +0.115 | 5 / 5 / 3 |
| P5 | 10 | 3.600 | 3.800 | +0.200 | 5 / 2 / 3 |

Refinement has a small positive mean effect but high variance. P4 candidate 2
fell from 7.0 to 6.0. P5 candidate 13 fell from 4.0 to 1.5 and candidate 25
fell from 4.0 to 1.0. Conversely, P5 candidates 29, 15, and 35 improved by
3.0, 2.0, and 1.5 points. The pipeline must retain historical versions while
continuing to refine.

## Internal-evaluation completion

At snapshot time, 18 candidates had all eight verifier and eight meta files.

| Problem | Internally complete mean | Incomplete mean | Best complete | Best incomplete |
|---|---:|---:|---:|---:|
| P4 | 4.167 (n=9) | 4.500 (n=18) | 5.5 | 6.0 |
| P5 | 3.722 (n=9) | 2.250 (n=8) | 7.0 | 3.5 |

Completion status is not itself a quality score for P4. For P5 it is useful
at this snapshot, but earlier exact-rank analysis showed weak correlation
between internal and external ordering. It should support selection, not
replace direct problem-obligation auditing.

## Pipeline consequence

The candidate pool is no longer the first bottleneck: it contains a 7/7 P5
proof at round 3 and a retained 7/7 P4 proof at round 2. The critical next test
is whether final selection preserves and chooses those proofs after four
rounds. The treatment branch therefore keeps historical proof versions and
adds P4/P5-specific completion gates to the selector. More refinement rounds
alone are not justified by this evidence.

Raw inputs are in `intermediate-r3-live-input/`; full grader responses and the
machine-readable summary are in
`intermediate-r3-grading-gpt56-c24-r2-newkeys/`.
