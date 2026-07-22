# P4/P5 intermediate round-one audit

This audit grades every complete round-one refinement that had eight follow-up
verifier calls when the four-round run was sampled. Each proof received two
independent `openai/gpt-5.6-sol` grades under the official IMO 2025 rubric.
All 22 calls completed successfully with no retries or invalid responses.

Eight of the eleven refinements have an externally graded initial proof from
the same run and therefore support a direct before/after comparison.

| Candidate | Initial / 7 | Round one / 7 | Delta | Grader calls |
| --- | ---: | ---: | ---: | --- |
| P4 c13 | 2.0 | 5.0 | +3.0 | 5, 5 |
| P4 c14 | 2.0 | 2.0 | 0.0 | 2, 2 |
| P4 c21 | 5.0 | 4.5 | -0.5 | 4, 5 |
| P4 c27 | 2.0 | 4.0 | +2.0 | 5, 3 |
| P4 c29 | 5.0 | 4.5 | -0.5 | 5, 4 |
| P4 c31 | 2.0 | 4.0 | +2.0 | 5, 3 |
| P4 c35 | 1.0 | 1.0 | 0.0 | 1, 1 |
| P5 c11 | 2.0 | 3.5 | +1.5 | 3, 4 |

Across the paired set, mean score rises from 2.625 to 3.563, a gain of 0.938
points. The seven paired P4 proofs rise from 2.714 to 3.571: three improve,
two are unchanged, and two regress. The only paired P5 proof improves from
2.0 to 3.5, which is encouraging but too small a sample for a rate estimate.

Three refinements do not have a gradeable initial proof because the initial
call reached its thinking budget or did not produce a parseable final proof.
Round one nevertheless produced complete outputs: P4 c7 scored 2, P4 c8
scored 5, and P5 c1 scored 1. P4 c8 demonstrates that refinement can recover
a credible proof from an initially unusable trajectory.

The result establishes that refinement is a real repair mechanism, but it also
exposes a retention problem. Both P4 proofs that started at 5/7 fell to 4.5,
while lower-scoring starts sometimes gained two or three points. A subsequent
round should therefore preserve already-supported lemmas and target only the
specific verifier-identified gap instead of freely reconstructing every proof.
Internal verifier score is useful for routing but remains too noisy to serve as
the sole acceptance gate: in the earlier two-proof sample it ranked the 4/7
refinement above the 5/7 refinement.

The reproducible candidate manifest and all grader responses are in
`intermediate-r1-grading-expanded/`.
