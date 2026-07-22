# P4 intermediate round-one check

This check externally grades the first two P4 candidates for which a complete
round-one refinement and eight follow-up verifier calls were available while
the four-round run was still active. Each proof received two independent
`openai/gpt-5.6-sol` grades under the IMO 2025 P4 rubric.

| Candidate | Initial external / 7 | Round-one external / 7 | Delta | Internal verifier mean r0 -> r1 |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 2.0 | 5.0 | +3.0 | 0.0000 -> 0.1875 |
| 31 | 2.0 | 4.0 | +2.0 | 0.3125 -> 0.6875 |

The first refinement materially improved both sampled proofs. Refinement is
therefore capable of repairing substantive errors and is not ruled out as an
effective stage. The full paired initial/final export is still required to
measure retention through later rounds, detect regressions, and evaluate the
selector.

The raw successful calls and aggregate are in `intermediate-r1-grading/`.
`failures.jsonl` preserves one initial HTTP 402 from a depleted key; the
missing call was rerouted and all four required successful calls completed.
