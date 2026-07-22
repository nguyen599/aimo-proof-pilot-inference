# Intermediate round-two proof quality

Every stage score is the arithmetic mean of two independent GPT-5.6 rubric grades. Paired deltas compare the same candidate.

| Problem | R2 n | R2 avg / 7 | R2 best / 7 | Initial pairs | Initial->R2 | R1 pairs | R1->R2 | R1->R2 gain/tie/loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 14 | 4.179 | 7.0 | 14 | +1.571 | 8 | +0.250 | 2/3/3 |
| 5 | 10 | 3.600 | 6.0 | 10 | +1.750 | 2 | +0.000 | 1/0/1 |

## Candidate-level paired scores

| Problem | Candidate | Initial | R1 | R2 | Initial->R2 | R1->R2 |
|---:|---:|---:|---:|---:|---:|---:|
| P4 | 2 | 4.5 | n/a | 7.0 | +2.5 | n/a |
| P4 | 3 | 0.0 | n/a | 3.5 | +3.5 | n/a |
| P4 | 7 | 1.0 | 2.0 | 2.0 | +1.0 | +0.0 |
| P4 | 8 | 0.5 | 5.0 | 5.0 | +4.5 | +0.0 |
| P4 | 12 | 2.5 | n/a | 4.0 | +1.5 | n/a |
| P4 | 13 | 2.0 | 5.0 | 5.0 | +3.0 | +0.0 |
| P4 | 17 | 4.5 | n/a | 5.0 | +0.5 | n/a |
| P4 | 20 | 4.0 | n/a | 4.5 | +0.5 | n/a |
| P4 | 21 | 5.0 | 4.5 | 4.0 | -1.0 | -0.5 |
| P4 | 27 | 2.0 | 4.0 | 5.0 | +3.0 | +1.0 |
| P4 | 29 | 5.0 | 4.5 | 4.0 | -1.0 | -0.5 |
| P4 | 31 | 2.0 | 4.0 | 2.0 | +0.0 | -2.0 |
| P4 | 33 | 2.5 | n/a | 2.5 | +0.0 | n/a |
| P4 | 35 | 1.0 | 1.0 | 5.0 | +4.0 | +4.0 |
| P5 | 1 | 1.0 | 1.0 | 1.5 | +0.5 | +0.5 |
| P5 | 2 | 1.0 | n/a | 4.0 | +3.0 | n/a |
| P5 | 3 | 3.0 | n/a | 1.0 | -2.0 | n/a |
| P5 | 11 | 2.0 | 3.5 | 3.0 | +1.0 | -0.5 |
| P5 | 13 | 2.0 | n/a | 4.0 | +2.0 | n/a |
| P5 | 15 | 1.0 | n/a | 4.0 | +3.0 | n/a |
| P5 | 21 | 1.0 | n/a | 6.0 | +5.0 | n/a |
| P5 | 25 | 3.0 | n/a | 4.0 | +1.0 | n/a |
| P5 | 29 | 1.5 | n/a | 3.0 | +1.5 | n/a |
| P5 | 35 | 3.0 | n/a | 5.5 | +2.5 | n/a |

Full half-point score distributions are in `summary.json`.

## Interpretation

Round two demonstrates that refinement can create a strong proof pool. P4
candidate 2 received `7,7`, and P5 candidate 21 received `6,6`. The main risk
is therefore no longer only initial proof quality: retention and final
selection must preserve these candidates through later rounds.

The recurring P4 defect is the necessity argument for even terms not divisible
by 3. Many proofs show one decreasing step but do not prove that the orbit
cannot later enter a viable multiple-of-6 class. The complete route in
candidate 2 closes this invariant before applying the `12 -> 13` descent.

The recurring P5 defect is quantification over arbitrary opponent play. Many
proofs establish Alice's upper-regime strategy only when Bazza exhausts his
quadratic budget. Candidate 21 uses the cumulative Cauchy-Schwarz bound and is
nearly complete; its remaining loss is the equality-case non-losing argument.

The paired round-one comparison also justifies conservative retention. Four of
ten paired candidates regressed in round two, including P4 candidate 31 by two
points, even though the aggregate pool improved. Future runs should replace an
earlier version only on strictly stronger verified evidence and should keep
the best boundary-scoring proof available to the selector.
