# P4/P5 four-round treatment

This treatment keeps the baseline two-node TP2/DP4 runtime, 36 candidates,
eight mixed-role verifiers, one meta review per verifier, and four mixed
refinement rounds. It changes only the measured P4/P5 interventions:

- adaptive initial-proof strategies for the missing P4 transition-closure and
  P5 game-regime obligations;
- P4/P5-specific mandatory checks for the four specialist verifiers while the
  four generalist prompts remain unchanged;
- the same decisive completion obligations carried into meta-audit, repair,
  and independent reconstruction prompts;
- meta-aware version retention without the duplicate challenge penalty; and
- conservative retention of the earlier proof when two verified versions have
  exactly equal internal evidence, except for strict-pass challenge survival;
- a `0.0` selector floor and an LLM tournament over every current candidate
  plus up to 36 earlier verified proof versions, initially taking at most one
  version per candidate. This prevents the noisy internal ranking from
  excluding the known externally strongest P4 history. Each comparison is
  capped at eight proofs; verifier-ranked seeding spreads the strongest
  internal candidates across groups before a final winner comparison. With 72
  entrants this requires at most 12 selector calls, compared with the thousands
  of verifier/meta calls in the four-round search. The
  exact meta-aware baseline reconstruction assigned internal
  scores `0.400` and `0.312` to P5 proofs graded `6.0` and `5.5` by GPT-5.6,
  so the former `0.5` floor removed the strongest known proof before the
  selector could compare it. The tournament also avoids making one noisy
  deterministic top-eight ranking the sole admission gate; and
- the final selector receives the same P4/P5 completion audit used by the
  verifier and refiner. It must therefore compare the actual closed-descent,
  transition, arbitrary-history, and equality-regime obligations instead of
  choosing from proof prose alone.

Do not launch it while the baseline run
`imo2025-p45-r4-mixed-challenge-p36-2node-20260721T212441Z` owns nodes 2 and 3.
After completion, export and grade every initial/final candidate with the same
two-call GPT-5.6 grader used for the baseline. `grader.yaml` assigns 24 reusable
slots round-robin across three keys, giving each key exactly eight concurrent
calls.

## Prompt audit

The problem-specific hints were checked against externally graded round-three
proofs before launch:

- The P4 orbit hints reproduce every structural step of the proof that received
  `7/7` in all 32 external grading calls in the completed July 18 run: closure
  of the non-`6Z` descent (including the `x=70` divisor-order case), the exact
  `13/12`, `31/30`, and fixed transitions, finite `13/12` iteration, and the
  final valuation parameterization. The modulo-three statements are taken in
  the field modulo 3, where the current state and its relevant divisors are
  invertible.
- The P5 hints reproduce the proof obligations in round-three candidate 35,
  which received `7/7` from both external grading calls: Bazza's pair-filling
  strategy below the threshold, Alice's Cauchy spike against an arbitrary
  Bazza history above it, and separate non-losing strategies for both players
  at equality.
- Selector clipping does not remove either known strong history: P4 candidate 2
  round 2 is 10,879 characters and P5 candidate 35 round 3 is 7,164
  characters, both below the 32,000-character per-candidate selector cap.

This audit establishes that the portfolio prompts point toward known strong
proof structures. It does not replace the round-zero external quality gate or
the final four-round comparison.
