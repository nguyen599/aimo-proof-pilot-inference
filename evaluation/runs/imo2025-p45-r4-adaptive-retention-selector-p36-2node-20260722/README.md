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
- a `0.0` selector floor and a saturation-aware stratified selector over every
  current candidate plus up to 36 earlier verified proof versions, initially
  taking at most one version per candidate. This prevents the noisy internal
  ranking from excluding the known externally strongest P4 history. If more
  than four candidates have capped internal score at least `0.5`, the selector
  runs 64 balanced, independently shuffled four-proof ballots over the best ten
  in that saturated band. Otherwise it runs 16 shuffled ballots over at most four
  proofs within 20% of the best internal score. Failed or malformed ballots are
  null votes, and ties fall back toward the better internal rank. The selector
  uses temperature `0.3`. The ballot schedule matches the tournament-selector
  branch in the teammate pipeline, but the saturation threshold is calibrated
  to this runtime's stricter meta-aware score. The teammate's `0.95` threshold
  applies to mean verifier scores; here validated critiques cap broad groups at
  `0.5`. In the measured round-three P4 pool four candidates were already tied
  at that cap, while externally strong proofs scored as low as `0.375`
  internally. Copying `0.95` would therefore disable the saturation path. This
  remains an opt-in mode; the legacy one-shot and elimination-tournament modes
  are unchanged. The
  exact meta-aware baseline reconstruction assigned internal
  scores `0.400` and `0.312` to P5 proofs graded `6.0` and `5.5` by GPT-5.6,
  so the former `0.5` floor removed the strongest known proof before the
  selector could compare it. The stratified ballots also avoid making one
  noisy comparison or deterministic top-four ranking the sole admission gate;
  and
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

## Teammate pipeline evidence

The teammate `feature/tournament-selector` branch reports that its step-225 OPD
checkpoint and deploy checkpoint are statistically tied on the graded IMO 2026
run, while increasing proof width from 32 to 64 per round produced a perfect P5
proof after one refinement round. It also found that the self-verifier became
anti-correlated with external quality inside a saturated `0.95`-to-`1.0` band.
That supports the selector design above, but it does not establish a checkpoint
win on IMO 2025. Checkpoint, search width, and selector must therefore be tested
as separate factors on the same P4/P5 inputs and graded with the same external
grader before replacing this run's model or candidate count.
