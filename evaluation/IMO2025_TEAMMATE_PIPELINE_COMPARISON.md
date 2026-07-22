# Teammate pipeline comparison for IMO 2025 P4/P5

## Evidence

The teammate `feature/tournament-selector` branch and the merged
`AIMO-Proof-Pilot` main branch report three relevant results on IMO 2026:

- the OPD step-225 and deploy checkpoints are statistically tied in the graded
  four-round setting (`18.19` versus `18.16` total);
- increasing width from 32 to 64 proofs per round produced a `7/7` P5 proof
  after one refinement round, while deeper 32-wide runs remained near `5/7`;
- inside the saturated verifier band, external quality was not ordered by the
  verifier score. The top ten contained several externally perfect proofs that
  a top-four admission rule could not see.

Their selector responds to this saturation by running repeated, independently
shuffled four-proof ballots. If more than four proofs score at least `0.95`, it
runs 64 balanced brackets over at most ten proofs; otherwise it runs 16 votes
over proofs within 20% of the best verifier score. Invalid ballots are ignored,
and ties prefer the better verifier rank. Selector temperature is `0.3`.

## Comparison

Our existing `llm_tournament` is a single-elimination bracket over the whole
admitted pool. It covers all retained histories cheaply, but one bad selector
vote can permanently eliminate a strong proof. The teammate selector spends
more calls only where the verifier is saturated and averages position and vote
noise instead of propagating one comparison.

The teammate result does **not** establish that its checkpoint is better for our
IMO 2025 P4/P5 workload. Its checkpoint comparison used different problems, and
the reported deploy/step-225 difference was within grader noise. Search width,
checkpoint, and selection must remain separate factors.

## Test plan

1. Finish and externally grade the current 36-wide baseline finals.
2. Run the existing adaptive round-zero checkpoint gate on the same P4/P5 input.
3. Use the opt-in `llm_stratified_tournament` selector in the four-round
   treatment, retaining current and historical proofs.
4. Grade every treatment final twice with the same external grader.
5. Only after that comparison, run a same-input checkpoint A/B and then a
   36-versus-64 width A/B if the checkpoint or generation pool remains the
   bottleneck.

Legacy `llm`, `llm_tournament`, and `score` modes remain unchanged.
