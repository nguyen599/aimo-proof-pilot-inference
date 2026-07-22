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

Their refinement topology also differs from ours. It keeps a cumulative pool of
the best verified proofs from previous rounds and creates each new proof by
combining four stratified parents with up to three randomly selected non-ideal
reviews per parent. Our current runtime keeps 36 independent candidate lineages:
each refinement receives that candidate's own retained proof and critique
history. Porting multi-parent refinement therefore requires a round-level global
pool and synchronization barrier; it is not a prompt-only change.

## Comparison

Our existing `llm_tournament` is a single-elimination bracket over the whole
admitted pool. It covers all retained histories cheaply, but one bad selector
vote can permanently eliminate a strong proof. The teammate selector spends
more calls only where the verifier is saturated and averages position and vote
noise instead of propagating one comparison.

The absolute threshold cannot be copied unchanged. Their `0.95` is applied to
mean verifier score, while our production rank uses a stricter meta-aware
`final_score` and caps candidates with validated critiques at `0.5`. In the
measured IMO 2025 round-three pool, four P4 candidates were tied at that cap and
externally strong proofs appeared below it. The treatment therefore uses a
`0.5` saturation threshold on our capped score while preserving the teammate
ballot schedule. The treatment additionally forces that balanced schedule over
the top ten internally ranked proofs: the known externally perfect P4 history
ranked seventh internally and would otherwise remain outside every ballot.
This is a score-scale and admission calibration, not a relaxation of proof
eligibility: all histories are admitted separately by the treatment's `0.0`
selector floor. Generic stratified selection keeps the teammate-compatible
saturation gate unless this override is enabled explicitly.

The teammate result does **not** establish that its checkpoint is better for our
IMO 2025 P4/P5 workload. Its checkpoint comparison used different problems, and
the reported deploy/step-225 difference was within grader noise. Search width,
checkpoint, and selection must remain separate factors.

Multi-parent refinement is also deferred from the first treatment. Adding it at
the same time would change candidate ancestry, verifier inputs, request volume,
and completion timing, preventing attribution of any score change to the new
selector or checkpoint.

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
