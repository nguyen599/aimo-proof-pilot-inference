# IMO 2026 P4/P5 pipeline checkpoint

Updated: 2026-07-22

## Objective

Improve the quality of the selected IMO 2026 P4 and P5 proofs after four
rounds. Measure the stages separately:

1. whether the initial candidate pool contains at least one correct proof;
2. whether verification and refinement preserve or improve that proof; and
3. whether final selection chooses the strongest proof already in the pool.

No prompt may fingerprint a benchmark statement, include a reference answer,
or inject a problem-specific lemma or proof skeleton. Commit `b40c900` removed
the known answer-bearing prompts. Commit `6d24a8e` additionally made prompt
portfolio assignment independent of the problem text. Candidate diversity now
depends only on candidate index and generic proof disciplines.

## Confirmed teammate findings

The teammate implementation is in `AIMO-Proof-Pilot` main and
`imo-inference` branch `feature/tournament-selector`.

- On IMO 2026, the deploy and OPD step-225 checkpoints were statistically tied
  in the reported four-round grading (`18.16` and `18.19` total). A checkpoint
  change is therefore not yet a demonstrated improvement.
- The 64-proof search found a P5 proof graded `7/7` after one refinement round.
  The 32-proof runs stayed near `5/7` despite greater depth. This is evidence
  that initial/round width is a stronger lever than extra refinement depth for
  P5.
- Strong P4 and P5 proofs existed below the verifier's exact top tie. In the
  reported saturated pools, raw mean verifier score did not reliably order
  external proof quality.
- The teammate refiner uses a cumulative global pool. Every new proof combines
  four stratified parents from the top verified pool and up to three randomly
  selected non-ideal verifier reviews per parent. Our current production path
  instead refines independent candidate lineages.

The teammate tiered selector uses **raw mean verifier score**:

- group size: 4;
- saturation trigger: more than four candidates with mean score at least
  `0.95`;
- saturated pool: top 10 candidates from that band;
- tournament: 64 balanced, independently shuffled four-proof ballots;
- winner: most ballot wins, with verifier rank as the tie-break;
- non-saturated path: repeated votes over at most four candidates whose scores
  are at least `best * (1 - 0.2)`; and
- invalid ballots are null votes, not run failures.

## Current harness comparison

`evaluation/harness_vllm/run.py` implements the same balanced bracket schedule
under `llm_stratified_tournament`, including null ballots and rank-based
tie-breaking. The default remains our existing treatment:

- it ranks and gates candidates using `final_score`, which is meta-weighted and
  can be capped at `0.5` after validated critiques;
- the teammate implementation ranks by the uncapped raw mean of verifier
  scores; and
- `candidate_selection_pool` may remove candidates using
  `selector_min_final_score` before the tournament sees them.

An opt-in compatibility score source is now available:

```text
--selector-mode llm_stratified_tournament
--selector-score-source raw_verifier_mean
--selector-tournament-group-size 4
--selector-tournament-rounds 64
--selector-tournament-max-candidates 10
--selector-tournament-threshold 0.95
--selector-score-window 0.2
```

`raw_verifier_mean` computes the unweighted mean from
`verifier_score_summaries[*].verifier_score`. The selected score source is used
consistently for candidate admission, ranking, the saturation test, the score
window, historical versions, and score fallbacks. Exact score ties preserve
the existing verifier rank. The default `final_score` behavior is unchanged.
`export_pipeline_candidates.py` preserves compact numeric verifier summaries,
and `replay_pipeline_selector.py` accepts the same score-source option, so a
stored-pool replay does not need to regenerate or reverify proofs.

The previous same-pool replay is **not** evidence against the teammate method.
That replay forced a wide top-ten tournament and used our capped meta-aware
score. It selected an IMO 2025 P4 proof graded `3.5/7` instead of the existing
selector's `6.5/7`, but it did not reproduce the teammate saturation gate or
score source. It remains evidence that an unconditional wide tournament is
unsafe.

## Required fair experiment

Use the same model checkpoint, sampling settings, and clean prover prompt in
both arms. Run only IMO 2026 P4 and P5.

### Gate A: initial pool

- Generate 64 independent round-zero proofs per problem.
- Do not run verifier, refiner, or selector.
- Externally grade every structurally complete, non-cutoff proof twice.
- Report eligible rate, mean score, best-of-64, score distribution, and count
  of `7/7` proofs.

If neither P4 nor P5 contains a correct proof, selection cannot solve the
failure. Improve generation width/checkpoint/sampling before spending on four
rounds.

The reproducible one-node launcher is
`scripts/launch_nii_imo2026_p45_gate_a.sh`. Its defaults reserve GPUs 4-7,
use TP2/DP2, run 64 baseline candidates for each of P4 and P5 concurrently,
and disable handoff, verification, refinement, and LLM selection. Set a unique
`AIMO_RUN_ID` before invoking it.

### Gate B: four-round topology

After Gate A establishes pool quality, compare:

1. current independent-lineage refinement and conservative selector; and
2. teammate-compatible cumulative four-parent refinement plus its raw-mean,
   saturation-gated tiered selector.

Both arms must use four total rounds, the same initial 64 proofs, the same
number of verifier calls, and the same external grader. Report, per round:

- externally best proof in the pool;
- externally best proof retained for the next round;
- internally top-ranked proof;
- selected final proof; and
- the score loss from pool-best to selected.

This separates generation, verifier/refinement retention, and selector loss.

## Next code step

The teammate-compatible selector score source is implemented and covered by
focused tests. The remaining topology difference is cumulative multi-parent
refinement. Add that mode behind an opt-in argument so the existing
independent-lineage pipeline remains available as the control. Before spending
on a four-round run, replay both selector score sources on an identical stored
candidate pool and run Gate A on clean 64-proof P4/P5 pools.

## NII snapshot

At the 2026-07-22 check, only the assigned Nguyen relay node was considered for
new work: `node3-hnode755`. GPUs 4-7 were free. GPUs 0-3 still held orphaned
vLLM workers from an earlier Nguyen-owned run, so no cleanup or launch should
touch them without a fresh PID/command audit. Model paths visible on the shared
filesystem included:

- `/tmp/models/olmo3-opd-sft-750-vllm`;
- `/tmp/chankhavu/models/opd-32b-bf16-step-225`; and
- `/tmp/models/dflash-32b-draft-v2test-phaseL`.

Node state is transient and must be rechecked immediately before launch.
