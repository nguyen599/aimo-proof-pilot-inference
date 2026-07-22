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
Their early selector also lost 5--9 of 16 ballots when reasoning consumed the
entire completion budget. Force-closing reasoning after 56,000 tokens with
`</think>\n\n<selected_id>` and reserving about 2,100 continuation tokens made
all 16 ballots parse. Our selector now exposes the same intervention as an
opt-in setting; legacy modes keep it disabled by default.

Their refinement topology also differs from ours. It keeps a cumulative pool of
the best verified proofs from previous rounds and creates each new proof by
combining four stratified parents with up to three randomly selected non-ideal
reviews per parent. Our current runtime keeps 36 independent candidate lineages:
each refinement receives that candidate's own retained proof and critique
history. Porting multi-parent refinement therefore requires a round-level global
pool and synchronization barrier; it is not a prompt-only change.

That topology difference is not yet the measured bottleneck on our P4/P5 run.
Our independent-lineage round-three snapshot already contained a GPT-5.6-graded
`7/7` P5 proof, three additional `6/7` P5 proofs, and a retained `7/7` P4 proof
from round two. The teammate's 64-wide run likewise found a `7/7` P5 proof, but
this does not establish that its multi-parent ancestry was necessary. Both
pipelines can create a correct proof; the unresolved end-to-end question is
whether the final selector finds the correct proof in a verifier-saturated pool.

The prior 64-wide SFT750 round-zero benchmark does not measure the adaptive
P4/P5 planning portfolio. All 84 externally graded P4/P5 candidates in that run
used the baseline trained prover prompt. It established baseline pool quality
(`2.489/7` mean and `5/7` best for P4; `1.385/7` mean and `3.5/7` best for P5),
but it cannot support a claim that the newer problem-specific planning prompts
help or hurt. The adaptive same-input checkpoint gate is also the first direct
measurement of those prompts.

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
selector or checkpoint. It remains an opt-in follow-up only if the checkpoint
and tournament-controlled run fails to put a correct proof in the pool, or if a
36-versus-64 width comparison shows that independent lineages cannot convert
additional initial diversity into stronger later proofs.

## Test plan

1. Finish and externally grade the current 36-wide baseline finals.
2. Run the adaptive round-zero SFT750 versus step-225 checkpoint gate on the
   same P4/P5 input. This simultaneously establishes adaptive-prompt pool quality
   for each checkpoint, but checkpoint is the only treatment difference.
3. Use the winning checkpoint with the opt-in `llm_stratified_tournament`
   selector in the four-round treatment, retaining current and historical
   proofs.
4. Grade every treatment final twice with the same external grader and compare
   it with the baseline final and with the best proof known to exist in each
   candidate pool.
5. If the final remains below the pool maximum, fix selection. If no correct
   proof exists in the pool, run a 36-versus-64 width A/B. If width improves the
   maximum but four independent repair rounds do not, then port cumulative
   four-parent refinement as a separate topology treatment.

Legacy `llm`, `llm_tournament`, and `score` modes remain unchanged.

## Same-pool selector replay

Use `export_pipeline_candidates.py` once a distributed run is complete, then
replay selector treatments without regenerating or reverifying any proof:

```bash
python evaluation/export_pipeline_candidates.py \
  --run-dir /path/to/completed-run \
  --rubrics-file imo-2025.parquet \
  --output-dir /path/to/completed-run/selector-replay-input \
  --problem-ids 4 5

python evaluation/replay_pipeline_selector.py \
  --candidate-dir /path/to/completed-run/selector-replay-input \
  --output-dir /path/to/completed-run/selector-replay-wide \
  --base-url http://127.0.0.1:8000/v1 \
  --model proof-model \
  --tokenizer-path /path/to/model \
  --selector-mode llm_stratified_tournament \
  --selector-tournament-group-size 4 \
  --selector-tournament-rounds 64 \
  --selector-tournament-max-candidates 10 \
  --selector-tournament-threshold 0.5 \
  --selector-tournament-force-wide-pool \
  --selector-max-new-tokens 58100 \
  --selector-thinking-budget-tokens 56000
```

The replay writes grader-ready `records.jsonl` and `rubrics.jsonl`, complete
ballot metadata in `selector_results.jsonl`, and raw selector calls under
`llm_calls/`. Candidate export preserves both capped `final_score` and uncapped
`pre_cap_score`; older exports without the latter remain compatible by falling
back to `final_score`.
Replay streams responses by default because the selector thinking cutoff is a
client-side streaming intervention. Pass
`--selector-thinking-budget-tokens 0` to disable it for a legacy-control
replay.
