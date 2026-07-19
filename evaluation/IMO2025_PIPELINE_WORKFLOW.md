# IMO 2025 proof pipeline workflow

This note records the production workflow for
`imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z` and
maps it to the implementation in `evaluation/harness_vllm/run.py`.

## Production shape

- Two NII nodes split 36 candidate indices by `attempt_idx % world_size`.
- Each node ran TP=2 and DP=4 across eight H200 GPUs.
- Each DP replica admitted 32 scheduled sequences; the node controller admitted
  256 HTTP requests.
- Six candidates used the DeepSeek-Math-V2 prompt family and 30 used the OPD
  XML prover prompt.
- Proof, verifier, and meta calls allowed 126,000 generated tokens in a
  262,144-token model context.
- Every parsed proof received four verifier calls and one meta-verifier call per
  verifier. Up to four refinement rounds used two validated critiques.
- Final selection used an LLM over candidates with internal score above 0.5.
  Each proof was clipped to 32,000 characters in the selector prompt.

The exact launch values are preserved in the run's `manifest.json`.

## Actual control flow

1. `run_async` schedules each problem and persists completed records.
2. `process_problem` calls `run_streaming_candidates` and later performs final
   selection.
3. `run_streaming_candidates` starts one asynchronous
   `run_candidate_pipeline` task per candidate assigned to the rank.
4. `generate_single_attempt` creates the initial proof. If an unfinished
   reasoning trace reaches its budget, lossless handoff finalizes the partial
   state, summarizes reusable evidence, and restarts proof generation. These
   restarts do not consume refinement rounds in this run.
5. `run_single_attempt` parses a proof and calls `run_verification_round`.
6. `run_verification_round` launches four independent proof verifiers. With
   `meta_policy=all-reviews`, every parsed verifier result also receives a meta
   review, including verifier score 1.
7. `aggregate_proof_label` multiplies each verifier score by its mean meta score
   and averages the weighted scores. A missing meta score contributes a factor
   of 0.6. Meta validation also controls which low-score critiques may be sent
   to refinement.
8. A refinement call receives the selected validated critiques. The refined
   proof is verified again. The candidate retains the best verification round
   and can roll back when a later refinement scores worse.
9. `candidate_selection_pool` keeps candidates with internal score greater than
   0.5 when any exist. Otherwise it keeps all candidates.
10. `select_best_candidate` asks the same policy model to choose one proof from
    the pool. A parse failure falls back to highest internal score and then
    longest proof.

Candidate pipelines are collected only after their own proof, verifier, meta,
and refinement stages finish. Other candidates continue concurrently.

## Audit method

Run the reducer against the authoritative node-side directory:

```bash
python evaluation/analyze_pipeline_run.py \
  --run-dir /path/to/imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z \
  --grader-summary /path/to/grading-pinference-gpt56-c8-r32-v5/summary.json \
  --output-dir /path/to/pipeline-analysis
```

The reducer reads all per-rank problem payloads and all raw `llm_calls` files.
It writes `analysis.json`, `REPORT.md`, `candidates.csv`, `calls.csv`, and
`failures.csv`. The decisive comparisons are:

- candidate-generation failure and valid-proof rates;
- internal score ceiling per problem;
- selected candidate's internal rank and external rubric grade;
- verifier/meta score trajectories before and after refinement;
- rollback, handoff, and selector-truncation rates;
- prompt-family completion and score differences;
- stage call volume, token use, failures, and budget interventions.

An external grade exists only for the selected proof. Candidate-level quality
comparisons therefore use internal scores; calibration conclusions require a
selected proof whose internal score and external grade disagree.
