# IMO 2025 P4/P5 refinement experiment

## Evidence

The 64-candidate round-zero benchmark shows that the two problems fail for
different reasons:

| Problem | Non-cutoff proofs | Structurally complete | Mean external score | Best observed score |
| --- | ---: | ---: | ---: | ---: |
| P4 | 47/64 | 45/64 | 2.489/7 | 5/7 |
| P5 | 39/64 | 39/64 | 1.385/7 | 3.5/7 |

P4 already produces useful proof skeletons and can benefit from targeted
repair. P5 needs more independent reconstruction: the four-round baseline
accepted an invalid round-zero game proof at internal score 1.0, while the
external grader assigned it 0.1875/7. Its central strategy claims were not
proved against arbitrary opponent responses.

## Refinement policy

`run.py` supports three refinement strategies:

- `repair`: preserve the existing critique-driven repair prompt.
- `reconstruct`: treat the proof and reviews as fallible notes and re-solve
  from first principles.
- `mixed`: route even candidate indexes to repair and odd indexes to
  reconstruction, preserving both local improvement and approach diversity.

A configurable strict-pass challenge addresses verifier false positives. A
proof that receives an internal strict pass may receive one shadow
reconstruction round. The original remains selected unless the reconstruction
also earns a strict pass under a fresh verifier pass. This rollback rule keeps
the known correct P4 proof while allowing P5 false passes to be challenged.

The NII launchers default to:

```text
AIMO_REFINEMENT_STRATEGY=mixed
AIMO_STRICT_PASS_CHALLENGE_ROUNDS=1
```

Legacy callers remain backward compatible because `run.py` defaults to
`repair` and zero strict-pass challenges unless the launcher or CLI opts in.

## Initial-proof portfolio

The external grader shows that refinement cannot recover when every initial
candidate shares the same false lemma. `run.py` therefore has an opt-in
`--proof-generation-strategy-portfolio diverse` mode for OPD-format initial
proofs. Candidate indexes are assigned deterministically in groups of eight:

| Candidate slots | Planning emphasis |
| --- | --- |
| 0-3 | Exact trained baseline prompt |
| 4 | Preserve adversarial quantifiers and prove a strategy against every reply |
| 5 | Prove an exhaustive transition classification, including boundary cases |
| 6 | Try to falsify universal lemmas and worst-case reductions before using them |
| 7 | Reframe the problem independently and prove the weakest essential lemma |

The four baseline slots preserve the current success distribution. The four
specialized slots target the observed failures: P5's unjustified replacement
of an adversary by a maximal move and P4's incomplete divisor-transition case
analysis. The emphasis is inserted as private planning guidance before the
unchanged XML response contract. DeepSeek-format candidates remain baseline
because they use a different trained prompt contract.

The default remains `baseline`. Before promoting `diverse`, run a controlled
proof-generation-only comparison on the same P4/P5 inputs, model, temperatures,
thinking budgets, and candidate count. Grade every structurally complete proof,
then compare complete-proof rate, mean score, best score, and the probability
that at least one candidate reaches each score threshold.

## Evaluation

Use the same P4/P5 candidate count, verifier mix, and token budgets for the
baseline and treatment. Compare:

1. external score distribution of all complete final proofs;
2. best and selected external score per problem;
3. internal strict-pass precision;
4. parse, cutoff, and rollback rates; and
5. whether P4 retains its correct proof while P5 improves beyond the
   round-zero candidate ceiling.

P5 grading uses three API keys with 24 global workers. The grader assigns
slots round-robin, so each key has exactly eight concurrent requests. Keep two
independent grading calls per proof and rerun only missing or failed records.

Export every completed candidate from the distributed rank payloads before
grading. Do not use only rank zero's `results.jsonl`; that file intentionally
contains the selector winner but not the full candidate bodies.

```bash
python evaluation/export_pipeline_candidates.py \
  --run-dir evaluation/runs/<run>/runtime \
  --rubrics-file evaluation/data/imo_2025.parquet \
  --output-dir evaluation/runs/<run>/grader_input/all_final_candidates \
  --problem-ids 4 5
```

The exporter requires a complete rank set, verifies that every configured
candidate attempt is completed, failed, skipped, or cancelled exactly once,
and emits matching `records.jsonl`, `rubrics.jsonl`, and
`candidate_manifest.jsonl`. The manifest preserves initial planning strategy,
handoff/refinement counts, verifier/meta counts, rollback round, internal final
score, and whether the pipeline selector chose the candidate. This makes the
external-grade comparison capable of distinguishing a weak initial pool from
failed refinement, verifier overconfidence, or selector error.
