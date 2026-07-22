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

Every repair and reconstruction prompt now begins with a compact indexed repair
ledger derived from the meta-validated verifier findings. Each item keeps the
identified defect and requested fix separate. The refiner must prove the fix,
remove the dependency, or concretely refute the critique, then report an
indexed status in its self-evaluation. This targets the observed P4 failure in
which the verifier correctly found a missing transition invariant but the next
proof merely repeated the unsupported descent claim. The response XML schema
is unchanged.

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

The follow-up baseline pool made the missing obligations more specific: P4
peaked at `5/7` because its one-step descent claim was never proved closed under
the divisor transition, while P5 peaked at `3/7` because its large-lambda
strategy replaced arbitrary Bazza play by budget saturation. The opt-in
`adaptive` portfolio routes problems by their prompt-visible structure. Exact
fingerprints give IMO 2025 P4/P5 a targeted cycle while preserving generic
cycles for other problems:

| Problem structure | 12-candidate cycle |
| --- | --- |
| IMO 2025 P4 | 2 baseline, 4 orbit normal form, 2 backward divisibility, 2 transition classification, 1 counterexample audit, 1 independent reformulation |
| IMO 2025 P5 | 2 baseline, 4 paired threshold proof, 2 Alice Cauchy/spike, 2 Bazza pairing/slack, 1 regime completeness, 1 independent reformulation |
| Other adversarial game | 2 baseline, 3 quantifier/history, 3 joint-state inequality, 2 proof-obligation ledger, 1 regime completeness, 1 independent reformulation |
| Iterated sequence | 3 baseline, 3 exhaustive transition, 2 state invariant, 2 proof-obligation ledger, 1 counterexample audit, 1 independent reformulation |
| Other | Existing eight-slot `diverse` cycle |

The P4 treatment follows the complete proof architecture: establish
divisibility using backward implications and a closed descent argument,
classify all transitions on multiples of six, eliminate the parity-breaking
branch, and prove the growth branch occurs only finitely often before deriving
and checking the initial-value parameterization. The P5 treatment pairs moves:
Cauchy--Schwarz controls arbitrary even moves for Alice, while Bazza fills each
pair's remaining square budget and tracks the resulting linear slack. It also
requires both non-losing strategies at equality. These are private planning
instructions; the trained XML output contract is unchanged. `baseline` and
`diverse` retain their previous behavior.

The next controlled comparison should use:

```text
AIMO_PROOF_GENERATION_ONLY=true
AIMO_PROOF_GENERATION_STRATEGY_PORTFOLIO=adaptive
```

Do not promote `adaptive` to the launcher default unless two-call external
grading improves the best-of-36 score or produces a correct candidate without
materially reducing the structurally complete proof rate.

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
  --proof-versions initial-final \
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

`--proof-versions initial-final` emits a paired grader record before and after
the verification/refinement loop for every completed candidate. The final
proof is the rollback-aware proof actually offered to the selector. Keep
`--proof-versions final` when only final-candidate grading is required.

After grading the paired records, reduce them into stage gates with:

```bash
python evaluation/analyze_paired_candidate_grades.py \
  --candidate-manifest evaluation/runs/<run>/grader_input/all_final_candidates/candidate_manifest.jsonl \
  --grader-summary evaluation/runs/<run>/grading-paired/summary.json \
  --output-dir evaluation/runs/<run>/paired-analysis
```

The paired report separates the initial candidate-pool ceiling from refinement
gain or regression, strict-pass calibration, and selector ranking. It reports
both a configurable credible-proof threshold (default `5/7`) and the stronger
count of candidates receiving `7/7` from every independent grader call.
