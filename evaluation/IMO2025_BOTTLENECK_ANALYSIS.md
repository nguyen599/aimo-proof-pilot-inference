# IMO 2025 proof-pipeline bottleneck

This note summarizes the full trace audit of
`imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z`.
It covers all 216 assigned candidates and every raw proof, verifier, meta,
refinement, handoff, and selector call.

## Diagnosis

The primary bottleneck is **verifier calibration**, not final selection.

- 169/216 candidate pipelines produced a parsed proof; 47 failed during proof
  generation.
- Only 50/169 valid candidates crossed the internal selector threshold.
- The selector chose an internally top-ranked candidate for all six problems.
- No selected proof was clipped by the selector's 32,000-character limit.
- Internal selected-proof score and external MathArena grade had Pearson
  correlation `0.0217`.
- Four selected proofs scored above `0.5` internally but at most `2/7`
  externally.

Generation is therefore a secondary bottleneck, but selector tuning cannot fix
the dominant error: the verifier confidently promotes invalid proofs.

## Bottleneck priority

1. **Verifier calibration and same-model self-confirmation.** The selected
   internal scores had Pearson correlation `0.0217` with the external grades.
   Problems 3, 5, and 6 retained explicit fatal claims while scoring above the
   selector threshold; Problems 5 and 6 scored `1.0` internally.
2. **Generation validity and token efficiency.** Candidate pipelines completed
   in 169/216 cases. The trained OPD prompt completed in 166/180 cases, but the
   DeepSeek-Math-V2 prompt completed in only 3/36. Thinking-budget restarts were
   also associated with a lower mean internal score (`0.270` versus `0.449`).
3. **Meta-verifier cost without independent evidence.** Meta review consumed
   29.3 million completion tokens but could only assess the stated review, not
   independently certify a proof that a verifier had accepted.
4. **Final selection is ruled out as the first bottleneck.** Every selected
   candidate had internal rank one, and no selected proof was clipped by the
   selector prompt.

The external grader evaluated only the selected proof for each problem. It is
therefore not yet possible to conclude whether proof generation failed to
produce any correct alternative in the other 210 candidates. Candidate-level
external grading or a validated adversarial re-verification pass is required
to separate "no correct proof was generated" from "a correct proof was
generated but ranked below a false positive."

## Failure mechanism

The four verifier calls used the same prompt and model. Refinement then optimized
against their critiques, after which the next verifier round started from
scratch and forgot earlier issues. This allowed polished rewrites to erase the
warning without repairing the underlying claim.

Problem 6 is the clearest example. Its internal trajectory rose from `0.125` to
`0.1875` to `1.0`. The final proof still assumed that arbitrary row and column
permutations preserve rectangular geometry. All four final verifiers praised
that false symmetry; the external grader correctly rejected it because such
permutations do not preserve consecutiveness or rectangles.

Problem 5 received an immediate internal `1.0`, although its proposed game
strategy illegally chose a value independent of Alice's preceding move and
used one cooperative infinite play to infer that neither player has a winning
strategy. Problem 3 similarly retained false power-of-two and divisibility
claims. These are verifier false negatives, not selector mistakes.

Problem 2 is a separate rubric-alignment case. Its coordinate proof may be
mathematically sound, while the external MathArena rubric awarded points only
for a prescribed synthetic checkpoint path. This discrepancy should not be
used as evidence that the proof itself is false.

## Ineffective cost

The meta verifier consumed about 29.3 million completion tokens. Its prompt
checks whether a review's stated defects are reasonable and explicitly does not
independently reject false positive reviews. It therefore cannot certify a
strict pass when a verifier reports no defect.

The first six candidates per problem used the DeepSeek-Math-V2 prompt. Only
3/36 produced valid candidates, versus 166/180 for the trained OPD prompt. None
of the DeepSeek candidates crossed the selector threshold.

Across all stages, initial proof generation used 48.5 million completion
tokens, refinement used 30.2 million, proof verification used 35.1 million,
and meta verification used 29.3 million. Improving verifier precision therefore
affects both correctness and a larger token budget than changing the final
selector.

## Pipeline changes

The corrected verifier path now:

1. assigns the four verifier calls different dependency, counterexample,
   quantifier/algebra, and statement-coverage audit roles;
2. carries meta-validated critiques into every later refinement and verifier
   round, requiring each issue to be marked resolved, unresolved, or invalid;
3. caps the aggregate score at `0.5` when meta review validates any verifier
   score below `1`, while retaining a separate fatal-error marker for score
   `0`; and
4. defaults the DeepSeek-Math-V2 candidate count to zero.

These changes target the observed causal failures. The replay below validates
them on the known false-positive proofs before a new IMO 2026 generation run.

## Adversarial replay validation

The six selected proofs were replayed through the four-role verifier and one
meta review per role. This replay reused the original proof text and did not
generate or refine a new proof.

| Problem | Old internal | Role scores | Old aggregation | Strict aggregation |
| --- | ---: | --- | ---: | ---: |
| 1 | 0.625 | 0, 1, 0.5, 0.5 | 0.375 | 0.375 |
| 2 | 1.0 | 1, 1, 1, 1 | 0.875 | 0.875 |
| 3 | 0.5625 | 0.5, 0.5, 0.5, 0.5 | 0.375 | 0.375 |
| 4 | 1.0 | 1, 1, 1, 1 | 1.0 | 1.0 |
| 5 | 1.0 | 0.5, 1, 1, 1 | 0.875 | 0.5 |
| 6 | 1.0 | 0, 1, 1, 0 | 0.5 | 0.5 |

The dependency audit found Problem 1's repeated-boundary-point gap. Every role
downgraded Problem 3's divisibility proof. The correct Problem 4 proof remained
a unanimous strict pass. Dependency and coverage audits rejected Problem 6's
invalid row/column-permutation symmetry.

Problem 5 isolated the remaining aggregation defect: dependency review found a
gap and meta review validated it, but averaging with three score-1 reviews still
produced `0.875`. Strict aggregation now caps any meta-validated non-perfect
review at `0.5`, so that proof cannot enter the selector pool. The game-strategy
role also now requires legality under arbitrary prior play and rejects using one
cooperative infinite play as proof of a draw.

This replay validates the known selected-proof failure set: all four externally
false proofs are now at or below the selector threshold, while Problem 4 stays
at `1.0`. It does not establish candidate-pool recall because the external
grader did not score the other 210 generated candidates.

## Reproduce the audit

```bash
python evaluation/analyze_pipeline_run.py \
  --run-dir /path/to/imo2025-run \
  --grader-summary /path/to/grading/summary.json \
  --output-dir /path/to/pipeline-analysis
```

See `IMO2025_PIPELINE_WORKFLOW.md` for the corresponding `run.py` control flow.
