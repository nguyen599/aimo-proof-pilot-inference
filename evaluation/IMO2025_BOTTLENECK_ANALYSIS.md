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

## Pipeline changes

The corrected verifier path now:

1. assigns the four verifier calls different dependency, counterexample,
   quantifier/algebra, and statement-coverage audit roles;
2. carries meta-validated critiques into every later refinement and verifier
   round, requiring each issue to be marked resolved, unresolved, or invalid;
3. caps the aggregate score at `0.5` when any verifier score `0` is validated by
   meta review; and
4. defaults the DeepSeek-Math-V2 candidate count to zero.

These changes target the observed causal failures. They still require a replay
on the known false-positive proofs before a new IMO 2026 generation run.

## Reproduce the audit

```bash
python evaluation/analyze_pipeline_run.py \
  --run-dir /path/to/imo2025-run \
  --grader-summary /path/to/grading/summary.json \
  --output-dir /path/to/pipeline-analysis
```

See `IMO2025_PIPELINE_WORKFLOW.md` for the corresponding `run.py` control flow.
