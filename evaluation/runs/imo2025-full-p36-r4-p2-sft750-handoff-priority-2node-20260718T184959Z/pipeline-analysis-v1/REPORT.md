# Proof pipeline bottleneck report

Run: `imo2025-full-p36-r4-p2-sft750-handoff-priority-2node-20260718T184959Z`

## Diagnosis

Primary bottleneck: verifier/selector calibration. The pipeline assigned passing internal scores to selected proofs that an independent rubric grader found fatally incomplete. Generation quality is a secondary bottleneck because many candidates also failed or never crossed the selector threshold.

- 47/216 candidate pipelines failed before selection (21.8%).
- The selector chose a non-top internal score on 0/6 problems.
- 4 selected proofs passed the internal selector threshold but received at most 2/7 from the external grader.
- 0 low-scoring problems had no candidate above the internal selector threshold.

## Problem outcomes

| Problem | Valid / assigned | Selected | Internal | Internal rank | Best internal | External / 7 | Proof chars | Selector clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 24/36 | 7 | 0.625 | 1 | 0.625 | 3.25 | 9000 | no |
| 2 | 31/36 | 17 | 1 | 1 | 1 | 0.375 | 11856 | no |
| 3 | 26/36 | 25 | 0.562 | 1 | 0.562 | 1.062 | 7481 | no |
| 4 | 30/36 | 24 | 1 | 1 | 1 | 7 | 7793 | no |
| 5 | 30/36 | 15 | 1 | 1 | 1 | 0.188 | 6316 | no |
| 6 | 28/36 | 8 | 1 | 1 | 1 | 1 | 4842 | no |

## Candidate health

- Pipeline completion: 169/216 (78.2%); failures: 47.
- Candidates above selector threshold: 50/169 (29.6%).
- Refinement improved the best round score for 79 candidates; 47 candidates rolled back from a later round.
- Thinking-budget restarts occurred in 99 candidates; their mean internal score was 0.27 versus 0.449 without restarts.
- 0 candidate proofs exceeded the selector's character window; 0 selected proofs were clipped.

## Prompt families

| Family | Assigned | Valid | Completion | Mean score | Eligible |
|---|---:|---:|---:|---:|---:|
| deepseek_math_v2 | 36 | 3 | 8.3% | 0.292 | 0 |
| opd | 180 | 166 | 92.2% | 0.345 | 50 |

## LLM stages

| Stage | Calls | Failed | Prompt tokens | Completion tokens | Budget interventions |
|---|---:|---:|---:|---:|---:|
| proof_finalize | 255 | 0 | 31484759 | 755949 | 0 |
| proof_generation | 438 | 0 | 770888 | 48510287 | 281 |
| proof_handoff | 39 | 0 | 4933913 | 143970 | 0 |
| proof_meta_verify | 2120 | 0 | 8708630 | 29291834 | 0 |
| proof_refine | 438 | 0 | 1645929 | 30222158 | 0 |
| proof_verify | 2120 | 0 | 6434912 | 35079830 | 3 |
| selector | 2 | 0 | 6488 | 32122 | 0 |

## Failure reasons

- `invalid_generation`: 47

## Interpretation boundary

The external grader evaluates only the selected proof, not every candidate. Candidate-level claims therefore use internal verifier scores, while causal claims about final quality use the selected proof's external grade. Raw prompts and completions remain in the source run.
