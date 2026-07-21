# IMO 2025 P2/P4/P5 round-zero benchmark

This run measures initial proof-generation quality before verifier, meta,
refinement, handoff, or selector-model stages can affect a candidate.

- Problems: IMO 2025 P2, P4, and P5.
- Candidates: 64 independent round-zero calls per problem.
- Runtime: two NII nodes, each TP2/DP4 over eight H200 GPUs.
- Generation: OLMo3 OPD SFT-750 with online FP8 and DFlash until 65,536 context.
- Proof limit: 126,000 completion tokens; the production thinking-budget
  controller records cutoff status in `usage.thinking_budget_applied`.
- Grading: only non-cutoff, structurally parseable final proofs; two independent
  `openai/gpt-5.6-sol` calls per proof and arithmetic-mean aggregation. P2/P4
  use six concurrent calls; P5 uses 24, or eight per API key.

If one API account is out of balance, `grader-p5-funded.yaml` preserves eight
parallel calls per funded key while excluding the depleted account. The July
21 run used this fallback after key slot 0 returned HTTP 402.

After generation, prepare the exact grader inputs with:

```bash
python -m evaluation.report_round0_proof_quality prepare \
  --run-dir evaluation/runs/imo2025-p245-round0-p64-sft750-20260721 \
  --rubrics-file evaluation/data/imo_2025.parquet \
  --problem-ids 2 4 5 \
  --expected-candidates 64
```

The final `REPORT.md`, raw LLM calls, runtime/vLLM logs, grader records, and
SHA-256 manifest are committed after the run completes.
