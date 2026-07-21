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
  `openai/gpt-5.6-sol` calls per proof, arithmetic mean, six concurrent calls
  spread across three API keys.

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
