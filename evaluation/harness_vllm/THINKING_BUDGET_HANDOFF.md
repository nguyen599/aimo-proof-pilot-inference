# Thinking-budget handoff

When proof generation reaches its thinking budget, `run.py` can use the
remaining refine round as a context reset:

1. Stop the unfinished assistant reasoning without emitting a partial proof.
2. Build a handoff either with a model-written structured summary or by
   preserving the strongest partial-progress report losslessly as untrusted
   evidence.
3. Add a real user turn containing the handoff.
4. Start a fresh proof-generation call from the original problem plus the
   handoff.
5. Optionally give the final restart a smaller thinking budget, force the
   transition to visible proof output at that boundary, and spend the remaining
   completion allowance on the proof.
6. Verify the restarted proof. If no refine round remains, keep the existing
   strongest-partial completion behavior unless a final-round budget is set.

In `model` mode, an invalid handoff is repaired once. If it is still invalid,
the pipeline falls back to the previous strongest-partial completion path. In
`lossless_partial` mode, the bounded strongest-partial report is wrapped
deterministically and no additional summarizer call is made.

## Configuration

```bash
python evaluation/harness_vllm/run.py \
  --thinking-budget-handoff-enabled \
  --thinking-budget-handoff-mode lossless_partial \
  --thinking-budget-handoff-preserve-refine-rounds \
  --thinking-budget-restart-strategy deadline_aware \
  --thinking-budget-final-round-tokens 100000 \
  --thinking-budget-handoff-max-tokens 4096 \
  --thinking-budget-handoff-temperature 0.7 \
  --thinking-budget-handoff-prompt-variant evidence_first
```

Handoff modes are:

- `model`: generate the six-section handoff with the policy model.
- `lossless_partial`: preserve the complete forced partial-progress report
  inside an explicitly untrusted deterministic wrapper.

Restart strategies are `standard` and `deadline_aware`. Prompt variants
`evidence_first`, `lemma_ledger`, and `continuation_frontier` apply to model
handoffs. `--thinking-budget-final-round-tokens 0` preserves the previous final
round behavior; a positive value reserves the rest of `--max-new-tokens` for
visible output. Disable the feature with
`--no-thinking-budget-handoff-enabled`.

By default, each budget restart consumes one configured refinement round to
preserve the original behavior. Enable
`--thinking-budget-handoff-preserve-refine-rounds` to count the restart
separately. With one restart and `--refine-rounds 1`, the restarted proof can
then receive verification, one verifier-guided refinement, and re-verification.
The reported `budget_restart_count` remains the actual number of context
restarts in both modes.

The 100,000-token value above is an experiment configuration for a
126,000-token completion allowance, not a universal ratio. See
[`EXPERIMENT.md`](../runs/thinking-handoff-opt-sft750-20260717/EXPERIMENT.md)
for the measured completion and proof-quality results before adopting it.

Current live evidence does not justify a nonzero production default. On the
same difficult proof, forcing at 100,000 tokens produced valid XML with an
incorrect proof, while forcing at 80,000 or 60,000 tokens exhausted the full
125,000-token allowance without closing `<solution>`. Keep the final-round
budget opt-in and evaluate mathematical correctness separately from parser
validity.

## Prompt optimization

`optimize_thinking_handoff.py` reuses saved long reasoning contexts. It removes
the old forced partial-proof suffix, appends a tokenizer-correct user turn, and
calls only the short handoff stage. It does not regenerate the original proof.

The default experiment evaluates all three prompt variants at temperatures
`1.0`, `0.7`, and `0.6` over eight balanced saved cases:

```bash
python evaluation/harness_vllm/optimize_thinking_handoff.py \
  --logs-root evaluation/runs/imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717/logs \
  --model-path /tmp/models/olmo3-opd-sft-750-vllm \
  --base-url http://127.0.0.1:8000 \
  --base-url http://PEER_NODE_IP:8000 \
  --output-dir /tmp/thinking-handoff-optimization
```

The optimizer requires the canonical re-tokenization to decode to exactly the
saved prompt text. Equivalent token segmentations may differ slightly in token
count; `--max-token-drift` defaults to 4 and rejects larger differences.
It writes each full call under `calls/` and checkpoints `results.jsonl`,
`results.csv`, and `summary.json` after every completed or failed request.

## Replay verifier refinement

Use `evaluate_thinking_handoff_refinement.py` to test a saved parser-valid
restart without regenerating its long proof:

```bash
python evaluation/harness_vllm/evaluate_thinking_handoff_refinement.py \
  --logs-root evaluation/runs/imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717/logs \
  --restart-results /tmp/thinking-handoff-restart-passthrough-force100k-sft750-20260717/results.jsonl \
  --model-path /tmp/models/olmo3-opd-sft-750-vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir /tmp/thinking-handoff-refinement-replay
```

The replay runs the normal verifier, optional meta-verifier, proof refiner, and
final verifier over the saved proof. It writes the complete candidate record
to `result.json` and a compact call-count and score summary to `summary.json`.

Refinement can use the same lossless context-reset mechanism independently of
proof generation:

```bash
python evaluation/harness_vllm/evaluate_thinking_handoff_refinement.py \
  ... \
  --thinking-budget-refine-handoff-enabled \
  --thinking-budget-refine-tokens 50000 \
  --thinking-budget-refine-final-round-tokens 50000 \
  --thinking-budget-refine-max-restarts 1
```

The first refinement stops at the initial budget and writes a bounded,
untrusted partial-progress report. A fresh refinement receives the original
proof, the selected verifier critiques, and that report. Its final budget
forces the transition to XML while reserving the remaining completion tokens
for the repaired proof. The feature is disabled by default. Audit fields are
`proof_refine_attempt_output`, `proof_refine_handoff_output`,
`proof_refine_handoffs`, and `refine_budget_restart_count`.

To retest only the final repair from a saved refinement handoff, without
repeating the initial verification and first stopped refinement:

```bash
python evaluation/harness_vllm/evaluate_thinking_handoff_refinement.py \
  ... \
  --resume-refinement-result /tmp/previous-replay/result.json \
  --thinking-budget-refine-handoff-enabled \
  --thinking-budget-refine-final-round-tokens 20000
```

Resume mode rebuilds the original refinement prompt, attaches the saved
lossless handoff and selected validated critiques, and runs one final repair.
It invokes the verifier and meta-verifier only when that repair is
parser-valid, then keeps the higher-scoring verified round.

For refinement restarts, the selected critiques are also repeated after the
handoff as mandatory repair obligations. This keeps concrete verifier findings
near the generation boundary instead of burying them before a long transferred
research report. The model must repair each item or report it as unresolved;
the downstream verifier remains the authority on whether that instruction was
actually followed.
