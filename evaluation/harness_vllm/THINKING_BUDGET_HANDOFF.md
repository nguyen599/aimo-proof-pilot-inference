# Thinking-budget handoff

When proof generation reaches its thinking budget, `run.py` now uses the
remaining refine round as a context reset:

1. Stop the unfinished assistant reasoning without emitting a partial proof.
2. Close `</think>` and add a real user turn requesting a structured handoff.
3. Parse the handoff into established facts, promising work, failed routes,
   uncertain claims, the bottleneck, and next steps.
4. Start a fresh proof-generation call from the original problem plus the
   handoff.
5. Verify that restarted proof in the final round. If no refine round remains,
   keep the existing strongest-partial completion behavior.

An invalid handoff is repaired once. If it is still invalid, the pipeline falls
back to the previous strongest-partial completion path.

## Configuration

```bash
python evaluation/harness_vllm/run.py \
  --thinking-budget-handoff-enabled \
  --thinking-budget-handoff-max-tokens 4096 \
  --thinking-budget-handoff-temperature 0.7 \
  --thinking-budget-handoff-prompt-variant evidence_first
```

Prompt variants are `evidence_first`, `lemma_ledger`, and
`continuation_frontier`. Disable the feature with
`--no-thinking-budget-handoff-enabled`.

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
