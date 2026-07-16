# SFT-750 thinking-budget handoff experiments

## Goal

Increase the fraction of proof candidates that finish with a parseable final
proof after an initial proof-generation call reaches its thinking budget.
With one refine round configured, the new path summarizes the unfinished round
0 research, resets the context, and spends round 1 on a fresh proof attempt.
The final success metric is not handoff validity alone: round 1 should finish
before its own thinking cutoff and emit valid proof XML.

## Fixed runtime

- Target: `/tmp/models/olmo3-opd-sft-750-vllm`
- Draft: `/tmp/models/dflash-32b-draft-v2test-phaseL`
- Runtime: vLLM 0.25.1
- Per node: 8 H200, `TP=2`, `DP=4`
- Target quantization: online FP8
- KV cache: FP8
- DFlash speculative tokens: 10
- DFlash context cutoff: 65,536
- Model context: 262,144
- Saved source calls:
  `imo2025-full-p16-r1-p2-sft750-parserfix4-20260716T173047Z-partial-20260717`

## Experiment 0: local parser and control-flow validation

Commit `aa3c350` added the handoff/restart path, standalone optimizer, and
focused tests.

Results:

- Parsed all 17 checked-in proof calls containing the old forced-partial marker.
- Confirmed the marker is absent after reconstruction.
- Confirmed a budget hit with one round remaining calls:
  proof generation, handoff, fresh proof generation, then verification.
- Confirmed the final round retains the old strongest-partial fallback.
- Full repository suite: 112 tests passed, 36 subtests passed.

## Experiment 1: exact SFT-750 tokenizer reconstruction

The first NII tokenizer dry run failed before any GPU server was started:

```text
logged continuation tokens: 121992
re-encoded continuation tokens: 121991
```

Root cause: the saved-call parser used `rstrip()` on the decoded continuation
prompt. The real prompt ended with a significant newline from the forced text,
so reconstruction dropped one token. The parser was changed to remove exactly
the two debug-log section delimiters while preserving prompt trailing
whitespace. A regression test now covers this case.

After preserving whitespace, canonical SFT-750 re-tokenization still used
between zero and three fewer tokens across the 17 contexts. For every context,
decoding the canonical token IDs reproduced the saved prompt text exactly.
This is equivalent tokenizer segmentation rather than parser loss. The
optimizer now requires exact decoded-text equality and separately bounds the
absolute token-count drift to four.

## Experiment 2: handoff prompt and temperature sweep

Planned matrix:

- Eight balanced saved contexts: both ranks, both IMO problems, and both
  parseable/unparseable old forced-partial outcomes.
- Prompt variants: `evidence_first`, `lemma_ledger`,
  `continuation_frontier`.
- Temperatures: `1.0`, `0.7`, `0.6`.
- Total short handoff calls: 72.

Record XML validity, fidelity, mathematical information retained, failed-route
precision, bottleneck quality, next-step usefulness, completion tokens, and
latency. Results will be appended below.

## Experiment 3: fresh round-1 proof completion

After selecting the strongest handoff configuration, restart fresh proof
generation from the original problem plus handoff. Record:

- final `finish_reason`;
- whether the final thinking budget was reached;
- presence of `</think>`;
- parseable `<solution>`, `<self_evaluation>`, and `<score>`;
- proof character and token counts;
- whether verification can start.

This experiment determines whether the handoff improves proof completion rate,
rather than only producing cleaner summaries.
