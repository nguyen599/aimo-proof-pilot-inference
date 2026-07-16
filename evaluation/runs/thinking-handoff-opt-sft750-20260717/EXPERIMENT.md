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
- All 17 calls reached the round-0 unfinished-thinking cutoff. Only 6/17 old
  forced-partial continuations emitted structurally parseable final sections;
  this is a formatting baseline, not evidence of a complete proof.
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

### Server launch attempt 1

Both NII nodes used vLLM 0.25.1 with the fixed runtime above. Target and draft
weights loaded, but startup failed during CUDA-graph memory profiling:

```text
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455
'an illegal memory access was encountered'
```

Node 6 also reported one secondary Inductor autotuning failure while the DP
engines were aborting:

```text
Failed to run autotuning code block: CUDA driver error: file not found
```

No handoff requests were submitted. The retry preserves compiled execution,
CUDA graphs, `TP=2`, `DP=4`, FP8 target/KV, and DFlash, but adds
`--disable-custom-all-reduce` and uses new per-node compile-cache directories.

### Live smoke 1: unconstrained handoff length

The retry started successfully on both nodes. One `evidence_first`,
temperature-0.7 handoff was generated from a 122,227-token prompt:

```text
finish_reason=length
completion_tokens=4096
is_valid=false
missing_sections=established,promising,failed,uncertain,bottleneck,next_steps
```

This was not a turn-transition or prefix failure. The output correctly began
with `</think><handoff><established>`, but it used all 4,096 tokens in the first
section. It repeatedly restated the problem and used phrases such as "the
previous attempt attempted" instead of compressing the mathematical state.

The next prompt revision keeps 4,096 as a safety ceiling but requires:

- fewer than 1,200 words total;
- hard bullet limits for every list section;
- at most two sentences per bullet;
- one bottleneck paragraph of at most 120 words;
- no full problem restatement, duplicated facts, or attempt narration;
- all XML tags to be closed.

### Live smoke 2: strict single-call compression

The same source context was retried with the strict compression contract:

```text
prompt_tokens=122377
completion_tokens=4096
finish_reason=length
is_valid=false
```

The output was denser and dropped most attempt narration, but it again remained
inside `<established>` until the token cap. It expanded into small-case
calculations instead of obeying the bullet and section limits. Increasing the
single-call cap would preserve this failure mode and consume more context.

The optimizer is therefore being aligned with the production path in `run.py`:
after an invalid first handoff, it makes one repair turn over the exact prior
context and draft. The repair instruction requires the same mathematical
content to be re-emitted in all six XML sections. Results retain both attempts
and aggregate their token cost, while the repaired artifact is what the restart
evaluation consumes.

### Live smoke 3: production-equivalent repair

The repair turn did not recover the XML:

```text
initial: prompt=122377 completion=4096 finish=length valid=false
repair:  prompt=126579 completion=4096 finish=length valid=false
aggregate completion tokens=8192
```

Both calls remained inside `<established>`. The repair was therefore more
expensive than the original forced-partial fallback without creating a usable
restart artifact.

The next design generates each handoff section independently from the same
pre-force context. Each request has a focused extraction prompt and a hard
section-specific cap:

| Section | Max tokens |
| --- | ---: |
| `established` | 768 |
| `promising` | 640 |
| `failed` | 512 |
| `uncertain` | 384 |
| `bottleneck` | 256 |
| `next_steps` | 384 |

The harness strips any accidental section tag, fills only genuinely empty
sections with an explicit no-information sentence, and assembles the six
sections into valid XML. This bounds total generated handoff content at 2,944
tokens and prevents one category from consuming the whole budget. All section
requests retain the full original reasoning context; vLLM prefix caching can
reuse their common 122K-token prefix.

### Live smoke 4: bounded sections in the original conversation

The six calls completed in 37.2 seconds and produced structurally valid XML:

```text
aggregate prompt tokens=732745
aggregate completion tokens=2838
valid=true
```

This established that prefix caching makes six extraction calls practical, but
manual review rejected the handoff quality:

- five of six sections reached their token cap;
- `established`, `promising`, and `uncertain` restated the problem and mixed in
  new speculative solving;
- `failed` repeated generic failure statements;
- `bottleneck` echoed its extraction instruction;
- `next_steps` was generic and unfinished.

The likely cause is role framing. The 122K-token attempt remains the model's own
previous assistant turn, so even a new user instruction leaves a strong
continuation bias.

### Fresh-context section extraction

The next smoke reframes the attempt as quoted, untrusted research notes in a
new conversation. It:

1. extracts the original problem without the final-answer XML contract;
2. removes an exact consecutive repeated-token suffix when detected;
3. samples eight chronological 4,096-token reasoning windows, at most 32,768
   tokens total;
4. places the focused section instruction after those windows;
5. starts a fresh assistant turn with the section tag already opened.

This reduces each extraction prompt from roughly 122K to roughly 33K tokens,
keeps early, middle, and late research visible, and moves the current
instruction into a clean role hierarchy.

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

Implementation:

- `evaluate_thinking_handoff_restart.py` consumes one selected
  prompt/temperature group from the short handoff sweep.
- It inserts the handoff into the original rendered problem prompt before the
  assistant generation marker, then starts a fresh round-1 completion.
- It stops an unfinished reasoning stream at 122,000 tokens without applying
  the old strongest-partial force text. If `</think>` was genuinely emitted
  before that boundary, it lets the visible proof continue up to 126,000 total
  completion tokens.
- It records the raw prompt, handoff, output, finish reason, real cutoff state,
  parseable proof state, proof length, latency, and throughput for every case.
- The focused handoff suite passes with 11 tests after adding rendered-prompt
  insertion coverage.
